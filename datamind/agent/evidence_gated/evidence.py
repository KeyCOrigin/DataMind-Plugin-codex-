"""Server-owned registry, exact selectors, and scoped import identities."""
from __future__ import annotations
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope
from .schemas import AnswerCandidate, EvidenceGroup, EvidenceRecord, ReasoningDraft, ToolRecord

def canonical(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()

def candidate_digest(group: EvidenceGroup, candidate: AnswerCandidate) -> str:
    return digest({"group": group.model_dump(mode="json"),
        "candidate": candidate.model_dump(mode="json", exclude={"candidate_sha256"})})

def candidate_binding_errors(group, candidate, expected_sha256=None):
    reasons = []
    if (not candidate.candidate_id or candidate.request_id != group.request_id
            or candidate.group_id != group.group_id):
        reasons.append("CANDIDATE_BOUND_MISMATCH")
    if (not candidate.candidate_sha256 or candidate.candidate_sha256 != candidate_digest(group, candidate)
            or (expected_sha256 is not None and candidate.candidate_sha256 != expected_sha256)):
        reasons.append("CANDIDATE_CHANGED")
    return reasons

def subject(group: EvidenceGroup, candidate: AnswerCandidate | None, draft: ReasoningDraft | None) -> str:
    return digest({"group": group.model_dump(mode="json"),
        "candidate": candidate.model_dump(mode="json") if candidate else None,
        "draft": draft.model_dump(mode="json") if draft else None})

def locators(value: Any, pointer: str = "") -> list[dict]:
    result = [{"kind": "json_pointer", "pointer": pointer}]
    if isinstance(value, dict):
        for key, child in value.items():
            result.extend(locators(child, pointer + "/" + key.replace("~", "~0").replace("/", "~1")))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            result.extend(locators(child, pointer + f"/{i}"))
    return result

def resolve(payload: Any, selector: dict) -> Any:
    kind = selector.get("kind")
    if kind == "json_pointer":
        if set(selector) != {"kind", "pointer"}:
            raise ValueError("INVALID_SELECTOR")
        pointer = selector["pointer"]
        if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
            raise ValueError("INVALID_SELECTOR")
        value = payload
        if pointer:
            for token in pointer[1:].split("/"):
                # RFC 6901: malformed escapes are invalid.
                import re
                if re.search(r"~(?![01])", token):
                    raise ValueError("INVALID_SELECTOR")
                key = token.replace("~1", "/").replace("~0", "~")
                if isinstance(value, list):
                    if not key.isascii() or not key.isdigit() or (len(key) > 1 and key[0] == "0"):
                        raise ValueError("INVALID_SELECTOR")
                    value = value[int(key)]
                elif isinstance(value, dict):
                    value = value[key]
                else:
                    raise ValueError("INVALID_SELECTOR")
        return value
    if kind == "text_span":
        allowed = {"kind", "pointer", "start", "end", "occurrence"}
        if not set(selector).issubset(allowed) or "start" not in selector or "end" not in selector:
            raise ValueError("INVALID_SELECTOR")
        if "occurrence" in selector and type(selector["occurrence"]) is not int:
            raise ValueError("INVALID_SELECTOR")
        text = resolve(payload, {"kind": "json_pointer", "pointer": selector.get("pointer", "")})
        start, end = selector.get("start"), selector.get("end")
        if not isinstance(text, str) or type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            raise ValueError("INVALID_SELECTOR")
        return text[start:end]  # Python Unicode code points, end-exclusive.
    raise ValueError("INVALID_SELECTOR")


def _str_occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    """All non-overlapping exact occurrences of ``needle`` in ``text``.

    Returns ``(start, end)`` pairs with Python Unicode code-point offsets and
    an end-exclusive ``end``. Overlap is not allowed so the list is
    deterministic and each span identifies one occurrence unambiguously.
    """
    if not needle:
        return []
    occurrences: list[tuple[int, int]] = []
    cursor = 0
    width = len(needle)
    while True:
        idx = text.find(needle, cursor)
        if idx == -1:
            break
        occurrences.append((idx, idx + width))
        cursor = idx + width
    return occurrences


def resolve_text_span(
    payload: Any,
    pointer: str,
    quote: str,
    occurrence: int | None = None,
) -> dict:
    """Anchor an exact substring to a concrete, traceable ``text_span``.

    The server computes the real start/end offsets; the model never needs to
    count characters. Matching is exact substring search only — never fuzzy
    similarity. Multiple exact occurrences must be disambiguated with a
    zero-based ``occurrence`` index.

    Raises ``ValueError`` whose message is a stable reason code:
      NO_EXACT_MATCH       — quote absent from the pointed text
      AMBIGUOUS_TEXT_SPAN  — several occurrences and no occurrence given
      INVALID_OCCURRENCE   — occurrence not an in-range integer
      INVALID_SELECTOR     — pointer does not resolve to a string
    """
    text = resolve(payload, {"kind": "json_pointer", "pointer": pointer})
    if not isinstance(text, str):
        raise ValueError("INVALID_SELECTOR")
    matches = _str_occurrences(text, quote)
    if not matches:
        raise ValueError("NO_EXACT_MATCH")
    if occurrence is None:
        if len(matches) != 1:
            raise ValueError("AMBIGUOUS_TEXT_SPAN")
        occurrence = 0
    if type(occurrence) is not int or not 0 <= occurrence < len(matches):
        raise ValueError("INVALID_OCCURRENCE")
    start, end = matches[occurrence]
    return {
        "kind": "text_span",
        "pointer": pointer,
        "start": start,
        "end": end,
        "occurrence": occurrence,
    }

def _tables(call: ToolRecord) -> set[str]:
    result = call.result if isinstance(call.result, dict) else {}
    sql = result.get("sql") or (call.args.get("sql") if call.tool_name == "db_query_sql" else None)
    if sql:
        try:
            tables = set()
            for tree in sqlglot.parse(sql):
                for scope in traverse_scope(tree):
                    for source in scope.sources.values():
                        if isinstance(source, exp.Table):
                            tables.add(".".join(p for p in (source.catalog, source.db, source.name) if p))
            return tables
        except sqlglot.errors.ParseError:
            return set()
    # Search hints such as NL2SQL's args.tables are not executed table identities.
    table = result.get("table") or (call.args.get("table") if call.tool_name == "db_describe" else None)
    return {table} if isinstance(table, str) else set()

class SourceMappingResolver:
    def __init__(self, path: Path | None = None):
        self.path = path

    def matching(self, tables: set[str], profile: str | None) -> list[dict]:
        if not tables or self.path is None or not self.path.is_file():
            return []
        matches = {}
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                receipt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if profile is not None and receipt.get("profile") != profile:
                continue
            def walk(value, file=None, pointer=""):
                if isinstance(value, dict):
                    file = value.get("source_file", file)
                    table = value.get("table")
                    if table in tables and not value.get("error"):
                        matches[table] = {"table": table, "file": file,
                            "sheet": value.get("sheet"), "details": value,
                            "source": receipt.get("source"), "revision": receipt.get("revision"),
                            "receipt_id": receipt.get("receipt_id"), "receipt_line": line_number,
                            "receipt_pointer": pointer, "receipt_sha256": digest(receipt)}
                    for key, child in value.items():
                        walk(child, file, pointer + "/" + str(key))
                elif isinstance(value, list):
                    for i, child in enumerate(value):
                        walk(child, file, pointer + f"/{i}")
            for index, outcome in enumerate(receipt.get("results", [])):
                if outcome.get("surface") == "db" and outcome.get("status") in {"stored", "unchanged"}:
                    walk(outcome.get("details", {}), (receipt.get("source") or {}).get("uri"), f"/results/{index}/details")
        return list(matches.values())

class EvidenceRegistry:
    def __init__(self, resolver: SourceMappingResolver | None = None):
        self.resolver = resolver or SourceMappingResolver()

    def register(self, trace: list[ToolRecord], context: dict) -> list[EvidenceRecord]:
        records, seen = [], set()
        for call in trace:
            payload = call.result
            data = payload if isinstance(payload, dict) else {}
            error = call.error or (str(data["error"]) if data.get("error") else None)
            if data.get("status") in {"denied", "ask_user", "failed"} or data.get("is_error") is True or data.get("kind") in {"denied", "asks_user"}:
                error = error or "TOOL_FAILED"
            bindings = {k: data[k] for k in ("table", "sheet", "source_file", "column_map", "orig", "source_id", "edge_id") if k in data}
            tables = _tables(call)
            if tables:
                bindings["queried_tables"] = sorted(tables)
            mappings = self.resolver.matching(tables, context.get("profile"))
            if mappings:
                bindings["import_mappings"] = mappings
            fingerprint = digest({"tool": call.tool_name, "args": call.args, "payload": payload, "bindings": bindings, "error": error})
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            partial = data.get("truncated") is True or data.get("has_more") is True
            complete = data.get("truncated") is False or data.get("has_more") is False or data.get("completeness") == "complete"
            records.append(EvidenceRecord(evidence_id="ev_" + call.call_id, call_id=call.call_id,
                surface=call.tool_name.split("_", 1)[0], tool_name=call.tool_name, tool_args=call.args,
                raw_payload=payload, payload_sha256=digest(payload), source_bindings=bindings,
                available_locators=locators(payload), completeness="partial" if partial else "complete" if complete else "unknown",
                source_id=data.get("source_id"), source_version=str(data["source_version"]) if data.get("source_version") is not None else None,
                workspace_revision=str(data["workspace_revision"]) if data.get("workspace_revision") is not None else None,
                surface_version=str(data["surface_version"]) if data.get("surface_version") is not None else None, error=error))
        return records


def locator_context(payload, selector):
    """Keep actual ancestor identities when citing a cell, chunk text, or edge."""
    pointer = selector.get("pointer", "")
    keys = {"id", "source_id", "source", "target", "subject", "object", "relation",
            "from", "to", "rel", "head", "tail",
            "edge_id", "chunk_id", "source_file", "file", "file_name", "path", "uri",
            "metadata", "properties", "page", "page_number", "sheet", "table", "name",
            "columns", "column_map", "orig", "source_version", "workspace_revision", "surface_version"}
    ancestors = []
    current = payload
    prefix = ""
    parts = pointer[1:].split("/") if pointer else []
    for part in parts + [None]:
        if isinstance(current, dict):
            identity = {k: v for k, v in current.items() if k in keys}
            if identity:
                ancestors.append({"pointer": prefix, "identity": identity})
        if part is None:
            break
        prefix += "/" + part
        current = resolve(payload, {"kind": "json_pointer", "pointer": prefix})
    return ancestors


def record_context(payload, selector):
    """Keep the cited list item, or enclosing object, without mixing sibling records.

    Choosing a record by structure preserves unfamiliar field names too. Direct
    record citations already carry that record in content and need no duplicate.
    """
    pointer = selector.get("pointer", "")
    current, prefix = payload, ""
    enclosing = ("", payload) if isinstance(payload, dict) else None
    record = None
    for part in (pointer[1:].split("/") if pointer else []):
        parent = current
        prefix += "/" + part
        current = resolve(payload, {"kind": "json_pointer", "pointer": prefix})
        if isinstance(parent, list) and isinstance(current, (dict, list)):
            record = (prefix, current)
        if isinstance(current, dict):
            enclosing = (prefix, current)
    selected = record or enclosing
    if selected is None:
        return None
    context_pointer, value = selected
    if context_pointer == pointer and selector.get("kind") == "json_pointer":
        return None
    return {"pointer": context_pointer, "content": deepcopy(value)}
