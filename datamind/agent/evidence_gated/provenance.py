"""Deterministic authenticity and structure checks, never semantic scoring."""
from copy import deepcopy
from jsonschema import Draft202012Validator
from .evidence import (canonical, digest, resolve, resolve_text_span,
    locator_context, record_context, candidate_binding_errors)
from .schemas import CitationCheck, ResolvedEvidence

def contract_errors(value, contract):
    if value is None and not contract.get("allow_null", False):
        return ["UNKNOWN_ANSWER"]
    if value == "" and not contract.get("allow_empty_string", False):
        return ["EMPTY_ANSWER"]
    schema = contract.get("json_schema") or ({"type": contract["type"]} if contract.get("type") else None)
    if schema is not None:
        try:
            Draft202012Validator.check_schema(schema)
            if list(Draft202012Validator(schema).iter_errors(value)):
                return ["CONTRACT_MISMATCH"]
        except Exception:
            return ["INVALID_CONTRACT"]
    return []

def dependency_order(draft):
    lookup = {a.atom_id: a for a in draft.atoms}
    visiting, visited, ordered = set(), set(), []
    def visit(atom):
        if atom.atom_id in visiting:
            raise ValueError("CYCLIC_DEPENDENCY")
        if atom.atom_id in visited:
            return
        visiting.add(atom.atom_id)
        for key in atom.depends_on:
            if key not in lookup:
                raise ValueError("INVALID_DEPENDENCY")
            visit(lookup[key])
        visiting.remove(atom.atom_id)
        visited.add(atom.atom_id)
        ordered.append(atom)
    for atom in draft.atoms:
        visit(atom)
    return ordered

def structure_errors(draft, group, candidate, max_atoms=24):
    reasons = candidate_binding_errors(group, candidate)
    if (draft.candidate_id != candidate.candidate_id
            or draft.candidate_sha256 != candidate.candidate_sha256):
        reasons.append("CANDIDATE_DRAFT_MISMATCH")
    if candidate.status != "candidate" or candidate.gaps or draft.gaps:
        reasons.append("UNRESOLVED_GAPS")
    if canonical(draft.answer_value) != canonical(candidate.answer_value):
        reasons.append("ANSWER_BINDING_MISMATCH")
    if draft.group_id != group.group_id or not draft.draft_id:
        reasons.append("DRAFT_BOUND_MISMATCH")
    if not draft.atoms:
        reasons.append("EMPTY_ATOMS")
    if len(draft.atoms) > max_atoms:
        reasons.append("ATOM_BUDGET_EXCEEDED")
    ids = [a.atom_id for a in draft.atoms]
    if len(set(ids)) != len(ids) or any(not x for x in ids):
        reasons.append("INVALID_ATOM_ID")
    finals = [a for a in draft.atoms if a.kind == "answer"]
    if len(finals) != 1 or finals[0].atom_id != draft.final_atom_id:
        reasons.append("INVALID_FINAL_ATOM")
    elif canonical(finals[0].value) != canonical(draft.answer_value):
        reasons.append("ANSWER_BINDING_MISMATCH")
    for atom in draft.atoms:
        if not atom.statement.strip():
            reasons.append("EMPTY_STATEMENT")
        if atom.kind != "answer" and draft.final_atom_id in atom.depends_on:
            reasons.append("INVALID_DEPENDENCY")
        if not atom.citations:
            reasons.append("NO_CITATION")
    statements = [" ".join(a.statement.split()).casefold() for a in draft.atoms]
    if len(set(statements)) != len(statements):
        reasons.append("DUPLICATE_ATOM_CLAIM")
    if draft.final_atom_id in ids and len(set(ids)) == len(ids):
        lookup = {a.atom_id: a for a in draft.atoms}
        reachable = set()
        def visit(key):
            if key in reachable or key not in lookup:
                return
            reachable.add(key)
            for dep in lookup[key].depends_on:
                visit(dep)
        visit(draft.final_atom_id)
        if reachable != set(ids):
            reasons.append("UNRELATED_ATOM")
    try:
        dependency_order(draft)
    except ValueError as exc:
        reasons.append(str(exc))
    reasons.extend(contract_errors(draft.answer_value, group.answer_contract))
    return list(dict.fromkeys(reasons))

class ProvenanceValidator:
    def validate(self, draft, group):
        registry = {e.evidence_id: e for e in group.evidence}
        trace = {c.call_id: c for c in group.tool_trace}
        checks = []
        duplicate = len(registry) != len(group.evidence) or len(trace) != len(group.tool_trace)
        for atom in draft.atoms:
            reasons, resolved = [], []
            if duplicate:
                reasons.append("DUPLICATE_EVIDENCE_ID")
            if not atom.citations:
                reasons.append("NO_CITATION")
            for citation in atom.citations:
                ev = registry.get(citation.evidence_id)
                if ev is None:
                    reasons.append("UNKNOWN_EVIDENCE")
                    continue
                call = trace.get(ev.call_id)
                if call is None or call.tool_name != ev.tool_name or canonical(call.args) != canonical(ev.tool_args):
                    reasons.append("SOURCE_ID_MISMATCH")
                    continue
                if digest(ev.raw_payload) != ev.payload_sha256 or digest(call.result) != ev.payload_sha256:
                    reasons.append("PAYLOAD_HASH_MISMATCH")
                    continue
                if ev.error or call.error:
                    reasons.append("TOOL_FAILED")
                    continue
                raw = ev.raw_payload if isinstance(ev.raw_payload, dict) else {}
                if ev.source_id != raw.get("source_id"):
                    reasons.append("SOURCE_ID_MISMATCH")
                for field in ("source_version", "workspace_revision", "surface_version"):
                    registered = str(raw[field]) if raw.get(field) is not None else None
                    if getattr(ev, field) != registered:
                        reasons.append("VERSION_MISMATCH")
                for field in ("table", "sheet", "source_file", "column_map", "orig", "source_id", "edge_id"):
                    if field in raw and canonical(ev.source_bindings.get(field)) != canonical(raw[field]):
                        reasons.append("SOURCE_ID_MISMATCH")
                context = group.context
                expected = context.get("workspace_revision")
                if expected is not None and str(expected) != ev.workspace_revision:
                    reasons.append("VERSION_MISMATCH")
                expected = context.get("source_versions", {}).get(ev.source_id)
                if expected is not None and str(expected) != ev.source_version:
                    reasons.append("VERSION_MISMATCH")
                expected = context.get("surface_versions", {}).get(ev.surface)
                if expected is not None and str(expected) != ev.surface_version:
                    reasons.append("VERSION_MISMATCH")
                if context.get("require_complete") is True and ev.completeness != "complete":
                    reasons.append("COMPLETENESS_UNVERIFIED")
                effective_selector = dict(citation.selector)
                # Server-side anchoring: when a quote is supplied and the
                # selector points at a string, derive the concrete text_span
                # and record the real start/end offsets. Exact substring only;
                # no fuzzy similarity fallback.
                if citation.quote is not None and citation.selector.get("kind") in ("json_pointer", "text_span"):
                    try:
                        target = resolve(ev.raw_payload, {
                            "kind": "json_pointer",
                            "pointer": citation.selector.get("pointer", ""),
                        })
                    except (ValueError, KeyError, IndexError, TypeError):
                        target = None
                    if isinstance(target, str):
                        try:
                            effective_selector = resolve_text_span(
                                ev.raw_payload,
                                citation.selector.get("pointer", ""),
                                citation.quote,
                                occurrence=citation.selector.get("occurrence"),
                            )
                        except ValueError as exc:
                            reasons.append(str(exc))
                            continue
                try:
                    content = resolve(ev.raw_payload, effective_selector)
                except (ValueError, KeyError, IndexError, TypeError):
                    reasons.append("INVALID_SELECTOR")
                    continue
                scoped = locator_context(ev.raw_payload, effective_selector)
                for ancestor in scoped:
                    identity = ancestor["identity"]
                    declared = [identity]
                    if isinstance(identity.get("metadata"), dict):
                        declared.append(identity["metadata"])
                    for item in declared:
                        expected = context.get("source_versions", {}).get(item.get("source_id"))
                        actual = item.get("source_version")
                        if expected is not None and (actual is None or str(actual) != str(expected)):
                            reasons.append("VERSION_MISMATCH")
                if context.get("source_version") is not None and str(context["source_version"]) != ev.source_version:
                    reasons.append("VERSION_MISMATCH")
                if citation.quote is not None and citation.quote != (content if isinstance(content, str) else canonical(content)):
                    reasons.append("QUOTE_MISMATCH")
                    continue
                resolved.append(ResolvedEvidence(evidence_id=ev.evidence_id, call_id=ev.call_id,
                    selector=effective_selector, content=content, source_bindings={**ev.source_bindings,
                    "locator_context": scoped},
                    record_context=record_context(ev.raw_payload, effective_selector),
                    tool_context={"tool_name": call.tool_name, "tool_args": deepcopy(call.args)},
                    completeness=ev.completeness, versions={"source_id": ev.source_id,
                    "source_version": ev.source_version, "workspace_revision": ev.workspace_revision,
                    "surface_version": ev.surface_version}))
            checks.append(CitationCheck(atom_id=atom.atom_id, valid=not reasons,
                                       resolved_evidence=resolved, reason_codes=list(dict.fromkeys(reasons))))
        return checks
