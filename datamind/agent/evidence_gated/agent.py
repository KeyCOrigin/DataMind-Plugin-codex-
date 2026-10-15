"""Retrieve → frozen candidate → atoms → audit → exact candidate publication."""
from __future__ import annotations
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from datamind.agent.base import AgentEvent
from datamind.core.context import RequestContext
from datamind.core.logging import current_context, bind_context
from .evidence import canonical, subject, digest, candidate_binding_errors
from .provenance import ProvenanceValidator, structure_errors, dependency_order, contract_errors
from .generation_common import ContextBudgetExceeded, GenerationError
from .atom_builder import assemble_draft
from .schemas import AuditReport, AtomAudit, QueryResult
from .gate import Gate
from .judge import audit_context, audited_atom

class EvidenceGatedAgent:
    def __init__(self, *, bridge, answer_generator, atom_builder, judge, profile="default", max_atoms=24,
                 total_timeout_s=300.0, audit_path: Path | None = None,
                 min_support_ratio=0.8, require_final_support=False):
        self.bridge, self.answer_generator, self.atom_builder, self.judge = bridge, answer_generator, atom_builder, judge
        self.profile, self.max_atoms, self.total_timeout_s = profile, max_atoms, total_timeout_s
        self.provenance = ProvenanceValidator()
        self.gate = Gate(support_threshold=judge.support_threshold, refute_threshold=judge.refute_threshold,
            min_support_ratio=min_support_ratio, require_final_support=require_final_support, max_atoms=max_atoms)
        self.audit_path = Path(audit_path) if audit_path else None
        self._log_lock = asyncio.Lock()

    async def _execute(self, message, history, context, contract, state, emit):
        async def phase(name, data=None):
            if emit:
                await emit(AgentEvent(name, data or {}))
        await phase("retrieving")
        start = perf_counter()
        group = await self.bridge.collect(question=message, context=context, answer_contract=contract,
            history=history, request_id=state["request_id"], state=state)
        state["group"] = group
        state["timings"]["retrieval"] = (perf_counter()-start)*1000
        evidence_hash = digest(group)
        state["evidence_group_sha256"] = evidence_hash
        if not group.evidence:
            return "insufficient", ["NO_EVIDENCE"]
        await phase("answer_generation")
        state["stage"] = "answer_generation"
        start = perf_counter()
        usage_start = len(current_context().extra.get("nested_model_usage", []))
        try:
            candidate = await self.answer_generator.generate(group, state=state)
        finally:
            state["answer_generation_usage"] = deepcopy(current_context().extra.get("nested_model_usage", [])[usage_start:])
            state["timings"]["answer_generation"] = (perf_counter()-start)*1000
        state["candidate"] = candidate
        expected_candidate_hash = candidate.candidate_sha256
        state["candidate_snapshot_sha256"] = expected_candidate_hash
        state["candidate_snapshot"] = candidate.model_dump(mode="json")
        def guard_candidate():
            if digest(group) != evidence_hash:
                return ["EVIDENCE_GROUP_CHANGED"]
            return candidate_binding_errors(group, candidate, expected_candidate_hash)
        reasons = guard_candidate()
        if reasons:
            return "insufficient", reasons
        if candidate.status == "insufficient":
            return "insufficient", ["INSUFFICIENT_CANDIDATE"]
        reasons = contract_errors(candidate.answer_value, group.answer_contract)
        if reasons:
            return "insufficient", reasons
        await phase("atom_building")
        state["stage"] = "atom_building"
        start = perf_counter()
        usage_start = len(current_context().extra.get("nested_model_usage", []))
        try:
            # The builder receives copies, never the live candidate snapshot.
            output = await self.atom_builder.build(group.model_copy(deep=True), candidate.model_copy(deep=True), state=state)
        finally:
            state["atom_building_usage"] = deepcopy(current_context().extra.get("nested_model_usage", [])[usage_start:])
            state["timings"]["atom_building"] = (perf_counter()-start)*1000
        state["atom_build"] = output
        reasons = guard_candidate()
        if reasons:
            return "insufficient", reasons
        draft = assemble_draft(group, candidate, output)
        state["draft"] = draft
        if draft.status == "insufficient":
            return "insufficient", ["INSUFFICIENT_ATOM_BUILD"]
        state["stage"] = "citation_check"
        reasons = structure_errors(draft, group, candidate, self.max_atoms)
        if reasons:
            return "insufficient", reasons
        bound_subject = subject(group, candidate, draft)
        state["audit_subject_sha256"] = bound_subject
        await phase("citation_check")
        start = perf_counter()
        checks = self.provenance.validate(draft, group)
        state["checks"] = checks
        state["timings"]["citation_check"] = (perf_counter()-start)*1000
        if not all(c.valid for c in checks):
            state["reports"] = [AtomAudit(atom_id=c.atom_id, execution_status="invalid",
                support_threshold=self.judge.support_threshold, refute_threshold=self.judge.refute_threshold,
                reason_codes=c.reason_codes or ["CITATION_STAGE_FAILED"]) for c in checks]
            return "insufficient", list(dict.fromkeys(code for c in checks for code in c.reason_codes))
        lookup = {a.atom_id: a for a in draft.atoms}
        references = {c.atom_id: c.resolved_evidence for c in checks}
        reports = {}
        def closure(atom):
            return audit_context(atom, lookup, reports, references)
        for atom in dependency_order(draft):
            if atom.kind == "answer":
                continue
            state["stage"] = "atom_audit"
            await phase("atom_audit", {"atom_id": atom.atom_id, "kind": atom.kind})
            report = await self.judge.score(group=group, atom=atom, citations=references[atom.atom_id],
                dependencies=closure(atom), all_reports=[])
            reports[atom.atom_id] = report
            state["reports"] = list(reports.values())
            await phase("atom_audit_result", report.model_dump(mode="json"))
        final = lookup[draft.final_atom_id]
        state["stage"] = "final_audit"
        await phase("atom_audit", {"atom_id": final.atom_id, "kind": "answer"})
        prior_audits = [audited_atom(key, lookup, reports, references) for key in reports]
        report = await self.judge.score(group=group, atom=final, citations=references[final.atom_id],
                    dependencies=closure(final), all_reports=prior_audits, final=True)
        reports[final.atom_id] = report
        state["reports"] = list(reports.values())
        await phase("atom_audit_result", report.model_dump(mode="json"))
        audit = AuditReport(subject_sha256=bound_subject, citation_checks=checks,
            atom_reports=list(reports.values()), timings_ms=state["timings"])
        state["audit"] = audit
        reasons = guard_candidate()
        if reasons:
            return "insufficient", reasons
        state["stage"] = "gate"
        await phase("gate")
        gate = self.gate.decide(group, candidate, draft, audit)
        state["gate"] = gate.model_dump(mode="json")
        await phase("gate_result", state["gate"])
        if not gate.allow:
            return "error" if any(r.execution_status == "error" for r in reports.values()) else "insufficient", gate.reason_codes
        reasons = guard_candidate()
        if reasons:
            return "insufficient", reasons
        # Publish the first call's value; the builder has no answer write path.
        state["published"] = deepcopy(candidate.answer_value)
        return "answer", []

    async def _query(self, message, *, history=None, context=None, final_contract=None, emit=None):
        runtime = current_context()
        explicit = deepcopy(context or {})
        if runtime:
            # Only explicitly named runtime context, never bench/gold attachments.
            inherited = runtime.extra.get("evidence_context", {})
            explicit = {**deepcopy(inherited), **explicit}
        forbidden = {"gold", "gold_answer", "required_evidence_nodes", "acceptable_evidence_sets", "supported_label"}
        if forbidden.intersection(explicit):
            raise ValueError("Evaluation labels are not request context")
        explicit.setdefault("profile", runtime.profile if runtime else self.profile)
        state = {"request_id": uuid4().hex, "question": message, "context": explicit,
                 "answer_contract": deepcopy(final_contract or {}), "timings": {}, "checks": [], "reports": [],
                 "generation_calls": 0, "stage": "retrieval", "protocol_version": "abc-v1"}
        start = perf_counter()
        decision, reasons = "error", ["INTERNAL_ERROR"]
        try:
            request_context = (RequestContext(session_id=runtime.session_id, profile=runtime.profile,
                user_id=runtime.user_id, trace_id=runtime.trace_id,
                extra={**deepcopy(runtime.extra), "nested_model_usage": []}) if runtime
                else RequestContext.new(profile=self.profile))
            with bind_context(request_context):
                async with asyncio.timeout(self.total_timeout_s):
                    decision, reasons = await self._execute(message, history, explicit, state["answer_contract"], state, emit)
        except asyncio.CancelledError:
            state["decision"], state["reason_codes"] = "error", ["CANCELLED"]
            await asyncio.shield(self._persist(state))
            raise
        except TimeoutError:
            reasons = ["TOTAL_TIMEOUT"]
        except ContextBudgetExceeded:
            reasons = ["CONTEXT_BUDGET_EXCEEDED"]
        except GenerationError as exc:
            reasons = [exc.code]
            state["exception"] = f"{type(exc).__name__}: {exc}"
        except (ValueError, TypeError, KeyError) as exc:
            reasons = [{"answer_generation": "INVALID_CANDIDATE", "atom_building": "INVALID_ATOM_BUILD"}.get(state["stage"], "INVALID_DRAFT")]
            state["exception"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            state["exception"] = f"{type(exc).__name__}: {exc}"
        state["timings"]["total"] = (perf_counter()-start)*1000
        group, candidate, draft = state.get("group"), state.get("candidate"), state.get("draft")
        audit = state.get("audit") or AuditReport(subject_sha256=state.get("audit_subject_sha256") or (subject(group, candidate, draft) if group else ""),
            citation_checks=state["checks"], atom_reports=state["reports"], reason_codes=reasons, timings_ms=state["timings"])
        audit.timings_ms = deepcopy(state["timings"])
        # Check the bound subject again immediately before constructing publication.
        if decision == "answer" and (audit.subject_sha256 != subject(group, candidate, draft)
                or candidate_binding_errors(group, candidate, state["candidate_snapshot_sha256"])
                or canonical(state.get("published")) != canonical(candidate.answer_value)):
            decision, reasons = "insufficient", ["AUDIT_SUBJECT_MISMATCH"]
            state.pop("published", None)
        if decision != "answer":
            audit.reason_codes = list(dict.fromkeys(audit.reason_codes + reasons))
            prior_reasons = state.get("gate", {}).get("reason_codes", [])
            state["gate"] = {**state.get("gate", {}), "allow": False, "reason_codes": list(dict.fromkeys(prior_reasons + reasons))}
        result = QueryResult(decision=decision, answer=state.get("published") if decision == "answer" else None,
            candidate_id=candidate.candidate_id if candidate else None,
            draft_id=draft.draft_id if draft else None, atoms=draft.atoms if draft else [], audit=audit,
            evidence=group.evidence if group else [], reason_codes=reasons).model_dump(mode="json")
        state.update(decision=decision, reason_codes=reasons, audit=audit, result=result)
        await self._persist(state)
        answer_text = serialize(result["answer"]) if decision == "answer" else ""
        result.update(answer_text=answer_text, history=list(history or []) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer_text if decision == "answer" else "执行失败，未发布答案。" if decision == "error" else "证据不足，未发布答案。"}],
            iterations=state.get("retrieval_iterations", 0) + state["generation_calls"] + len(state["reports"]), stop_reason="evidence_gate_allowed" if decision == "answer" else "evidence_gate_error" if decision == "error" else "evidence_gate_rejected",
            usage={"retrieval": state.get("retrieval_usage", {}),
                "answer_generation": state.get("answer_generation_usage", []), "atom_building": state.get("atom_building_usage", []),
                "reasoning": state.get("answer_generation_usage", []) + state.get("atom_building_usage", []),
                "scoring": [r.scores.usage for r in state["reports"] if r.scores]},
            tool_trace=[c.model_dump(mode="json") for c in group.tool_trace] if group else [],
            surfaces_used=sorted({e.surface for e in group.evidence}) if group else [],
            gate=state.get("gate", {"allow": False, "reason_codes": reasons}))
        return result

    async def _persist(self, state):
        if self.audit_path is None:
            return
        data = {k: v.model_dump(mode="json") if hasattr(v, "model_dump") else v for k, v in state.items()}
        for key in ("checks", "reports"):
            data[key] = [x.model_dump(mode="json") for x in state[key]]
        line = canonical(data) + "\n"
        async with self._log_lock:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    async def query(self, message, *, history=None, final_contract=None, context=None):
        return await self._query(message, history=history, final_contract=final_contract, context=context)

    async def stream_query(self, *, user_message, history=None, final_contract=None, context=None):
        queue = asyncio.Queue()
        async def run():
            try:
                result = await self._query(user_message, history=history, final_contract=final_contract,
                                           context=context, emit=queue.put)
                if result["decision"] == "answer":
                    await queue.put(AgentEvent("text", {"delta": result["answer_text"]}))
                # Rejected drafts stay in the audit log, not in the stream.
                streamed = result if result["decision"] == "answer" else {**result, "atoms": []}
                await queue.put(AgentEvent("done", streamed))
            except Exception as exc:
                await queue.put(AgentEvent("error", {"message": type(exc).__name__}))
                await queue.put(AgentEvent("done", {"decision": "error", "answer": None}))
            finally:
                await queue.put(None)
        task = asyncio.create_task(run())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

def serialize(value, contract=None):
    # Preserve string bytes and JSON scalar/container types; no semantic changes.
    return value if isinstance(value, str) else canonical(value)
