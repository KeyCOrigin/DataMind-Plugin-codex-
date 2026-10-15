"""Release the fixed audited subject by support ratio and zero refutations."""
import math
from .evidence import subject, canonical, digest
from .provenance import structure_errors, ProvenanceValidator, dependency_order
from .judge import audit_context, audited_atom, build_audit_prompt
from .prompts import TEMPLATE_VERSION
from .scoring import validate_thresholds
from .schemas import GateDecision, AtomAudit

class Gate:
    def __init__(self, *, support_threshold=0.3, refute_threshold=-0.3,
                 min_support_ratio=0.8, require_final_support=False, max_atoms=24):
        validate_thresholds(support_threshold, refute_threshold)
        if not math.isfinite(min_support_ratio) or not 0 < min_support_ratio <= 1:
            raise ValueError("Require 0 < min_support_ratio <= 1")
        self.support_threshold, self.refute_threshold = support_threshold, refute_threshold
        self.min_support_ratio, self.require_final_support = min_support_ratio, require_final_support
        self.max_atoms = max_atoms

    def decide(self, group, candidate, draft, audit):
        reasons = structure_errors(draft, group, candidate, self.max_atoms)
        if draft.status != "candidate":
            reasons.append("INSUFFICIENT_DRAFT")
        if audit.subject_sha256 != subject(group, candidate, draft):
            reasons.append("AUDIT_SUBJECT_MISMATCH")
        reasons.extend(audit.reason_codes)
        ids = {a.atom_id for a in draft.atoms}
        refs = {r.atom_id: r for r in audit.citation_checks}
        reports = {r.atom_id: r for r in audit.atom_reports}
        if set(refs) != ids or len(refs) != len(audit.citation_checks) or not all(r.valid for r in refs.values()):
            reasons.append("CITATION_CHECK_FAILED")
        # Re-resolve original payloads, rather than trusting a supplied valid flag/content.
        actual_checks = ProvenanceValidator().validate(draft, group)
        actual_refs = {r.atom_id: r for r in actual_checks}
        if (canonical({k: r.model_dump(mode="json") for k, r in refs.items()})
                != canonical({k: r.model_dump(mode="json") for k, r in actual_refs.items()})) or not all(r.valid for r in actual_checks):
            reasons.append("CITATION_CHECK_FAILED")
        if set(reports) != ids or len(reports) != len(audit.atom_reports):
            reasons.append("MISSING_AUDIT_REPORT")
        states = {"support": [], "uncertain": [], "refute": []}
        valid_reports = {}
        call_ids = set()
        for atom in draft.atoms:
            report = reports.get(atom.atom_id)
            if report is None:
                continue
            try:
                report = AtomAudit.model_validate(report.model_dump())
                if (report.support_threshold != self.support_threshold or report.refute_threshold != self.refute_threshold
                        or report.template_version != TEMPLATE_VERSION):
                    raise ValueError("Audit policy mismatch")
                valid_reports[atom.atom_id] = report
                if report.execution_status != "scored":
                    reasons.extend(report.reason_codes)
                    reasons.append("ATOM_AUDIT_FAILED")
                    continue
                if report.scores.call_id in call_ids:
                    raise ValueError("Reused scoring call")
                call_ids.add(report.scores.call_id)
                states[report.verdict].append(atom.atom_id)
            except (ValueError, TypeError, AttributeError):
                reasons.append("ATOM_AUDIT_FAILED")
        # Bind each report to its actual prompt, including the audited dependency states.
        if set(valid_reports) == ids and set(refs) == ids and not reasons:
            lookup = {a.atom_id: a for a in draft.atoms}
            references = {key: r.resolved_evidence for key, r in refs.items()}
            seen = {}
            ordered = [a for a in dependency_order(draft) if a.kind != "answer"]
            ordered += [lookup[draft.final_atom_id]]
            for atom in ordered:
                final = atom.kind == "answer"
                prompt = build_audit_prompt(group=group, atom=atom, citations=references[atom.atom_id],
                    dependencies=audit_context(atom, lookup, seen, references),
                    all_reports=[audited_atom(k, lookup, seen, references) for k in seen] if final else [], final=final)
                report = valid_reports[atom.atom_id]
                if report.prompt_sha256 != digest(prompt):
                    reasons.append("AUDIT_PROMPT_MISMATCH")
                seen[atom.atom_id] = report
        total = len(draft.atoms)
        ratio = len(states["support"]) / total if total else None
        final = valid_reports.get(draft.final_atom_id)
        if not total:
            reasons.append("NO_AUDITED_ATOMS")
        if states["refute"]:
            reasons.append("REFUTING_ATOM_PRESENT")
        if ratio is not None and ratio < self.min_support_ratio:
            reasons.append("SUPPORT_RATIO_BELOW_THRESHOLD")
        if self.require_final_support and (final is None or final.verdict != "support"):
            reasons.append("FINAL_ANSWER_NOT_SUPPORTED")
        return GateDecision(allow=not reasons, reason_codes=list(dict.fromkeys(reasons)),
            support_atom_ids=states["support"], uncertain_atom_ids=states["uncertain"], refute_atom_ids=states["refute"],
            support_count=len(states["support"]), uncertain_count=len(states["uncertain"]), refute_count=len(states["refute"]),
            total_count=total, support_ratio=ratio, final_verdict=final.verdict if final else None,
            min_support_ratio=self.min_support_ratio, require_final_support=self.require_final_support)
