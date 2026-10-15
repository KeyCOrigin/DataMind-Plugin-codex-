"""ABC semantic audit; execution failures are separate from uncertainty."""
from time import perf_counter
from .evidence import canonical, digest
from .prompts import LOCAL, FINAL, TEMPLATE_VERSION
from .generation_common import check_budget, ContextBudgetExceeded
from .scoring import ScoringUnavailable, normalize_abc, classify, validate_thresholds
from .schemas import AtomAudit


def audit_context(atom, lookup, reports, references):
    """Transitive dependency audits, including uncertain/refuted/error predecessors."""
    ids = set()
    def expand(a):
        for key in a.depends_on:
            if key not in ids:
                ids.add(key)
                expand(lookup[key])
    expand(atom)
    return [audited_atom(key, lookup, reports, references) for key in sorted(ids)]


def audited_atom(key, lookup, reports, references):
    return {"atom": lookup[key].model_dump(mode="json"),
            "audit": reports[key].model_dump(mode="json"),
            "evidence": [c.model_dump(mode="json") for c in references[key]]}


def build_audit_prompt(*, group, atom, citations, dependencies, all_reports, final=False):
    data = {"question": group.question, "context": group.context,
            "atom": atom.model_dump(mode="json"),
            "resolved_citations": [c.model_dump(mode="json") for c in citations],
            "dependency_audits": dependencies}
    if final:
        data.update(answer_contract=group.answer_contract, answer=atom.value,
                    atom_audits=all_reports,
                    all_evidence=[e.model_dump(mode="json") for e in group.evidence])
    return (FINAL if final else LOCAL) + "\n" + canonical(data)


class ABCJudge:
    def __init__(self, scoring, *, support_threshold=0.3, refute_threshold=-0.3, max_input_bytes=200000):
        validate_thresholds(support_threshold, refute_threshold)
        self.scoring = scoring
        self.support_threshold, self.refute_threshold = support_threshold, refute_threshold
        self.max_input_bytes = max_input_bytes

    async def score(self, *, group, atom, citations, dependencies, all_reports, final=False):
        prompt = build_audit_prompt(group=group, atom=atom, citations=citations,
                                   dependencies=dependencies, all_reports=all_reports, final=final)
        start = perf_counter()
        fields = dict(atom_id=atom.atom_id, support_threshold=self.support_threshold,
                      refute_threshold=self.refute_threshold,
                      template_version=TEMPLATE_VERSION, prompt_sha256=digest(prompt))
        try:
            check_budget(prompt, self.max_input_bytes)
            scores = await self.scoring.score_labels(prompt)
            probs = normalize_abc(scores.logprobs)
            score = probs["A"] - probs["C"]
            return AtomAudit(**fields, execution_status="scored",
                verdict=classify(score, self.support_threshold, self.refute_threshold),
                scores=scores, label_probs=probs, expected_score=score,
                elapsed_ms=(perf_counter()-start)*1000)
        except Exception as exc:
            code = str(exc) if isinstance(exc, (ScoringUnavailable, ContextBudgetExceeded)) else "SCORING_UNAVAILABLE"
            return AtomAudit(**fields, execution_status="error", reason_codes=[code],
                             elapsed_ms=(perf_counter()-start)*1000)
