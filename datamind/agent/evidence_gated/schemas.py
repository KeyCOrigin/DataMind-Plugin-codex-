"""Request-bound evidence and externally checkable claims. No plan protocol."""
from __future__ import annotations
from typing import Any, Literal
import math
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .scoring import normalize_abc, classify, validate_thresholds, SCORE_TYPE
from datamind.core.protocols import LabelScores
from .prompts import TEMPLATE_VERSION

class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

class ToolRecord(Schema):
    call_id: str
    tool_name: str
    args: dict[str, Any]
    result: Any
    error: str | None = None
    started_at: str | None = None
    read_at: str
    elapsed_ms: float | None = None

class EvidenceRecord(Schema):
    evidence_id: str
    call_id: str
    surface: str
    tool_name: str
    tool_args: dict[str, Any]
    raw_payload: Any
    payload_sha256: str
    source_id: str | None = None
    source_version: str | None = None
    workspace_revision: str | None = None
    surface_version: str | None = None
    source_bindings: dict[str, Any] = Field(default_factory=dict)
    available_locators: list[dict[str, Any]] = Field(default_factory=list)
    completeness: Literal["complete", "partial", "unknown"] = "unknown"
    error: str | None = None

class EvidenceGroup(Schema):
    request_id: str
    question: str
    context: dict[str, Any]
    answer_contract: dict[str, Any]
    group_id: str
    evidence: list[EvidenceRecord]
    tool_trace: list[ToolRecord]

class Citation(Schema):
    evidence_id: str
    selector: dict[str, Any]
    quote: str | None = None

class Atom(Schema):
    atom_id: str
    kind: Literal["fact", "derived", "answer"]
    statement: str
    value: Any = None
    citations: list[Citation]
    depends_on: list[str] = Field(default_factory=list)

class AnswerGenerationOutput(Schema):
    status: Literal["candidate", "insufficient"]
    answer_value: Any = None
    basis_notes: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)

class AnswerCandidate(AnswerGenerationOutput):
    candidate_id: str
    request_id: str
    group_id: str
    candidate_sha256: str = ""

class IntermediateAtom(Atom):
    kind: Literal["fact", "derived"]

class FinalSupport(Schema):
    citations: list[Citation] = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)

class AtomBuildOutput(Schema):
    status: Literal["candidate", "insufficient"]
    atoms: list[IntermediateAtom] = Field(default_factory=list)
    final_support: FinalSupport | None = None
    gaps: list[str] = Field(default_factory=list)

class ReasoningDraft(Schema):
    draft_id: str
    group_id: str
    candidate_id: str
    candidate_sha256: str
    status: Literal["candidate", "insufficient"]
    atoms: list[Atom]
    final_atom_id: str | None = None
    answer_value: Any = None
    gaps: list[str] = Field(default_factory=list)

class ResolvedEvidence(Schema):
    evidence_id: str
    call_id: str
    selector: dict[str, Any]
    content: Any
    source_bindings: dict[str, Any]
    completeness: str
    versions: dict[str, Any]
    record_context: dict[str, Any] | None = None
    tool_context: dict[str, Any] = Field(default_factory=dict)

class CitationCheck(Schema):
    atom_id: str
    valid: bool
    resolved_evidence: list[ResolvedEvidence] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

class AtomAudit(Schema):
    atom_id: str
    execution_status: Literal["scored", "invalid", "error"]
    verdict: Literal["support", "uncertain", "refute"] | None = None
    scores: LabelScores | None = None
    label_probs: dict[str, float] | None = None
    expected_score: float | None = None
    support_threshold: float
    refute_threshold: float
    score_type: Literal["raw_abc_expected_value"] = SCORE_TYPE
    template_version: str = TEMPLATE_VERSION
    prompt_sha256: str | None = None
    elapsed_ms: float | None = None
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_score(self):
        validate_thresholds(self.support_threshold, self.refute_threshold)
        if self.execution_status != "scored":
            if any(v is not None for v in (self.verdict, self.scores, self.label_probs, self.expected_score)):
                raise ValueError("Execution failure cannot carry a semantic verdict or scores")
            return self
        if self.scores is None or self.label_probs is None or self.expected_score is None or not self.prompt_sha256:
            raise ValueError("Scored audit requires complete probabilities and prompt binding")
        scores = LabelScores.model_validate(self.scores.model_dump())
        probs = normalize_abc(scores.logprobs)
        if set(self.label_probs) != set(probs) or any(
                not math.isfinite(self.label_probs[k]) or not math.isclose(probs[k], self.label_probs[k], rel_tol=1e-12, abs_tol=1e-12)
                for k in probs):
            raise ValueError("ABC probability mismatch")
        expected = probs["A"] - probs["C"]
        if not math.isfinite(self.expected_score) or not math.isclose(self.expected_score, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("ABC expected score mismatch")
        if self.verdict != classify(expected, self.support_threshold, self.refute_threshold):
            raise ValueError("ABC verdict mismatch")
        return self

class AuditReport(Schema):
    subject_sha256: str
    citation_checks: list[CitationCheck] = Field(default_factory=list)
    atom_reports: list[AtomAudit] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)

class GateDecision(Schema):
    allow: bool
    reason_codes: list[str] = Field(default_factory=list)
    protocol_version: Literal["abc-v1"] = "abc-v1"
    support_atom_ids: list[str] = Field(default_factory=list)
    uncertain_atom_ids: list[str] = Field(default_factory=list)
    refute_atom_ids: list[str] = Field(default_factory=list)
    support_count: int = 0
    uncertain_count: int = 0
    refute_count: int = 0
    total_count: int = 0
    support_ratio: float | None = None
    final_verdict: Literal["support", "uncertain", "refute"] | None = None
    min_support_ratio: float
    require_final_support: bool


class QueryResult(Schema):
    protocol_version: Literal["abc-v1"] = "abc-v1"
    decision: Literal["answer", "insufficient", "error"]
    answer: Any = None
    draft_id: str | None = None
    candidate_id: str | None = None
    atoms: list[Atom] = Field(default_factory=list)
    audit: AuditReport
    evidence: list[EvidenceRecord]
    reason_codes: list[str] = Field(default_factory=list)
