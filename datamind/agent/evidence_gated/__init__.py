"""New evidence gate entry points and evidence/atom protocols only."""
from .agent import EvidenceGatedAgent
from .schemas import (EvidenceGroup, EvidenceRecord, ToolRecord, Citation, Atom,
    ReasoningDraft, ResolvedEvidence, CitationCheck, LabelScores, AtomAudit,
    AuditReport, GateDecision, QueryResult, AnswerCandidate, AnswerGenerationOutput,
    AtomBuildOutput, FinalSupport)
from .retrieval import RetrievalBridge
from .evidence import EvidenceRegistry, SourceMappingResolver
from .answer_generator import AnswerGenerator
from .atom_builder import AtomBuilder
from .provenance import ProvenanceValidator
from .scoring import TokenScoringAdapter, ScoringUnavailable
from .judge import ABCJudge
from .gate import Gate

__all__ = ["EvidenceGatedAgent", "EvidenceGroup", "EvidenceRecord", "ToolRecord",
    "Citation", "Atom", "ReasoningDraft", "ResolvedEvidence", "CitationCheck",
    "LabelScores", "AtomAudit", "AuditReport", "GateDecision", "QueryResult",
    "RetrievalBridge", "EvidenceRegistry", "SourceMappingResolver", "AnswerGenerator", "AtomBuilder",
    "AnswerCandidate", "AnswerGenerationOutput", "AtomBuildOutput", "FinalSupport",
    "ProvenanceValidator", "TokenScoringAdapter", "ScoringUnavailable", "ABCJudge", "Gate"]
