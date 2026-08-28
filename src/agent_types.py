"""Serializable typed contracts shared by the G4A logical agents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class SerializableContract:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentEvent(SerializableContract):
    agent_name: str
    status: str
    input_identifiers: Dict[str, str]
    output_count: int
    decisions: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    error: Optional[str] = None


AgentTrace = AgentEvent


@dataclass(frozen=True)
class QueryPlan(SerializableContract):
    question_id: str
    paper_id: str
    split: str
    original_question: str
    normalized_question: str
    retrieval_query: str
    question_style: str
    rewritten: bool = False


@dataclass(frozen=True)
class RetrievalCandidate(SerializableContract):
    document_id: str
    source_type: str
    source_id: str
    paper_id: str
    split: str
    title: str
    section_name: str
    text: str
    hybrid_rank: int
    hybrid_score: float
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    chunk_id: str = ""
    paragraph_id: str = ""
    section_id: str = ""
    figure_table_id: str = ""


@dataclass(frozen=True)
class RetrievalResult(SerializableContract):
    question_id: str
    paper_id: str
    split: str
    candidates: List[RetrievalCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class SelectedEvidence(SerializableContract):
    citation_label: str
    document_id: str
    source_type: str
    source_id: str
    paper_id: str
    title: str
    section_name: str
    evidence_text: str
    final_rank: int
    final_evidence_score: float
    cross_encoder_score: float
    normalized_cross_encoder_score: float
    normalized_hybrid_score: float
    agreement_score: float
    retrieved_by_both: bool
    hybrid_rank: int
    hybrid_score: float
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    chunk_id: str = ""
    paragraph_id: str = ""
    section_id: str = ""
    figure_table_id: str = ""


@dataclass(frozen=True)
class Citation(SerializableContract):
    label: str
    document_id: str
    source_type: str
    source_id: str
    title: str
    section_name: str


@dataclass(frozen=True)
class AnswerDraft(SerializableContract):
    question_id: str
    answer: str
    citations: List[Citation]
    citation_labels: List[str]
    unanswerable: bool
    status: str
    error: Optional[str] = None
