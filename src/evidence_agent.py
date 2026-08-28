"""G4A wrapper around the selected G3 evidence-fused configuration."""

from __future__ import annotations

import time
from typing import Callable, List, Tuple

import pandas as pd

from src.agent_types import AgentEvent, QueryPlan, RetrievalResult, SelectedEvidence
from src.evidence_scoring import EVIDENCE_WEIGHTS, EvidenceSelector, score_evidence


def _nullable(value, cast):
    return None if value is None or pd.isna(value) else cast(value)


class EvidenceAgent:
    def __init__(self, reranker, scorer: Callable[[pd.DataFrame, str], pd.DataFrame] = score_evidence,
                 selector=None, top_n: int = 5, clock: Callable[[], float] = time.perf_counter):
        self.reranker = reranker; self.scorer = scorer
        self.selector = selector or EvidenceSelector(top_n); self.top_n = min(top_n, 5); self.clock = clock

    def select(self, plan: QueryPlan, retrieval: RetrievalResult) -> Tuple[List[SelectedEvidence], AgentEvent]:
        started = self.clock()
        if (retrieval.question_id, retrieval.paper_id, retrieval.split) != (plan.question_id, plan.paper_id, plan.split):
            raise ValueError("retrieval result identifiers do not match query plan")
        frame = pd.DataFrame([candidate.to_dict() for candidate in retrieval.candidates])
        if frame.empty:
            event = AgentEvent("evidence_agent", "ok", {"question_id": plan.question_id,
                               "paper_id": plan.paper_id, "split": plan.split}, 0,
                               ["configuration=70/25/5", "empty_candidates=true"],
                               round((self.clock() - started) * 1000, 3))
            return [], event
        frame["question_id"] = plan.question_id; frame["rank"] = frame["hybrid_rank"]
        frame["score"] = frame["hybrid_score"]; frame["fused_score"] = frame["hybrid_score"]
        reranked = self.reranker.rerank(plan.original_question, frame, top_k=len(frame))
        scored = self.scorer(reranked, "fused")
        selected = self.selector.select(scored).sort_values(["final_rank", "document_id"], kind="stable").head(self.top_n)
        evidence = []
        for index, row in enumerate(selected.itertuples(index=False), 1):
            evidence.append(SelectedEvidence(
                citation_label=f"E{index}", document_id=str(row.document_id), source_type=str(row.source_type),
                source_id=str(row.source_id), paper_id=str(row.paper_id), title=str(row.title),
                section_name=str(row.section_name), evidence_text=str(row.evidence_text), final_rank=index,
                final_evidence_score=float(row.final_evidence_score), cross_encoder_score=float(row.cross_encoder_score),
                normalized_cross_encoder_score=float(row.normalized_cross_encoder_score),
                normalized_hybrid_score=float(row.normalized_hybrid_score), agreement_score=float(row.agreement_score),
                retrieved_by_both=bool(row.retrieved_by_both), hybrid_rank=int(row.hybrid_rank),
                hybrid_score=float(row.hybrid_score), bm25_score=_nullable(row.bm25_score, float),
                bm25_rank=_nullable(row.bm25_rank, int), dense_score=_nullable(row.dense_score, float),
                dense_rank=_nullable(row.dense_rank, int), chunk_id=str(row.chunk_id),
                paragraph_id=str(row.paragraph_id), section_id=str(row.section_id),
                figure_table_id=str(row.figure_table_id)))
        event = AgentEvent("evidence_agent", "ok", {"question_id": plan.question_id, "paper_id": plan.paper_id,
                           "split": plan.split}, len(evidence), ["configuration=70/25/5",
                           f"max_sources={self.top_n}", "citation_labels=stable"],
                           round((self.clock() - started) * 1000, 3))
        return evidence, event
