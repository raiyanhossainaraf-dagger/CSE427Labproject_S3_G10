"""Paper/split-scoped adapter around the completed G2 hybrid retriever."""

from __future__ import annotations

import math
import time
from typing import Callable, Tuple

import pandas as pd

from src.agent_types import AgentEvent, QueryPlan, RetrievalCandidate, RetrievalResult
from src.hybrid_retrieval import canonical_source


def _optional_float(value):
    return None if value is None or pd.isna(value) else float(value)


def _optional_int(value):
    return None if value is None or pd.isna(value) else int(value)


class ExistingHybridBackend:
    """Adapt G2 HybridRetriever.search_embedding using an injected query encoder."""

    def __init__(self, hybrid_retriever, encode_query: Callable[[str], object]):
        self.hybrid_retriever = hybrid_retriever; self.encode_query = encode_query

    def search(self, query, paper_id, split, top_k=50, question_id=""):
        embedding = self.encode_query(query)
        return self.hybrid_retriever.search_embedding(query, embedding, paper_id, split, top_k, question_id)


class RetrievalAgent:
    def __init__(self, retriever, corpus: pd.DataFrame | None = None, candidate_depth: int = 50,
                 clock: Callable[[], float] = time.perf_counter):
        if candidate_depth < 1:
            raise ValueError("candidate_depth must be positive")
        if not hasattr(retriever, "search"):
            raise TypeError("retriever must provide search(query, paper_id, split, top_k, question_id)")
        self.retriever = retriever; self.candidate_depth = candidate_depth; self.clock = clock
        self.corpus = None if corpus is None else corpus.set_index("document_id", drop=False)

    def retrieve(self, plan: QueryPlan) -> Tuple[RetrievalResult, AgentEvent]:
        started = self.clock()
        frame = self.retriever.search(plan.retrieval_query, plan.paper_id, plan.split,
                                      self.candidate_depth, plan.question_id)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("retriever.search must return a pandas DataFrame")
        if self.corpus is not None and not frame.empty:
            missing = [column for column in ("title", "section_name", "text") if column not in frame]
            if missing:
                frame = frame.join(self.corpus[missing], on="document_id", validate="many_to_one")
        required = {"document_id", "paper_id", "split", "source_type", "source_id", "rank", "score",
                    "title", "section_name", "text"}
        missing = required - set(frame.columns)
        if missing and not frame.empty:
            raise ValueError(f"retrieval output missing columns: {sorted(missing)}")
        if not frame.empty and (frame.paper_id.astype(str).ne(plan.paper_id).any() or
                                frame.split.astype(str).ne(plan.split).any()):
            raise ValueError("retriever returned candidates outside the requested paper or split")
        rows, seen = [], set()
        ordered = frame.sort_values(["rank", "document_id"], kind="stable") if not frame.empty else frame
        for row in ordered.itertuples(index=False):
            key = canonical_source(row)
            if key in seen:
                continue
            seen.add(key)
            hybrid_score = getattr(row, "fused_score", getattr(row, "score"))
            if hybrid_score is None or (isinstance(hybrid_score, float) and math.isnan(hybrid_score)):
                hybrid_score = getattr(row, "score")
            rows.append(RetrievalCandidate(
                document_id=str(row.document_id), source_type=str(row.source_type), source_id=str(row.source_id),
                paper_id=str(row.paper_id), split=str(row.split), title=str(row.title),
                section_name=str(row.section_name), text=str(row.text), hybrid_rank=int(row.rank),
                hybrid_score=float(hybrid_score), bm25_score=_optional_float(getattr(row, "bm25_score", None)),
                bm25_rank=_optional_int(getattr(row, "bm25_rank", None)),
                dense_score=_optional_float(getattr(row, "dense_score", None)),
                dense_rank=_optional_int(getattr(row, "dense_rank", None)),
                chunk_id=str(getattr(row, "chunk_id", "")), paragraph_id=str(getattr(row, "paragraph_id", "")),
                section_id=str(getattr(row, "section_id", "")),
                figure_table_id=str(getattr(row, "figure_table_id", ""))))
            if len(rows) >= self.candidate_depth:
                break
        result = RetrievalResult(plan.question_id, plan.paper_id, plan.split, rows)
        event = AgentEvent("retrieval_agent", "ok", {"question_id": plan.question_id, "paper_id": plan.paper_id,
                           "split": plan.split}, len(rows), ["scope=paper+split", "canonical_deduplication=true",
                           f"candidate_depth={self.candidate_depth}"], round((self.clock() - started) * 1000, 3))
        return result, event
