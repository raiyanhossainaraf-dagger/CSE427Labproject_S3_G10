"""Run the complete G4A logical-agent flow with offline deterministic fakes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.answer_agent import AnswerAgent
from src.evidence_agent import EvidenceAgent
from src.llm_backend import MockLLMBackend
from src.query_agent import QueryAgent
from src.retrieval_agent import RetrievalAgent


class FakeRetriever:
    def search(self, query, paper_id, split, top_k, question_id):
        return pd.DataFrame([
            {"question_id": question_id, "paper_id": paper_id, "split": split, "document_id": "d1",
             "source_type": "chunk", "source_id": "c1", "chunk_id": "c1", "paragraph_id": "p1",
             "section_id": "s1", "figure_table_id": "", "rank": 1, "score": .03, "fused_score": .03,
             "bm25_score": 4., "bm25_rank": 1, "dense_score": .8, "dense_rank": 2,
             "title": "Demo Paper", "section_name": "Results", "text": "The proposed method improved accuracy."},
            {"question_id": question_id, "paper_id": paper_id, "split": split, "document_id": "d2",
             "source_type": "float", "source_id": "f1", "chunk_id": "", "paragraph_id": "",
             "section_id": "", "figure_table_id": "f1", "rank": 2, "score": .02, "fused_score": .02,
             "bm25_score": np.nan, "bm25_rank": pd.NA, "dense_score": .7, "dense_rank": 1,
             "title": "Demo Paper", "section_name": "", "text": "Figure 1 reports the accuracy gain."},
        ]).head(top_k)


class FakeReranker:
    def rerank(self, question, frame, top_k):
        output = frame.copy(); output["hybrid_rank"] = output["rank"]
        output["hybrid_score"] = output["fused_score"]
        output["cross_encoder_score"] = [2., 1.][:len(output)]
        output["reranked_rank"] = range(1, len(output) + 1)
        return output


def main():
    deterministic_clock = lambda: 1.0
    plan, query_trace = QueryAgent(clock=deterministic_clock).plan(
        "q-smoke", "  What did the method improve? ", "paper-1", "validation")
    retrieved, retrieval_trace = RetrievalAgent(FakeRetriever(), clock=deterministic_clock).retrieve(plan)
    evidence, evidence_trace = EvidenceAgent(FakeReranker(), clock=deterministic_clock).select(plan, retrieved)
    backend = MockLLMBackend('{"answer":"The method improved accuracy [E1].","citation_labels":["E1"],"unanswerable":false}')
    draft, answer_trace = AnswerAgent(backend, clock=deterministic_clock).answer(plan.original_question, plan, evidence)
    payload = {"query_plan": plan.to_dict(), "ranked_candidates": retrieved.to_dict(),
               "selected_evidence": [item.to_dict() for item in evidence], "answer_draft": draft.to_dict(),
               "traces": [event.to_dict() for event in (query_trace, retrieval_trace, evidence_trace, answer_trace)]}
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
