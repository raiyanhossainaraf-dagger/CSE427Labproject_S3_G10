import json

import numpy as np
import pandas as pd

from src.agent_types import AgentEvent, AnswerDraft, Citation, QueryPlan, RetrievalCandidate, RetrievalResult, SelectedEvidence
from src.answer_agent import AnswerAgent, build_grounded_prompt
from src.evidence_agent import EvidenceAgent
from src.evidence_scoring import score_evidence
from src.llm_backend import MockLLMBackend
from src.query_agent import QueryAgent
from src.retrieval_agent import RetrievalAgent


ZERO_CLOCK = lambda: 10.0


def candidate(doc="d1", paragraph="p1", rank=1, paper="paper", split="validation", text="selected text"):
    return {"question_id": "q1", "paper_id": paper, "split": split, "document_id": doc,
            "source_type": "chunk", "source_id": f"c-{doc}", "chunk_id": f"c-{doc}",
            "paragraph_id": paragraph, "section_id": "s1", "figure_table_id": "", "rank": rank,
            "score": .03 / rank, "fused_score": .03 / rank, "bm25_score": 2., "bm25_rank": rank,
            "dense_score": .8, "dense_rank": rank, "title": "Paper", "section_name": "Results", "text": text}


class FakeRetriever:
    def __init__(self, rows): self.rows = rows; self.calls = []
    def search(self, query, paper_id, split, top_k, question_id):
        self.calls.append((query, paper_id, split, top_k, question_id))
        return pd.DataFrame(self.rows).head(top_k)


class FakeReranker:
    def __init__(self): self.questions = []
    def rerank(self, question, frame, top_k):
        self.questions.append(question); output = frame.copy()
        output["hybrid_rank"] = output["rank"]; output["hybrid_score"] = output["fused_score"]
        output["cross_encoder_score"] = np.arange(len(output), 0, -1, dtype=float)
        output["reranked_rank"] = range(1, len(output) + 1)
        return output


def plan():
    return QueryAgent(clock=ZERO_CLOCK).plan("q1", " What was improved? ", "paper", "validation")[0]


def selected(label="E1", text="selected evidence", doc="d1"):
    return SelectedEvidence(label, doc, "chunk", f"c-{doc}", "paper", "Paper", "Results", text,
                            1, .9, 2., 1., .5, 1., True, 1, .03, 2., 1, .8, 1,
                            f"c-{doc}", "p1", "s1", "")


def test_all_contracts_are_json_serializable():
    objects = [plan(), RetrievalCandidate(**{k: v for k, v in {
        "document_id":"d", "source_type":"chunk", "source_id":"c", "paper_id":"p", "split":"validation",
        "title":"T", "section_name":"S", "text":"x", "hybrid_rank":1, "hybrid_score":.1}.items()}),
        RetrievalResult("q", "p", "validation", []), selected(), Citation("E1", "d", "chunk", "c", "T", "S"),
        AnswerDraft("q", "a", [], [], False, "ok"), AgentEvent("agent", "ok", {"question_id":"q"}, 1)]
    for obj in objects:
        json.dumps(obj.to_dict())


def test_query_agent_is_deterministic_classifies_styles_and_rewrite_is_off():
    agent = QueryAgent(rewriter=lambda _: "LEAK", clock=ZERO_CLOCK)
    first, trace1 = agent.plan("q1", "  Is\n it effective?  ", "p", "validation")
    second, trace2 = agent.plan("q1", "  Is\n it effective?  ", "p", "validation")
    assert first == second and trace1 == trace2
    assert first.original_question == "  Is\n it effective?  "
    assert first.retrieval_query == "Is it effective?" and first.question_style == "yes/no" and not first.rewritten
    assert QueryAgent(clock=ZERO_CLOCK).plan("q", "How does it work?", "p", "validation")[0].question_style == "free-form"


def test_retrieval_injection_scope_canonical_dedup_and_scores():
    rows = [candidate("d1", "same", 1), candidate("d2", "same", 2), candidate("d3", "other", 3)]
    fake = FakeRetriever(rows); result, trace = RetrievalAgent(fake, candidate_depth=50, clock=ZERO_CLOCK).retrieve(plan())
    assert fake.calls == [("What was improved?", "paper", "validation", 50, "q1")]
    assert [item.document_id for item in result.candidates] == ["d1", "d3"]
    assert result.candidates[0].bm25_score == 2. and result.candidates[0].dense_rank == 1
    assert trace.decisions[0] == "scope=paper+split"


def test_retrieval_rejects_cross_paper_or_split_results():
    for row in (candidate(paper="foreign"), candidate(split="test")):
        try:
            RetrievalAgent(FakeRetriever([row])).retrieve(plan())
            assert False, "scope violation was accepted"
        except ValueError as exc:
            assert "outside" in str(exc)


def test_evidence_agent_uses_injected_dependencies_caps_five_and_stable_labels():
    retrieval = RetrievalAgent(FakeRetriever([candidate(f"d{i}", f"p{i}", i) for i in range(1, 8)])).retrieve(plan())[0]
    calls = []
    def fake_scorer(frame, method):
        calls.append(method); return score_evidence(frame, method)
    evidence, trace = EvidenceAgent(FakeReranker(), scorer=fake_scorer, top_n=5, clock=ZERO_CLOCK).select(plan(), retrieval)
    assert calls == ["fused"] and len(evidence) == 5
    assert [item.citation_label for item in evidence] == ["E1", "E2", "E3", "E4", "E5"]
    assert len({(item.source_type, item.paragraph_id) for item in evidence}) == 5
    assert "configuration=70/25/5" in trace.decisions


def test_grounded_prompt_contains_selected_evidence_only_and_no_gold_leakage():
    evidence = [selected(text="ALLOWED SOURCE")]
    prompt = build_grounded_prompt("Question?", plan(), evidence)
    assert "[E1]" in prompt and "ALLOWED SOURCE" in prompt
    assert "UNSELECTED SOURCE" not in prompt and "gold evidence" not in prompt.lower()
    assert "use only the supplied evidence" in prompt.lower() and "never invent" in prompt.lower()


def test_valid_answer_and_citation_parsing():
    backend = MockLLMBackend('{"answer":"Supported [E1].","citation_labels":["E1","E1"],"unanswerable":false}')
    draft, trace = AnswerAgent(backend, clock=ZERO_CLOCK).answer("Question?", plan(), [selected()])
    assert draft.status == "ok" and draft.citation_labels == ["E1"]
    assert draft.citations[0].document_id == "d1" and trace.status == "ok"


def test_unknown_citation_and_malformed_json_are_rejected():
    unknown = MockLLMBackend('{"answer":"bad","citation_labels":["E9"],"unanswerable":false}')
    malformed = MockLLMBackend("not-json")
    for backend, expected in ((unknown, "unknown citation"), (malformed, "Expecting value")):
        draft, trace = AnswerAgent(backend, clock=ZERO_CLOCK).answer("Question?", plan(), [selected()])
        assert draft.status == "error" and draft.citations == [] and expected in draft.error
        assert trace.status == "error" and "response_rejected=true" in trace.decisions


def test_empty_evidence_does_not_call_backend():
    backend = MockLLMBackend("should not be called")
    draft, trace = AnswerAgent(backend, clock=ZERO_CLOCK).answer("Question?", plan(), [])
    assert draft.answer == "Insufficient evidence" and draft.unanswerable
    assert backend.prompts == [] and "llm_called=false" in trace.decisions


def test_all_agent_traces_are_deterministic_with_injected_clock():
    traces = []
    for _ in range(2):
        query_plan, q_trace = QueryAgent(clock=ZERO_CLOCK).plan("q1", "What was improved?", "paper", "validation")
        retrieval, r_trace = RetrievalAgent(FakeRetriever([candidate()]), clock=ZERO_CLOCK).retrieve(query_plan)
        evidence, e_trace = EvidenceAgent(FakeReranker(), clock=ZERO_CLOCK).select(query_plan, retrieval)
        backend = MockLLMBackend('{"answer":"Supported.","citation_labels":["E1"],"unanswerable":false}')
        _, a_trace = AnswerAgent(backend, clock=ZERO_CLOCK).answer(query_plan.original_question, query_plan, evidence)
        traces.append([event.to_dict() for event in (q_trace, r_trace, e_trace, a_trace)])
    assert traces[0] == traces[1]
    assert all(event["elapsed_ms"] == 0.0 for event in traces[0])
