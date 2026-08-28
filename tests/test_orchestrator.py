import json

from src.answer_agent import AnswerAgent
from src.critic_agent import CriticAgent
from src.evidence_agent import EvidenceAgent
from src.llm_backend import MockLLMBackend
from src.orchestrator import MultiAgentOrchestrator, to_qasper_prediction
from src.query_agent import QueryAgent
from src.retrieval_agent import RetrievalAgent
from tests.test_agents import FakeRetriever, FakeReranker, candidate, ZERO_CLOCK


def orchestrator(responses, rows=None):
    backend = MockLLMBackend(responses)
    flow = MultiAgentOrchestrator(QueryAgent(clock=ZERO_CLOCK),
        RetrievalAgent(FakeRetriever([candidate()] if rows is None else rows), clock=ZERO_CLOCK),
        EvidenceAgent(FakeReranker(), clock=ZERO_CLOCK), AnswerAgent(backend, clock=ZERO_CLOCK),
        CriticAgent(clock=ZERO_CLOCK), clock=ZERO_CLOCK)
    return flow, backend


def test_accepted_first_answer_and_exact_execution_order():
    flow, _ = orchestrator('{"answer":"Good [E1].","citation_labels":["E1"],"unanswerable":false}')
    result = flow.run("q1", "What?", "paper", "validation")
    assert result.final_status == "accepted" and result.retry_count == 0
    assert [x["agent_name"] for x in result.agent_traces] == ["query_agent", "retrieval_agent", "evidence_agent", "answer_agent", "critic_agent"]
    assert result.attempt_history == [{"attempt_number": 1, "answer": "Good [E1].",
        "validated_citations": ["E1"], "critic_verdict": "accept", "revision_instruction": "",
        "deterministic_checks_passed": True, "deterministic_failures": [], "critic_mode": "deterministic",
        "llm_critic_valid": None, "critic_schema_error": None, "fallback_used": False}]
    json.dumps(result.to_dict())


def test_accepted_revision_has_maximum_one_retry():
    responses = ['{"answer":"No citation.","citation_labels":[],"unanswerable":false}',
                 '{"answer":"Fixed [E1].","citation_labels":["E1"],"unanswerable":false}']
    flow, backend = orchestrator(responses)
    result = flow.run("q1", "What?", "paper", "validation")
    assert result.final_status == "accepted_revised" and result.retry_count == 1 and len(backend.prompts) == 2
    assert "Revision instruction:" in backend.prompts[1]
    assert [item["attempt_number"] for item in result.attempt_history] == [1, 2]
    assert len(result.attempt_history) == 2


def test_rejected_second_answer_is_safe_and_serializable():
    bad = '{"answer":"No citation.","citation_labels":[],"unanswerable":false}'
    flow, backend = orchestrator([bad, bad])
    result = flow.run("q1", "What?", "paper", "validation")
    assert result.final_status == "rejected" and result.answer == "Insufficient evidence"
    assert len(backend.prompts) == 2
    json.dumps(result.to_dict())
    assert to_qasper_prediction(result)["q1"]["unanswerable"]


def test_no_evidence_calls_no_llm_and_returns_insufficient():
    flow, backend = orchestrator("should not run", rows=[])
    result = flow.run("q1", "What?", "paper", "validation")
    assert result.final_status == "insufficient_evidence" and backend.prompts == []
    assert [x["agent_name"] for x in result.agent_traces] == ["query_agent", "retrieval_agent", "evidence_agent", "answer_agent", "critic_agent"]
    assert result.agent_traces[-2]["status"] == "skipped" and result.agent_traces[-1]["status"] == "skipped"
