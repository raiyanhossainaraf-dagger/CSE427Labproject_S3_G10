from src.answer_agent import AnswerAgent
from src.critic_agent import CriticAgent
from src.llm_backend import MockLLMBackend
from tests.test_agents import plan, selected, ZERO_CLOCK


def draft(response):
    return AnswerAgent(MockLLMBackend(response), clock=ZERO_CLOCK).answer("Question?", plan(), [selected()])[0]


def test_deterministic_critic_accepts_valid_grounded_answer():
    verdict, _ = CriticAgent(clock=ZERO_CLOCK).review(
        "Question?", plan(), draft('{"answer":"Supported [E1].","citation_labels":["E1"],"unanswerable":false}'), [selected()])
    assert verdict.verdict == "accept" and verdict.supported


def test_critic_requests_revision_for_missing_or_invalid_citations():
    verdict, _ = CriticAgent(clock=ZERO_CLOCK).review(
        "Question?", plan(), draft('{"answer":"Unsupported.","citation_labels":[],"unanswerable":false}'), [selected()])
    assert verdict.verdict == "revise" and "factual_answer_without_citations" in verdict.deterministic_failures


def test_insufficient_evidence_consistency_and_no_gold_in_llm_prompt():
    backend = MockLLMBackend('{"verdict":"accept","supported":true,"missing_citations":[],"unsupported_claims":[],"revision_instruction":""}')
    critic = CriticAgent(backend, use_llm=True, clock=ZERO_CLOCK)
    valid = draft('{"answer":"Supported [E1].","citation_labels":["E1"],"unanswerable":false}')
    critic.review("Question?", plan(), valid, [selected(text="ALLOWED")])
    assert "ALLOWED" in backend.prompts[0]
    assert "gold answer" not in backend.prompts[0].lower() and "gold evidence" not in backend.prompts[0].lower()


def test_invalid_verdict_with_deterministic_pass_uses_marked_fallback():
    backend = MockLLMBackend('{"verdict":"approved","supported":true,"missing_citations":[],"unsupported_claims":[],"revision_instruction":""}')
    verdict, _ = CriticAgent(backend, use_llm=True, clock=ZERO_CLOCK).review(
        "Question?", plan(), draft('{"answer":"Supported [E1].","citation_labels":["E1"],"unanswerable":false}'), [selected()])
    assert verdict.verdict == "accept" and verdict.critic_mode == "deterministic_fallback"
    assert verdict.fallback_used and verdict.llm_critic_valid is False
    assert "invalid critic verdict" in verdict.critic_schema_error


def test_malformed_critic_json_with_deterministic_pass_uses_fallback():
    verdict, _ = CriticAgent(MockLLMBackend("not-json"), use_llm=True, clock=ZERO_CLOCK).review(
        "Question?", plan(), draft('{"answer":"Supported [E1].","citation_labels":["E1"],"unanswerable":false}'), [selected()])
    assert verdict.verdict == "accept" and verdict.fallback_used and verdict.llm_critic_valid is False


def test_invalid_llm_cannot_override_deterministic_failure():
    backend = MockLLMBackend('{"verdict":"approved"}')
    verdict, _ = CriticAgent(backend, use_llm=True, clock=ZERO_CLOCK).review(
        "Question?", plan(), draft('{"answer":"Unsupported.","citation_labels":[],"unanswerable":false}'), [selected()])
    assert verdict.verdict == "revise" and not verdict.fallback_used
    assert verdict.critic_mode == "deterministic" and backend.prompts == []


def test_valid_llm_revise_and_insufficient_are_preserved():
    base = '{"supported":false,"missing_citations":[],"unsupported_claims":[],"revision_instruction":"Be precise.","verdict":"%s"}'
    valid = draft('{"answer":"Supported [E1].","citation_labels":["E1"],"unanswerable":false}')
    for name in ("revise", "insufficient"):
        verdict, _ = CriticAgent(MockLLMBackend(base % name), use_llm=True, clock=ZERO_CLOCK).review(
            "Question?", plan(), valid, [selected()])
        assert verdict.verdict == name and verdict.llm_critic_valid is True and not verdict.fallback_used
