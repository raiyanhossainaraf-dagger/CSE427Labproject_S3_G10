"""Grounding critic with deterministic safeguards and an optional LLM review."""

from __future__ import annotations

import time
from typing import Callable, List, Tuple

from src.agent_types import AgentEvent, AnswerDraft, CriticVerdict, QueryPlan, SelectedEvidence
from src.llm_backend import LLMBackend, extract_json_object


def _critic_prompt(question: str, draft: AnswerDraft, evidence: List[SelectedEvidence]) -> str:
    cited = {item.citation_label: item for item in evidence if item.citation_label in draft.citation_labels}
    blocks = "\n\n".join(f"[{label}] {item.evidence_text}" for label, item in cited.items())
    return (
        "Review grounding only. Do not reveal reasoning. Return JSON with fields: "
        '"verdict" (must be exactly one of: "accept", "revise", "insufficient"), "supported" (boolean), '
        '"missing_citations" (string list), "unsupported_claims" (string list), '
        'and "revision_instruction" (concise string).\n'
        f"Question: {question}\nProposed answer: {draft.answer}\n"
        f"Validated citations: {draft.citation_labels}\nCited evidence:\n{blocks}"
    )


class CriticAgent:
    def __init__(self, backend: LLMBackend | None = None, use_llm: bool = False,
                 clock: Callable[[], float] = time.perf_counter):
        self.backend = backend; self.use_llm = use_llm; self.clock = clock
        if use_llm and backend is None:
            raise ValueError("an LLM backend is required when use_llm=True")

    def review(self, question: str, plan: QueryPlan, draft: AnswerDraft,
               evidence: List[SelectedEvidence]) -> Tuple[CriticVerdict, AgentEvent]:
        started = self.clock()
        ids = {"question_id": plan.question_id, "paper_id": plan.paper_id, "split": plan.split}
        available = {item.citation_label: item for item in evidence}
        failures: List[str] = []
        if not draft.answer.strip(): failures.append("empty_answer")
        unknown = sorted(set(draft.citation_labels) - set(available))
        if unknown: failures.append("fabricated_citation")
        if any(citation.label not in available for citation in draft.citations): failures.append("invalid_citation")
        if any(item.paper_id != plan.paper_id for item in evidence): failures.append("cross_paper_evidence")
        sources = [(item.source_type, item.source_id) for item in evidence]
        if len(sources) != len(set(sources)): failures.append("duplicate_evidence_source")
        insufficient_text = draft.answer.strip().lower() == "insufficient evidence"
        if draft.unanswerable != insufficient_text: failures.append("inconsistent_insufficient_evidence")
        if not draft.unanswerable and not draft.citation_labels: failures.append("factual_answer_without_citations")
        if draft.status == "error": failures.append("malformed_answer_response")
        if failures:
            verdict_name = "insufficient" if draft.unanswerable and failures == ["malformed_answer_response"] else "revise"
            verdict = CriticVerdict(verdict_name, False, [], failures,
                                    "Return a non-empty answer grounded in the supplied evidence with valid citations.",
                                    failures, "deterministic", None, None, False)
        elif draft.unanswerable:
            verdict = CriticVerdict("insufficient", True, [], [], "", [],
                                    "deterministic", None, None, False)
        elif not self.use_llm:
            verdict = CriticVerdict("accept", True, [], [], "", [],
                                    "deterministic", None, None, False)
        else:
            try:
                payload = extract_json_object(self.backend.generate(_critic_prompt(question, draft, evidence)),
                    {"verdict", "supported", "missing_citations", "unsupported_claims", "revision_instruction"})
                if payload["verdict"] not in {"accept", "revise", "insufficient"}:
                    raise ValueError("invalid critic verdict")
                if not isinstance(payload["supported"], bool) or not isinstance(payload["revision_instruction"], str):
                    raise ValueError("invalid critic field types")
                if any(not isinstance(payload[key], list) or any(not isinstance(x, str) for x in payload[key])
                       for key in ("missing_citations", "unsupported_claims")):
                    raise ValueError("critic lists must contain strings")
                if set(payload["missing_citations"]) - set(available):
                    raise ValueError("critic returned unknown citation labels")
                verdict = CriticVerdict(**payload, deterministic_failures=[], critic_mode="llm",
                                        llm_critic_valid=True, critic_schema_error=None, fallback_used=False)
            except (TypeError, ValueError) as exc:
                # The deterministic checks above passed. Invalid LLM structure is
                # non-authoritative and cannot turn a grounded answer into rejection.
                verdict = CriticVerdict("accept", True, [], [], "", [],
                                        "deterministic_fallback", False, str(exc), True)
        event = AgentEvent("critic_agent", "ok", ids, 1,
                           [f"verdict={verdict.verdict}",
                            f"critic_mode={verdict.critic_mode}",
                            f"fallback_used={verdict.fallback_used}",
                            f"llm_called={self.use_llm and not failures and not draft.unanswerable}"],
                           round((self.clock() - started) * 1000, 3))
        return verdict, event
