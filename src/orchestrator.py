"""Bounded final G4 multi-agent orchestration and prediction adapters."""

from __future__ import annotations

import time
from typing import Callable, Dict

from src.agent_types import AgentEvent, CriticVerdict, FinalResult


def to_qasper_prediction(result: FinalResult) -> Dict[str, object]:
    """Convert inference output only; official scoring remains a separate concern."""
    style = result.query_plan.get("question_style", "free-form")
    unanswerable = result.final_status not in {"accepted", "accepted_revised"}
    answer = "Insufficient evidence" if unanswerable else result.answer
    if style == "yes/no" and not unanswerable:
        lowered = answer.strip().lower()
        yes_no = "yes" if lowered.startswith("yes") else "no" if lowered.startswith("no") else None
    else:
        yes_no = None
    return {result.question_id: {"answer": answer, "answer_type": "unanswerable" if unanswerable else style,
                                "yes_no": yes_no, "unanswerable": unanswerable}}


class MultiAgentOrchestrator:
    def __init__(self, query_agent, retrieval_agent, evidence_agent, answer_agent, critic_agent,
                 clock: Callable[[], float] = time.perf_counter):
        self.query_agent = query_agent; self.retrieval_agent = retrieval_agent
        self.evidence_agent = evidence_agent; self.answer_agent = answer_agent
        self.critic_agent = critic_agent; self.clock = clock

    def run(self, question_id: str, question: str, paper_id: str, split: str) -> FinalResult:
        started = self.clock(); traces = []; attempts = []
        plan, event = self.query_agent.plan(question_id, question, paper_id, split); traces.append(event)
        retrieval, event = self.retrieval_agent.retrieve(plan); traces.append(event)
        evidence, event = self.evidence_agent.select(plan, retrieval); traces.append(event)
        backend = self.answer_agent.backend
        empty_verdict = CriticVerdict("insufficient", True, [], [], "")
        if not evidence:
            ids = {"question_id": plan.question_id, "paper_id": plan.paper_id, "split": plan.split}
            traces.append(AgentEvent("answer_agent", "skipped", ids, 0,
                                     ["llm_called=false", "empty_evidence=true"]))
            traces.append(AgentEvent("critic_agent", "skipped", ids, 0,
                                     ["llm_called=false", "empty_evidence=true"]))
            final_answer, verdict, status, retry_count = "Insufficient evidence", empty_verdict, "insufficient_evidence", 0
        else:
            instruction = ""; retry_count = 0
            for attempt in range(2):
                draft, event = self.answer_agent.answer(question, plan, evidence, instruction); traces.append(event)
                verdict, event = self.critic_agent.review(question, plan, draft, evidence); traces.append(event)
                attempts.append({"attempt_number": attempt + 1, "answer": draft.answer,
                                 "validated_citations": list(draft.citation_labels),
                                 "critic_verdict": verdict.verdict,
                                 "revision_instruction": verdict.revision_instruction,
                                 "deterministic_checks_passed": not verdict.deterministic_failures,
                                 "deterministic_failures": list(verdict.deterministic_failures),
                                 "critic_mode": verdict.critic_mode,
                                 "llm_critic_valid": verdict.llm_critic_valid,
                                 "critic_schema_error": verdict.critic_schema_error,
                                 "fallback_used": verdict.fallback_used})
                if verdict.verdict == "accept":
                    final_answer = draft.answer; status = "accepted" if attempt == 0 else "accepted_revised"; break
                if verdict.verdict == "insufficient" and draft.status != "error":
                    final_answer = "Insufficient evidence"; status = "insufficient_evidence"; break
                if attempt == 0:
                    retry_count = 1
                    instruction = verdict.revision_instruction or "Fix grounding and citations."
                    continue
                final_answer = "Insufficient evidence"; status = "rejected"
            else:  # pragma: no cover - bounded loop always exits above
                final_answer = "Insufficient evidence"; status = "rejected"
        citations = draft.citations if evidence and status in {"accepted", "accepted_revised"} else []
        return FinalResult(str(question_id), question, str(paper_id), str(split), plan.to_dict(),
                           [item.to_dict() for item in evidence], final_answer,
                           [item.to_dict() for item in citations], verdict.to_dict(), status, retry_count,
                           [item.to_dict() for item in traces], getattr(backend, "model_name", "unknown"),
                           dict(getattr(backend, "generation_config", {})), round(self.clock() - started, 6), attempts)
