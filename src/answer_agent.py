"""Strict evidence-grounded answer drafting with safe structured parsing."""

from __future__ import annotations

import json
import time
from typing import Callable, List, Tuple

from src.agent_types import AgentEvent, AnswerDraft, Citation, QueryPlan, SelectedEvidence
from src.llm_backend import LLMBackend


def build_grounded_prompt(question: str, plan: QueryPlan, evidence: List[SelectedEvidence]) -> str:
    blocks = []
    for item in evidence:
        blocks.append(f"[{item.citation_label}] Title: {item.title}\nSection: {item.section_name}\nEvidence: {item.evidence_text}")
    context = "\n\n".join(blocks)
    return (
        "You are an evidence-grounded scientific question answering component.\n"
        "Use only the supplied evidence. Cite every supported claim using only the supplied labels. "
        "Never invent a citation. If the evidence is inadequate, answer exactly \"Insufficient evidence\".\n"
        "Return JSON only with this schema: "
        '{"answer": "string", "citation_labels": ["E1"], "unanswerable": false}.\n'
        f"Question: {question}\nQuestion style: {plan.question_style}\n\nEvidence:\n{context}"
    )


class AnswerAgent:
    def __init__(self, backend: LLMBackend, clock: Callable[[], float] = time.perf_counter):
        self.backend = backend; self.clock = clock

    def answer(self, question: str, plan: QueryPlan,
               evidence: List[SelectedEvidence]) -> Tuple[AnswerDraft, AgentEvent]:
        started = self.clock()
        ids = {"question_id": plan.question_id, "paper_id": plan.paper_id, "split": plan.split}
        if not evidence:
            draft = AnswerDraft(plan.question_id, "Insufficient evidence", [], [], True, "insufficient_evidence")
            return draft, AgentEvent("answer_agent", "ok", ids, 1, ["llm_called=false", "empty_evidence=true"],
                                     round((self.clock() - started) * 1000, 3))
        prompt = build_grounded_prompt(question, plan, evidence)
        try:
            payload = json.loads(self.backend.generate(prompt))
            if not isinstance(payload, dict):
                raise ValueError("response must be a JSON object")
            if not isinstance(payload.get("answer"), str) or not isinstance(payload.get("unanswerable"), bool):
                raise ValueError("answer must be a string and unanswerable must be a boolean")
            labels = payload.get("citation_labels")
            if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
                raise ValueError("citation_labels must be a list of strings")
            available = {item.citation_label: item for item in evidence}
            unknown = sorted(set(labels) - set(available))
            if unknown:
                raise ValueError(f"unknown citation labels: {unknown}")
            labels = list(dict.fromkeys(labels))
            citations = [Citation(label, available[label].document_id, available[label].source_type,
                                  available[label].source_id, available[label].title,
                                  available[label].section_name) for label in labels]
            answer = "Insufficient evidence" if payload["unanswerable"] else payload["answer"].strip()
            if not answer:
                raise ValueError("answer must not be empty")
            status = "insufficient_evidence" if payload["unanswerable"] else "ok"
            draft = AnswerDraft(plan.question_id, answer, citations, labels, payload["unanswerable"], status)
            event = AgentEvent("answer_agent", "ok", ids, 1, ["llm_called=true",
                               f"validated_citations={len(labels)}"], round((self.clock() - started) * 1000, 3))
            return draft, event
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            message = str(exc)
            draft = AnswerDraft(plan.question_id, "Insufficient evidence", [], [], True, "error", message)
            event = AgentEvent("answer_agent", "error", ids, 0, ["llm_called=true", "response_rejected=true"],
                               round((self.clock() - started) * 1000, 3), message)
            return draft, event
