"""Leakage-free deterministic query planning."""

from __future__ import annotations

import re
import time
from typing import Callable, Optional, Tuple

from src.agent_types import AgentEvent, QueryPlan


def _style(question: str) -> str:
    lowered = question.lower().strip()
    first = lowered.split(maxsplit=1)[0] if lowered else ""
    if first in {"is", "are", "was", "were", "do", "does", "did", "can", "could", "has", "have", "had"}:
        return "yes/no"
    if first in {"what", "which", "who", "where", "when", "how", "why"}:
        return "extractive" if first in {"what", "which", "who", "where", "when"} else "free-form"
    return "unknown"


class QueryAgent:
    def __init__(self, rewriter: Optional[Callable[[str], str]] = None, rewriting_enabled: bool = False,
                 clock: Callable[[], float] = time.perf_counter):
        self.rewriter = rewriter; self.rewriting_enabled = rewriting_enabled; self.clock = clock

    def plan(self, question_id: str, question: str, paper_id: str, split: str) -> Tuple[QueryPlan, AgentEvent]:
        started = self.clock()
        normalized = re.sub(r"\s+", " ", str(question)).strip()
        if not normalized:
            raise ValueError("question must contain non-whitespace text")
        rewritten = bool(self.rewriting_enabled and self.rewriter)
        retrieval_query = re.sub(r"\s+", " ", self.rewriter(normalized)).strip() if rewritten else normalized
        plan = QueryPlan(str(question_id), str(paper_id), str(split), str(question), normalized,
                         retrieval_query, _style(normalized), rewritten)
        event = AgentEvent("query_agent", "ok", {"question_id": plan.question_id, "paper_id": plan.paper_id,
                           "split": plan.split}, 1, [f"style={plan.question_style}", f"rewritten={rewritten}"],
                           round((self.clock() - started) * 1000, 3))
        return plan, event
