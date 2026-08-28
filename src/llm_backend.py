"""Shared generation interface; G4A provides only an offline deterministic mock."""

from __future__ import annotations

from typing import Callable, Protocol, Union


class LLMBackend(Protocol):
    def generate(self, prompt: str) -> str: ...


class MockLLMBackend:
    """Return a fixed response or a deterministic function of the prompt."""

    def __init__(self, response: Union[str, Callable[[str], str]]):
        self.response = response
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response(prompt) if callable(self.response) else self.response
