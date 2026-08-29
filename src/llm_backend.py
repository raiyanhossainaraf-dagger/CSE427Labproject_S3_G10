"""Lazy, shared text-generation backends and safe JSON extraction."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Protocol, Union

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


class LLMBackend(Protocol):
    model_name: str
    generation_config: Dict[str, Any]
    def generate(self, prompt: str) -> str: ...


def extract_json_object(text: str, expected_fields: Iterable[str]) -> Dict[str, Any]:
    """Extract one JSON object without evaluating any generated content."""
    if not isinstance(text, str):
        raise ValueError("model response must be text")
    fenced = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, flags=re.I | re.S)
    candidate = fenced.group(1) if fenced else text.strip()
    decoder = json.JSONDecoder()
    value = None
    for match in re.finditer(r"\{", candidate):
        try:
            parsed, _ = decoder.raw_decode(candidate[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            value = parsed
            break
    if value is None:
        raise ValueError("Expecting value: response does not contain a valid JSON object")
    expected = set(expected_fields)
    missing = expected - set(value)
    if missing:
        raise ValueError(f"response missing required fields: {sorted(missing)}")
    return {key: value[key] for key in expected}


@dataclass
class _LoadedModel:
    tokenizer: Any
    model: Any
    device: str
    load_seconds: float


_MODEL_CACHE: Dict[str, _LoadedModel] = {}
_MODEL_LOCK = threading.Lock()


class TransformersLLMBackend:
    """Lazy Qwen backend. Equal model names share one tokenizer/model pair."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, max_new_tokens: int = 256,
                 input_token_limit: int = 4096, device: str | None = None,
                 local_files_only: bool = False, seed: int = 427):
        if max_new_tokens < 1 or input_token_limit < 128:
            raise ValueError("invalid generation limits")
        self.model_name = model_name
        self.input_token_limit = input_token_limit
        self.requested_device = device
        self.local_files_only = bool(local_files_only)
        self.seed = int(seed)
        self.generation_config = {"do_sample": False, "max_new_tokens": max_new_tokens, "batch_size": 1}

    @property
    def is_loaded(self) -> bool:
        return self.model_name in _MODEL_CACHE

    def _load(self) -> _LoadedModel:
        if self.model_name in _MODEL_CACHE:
            return _MODEL_CACHE[self.model_name]
        with _MODEL_LOCK:
            if self.model_name in _MODEL_CACHE:
                return _MODEL_CACHE[self.model_name]
            try:
                started = time.perf_counter()
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                device = self.requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
                if self.requested_device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA was explicitly requested but is not available")
                kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}
                if device == "cuda":
                    kwargs.update({"torch_dtype": torch.float16})
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, local_files_only=self.local_files_only)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, local_files_only=self.local_files_only, **kwargs)
                model.to(device)
                model.eval()
                loaded = _LoadedModel(tokenizer, model, device, time.perf_counter() - started)
                _MODEL_CACHE[self.model_name] = loaded
                return loaded
            except Exception as exc:
                raise RuntimeError(f"Could not load requested model '{self.model_name}': {exc}") from exc

    @property
    def device(self) -> str:
        return self._load().device

    @property
    def load_seconds(self) -> float:
        return self._load().load_seconds

    def generate(self, prompt: str) -> str:
        loaded = self._load()
        import torch
        torch.manual_seed(self.seed)
        if loaded.device == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        rendered = loaded.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        encoded = loaded.tokenizer(rendered, return_tensors="pt", truncation=True,
                                   max_length=self.input_token_limit)
        model_device = next(loaded.model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = loaded.model.generate(**encoded, do_sample=False,
                                           max_new_tokens=self.generation_config["max_new_tokens"])
        new_tokens = output[0, encoded["input_ids"].shape[1]:]
        return loaded.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class MockLLMBackend:
    """Return a fixed response or a deterministic function of the prompt."""

    model_name = "mock"
    generation_config = {"do_sample": False, "max_new_tokens": 0, "batch_size": 1}

    def __init__(self, response: Union[str, list[str], Callable[[str], str]]):
        self.response = response
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if callable(self.response):
            return self.response(prompt)
        if isinstance(self.response, list):
            return self.response[min(len(self.prompts) - 1, len(self.response) - 1)]
        return self.response
