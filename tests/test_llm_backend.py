import json
import sys

import pytest

from src.llm_backend import TransformersLLMBackend, extract_json_object


def test_real_backend_is_lazy_and_generation_is_deterministic():
    sys.modules.pop("transformers", None)
    backend = TransformersLLMBackend(model_name="unit-test-never-load", max_new_tokens=257)
    assert not backend.is_loaded and "transformers" not in sys.modules
    assert backend.generation_config == {"do_sample": False, "max_new_tokens": 257, "batch_size": 1}
    assert backend.input_token_limit == 4096
    assert backend.local_files_only is False


def test_offline_loading_is_explicit_opt_in():
    backend = TransformersLLMBackend(model_name="unit-test-offline-never-load", local_files_only=True)
    assert backend.local_files_only is True and not backend.is_loaded


def test_plain_and_fenced_json_extraction_uses_expected_object_only():
    expected = {"answer", "citation_labels", "unanswerable"}
    plain = extract_json_object('{"answer":"a","citation_labels":[],"unanswerable":false,"ignored":1}', expected)
    fenced = extract_json_object('```json\n{"answer":"b","citation_labels":["E1"],"unanswerable":false}\n```', expected)
    assert plain == {"answer": "a", "citation_labels": [], "unanswerable": False}
    assert fenced["answer"] == "b"


def test_malformed_json_and_missing_fields_are_rejected():
    with pytest.raises(ValueError, match="Expecting value"):
        extract_json_object("not json", {"answer"})
    with pytest.raises(ValueError, match="missing required"):
        extract_json_object('{"answer":"a"}', {"answer", "citation_labels"})


@pytest.mark.skip(reason="optional: explicitly enable when the Qwen weights are locally available")
def test_real_qwen_optional_integration():
    backend = TransformersLLMBackend()
    assert backend.generate("Reply with the word OK.")
