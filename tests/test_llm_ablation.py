import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_llm_ablation as g7
from src.llm_backend import MockLLMBackend, TransformersLLMBackend, DEFAULT_MODEL_NAME


ROOT = Path(__file__).resolve().parent.parent


def test_model_contract_preserves_default_and_explicit_qwen3_is_lazy():
    assert DEFAULT_MODEL_NAME == "Qwen/Qwen2.5-1.5B-Instruct"
    default = TransformersLLMBackend()
    qwen3 = TransformersLLMBackend(model_name=g7.MODEL_ID, device="cuda", seed=427)
    assert default.model_name == g7.BASELINE_MODEL_ID
    assert qwen3.model_name == g7.MODEL_ID and not qwen3.is_loaded
    assert qwen3.generation_config == {"do_sample": False, "max_new_tokens": 256, "batch_size": 1}
    assert qwen3.input_token_limit == 4096 and qwen3.seed == 427


def test_frozen_inputs_are_exact_ordered_g5_evidence():
    rows, frozen = g7.load_frozen_inputs(ROOT, 3)
    sample = pd.read_csv(ROOT / "outputs/tables/g5_validation_sample.csv").sort_values("sample_order")
    assert [row["question_id"] for row in rows] == sample.question_id.astype(str).tolist()[:3]
    for name, filename in g7.FROZEN_SOURCES.items():
        source = json.loads((ROOT / "outputs/predictions" / filename).read_text(encoding="utf-8"))
        assert [row["selected_evidence"] for row in frozen[name]] == [
            row["selected_evidence"] for row in source["predictions"][:3]]


def test_dense_mock_uses_answer_agent_without_critic(monkeypatch):
    rows, frozen = g7.load_frozen_inputs(ROOT, 3)
    backend = MockLLMBackend('{"answer":"Grounded [E1].","citation_labels":["E1"],"unanswerable":false}')
    record = g7.infer_one("qwen3_dense_single_agent", rows[0], frozen["qwen3_dense_single_agent"][0], backend)
    assert record["final_status"] == "accepted" and len(backend.prompts) == 1
    assert record["selected_evidence"] == frozen["qwen3_dense_single_agent"][0]["selected_evidence"]


def test_full_mock_has_at_most_one_revision():
    rows, frozen = g7.load_frozen_inputs(ROOT, 3)
    responses = [
        '{"answer":"Grounded [E1].","citation_labels":["E1"],"unanswerable":false}',
        '{"verdict":"revise","supported":false,"missing_citations":[],"unsupported_claims":["claim"],"revision_instruction":"Revise."}',
        '{"answer":"Revised [E1].","citation_labels":["E1"],"unanswerable":false}',
        '{"verdict":"accept","supported":true,"missing_citations":[],"unsupported_claims":[],"revision_instruction":""}',
    ]
    record = g7.infer_one("qwen3_full_multi_agent", rows[0], frozen["qwen3_full_multi_agent"][0],
                          MockLLMBackend(responses))
    assert record["final_status"] == "accepted_revised"
    assert len(record["attempt_history"]) == 2


def test_evaluation_refuses_to_load_gold_until_all_predictions_exist(tmp_path, monkeypatch):
    called = False
    def forbidden(_root):
        nonlocal called
        called = True
    monkeypatch.setattr(g7, "_official_module", forbidden)
    paths = {name: tmp_path / f"{name}.json" for name in g7.CONFIGURATIONS}
    paths[g7.CONFIGURATIONS[0]].write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="every Qwen3 prediction"):
        g7.evaluate(tmp_path, paths)
    assert not called


def test_atomic_checkpoint_resume_and_cleanup(monkeypatch, tmp_path):
    rows = [{"question_id": f"q{i}", "question": "Q?", "paper_id": "p", "split": "validation"}
            for i in range(3)]
    frozen = {name: [{"question_id": f"q{i}", "selected_evidence": []} for i in range(3)]
              for name in g7.CONFIGURATIONS}
    monkeypatch.setattr(g7, "load_frozen_inputs", lambda *args: (rows, frozen))
    monkeypatch.setattr(g7, "infer_one", lambda name, row, source, backend: {
        "question_id": row["question_id"], "paper_id": "p", "split": "validation",
        "answer": "Insufficient evidence", "unanswerable": True, "final_status": "insufficient_evidence",
        "citations": [], "selected_evidence": [], "evidence_count": 0, "runtime_seconds": 0.0,
        "attempt_history": [], "critic_verdict": None})
    paths = g7.run_inference(tmp_path, MockLLMBackend("{}"), 3, "_smoke")
    assert all(path.exists() for path in paths.values())
    assert not list((tmp_path / "outputs/checkpoints").glob("*.checkpoint.json"))
    assert not list(tmp_path.rglob("*.tmp"))
    for path in paths.values():
        text = path.read_text(encoding="utf-8").lower()
        assert "gold_answer" not in text and "gold_evidence" not in text


def test_transformers_requirement_supports_qwen3():
    requirement = (ROOT / "requirements-llm.txt").read_text(encoding="utf-8")
    assert "transformers>=4.51.0" in requirement
