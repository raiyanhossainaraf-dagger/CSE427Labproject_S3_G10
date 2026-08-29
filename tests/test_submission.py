from pathlib import Path
import json

import nbformat

from scripts.validate_submission import ROOT, validate


def test_submission_validator_passes():
    assert validate(ROOT) == []


def test_final_notebook_is_valid_and_unexecuted():
    path = ROOT / "CSE427_Final_Project.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert len(notebook.cells) >= 30
    assert all(cell.get("execution_count") is None for cell in notebook.cells if cell.cell_type == "code")
    assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")


def test_saved_inference_has_no_gold_fields():
    forbidden = {"gold", "gold_answer", "gold_answers", "gold_evidence", "reference_answer", "reference_answers"}
    paths = list((ROOT / "outputs/predictions").glob("g5_*.json")) + list(
        (ROOT / "outputs/predictions").glob("g7_*.json"))
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        keys = set()
        stack = [payload]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                keys.update(str(key).lower() for key in value); stack.extend(value.values())
            elif isinstance(value, list): stack.extend(value)
        assert not forbidden.intersection(keys), path


def test_g7_exact_metrics_and_validation_scope():
    summary = json.loads((ROOT / "outputs/summaries/g7_llm_ablation_summary.json").read_text(encoding="utf-8"))
    assert summary["question_count"] == 100
    assert summary["test_split_accessed"] is False
    assert summary["model"] == "Qwen/Qwen3-4B-Instruct-2507"
    results = {row["configuration"]: row for row in summary["results"]}
    assert results["qwen3_dense_single_agent"]["official_answer_f1"] == .319854675160875
    assert results["qwen3_full_multi_agent"]["official_answer_f1"] == .4082184272132053
    assert results["qwen3_full_multi_agent"]["accepted_first_attempt"] == 89
    assert results["qwen3_full_multi_agent"]["accepted_after_revision"] == 4
    assert results["qwen3_full_multi_agent"]["rejected"] == 7
    for filename in ("g7_qwen3_dense_single_agent.json", "g7_qwen3_full_multi_agent.json"):
        payload = json.loads((ROOT / "outputs/predictions" / filename).read_text(encoding="utf-8"))
        assert payload["split"] == "validation" and len(payload["predictions"]) == 100
        assert all(row["split"] == "validation" for row in payload["predictions"])


def test_final_notebook_qwen3_colab_contract():
    text = (ROOT / "CSE427_Final_Project.ipynb").read_text(encoding="utf-8")
    assert "Qwen/Qwen3-4B-Instruct-2507" in text
    assert "transformers>=4.51.0" in text
    assert "CUDA is required" in text
    assert "QUICK_DEMO = True" in text
    assert "RUN_FULL_GENERATION = False" in text


def test_final_figures_are_nonempty():
    for path in (ROOT / "figures").glob("*.png"):
        assert path.stat().st_size > 10_000
