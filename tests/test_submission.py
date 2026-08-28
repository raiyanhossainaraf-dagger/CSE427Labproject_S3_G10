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
    for path in (ROOT / "outputs/predictions").glob("g5_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        keys = set()
        stack = [payload]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                keys.update(str(key).lower() for key in value); stack.extend(value.values())
            elif isinstance(value, list): stack.extend(value)
        assert not forbidden.intersection(keys), path


def test_final_figures_are_nonempty():
    for path in (ROOT / "figures").glob("*.png"):
        assert path.stat().st_size > 10_000
