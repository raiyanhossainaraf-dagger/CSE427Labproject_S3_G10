import json

import numpy as np
import pandas as pd
import pytest

from scripts import analyze_g5_results as diagnostics


def test_paired_statistics_are_deterministic_and_directional():
    differences = np.array([1.0, 0.0, -0.5, 0.25])
    first = diagnostics.paired_statistics(differences, samples=1000)
    second = diagnostics.paired_statistics(differences, samples=1000)
    assert first == second
    assert first["mean_difference"] == pytest.approx(0.1875)
    assert (first["wins"], first["ties"], first["losses"]) == (2, 1, 1)
    assert 0 <= first["permutation_p_value_two_sided"] <= 1


def test_validation_requires_same_ordered_validation_ids():
    sample = pd.DataFrame({"sample_order": range(1, 101),
                           "question_id": [f"q{i}" for i in range(100)],
                           "split": ["validation"] * 100})
    records = [{"question_id": qid, "split": "validation"} for qid in sample.question_id]
    payloads = {"a": {"split": "validation", "predictions": records}}
    assert diagnostics.validate_prediction_payloads(payloads, sample) == sample.question_id.tolist()
    bad = json.loads(json.dumps(payloads))
    bad["a"]["predictions"][0]["split"] = "test"
    with pytest.raises(ValueError, match="non-validation"):
        diagnostics.validate_prediction_payloads(bad, sample)
    reversed_payload = {"a": {"split": "validation", "predictions": records[::-1]}}
    with pytest.raises(ValueError, match="ordered"):
        diagnostics.validate_prediction_payloads(reversed_payload, sample)


def test_evidence_conversion_and_source_counts_do_not_include_text():
    records = [{"question_id": "q", "paper_id": "p", "split": "validation",
                "selected_evidence": [
                    {"source_type": "chunk", "chunk_id": "c", "paragraph_id": "r", "final_rank": 1,
                     "final_evidence_score": .8, "evidence_text": "must not be copied"},
                    {"source_type": "float", "figure_table_id": "f", "final_rank": 2,
                     "final_evidence_score": .7, "evidence_text": "must not be copied"}]}]
    frame = diagnostics.evidence_frame(records)
    assert "evidence_text" not in frame
    assert frame.chunk_id.tolist() == ["c", ""]
    assert frame.float_id.tolist() == ["", "f"]
    counts = diagnostics.source_counts(records, "x").iloc[0]
    assert counts.x_paragraph_count == 1 and counts.x_figure_table_count == 1
    assert bool(counts.x_mixed_sources)
