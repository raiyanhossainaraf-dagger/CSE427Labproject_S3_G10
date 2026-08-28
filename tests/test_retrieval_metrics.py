import math

import pandas as pd
import pytest

from src.retrieval_metrics import evaluate_retrieval, normalize_predictions


def dataset(evidence_specs, extra_questions=False):
    questions = [{"question_id": "q1", "paper_id": "p1", "split": "test"}]
    if extra_questions:
        questions.append({"question_id": "q-empty", "paper_id": "p1", "split": "test"})
    annotations = sorted({spec[0] for spec in evidence_specs})
    answers = pd.DataFrame([{"annotation_id": aid, "question_id": "q1", "paper_id": "p1", "split": "test"} for aid in annotations])
    evidence_rows, mapping_rows = [], []
    for index, (annotation_id, evidence_id, candidates) in enumerate(evidence_specs):
        evidence_rows.append({"evidence_id": evidence_id, "annotation_id": annotation_id, "question_id": "q1",
                              "paper_id": "p1", "split": "test"})
        for source_type, source_id in candidates:
            mapping_rows.append({"evidence_id": evidence_id, "paper_id": "p1", "split": "test", "source_type": source_type,
                                 "paragraph_id": source_id if source_type == "paragraph" else "",
                                 "section_id": source_id if source_type == "section" else "",
                                 "float_id": source_id if source_type == "float" else ""})
    chunks = pd.DataFrame([
        {"chunk_id": "c1", "paper_id": "p1", "split": "test", "section_id": "s1", "paragraph_ids": ["p1"]},
        {"chunk_id": "c2", "paper_id": "p1", "split": "test", "section_id": "s2", "paragraph_ids": ["p2"]},
        {"chunk_id": "foreign", "paper_id": "p2", "split": "test", "section_id": "sx", "paragraph_ids": ["px"]},
    ])
    return (pd.DataFrame(questions), answers, pd.DataFrame(evidence_rows), pd.DataFrame(mapping_rows), chunks)


def evaluate(predictions, specs, k=(1,), extra_questions=False, **kwargs):
    return evaluate_retrieval(pd.DataFrame(predictions), *dataset(specs, extra_questions), k_values=k, split="test", **kwargs)


def metric(result, name, k=1):
    return result["metrics"][str(k)][name]


def test_hand_calculated_single_gold_and_perfect_ranking():
    result = evaluate([{"question_id": "q1", "paper_id": "p1", "rank": 1, "score": .9, "chunk_id": "c1"}],
                      [("a1", "e1", [("paragraph", "p1")])], k=(1, 3))
    assert all(metric(result, name) == 1.0 for name in ("hit_rate", "precision", "recall", "evidence_f1", "mrr", "map", "ndcg"))
    assert metric(result, "precision", 3) == pytest.approx(1 / 3)
    assert metric(result, "evidence_f1", 3) == pytest.approx(0.5)


def test_multiple_gold_hand_calculation():
    predictions = [
        {"question_id": "q1", "paper_id": "p1", "rank": 1, "chunk_id": "c1"},
        {"question_id": "q1", "paper_id": "p1", "rank": 2, "section_id": "wrong"},
        {"question_id": "q1", "paper_id": "p1", "rank": 3, "chunk_id": "c2"},
    ]
    result = evaluate(predictions, [("a1", "e1", [("paragraph", "p1")]), ("a1", "e2", [("paragraph", "p2")])], k=(3,))
    assert metric(result, "precision", 3) == pytest.approx(2 / 3)
    assert metric(result, "recall", 3) == 1.0
    assert metric(result, "evidence_f1", 3) == pytest.approx(0.8)
    assert metric(result, "map", 3) == pytest.approx(5 / 6)
    assert metric(result, "ndcg", 3) == pytest.approx((1 + 0.5) / (1 + 1 / math.log2(3)))


def test_ambiguous_candidates_count_one_evidence_once():
    specs = [("a1", "e1", [("paragraph", "p1"), ("paragraph", "p2")])]
    predictions = [{"question_id": "q1", "paper_id": "p1", "rank": 1, "chunk_id": "c2"},
                   {"question_id": "q1", "paper_id": "p1", "rank": 2, "chunk_id": "c1"}]
    result = evaluate(predictions, specs, k=(2,))
    assert metric(result, "recall", 2) == 1.0
    assert metric(result, "precision", 2) == 0.5


def test_multiple_annotations_best_reference_and_union_diagnostic():
    specs = [("a1", "e1", [("paragraph", "p1")]), ("a2", "e2", [("paragraph", "p2")])]
    prediction = [{"question_id": "q1", "paper_id": "p1", "rank": 1, "chunk_id": "c2"}]
    result = evaluate(prediction, specs)
    assert metric(result, "recall") == 1.0
    assert result["union_diagnostic"]["1"]["recall"] == 0.5


def test_paragraph_section_and_float_sources():
    specs = [("a1", "ep", [("paragraph", "p1")]), ("a1", "es", [("section", "s2")]),
             ("a1", "ef", [("float", "f1")])]
    predictions = [{"question_id": "q1", "paper_id": "p1", "rank": 1, "chunk_id": "c1"},
                   {"question_id": "q1", "paper_id": "p1", "rank": 2, "chunk_id": "c2"},
                   {"question_id": "q1", "paper_id": "p1", "rank": 3, "float_id": "f1"}]
    result = evaluate(predictions, specs, k=(3,))
    assert metric(result, "recall", 3) == 1.0


def test_duplicate_results_and_duplicate_ranks_are_normalized():
    specs = [("a1", "e1", [("paragraph", "p1")])]
    predictions = [{"question_id": "q1", "paper_id": "p1", "rank": 1, "score": 1.0, "chunk_id": "c1"},
                   {"question_id": "q1", "paper_id": "p1", "rank": 1, "score": 0.5, "chunk_id": "c1"}]
    normalized = normalize_predictions(pd.DataFrame(predictions), dataset(specs)[-1])
    assert normalized["rank"].tolist() == [1]
    result = evaluate(predictions, specs, k=(2,))
    assert metric(result, "recall", 2) == 1.0
    assert metric(result, "precision", 2) == 0.5


def test_cross_paper_false_positive_and_completely_incorrect_are_zero():
    specs = [("a1", "e1", [("paragraph", "px")])]
    prediction = [{"question_id": "q1", "paper_id": "p2", "rank": 1, "chunk_id": "foreign"}]
    result = evaluate(prediction, specs)
    assert all(metric(result, name) == 0.0 for name in ("hit_rate", "precision", "recall", "evidence_f1", "mrr", "map", "ndcg"))


def test_empty_predictions_and_questions_without_evidence():
    result = evaluate([], [("a1", "e1", [("paragraph", "p1")])], extra_questions=True)
    assert result["evaluated_questions"] == 1
    assert result["excluded_questions"] == 1
    assert all(metric(result, name) == 0.0 for name in ("hit_rate", "precision", "recall", "evidence_f1", "mrr", "map", "ndcg"))


def test_deterministic_results_and_prediction_validation():
    specs = [("a1", "e1", [("paragraph", "p1")])]
    predictions = [{"question_id": "q1", "paper_id": "p1", "rank": 2, "chunk_id": "c1"}]
    assert evaluate(predictions, specs) == evaluate(predictions, specs)
    with pytest.raises(ValueError, match="positive integers"):
        evaluate([{**predictions[0], "rank": 0}], specs)
    with pytest.raises(ValueError, match="disagrees"):
        evaluate([{**predictions[0], "paper_id": "p2"}], specs)

    tied = [{"question_id": "q1", "paper_id": "p1", "rank": 1, "score": 0.5, "chunk_id": "c2"},
            {"question_id": "q1", "paper_id": "p1", "rank": 1, "score": 0.5, "chunk_id": "c1"}]
    forward = normalize_predictions(pd.DataFrame(tied), dataset(specs)[-1])
    reversed_rows = normalize_predictions(pd.DataFrame(list(reversed(tied))), dataset(specs)[-1])
    pd.testing.assert_frame_equal(forward, reversed_rows)


def test_split_mixing_is_rejected():
    specs = [("a1", "e1", [("paragraph", "p1")])]
    prediction = [{"question_id": "q1", "paper_id": "p1", "rank": 1, "chunk_id": "c1", "split": "train"}]
    with pytest.raises(ValueError, match="another split"):
        evaluate(prediction, specs)
