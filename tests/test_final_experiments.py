import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_final_experiments as g5


def question_frame(count=120):
    return pd.DataFrame({"question_id": [f"q{i:03}" for i in range(count)],
                         "question": [f"Question {i}?" for i in range(count)],
                         "paper_id": [f"p{i % 9}" for i in range(count)],
                         "split": ["validation"] * count})


def test_stable_sample_is_exact_seeded_and_input_order_independent():
    questions = question_frame()
    first = g5.stable_sample(questions, 100)
    second = g5.stable_sample(questions.sample(frac=1, random_state=8), 100)
    assert first.question_id.tolist() == second.question_id.tolist()
    assert len(first) == first.question_id.nunique() == 100
    expected = questions.assign(sampling_hash=questions.question_id.map(
        lambda qid: __import__("hashlib").sha256(f"427:{qid}".encode()).hexdigest()))
    assert first.question_id.tolist() == expected.sort_values(
        ["sampling_hash", "question_id"], kind="stable").head(100).question_id.tolist()


def test_sampling_rejects_duplicate_or_too_small_pool():
    with pytest.raises(ValueError):
        g5.stable_sample(question_frame(2), 3)
    duplicated = pd.concat([question_frame(3), question_frame(3).head(1)])
    with pytest.raises(ValueError):
        g5.stable_sample(duplicated, 3)


def test_frozen_configuration_contract():
    configs = g5.configuration_metadata()
    assert tuple(configs) == g5.CONFIGURATIONS
    assert configs["dense_single_agent"]["cross_encoder"] is None
    assert configs["dense_single_agent"]["critic_agent"] is False
    assert configs["evidence_aware_no_critic"]["evidence_weights"] == {
        "cross_encoder": .70, "hybrid": .25, "agreement": .05}
    assert configs["full_multi_agent"]["maximum_revisions"] == 1
    assert configs["full_multi_agent"]["agents"] == ["query", "retrieval", "evidence", "answer", "critic"]


def test_evaluation_waits_for_every_prediction(tmp_path):
    paths = {name: tmp_path / f"{name}.json" for name in g5.CONFIGURATIONS}
    for path in list(paths.values())[:-1]:
        path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="all configurations"):
        g5.evaluate_saved_predictions(tmp_path, paths, ["q1"])


def test_mocked_run_saves_all_predictions_before_evaluation(monkeypatch, tmp_path):
    (tmp_path / "data/processed").mkdir(parents=True)
    questions = question_frame(5)
    monkeypatch.setattr(pd, "read_parquet", lambda path: questions.copy())
    monkeypatch.setattr(g5, "build_components", lambda *args: {"backend": g5.MockLLMBackend("{}")})
    def fake_infer(name, sample, components):
        return [{"question_id": q.question_id, "paper_id": q.paper_id, "split": "validation",
                 "answer": "Insufficient evidence", "unanswerable": True,
                 "final_status": "insufficient_evidence", "citations": [], "selected_evidence": [],
                 "evidence_count": 0, "runtime_seconds": 0.1, "attempt_history": [],
                 "critic_verdict": None} for q in sample.itertuples(index=False)]
    monkeypatch.setattr(g5, "infer_configuration", fake_infer)
    monkeypatch.setattr(g5, "consolidate_retrieval", lambda root: pd.DataFrame(
        [{"scope": "full_validation", "stage": "retrieval", "method": "dense", "k": 5}]))
    observed = {}
    def fake_evaluate(root, paths, selected_ids):
        observed["all_exist"] = all(path.exists() for path in paths.values())
        observed["ids"] = selected_ids
        return pd.DataFrame([{"configuration": name, "official_answer_f1": 0.0}
                             for name in g5.CONFIGURATIONS])
    monkeypatch.setattr(g5, "evaluate_saved_predictions", fake_evaluate)
    outputs = g5.run(tmp_path, sample_size=3, suffix="smoke", mock=True)
    assert observed == {"all_exist": True, "ids": g5.stable_sample(questions, 3).question_id.tolist()}
    assert all(path.exists() for path in outputs["predictions"].values())
    sample = pd.read_csv(outputs["sample"])
    assert len(sample) == 3 and set(sample.split) == {"validation"}
    for path in outputs["predictions"].values():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
        assert all("question" not in record and "gold" not in json.dumps(record).lower()
                   for record in payload["predictions"])


def test_retrieval_consolidation_labels_full_validation(tmp_path):
    tables = tmp_path / "outputs/tables"; tables.mkdir(parents=True)
    row = {"method": "dense", "k": 5, "hit_rate": .1, "precision": .1, "recall": .1,
           "evidence_f1": .1, "mrr": .1, "map": .1, "ndcg": .1,
           "evaluated_questions": 927, "excluded_questions": 78}
    pd.DataFrame([row]).to_csv(tables / "retrieval_validation_metrics.csv", index=False)
    pd.DataFrame([{**row, "method": "evidence_fused"}]).to_csv(
        tables / "reranking_validation_metrics.csv", index=False)
    result = g5.consolidate_retrieval(tmp_path)
    assert set(result.scope) == {"full_validation"}
    assert set(result.stage) == {"retrieval", "reranking"}
    assert set(result.evaluated_questions) == {927}
