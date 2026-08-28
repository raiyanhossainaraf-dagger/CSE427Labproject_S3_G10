"""Post-hoc diagnostics for the frozen G5 validation predictions.

This script only reads saved predictions and validation gold data.  It does not
load a model or retriever, perform inference, alter predictions, or inspect test
records.
"""

from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILES = {
    "dense_single_agent": "g5_dense_single_agent.json",
    "evidence_aware_no_critic": "g5_evidence_aware.json",
    "full_multi_agent": "g5_full_multi_agent.json",
}
SEED = 427
N_RESAMPLES = 10_000
TYPE_NAMES = {"extractive": "extractive", "abstractive": "abstractive",
              "boolean": "yes/no", "none": "unanswerable"}


def load_official_evaluator(root: Path):
    path = root / "data/raw/qasper_v0.3/qasper_evaluator.py"
    spec = importlib.util.spec_from_file_location("official_qasper_evaluator_g5_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def validate_prediction_payloads(payloads: dict, sample: pd.DataFrame) -> list[str]:
    expected = sample.sort_values("sample_order").question_id.astype(str).tolist()
    if len(expected) != 100 or len(set(expected)) != 100:
        raise ValueError("G5 sample must contain exactly 100 unique ordered question IDs")
    if set(sample.split.astype(str)) != {"validation"}:
        raise ValueError("G5 sample contains a non-validation record")
    for name, payload in payloads.items():
        records = payload.get("predictions", [])
        ids = [str(row["question_id"]) for row in records]
        splits = {str(row.get("split")) for row in records}
        if payload.get("split") != "validation" or splits != {"validation"}:
            raise ValueError(f"{name} contains a non-validation record")
        if ids != expected:
            raise ValueError(f"{name} does not have the same ordered G5 question IDs")
    return expected


def per_question_official(records: list[dict], gold: dict, evaluator) -> pd.DataFrame:
    """Reproduce the official evaluator's max-over-references answer scoring."""
    rows = []
    for record in records:
        qid = str(record["question_id"])
        prediction = "Unanswerable" if record["unanswerable"] else record["answer"]
        scored = [(evaluator.token_f1_score(prediction, ref["answer"]), ref["type"])
                  for ref in gold[qid]]
        best_f1, best_type = sorted(scored, key=lambda item: item[0], reverse=True)[0]
        rows.append({"question_id": qid, "answer_f1": float(best_f1),
                     "answer_type": TYPE_NAMES[best_type]})
    return pd.DataFrame(rows)


def paired_statistics(differences, seed: int = SEED, samples: int = N_RESAMPLES) -> dict:
    differences = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    # The paired randomization null swaps dense/full labels, equivalent to random signs.
    signs = rng.choice(np.array([-1.0, 1.0]), size=(samples, len(differences)))
    permuted = (signs * differences).mean(axis=1)
    observed = float(differences.mean())
    return {
        "mean_difference": observed,
        "wins": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "losses": int((differences < 0).sum()),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "bootstrap_ci_95": [float(x) for x in np.quantile(bootstrap, [.025, .975])],
        "permutation_samples": samples,
        "permutation_p_value_two_sided": float((np.count_nonzero(
            np.abs(permuted) >= abs(observed)) + 1) / (samples + 1)),
    }


def evidence_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        for position, item in enumerate(record["selected_evidence"], 1):
            source_type = str(item["source_type"])
            rows.append({"question_id": str(record["question_id"]), "paper_id": str(record["paper_id"]),
                         "split": str(record["split"]), "rank": int(item.get("final_rank", position)),
                         "score": item.get("final_evidence_score"),
                         "chunk_id": item.get("chunk_id", "") if source_type == "chunk" else "",
                         "paragraph_id": item.get("paragraph_id", "") if source_type == "chunk" else "",
                         "section_id": item.get("section_id", "") if source_type == "section" else "",
                         "float_id": item.get("figure_table_id", "") if source_type == "float" else ""})
    return pd.DataFrame(rows)


def source_counts(records: list[dict], prefix: str) -> pd.DataFrame:
    rows = []
    for record in records:
        counts = Counter({"chunk": 0, "section": 0, "float": 0})
        counts.update(str(item["source_type"]) for item in record["selected_evidence"])
        nonzero = sum(counts[key] > 0 for key in ("chunk", "section", "float"))
        rows.append({"question_id": str(record["question_id"]),
                     f"{prefix}_paragraph_count": counts["chunk"],
                     f"{prefix}_section_count": counts["section"],
                     f"{prefix}_figure_table_count": counts["float"],
                     f"{prefix}_mixed_sources": nonzero > 1})
    return pd.DataFrame(rows)


def mixed_source_analysis(paired: pd.DataFrame) -> dict:
    mixed = paired.full_multi_agent_mixed_sources.astype(bool).to_numpy()
    delta = paired.f1_difference.to_numpy(dtype=float)
    correlation = float(np.corrcoef(mixed.astype(float), delta)[0, 1]) if mixed.any() and (~mixed).any() else None
    return {
        "definition": "full prediction uses more than one of paragraph, section, figure/table",
        "mixed_question_count": int(mixed.sum()),
        "single_source_question_count": int((~mixed).sum()),
        "mean_f1_change_mixed": float(delta[mixed].mean()) if mixed.any() else None,
        "mean_f1_change_single_source": float(delta[~mixed].mean()) if (~mixed).any() else None,
        "point_biserial_correlation_with_f1_change": correlation,
        "interpretation_constraint": "association only; no causal claim",
    }


def run(root: Path = PROJECT_ROOT) -> dict:
    prediction_dir = root / "outputs/predictions"
    payloads = {name: json.loads((prediction_dir / filename).read_text(encoding="utf-8"))
                for name, filename in CONFIG_FILES.items()}
    sample = pd.read_csv(root / "outputs/tables/g5_validation_sample.csv")
    ordered_ids = validate_prediction_payloads(payloads, sample)

    evaluator = load_official_evaluator(root)
    raw_gold = json.loads((root / "data/raw/qasper_v0.3/qasper-dev-v0.3.json").read_text(encoding="utf-8"))
    all_gold = evaluator.get_answers_and_evidence(raw_gold, False)
    gold = {qid: all_gold[qid] for qid in ordered_ids}
    per_config = {name: per_question_official(payload["predictions"], gold, evaluator)
                  for name, payload in payloads.items()}

    # Independently call the unmodified official aggregate evaluator as a second path.
    official_aggregate = {}
    for name, payload in payloads.items():
        predicted = {str(row["question_id"]): {
            "answer": "Unanswerable" if row["unanswerable"] else row["answer"], "evidence": []}
            for row in payload["predictions"]}
        official_aggregate[name] = float(evaluator.evaluate(gold, predicted)["Answer F1"])

    saved_summary = json.loads((root / "outputs/summaries/g5_experiment_summary.json").read_text(encoding="utf-8"))
    saved_f1 = {row["configuration"]: float(row["official_answer_f1"])
                for row in saved_summary["generation_experiment"]}
    aggregate = {name: float(frame.answer_f1.mean()) for name, frame in per_config.items()}
    for name in CONFIG_FILES:
        if not np.isclose(aggregate[name], saved_f1[name], rtol=0, atol=1e-15):
            raise AssertionError(f"recomputed {name} F1 differs from saved G5 value")
        if not np.isclose(official_aggregate[name], aggregate[name], rtol=0, atol=1e-15):
            raise AssertionError(f"official aggregate and per-question paths disagree for {name}")

    questions = pd.read_parquet(root / "data/processed/questions.parquet")
    selected_questions = questions[questions.question_id.astype(str).isin(ordered_ids)].copy()
    selected_questions["question_id"] = selected_questions.question_id.astype(str)
    selected_questions = selected_questions.set_index("question_id").loc[ordered_ids].reset_index()
    if len(selected_questions) != 100 or set(selected_questions.split.astype(str)) != {"validation"}:
        raise AssertionError("selected processed questions are not exactly 100 validation records")

    paired = selected_questions[["question_id", "question"]].copy()
    for name, frame in per_config.items():
        keyed = frame.set_index("question_id")
        paired[f"{name}_f1"] = paired.question_id.map(keyed.answer_f1)
        paired[f"{name}_answer_type"] = paired.question_id.map(keyed.answer_type)
        paired = paired.merge(source_counts(payloads[name]["predictions"], name), on="question_id", validate="one_to_one")
    paired["f1_difference"] = paired.full_multi_agent_f1 - paired.dense_single_agent_f1
    # The official type associated with the full-system score is used for paired case listings.
    paired["answer_type"] = paired.full_multi_agent_answer_type
    paired.insert(0, "sample_order", np.arange(1, len(paired) + 1))

    from src.retrieval_metrics import evaluate_retrieval
    processed = {name: pd.read_parquet(root / f"data/processed/{name}.parquet") for name in
                 ("answers", "evidence", "evidence_mappings", "chunks")}
    retrieval = {}
    for name, payload in payloads.items():
        result = evaluate_retrieval(evidence_frame(payload["predictions"]), selected_questions,
            processed["answers"], processed["evidence"], processed["evidence_mappings"], processed["chunks"],
            k_values=(5,), reference_policy="best", include_union=False, split="validation")
        retrieval[name] = {"recall_at_5": result["metrics"]["5"]["recall"],
                           "evaluated_questions": result["evaluated_questions"],
                           "excluded_questions": result["excluded_questions"]}

    breakdown_rows = []
    for name, frame in per_config.items():
        for answer_type in ("extractive", "abstractive", "yes/no", "unanswerable"):
            values = frame.loc[frame.answer_type.eq(answer_type), "answer_f1"]
            breakdown_rows.append({"configuration": name, "answer_type": answer_type,
                                   "question_count": int(len(values)),
                                   "official_answer_f1": float(values.mean()) if len(values) else 0.0})
    breakdown = pd.DataFrame(breakdown_rows)
    statistics = paired_statistics(paired.f1_difference)
    paired_by_type = {}
    for answer_type, group in paired.groupby("answer_type", sort=True):
        delta = group.f1_difference
        paired_by_type[answer_type] = {"question_count": int(len(group)),
            "mean_f1_difference": float(delta.mean()), "wins": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()), "losses": int((delta < 0).sum())}
    composition = mixed_source_analysis(paired)

    table_dir = root / "outputs/tables"
    summary_dir = root / "outputs/summaries"
    table_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(table_dir / "g5_paired_diagnostics.csv", index=False)
    breakdown.to_csv(table_dir / "g5_answer_type_breakdown.csv", index=False)
    case_columns = ["question_id", "question", "dense_single_agent_f1", "full_multi_agent_f1",
                    "f1_difference", "answer_type", "dense_single_agent_paragraph_count",
                    "dense_single_agent_section_count", "dense_single_agent_figure_table_count",
                    "full_multi_agent_paragraph_count", "full_multi_agent_section_count",
                    "full_multi_agent_figure_table_count"]
    paired.nsmallest(10, ["f1_difference", "sample_order"])[case_columns].to_csv(
        table_dir / "g5_largest_regressions.csv", index=False)
    paired.nlargest(10, ["f1_difference", "sample_order"])[case_columns].to_csv(
        table_dir / "g5_largest_improvements.csv", index=False)

    summary = {
        "scope": "post-hoc G5 validation diagnostics only",
        "seed": SEED,
        "question_count": 100,
        "ordered_question_ids_match": True,
        "test_records_present": False,
        "official_answer_f1": aggregate,
        "independent_official_evaluator_answer_f1": official_aggregate,
        "saved_g5_answer_f1": saved_f1,
        "aggregate_f1_exactly_reproduced": True,
        "dense_to_full": statistics,
        "dense_to_full_by_full_official_answer_type": paired_by_type,
        "retrieval_recall_at_5_t06": retrieval,
        "evidence_source_composition": {
            "categories": {"chunk": "paragraph", "section": "section", "float": "figure/table"},
            "totals": {name: source_counts(payloads[name]["predictions"], name).filter(like="_count").sum().to_dict()
                       for name in CONFIG_FILES},
            "mixed_source_association": composition,
        },
        "answer_type_note": ("Official max-over-gold-reference type is computed separately for each configuration; "
                             "paired case tables use the type selected for the full-system official score."),
        "gold_or_sensitive_content_saved": {"gold_answers": False, "hidden_prompts_or_reasoning": False},
    }
    (summary_dir / "g5_diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
