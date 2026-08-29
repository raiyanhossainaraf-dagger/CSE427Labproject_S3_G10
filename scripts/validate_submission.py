"""Lightweight, read-only validation for the faculty submission bundle."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/raiyanhossainaraf-dagger/CSE427Labproject_S3_G10.git"
COLAB_URL = "https://colab.research.google.com/github/raiyanhossainaraf-dagger/CSE427Labproject_S3_G10/blob/main/CSE427_Final_Project.ipynb"
GITHUB_LIMIT = 100 * 1024 * 1024

REQUIRED = [
    "CSE427_Final_Project.ipynb", "README.md", "report/final_project_report.md",
    "scripts/validate_submission.py", "scripts/build_retrieval_corpus.py", "scripts/build_dense_index.py",
    "scripts/run_multi_agent_demo.py", "tests/test_submission.py",
    "outputs/tables/g5_answer_comparison.csv", "outputs/tables/g5_retrieval_comparison.csv",
    "outputs/tables/g5_answer_type_breakdown.csv", "outputs/tables/g5_validation_sample.csv",
    "outputs/summaries/g5_diagnostic_summary.json", "outputs/summaries/g5_experiment_summary.json",
    "outputs/predictions/g5_dense_single_agent.json", "outputs/predictions/g5_evidence_aware.json",
    "outputs/predictions/g5_full_multi_agent.json",
    "outputs/predictions/g7_qwen3_dense_single_agent.json", "outputs/predictions/g7_qwen3_full_multi_agent.json",
    "outputs/tables/g7_llm_ablation.csv", "outputs/tables/g7_llm_paired_diagnostics.csv",
    "outputs/summaries/g7_llm_ablation_summary.json",
    "figures/retrieval_comparison.png", "figures/answer_f1_comparison.png",
    "figures/runtime_comparison.png", "figures/answer_type_diagnostic.png",
]
SECTIONS = [
    "Project objective", "System architecture", "Environment and dependency setup",
    "QASPER dataset statistics and schema", "Evidence mapping and validation",
    "Paragraph-aware chunking", "Unified paragraph/section/figure-table retrieval corpus",
    "Dense index construction", "BM25, dense, hybrid RRF and evidence-aware reranking",
    "Query, Retrieval, Evidence, Answer and Critic Agents", "Small real end-to-end validation demonstration",
    "Questions, E1–E5 evidence", "Saved G5 baseline and G7 Qwen3 results", "comparison charts",
    "Paired statistical diagnostic", "Limitations, reproducibility and conclusion",
]
CONTROLS = {"QUICK_DEMO": "True", "RUN_FULL_GENERATION": "False",
            "REBUILD_INDEX_IF_MISSING": "True", "DEMO_QUESTION_COUNT": "3"}
PREDICTION_FILES = ["g5_dense_single_agent.json", "g5_evidence_aware.json", "g5_full_multi_agent.json"]
G7_PREDICTION_FILES = ["g7_qwen3_dense_single_agent.json", "g7_qwen3_full_multi_agent.json"]
QWEN3_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
G7_F1 = {"qwen3_dense_single_agent": .319854675160875,
         "qwen3_full_multi_agent": .4082184272132053,
         "qwen2_dense_single_agent": .29542506768400517,
         "qwen2_full_multi_agent": .2707444183923975}


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True)
    return [root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def walk_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).lower()
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def validate(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    for relative in REQUIRED:
        if not (root / relative).is_file(): errors.append(f"missing required file: {relative}")

    notebook_path = root / "CSE427_Final_Project.ipynb"
    notebook_text = ""
    if notebook_path.is_file():
        try:
            raw = json.loads(notebook_path.read_text(encoding="utf-8"))
            nbformat.validate(raw)
            notebook_text = "\n".join("".join(c.get("source", "")) for c in raw.get("cells", []))
        except Exception as exc: errors.append(f"invalid notebook JSON/schema: {exc}")
    for section in SECTIONS:
        if section.lower() not in notebook_text.lower(): errors.append(f"notebook section missing: {section}")
    for name, default in CONTROLS.items():
        if not re.search(rf"(?m)^\s*{name}\s*=\s*{default}\s*$", notebook_text):
            errors.append(f"notebook control/default missing: {name}={default}")
    if re.search(r"(?i)(?:(?<![A-Za-z])[A-Z]:[\\/]|[\\/](?:Users|home)[\\/])", notebook_text):
        errors.append("notebook contains an absolute local path")
    if REPO_URL not in notebook_text: errors.append("notebook repository URL is incorrect or absent")
    if QWEN3_MODEL not in notebook_text: errors.append("notebook does not explicitly use the recommended Qwen3 model")
    if "transformers>=4.51.0" not in notebook_text: errors.append("notebook does not require transformers>=4.51.0")
    if not re.search(r"(?i)cuda.*required|required.*cuda", notebook_text):
        errors.append("notebook does not state that CUDA is required")

    secret_pattern = re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|secret)\s*[=:]\s*['\"][A-Za-z0-9_\-]{12,}")
    for path in [notebook_path, root / "README.md", root / "report/final_project_report.md"]:
        if path.is_file() and secret_pattern.search(path.read_text(encoding="utf-8")):
            errors.append(f"possible secret/token in {path.relative_to(root).as_posix()}")

    forbidden_gold = {"gold", "gold_answer", "gold_answers", "gold_evidence", "reference_answer", "reference_answers"}
    for filename in PREDICTION_FILES + G7_PREDICTION_FILES:
        path = root / "outputs/predictions" / filename
        if not path.is_file(): continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        found = forbidden_gold.intersection(walk_keys(payload))
        if found: errors.append(f"forbidden gold fields in {filename}: {sorted(found)}")
        predictions = payload.get("predictions", [])
        if len(predictions) != 100: errors.append(f"{filename} does not contain exactly 100 predictions")
        if payload.get("split") != "validation" or any(p.get("split") != "validation" for p in predictions):
            errors.append(f"non-validation prediction in {filename}")

    sample_path = root / "outputs/tables/g5_validation_sample.csv"
    g7_summary_path = root / "outputs/summaries/g7_llm_ablation_summary.json"
    g7_table_path = root / "outputs/tables/g7_llm_ablation.csv"
    g7_diag_path = root / "outputs/tables/g7_llm_paired_diagnostics.csv"
    if g7_summary_path.is_file() and g7_table_path.is_file() and g7_diag_path.is_file():
        summary = json.loads(g7_summary_path.read_text(encoding="utf-8"))
        table = pd.read_csv(g7_table_path).set_index("configuration")
        results = {row["configuration"]: row for row in summary.get("results", [])}
        if summary.get("question_count") != 100 or summary.get("test_split_accessed") is not False:
            errors.append("G7 scope is not exactly 100 validation questions with test access disabled")
        if summary.get("model") != QWEN3_MODEL: errors.append("G7 summary model mismatch")
        for name, expected in G7_F1.items():
            if name not in results or name not in table.index:
                errors.append(f"missing G7 result: {name}"); continue
            if abs(float(results[name]["official_answer_f1"]) - expected) > 1e-12:
                errors.append(f"G7 summary Answer F1 mismatch: {name}")
            if abs(float(table.loc[name, "official_answer_f1"]) - expected) > 1e-12:
                errors.append(f"G7 table Answer F1 mismatch: {name}")
        expected_rows = {
            "qwen3_dense_single_agent": (100, 0, 0, 0, 5.83156263, 9023.42),
            "qwen3_full_multi_agent": (89, 4, 7, 0, 9.84021109, 9008.56),
        }
        for name, values in expected_rows.items():
            row = results.get(name, {})
            actual = tuple(row.get(key) for key in ("accepted_first_attempt", "accepted_after_revision",
                           "rejected", "insufficient", "runtime_per_question_seconds", "peak_gpu_memory_mb"))
            if actual != values or row.get("citation_label_valid_rate") != 1.0:
                errors.append(f"G7 aggregate metrics mismatch: {name}")
        expected_pairs = {
            "qwen3_dense_single_agent_vs_qwen3_full_multi_agent": (.08836375205233037, 40, 36, 24, .03294533910756293, .14950672247153504, .0018998100189981002),
            "qwen2_full_multi_agent_vs_qwen3_full_multi_agent": (.13747400882080782, 51, 26, 23, .0495685217982756, .2248217011476312, .001999800019998),
            "qwen2_dense_single_agent_vs_qwen3_dense_single_agent": (.024429607476869807, 42, 30, 28, -.04878415814611686, .09501264412081427, .5098490150984901),
        }
        comparisons = summary.get("paired_comparisons", {})
        for name, values in expected_pairs.items():
            row = comparisons.get(name, {}); ci = row.get("bootstrap_ci_95", [None, None])
            actual = (row.get("mean_difference"), row.get("wins"), row.get("ties"), row.get("losses"),
                      ci[0], ci[1], row.get("permutation_p_value_two_sided"))
            if actual != values: errors.append(f"G7 paired metrics mismatch: {name}")
        diagnostics = pd.read_csv(g7_diag_path)
        if len(diagnostics) != 300 or diagnostics.question_id.nunique() != 100:
            errors.append("G7 paired diagnostics must contain three comparisons over 100 questions")

    if sample_path.is_file():
        sample = pd.read_csv(sample_path)
        if len(sample) != 100 or sample.question_id.nunique() != 100: errors.append("sample is not exactly 100 unique questions")
        if not sample.split.eq("validation").all(): errors.append("sample contains a non-validation split")

    answer_path = root / "outputs/tables/g5_answer_comparison.csv"
    diag_path = root / "outputs/summaries/g5_diagnostic_summary.json"
    if answer_path.is_file() and diag_path.is_file():
        answer = pd.read_csv(answer_path).set_index("configuration")
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        expected = {"dense_single_agent": .29542506768400517, "evidence_aware_no_critic": .2694944183923975,
                    "full_multi_agent": .2707444183923975}
        for name, value in expected.items():
            if abs(float(answer.loc[name, "official_answer_f1"]) - value) > 1e-12: errors.append(f"Answer F1 mismatch: {name}")
            if abs(float(diag["official_answer_f1"][name]) - value) > 1e-12: errors.append(f"diagnostic Answer F1 mismatch: {name}")
        if not (answer.question_count.eq(100).all() and answer.citation_valid_record_rate.eq(1.0).all()):
            errors.append("generation aggregate counts/rates do not match verified results")

    retrieval_path = root / "outputs/tables/g5_retrieval_comparison.csv"
    if retrieval_path.is_file():
        ret = pd.read_csv(retrieval_path)
        expected = {"dense": (.5753953834050921, .4311638676523843),
                    "hybrid": (.5965328176493224, .4521732013469213),
                    "evidence_fused": (.6580198636832941, .5010479916425552)}
        for method, (recall, ndcg) in expected.items():
            rows = ret[(ret.method == method) & (ret.k == 5)]
            row = rows.iloc[-1] if len(rows) else None
            if row is None or abs(row.recall - recall) > 1e-12 or abs(row.ndcg - ndcg) > 1e-12:
                errors.append(f"retrieval aggregate mismatch: {method}")
            elif int(row.evaluated_questions) != 927: errors.append(f"retrieval scope mismatch: {method}")

    readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").is_file() else ""
    if COLAB_URL not in readme: errors.append("README Colab link is incorrect or absent")
    if REPO_URL.removesuffix(".git") not in readme: errors.append("README repository URL is incorrect or absent")

    artifact_re = re.compile(r"(?i)(?:embedding.*\.npy$|\.faiss$|\.safetensors$|\.pt$|\.pth$|pytorch_model.*\.bin$|(?:^|/)cache/)")
    try: tracked = tracked_files(root)
    except Exception as exc:
        errors.append(f"could not inspect tracked files: {exc}"); tracked = []
    for path in tracked:
        relative = path.relative_to(root).as_posix()
        if artifact_re.search(relative): errors.append(f"generated artifact/model/cache is tracked: {relative}")
        if path.is_file() and path.stat().st_size > GITHUB_LIMIT: errors.append(f"tracked file exceeds GitHub 100 MB limit: {relative}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    errors = validate(args.project_root.resolve())
    if errors:
        print("Submission validation FAILED:")
        for error in errors: print(f"- {error}")
        return 1
    print("Submission validation PASSED: notebook, results, predictions, links, splits, and tracked-file policy verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
