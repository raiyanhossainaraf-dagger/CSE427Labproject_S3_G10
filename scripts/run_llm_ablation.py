"""Frozen-evidence Qwen3 generation ablation for the ordered G5 validation sample."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_g5_results import paired_statistics, per_question_official
from src.agent_types import SelectedEvidence
from src.answer_agent import AnswerAgent
from src.critic_agent import CriticAgent
from src.llm_backend import MockLLMBackend, TransformersLLMBackend
from src.query_agent import QueryAgent

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
BASELINE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 427
SPLIT = "validation"
SAMPLE_SIZE = 100
CONFIGURATIONS = ("qwen3_dense_single_agent", "qwen3_full_multi_agent")
FROZEN_SOURCES = {
    "qwen3_dense_single_agent": "g5_dense_single_agent.json",
    "qwen3_full_multi_agent": "g5_full_multi_agent.json",
}
BASELINES = {
    "qwen2_dense_single_agent": "g5_dense_single_agent.json",
    "qwen2_full_multi_agent": "g5_full_multi_agent.json",
}


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen_inputs(root: Path, sample_size: int) -> tuple[list[dict], dict[str, list[dict]]]:
    """Load question metadata and frozen G5 evidence only (never gold fields)."""
    sample = pd.read_csv(root / "outputs/tables/g5_validation_sample.csv").sort_values("sample_order")
    if len(sample) != SAMPLE_SIZE or sample.question_id.astype(str).nunique() != SAMPLE_SIZE:
        raise ValueError("the frozen G5 sample must contain 100 unique ordered questions")
    if set(sample.split.astype(str)) != {SPLIT} or sample_size not in {3, SAMPLE_SIZE}:
        raise ValueError("only the ordered validation smoke (3) or frozen experiment (100) is allowed")
    ordered_ids = sample.question_id.astype(str).tolist()[:sample_size]
    questions = pd.read_parquet(root / "data/processed/questions.parquet",
                                columns=["question_id", "question", "paper_id", "split"])
    questions["question_id"] = questions.question_id.astype(str)
    keyed = questions.set_index("question_id", verify_integrity=True)
    selected = keyed.loc[ordered_ids].reset_index()
    if selected.question_id.tolist() != ordered_ids or set(selected.split.astype(str)) != {SPLIT}:
        raise ValueError("question metadata does not match the ordered validation sample")
    rows = selected[["question_id", "question", "paper_id", "split"]].to_dict("records")

    frozen = {}
    for name, filename in FROZEN_SOURCES.items():
        payload = _load_json(root / "outputs/predictions" / filename)
        records = payload.get("predictions", [])[:sample_size]
        if payload.get("model") != BASELINE_MODEL_ID:
            raise ValueError(f"{filename} is not the completed Qwen2.5 baseline")
        if [str(row["question_id"]) for row in records] != ordered_ids:
            raise ValueError(f"{filename} does not match the frozen ordered G5 sample")
        if any(str(row.get("split")) != SPLIT for row in records):
            raise ValueError(f"{filename} contains a non-validation record")
        frozen[name] = records
    return rows, frozen


def _evidence(items: list[dict]) -> list[SelectedEvidence]:
    return [SelectedEvidence(**item) for item in items]


def _record(row: dict, evidence: list[SelectedEvidence], answer, status: str, runtime: float,
            attempts=None, verdict=None) -> dict:
    citations = [item.to_dict() for item in answer.citations] if status in {"accepted", "accepted_revised"} else []
    return {"question_id": str(row["question_id"]), "paper_id": str(row["paper_id"]), "split": SPLIT,
            "answer": answer.answer if status in {"accepted", "accepted_revised"} else "Insufficient evidence",
            "unanswerable": status not in {"accepted", "accepted_revised"}, "final_status": status,
            "citations": citations, "selected_evidence": [item.to_dict() for item in evidence],
            "evidence_count": len(evidence), "runtime_seconds": round(runtime, 6),
            "attempt_history": attempts or [], "critic_verdict": verdict}


def infer_one(name: str, row: dict, frozen_record: dict, backend) -> dict:
    started = time.perf_counter()
    plan, _ = QueryAgent().plan(str(row["question_id"]), str(row["question"]), str(row["paper_id"]), SPLIT)
    evidence = _evidence(frozen_record["selected_evidence"])
    answer_agent = AnswerAgent(backend)
    if name == "qwen3_dense_single_agent":
        answer, _ = answer_agent.answer(str(row["question"]), plan, evidence)
        status = "insufficient_evidence" if answer.unanswerable else "accepted"
        return _record(row, evidence, answer, status, time.perf_counter() - started)

    critic = CriticAgent(backend, use_llm=True)
    attempts, instruction, final_verdict = [], "", None
    for attempt_number in (1, 2):
        answer, _ = answer_agent.answer(str(row["question"]), plan, evidence, instruction)
        final_verdict, _ = critic.review(str(row["question"]), plan, answer, evidence)
        attempts.append({"attempt_number": attempt_number, "answer": answer.answer,
                         "validated_citations": list(answer.citation_labels),
                         "critic_verdict": final_verdict.verdict,
                         "revision_instruction": final_verdict.revision_instruction,
                         "deterministic_checks_passed": not final_verdict.deterministic_failures,
                         "deterministic_failures": list(final_verdict.deterministic_failures),
                         "critic_mode": final_verdict.critic_mode,
                         "llm_critic_valid": final_verdict.llm_critic_valid,
                         "critic_schema_error": final_verdict.critic_schema_error,
                         "fallback_used": final_verdict.fallback_used})
        if final_verdict.verdict == "accept":
            status = "accepted" if attempt_number == 1 else "accepted_revised"
            break
        if final_verdict.verdict == "insufficient" and answer.status != "error":
            status = "insufficient_evidence"
            break
        if attempt_number == 1:
            instruction = final_verdict.revision_instruction or "Fix grounding and citations."
        else:
            status = "rejected"
    return _record(row, evidence, answer, status, time.perf_counter() - started,
                   attempts, final_verdict.to_dict() if final_verdict else None)


def _output_path(root: Path, name: str, suffix: str) -> Path:
    return root / "outputs/predictions" / f"g7_{name}{suffix}.json"


def run_inference(root: Path, backend, sample_size: int = SAMPLE_SIZE, suffix: str = "") -> dict[str, Path]:
    rows, frozen = load_frozen_inputs(root, sample_size)
    checkpoint_dir = root / "outputs/checkpoints"
    paths = {}
    for name in CONFIGURATIONS:
        output = _output_path(root, name, suffix)
        checkpoint = checkpoint_dir / f"g7_{name}{suffix}.checkpoint.json"
        if output.exists():
            completed = _load_json(output)
            completed_ids = [str(record["question_id"]) for record in completed.get("predictions", [])]
            expected_ids = [str(row["question_id"]) for row in rows]
            if (completed.get("model") == MODEL_ID and completed.get("configuration") == name
                    and completed_ids == expected_ids):
                paths[name] = output
                checkpoint.unlink(missing_ok=True)
                print(f"{name}: using complete saved prediction file", flush=True)
                continue
            raise ValueError(f"existing output is incompatible; refusing to overwrite: {output}")
        records = []
        wall_elapsed = 0.0
        saved_peak = 0.0
        if checkpoint.exists():
            saved = _load_json(checkpoint)
            if saved.get("model") != MODEL_ID or saved.get("configuration") != name:
                raise ValueError(f"incompatible checkpoint: {checkpoint}")
            records = saved["predictions"]
            wall_elapsed = float(saved.get("wall_runtime_seconds", 0.0))
            saved_peak = float(saved.get("peak_gpu_memory_mb") or 0.0)
        expected_ids = [str(row["question_id"]) for row in rows]
        if [str(record["question_id"]) for record in records] != expected_ids[:len(records)]:
            raise ValueError(f"checkpoint order mismatch: {checkpoint}")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            phase_started = time.perf_counter()
            payload = {"schema_version": 1, "configuration": name, "split": SPLIT, "seed": SEED,
                       "sample_size": sample_size, "model": MODEL_ID, "device": "cuda", "dtype": "float16",
                       "generation": dict(backend.generation_config), "frozen_evidence_source": FROZEN_SOURCES[name],
                       "wall_runtime_seconds": wall_elapsed, "predictions": records}
            for index in range(len(records), len(rows)):
                records.append(infer_one(name, rows[index], frozen[name][index], backend))
                payload["wall_runtime_seconds"] = wall_elapsed + time.perf_counter() - phase_started
                current_peak = (torch.cuda.max_memory_allocated() / 1048576 if torch.cuda.is_available() else 0.0)
                payload["peak_gpu_memory_mb"] = round(max(saved_peak, current_peak), 2) or None
                _atomic_json(checkpoint, payload)
                print(f"{name}: {index + 1}/{sample_size}", flush=True)
            payload["wall_runtime_seconds"] = round(float(payload["wall_runtime_seconds"]), 6)
            current_peak = torch.cuda.max_memory_allocated() / 1048576 if torch.cuda.is_available() else 0.0
            payload["peak_gpu_memory_mb"] = round(max(saved_peak, current_peak), 2) or None
            _atomic_json(output, payload)
            checkpoint.unlink(missing_ok=True)
            paths[name] = output
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError("CUDA memory is insufficient; stopped without fallback or substitution") from exc
            raise
    return paths


def _official_module(root: Path):
    path = root / "data/raw/qasper_v0.3/qasper_evaluator.py"
    spec = importlib.util.spec_from_file_location("official_qasper_evaluator_g7", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def evaluate(root: Path, qwen3_paths: dict[str, Path], suffix: str = "") -> dict[str, Path]:
    """Separate gold-loading phase; called only after both Qwen3 predictions exist."""
    if any(not path.is_file() for path in qwen3_paths.values()):
        raise RuntimeError("save every Qwen3 prediction before evaluation")
    payloads = {name: _load_json(path) for name, path in qwen3_paths.items()}
    ids = [str(row["question_id"]) for row in payloads[CONFIGURATIONS[0]]["predictions"]]
    for name, filename in BASELINES.items():
        baseline = _load_json(root / "outputs/predictions" / filename)
        baseline["predictions"] = baseline["predictions"][:len(ids)]
        payloads[name] = baseline
    for name, payload in payloads.items():
        if [str(row["question_id"]) for row in payload["predictions"]] != ids:
            raise ValueError(f"ordered IDs differ for {name}")
    evaluator = _official_module(root)
    raw_gold = _load_json(root / "data/raw/qasper_v0.3/qasper-dev-v0.3.json")
    all_gold = evaluator.get_answers_and_evidence(raw_gold, False)
    gold = {qid: all_gold[qid] for qid in ids}
    frames = {name: per_question_official(payload["predictions"], gold, evaluator)
              for name, payload in payloads.items()}
    rows = []
    for name, payload in payloads.items():
        records = payload["predictions"]
        statuses = Counter(row["final_status"] for row in records)
        valid = sum(all(citation["label"] in {item["citation_label"] for item in row["selected_evidence"]}
                        for citation in row["citations"]) for row in records)
        rows.append({"configuration": name, "model": payload["model"], "question_count": len(records),
                     "official_answer_f1": float(frames[name].answer_f1.mean()),
                     "citation_label_valid_count": valid, "citation_label_valid_rate": valid / len(records),
                     "accepted_first_attempt": statuses["accepted"],
                     "accepted_after_revision": statuses["accepted_revised"], "rejected": statuses["rejected"],
                     "insufficient": statuses["insufficient_evidence"],
                     "runtime_per_question_seconds": sum(r["runtime_seconds"] for r in records) / len(records),
                     "total_runtime_seconds": payload["wall_runtime_seconds"],
                     "peak_gpu_memory_mb": payload.get("peak_gpu_memory_mb")})
    comparison = pd.DataFrame(rows)
    pairs = (("qwen2_dense_single_agent", "qwen3_dense_single_agent"),
             ("qwen2_full_multi_agent", "qwen3_full_multi_agent"),
             ("qwen3_dense_single_agent", "qwen3_full_multi_agent"))
    diagnostics, pair_summary = [], {}
    for left, right in pairs:
        left_f = frames[left].set_index("question_id").loc[ids].answer_f1.to_numpy()
        right_f = frames[right].set_index("question_id").loc[ids].answer_f1.to_numpy()
        delta = right_f - left_f
        label = f"{left}_vs_{right}"
        stats = paired_statistics(delta, seed=SEED, samples=10_000)
        pair_summary[label] = stats
        diagnostics.extend({"comparison": label, "sample_order": i + 1, "question_id": qid,
                            "left_f1": float(left_f[i]), "right_f1": float(right_f[i]),
                            "f1_difference_right_minus_left": float(delta[i])}
                           for i, qid in enumerate(ids))
    tables = root / "outputs/tables"; summaries = root / "outputs/summaries"
    tables.mkdir(parents=True, exist_ok=True); summaries.mkdir(parents=True, exist_ok=True)
    table_path = tables / f"g7_llm_ablation{suffix}.csv"
    diagnostic_path = tables / f"g7_llm_paired_diagnostics{suffix}.csv"
    summary_path = summaries / f"g7_llm_ablation_summary{suffix}.json"
    comparison.to_csv(table_path, index=False)
    pd.DataFrame(diagnostics).to_csv(diagnostic_path, index=False)
    _atomic_json(summary_path, {"scope": "frozen-evidence G5 validation generation ablation",
        "seed": SEED, "question_count": len(ids), "model": MODEL_ID, "test_split_accessed": False,
        "generation": {"do_sample": False, "max_input_tokens": 4096, "max_new_tokens": 256,
                       "batch_size": 1, "dtype": "float16", "device": "cuda"},
        "results": comparison.to_dict("records"), "paired_comparisons": pair_summary,
        "gold_fields_in_predictions_or_checkpoints": False})
    return {"table": table_path, "diagnostics": diagnostic_path, "summary": summary_path}


def run(root: Path = PROJECT_ROOT, sample_size: int = SAMPLE_SIZE, suffix: str = "", mock: bool = False):
    backend = (MockLLMBackend('{"answer":"Insufficient evidence","citation_labels":[],"unanswerable":true}')
               if mock else TransformersLLMBackend(model_name=MODEL_ID, device="cuda", max_new_tokens=256,
                                                    input_token_limit=4096, seed=SEED))
    predictions = run_inference(root, backend, sample_size, suffix)
    evaluated = evaluate(root, predictions, suffix)
    return {"predictions": predictions, **evaluated}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--sample-size", type=int, choices=(3, SAMPLE_SIZE), default=SAMPLE_SIZE)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    if args.sample_size == 3 and not args.suffix:
        raise ValueError("smoke runs require a suffix so final outputs cannot be overwritten")
    result = run(args.project_root.resolve(), args.sample_size, args.suffix, args.mock)
    print(json.dumps({key: ({k: str(v) for k, v in value.items()} if isinstance(value, dict) else str(value))
                      for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
