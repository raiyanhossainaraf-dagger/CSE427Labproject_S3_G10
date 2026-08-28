"""Run the frozen G5 validation generation comparison without inference leakage."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_types import SelectedEvidence
from src.answer_agent import AnswerAgent
from src.bm25_retrieval import PaperScopedBM25Retriever
from src.config import DEFAULT_DENSE_MODEL
from src.critic_agent import CriticAgent
from src.embeddings import encode_queries, load_embedding_model
from src.evidence_agent import EvidenceAgent
from src.evidence_scoring import EVIDENCE_WEIGHTS
from src.hybrid_retrieval import HybridRetriever, canonical_source
from src.llm_backend import DEFAULT_MODEL_NAME, MockLLMBackend, TransformersLLMBackend
from src.orchestrator import MultiAgentOrchestrator
from src.query_agent import QueryAgent
from src.reranker import DEFAULT_CROSS_ENCODER_MODEL, CrossEncoderReranker
from src.retrieval import DenseRetriever
from src.retrieval_agent import ExistingHybridBackend, RetrievalAgent

SEED = 427
SPLIT = "validation"
SAMPLE_SIZE = 100
CONFIGURATIONS = ("dense_single_agent", "evidence_aware_no_critic", "full_multi_agent")
PREDICTION_FILENAMES = {
    "dense_single_agent": "g5_dense_single_agent.json",
    "evidence_aware_no_critic": "g5_evidence_aware.json",
    "full_multi_agent": "g5_full_multi_agent.json",
}


def stable_sample(questions: pd.DataFrame, size: int = SAMPLE_SIZE, seed: int = SEED) -> pd.DataFrame:
    """Select from question metadata only, ordered by SHA-256(seed || question_id)."""
    required = {"question_id", "question", "paper_id", "split"}
    if required - set(questions):
        raise ValueError(f"question metadata missing columns: {sorted(required - set(questions))}")
    pool = questions.loc[questions.split.eq(SPLIT), list(required)].copy()
    if pool.question_id.astype(str).duplicated().any() or len(pool) < size:
        raise ValueError("validation question IDs must be unique and numerous enough")
    pool["sampling_hash"] = pool.question_id.astype(str).map(
        lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    return pool.sort_values(["sampling_hash", "question_id"], kind="stable").head(size).reset_index(drop=True)


def sample_artifact(sample: pd.DataFrame, total_validation_questions: int) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_order": range(1, len(sample) + 1), "question_id": sample.question_id.astype(str),
        "paper_id": sample.paper_id.astype(str), "split": SPLIT, "sampling_hash": sample.sampling_hash,
        "sampling_algorithm": "ascending_sha256(seed:question_id)", "seed": SEED,
        "sample_size": len(sample), "validation_pool_size": total_validation_questions,
        "source": "data/processed/questions.parquet (metadata only)",
    })


def _selected_from_frame(frame: pd.DataFrame, top_n: int = 5) -> list[SelectedEvidence]:
    """Canonical top-k dense sources, represented using the established G4 contract."""
    selected, seen = [], set()
    for row in frame.sort_values(["rank", "document_id"], kind="stable").itertuples(index=False):
        key = canonical_source(row)
        if key in seen:
            continue
        seen.add(key)
        selected.append(SelectedEvidence(
            citation_label=f"E{len(selected)+1}", document_id=str(row.document_id),
            source_type=str(row.source_type), source_id=str(row.source_id), paper_id=str(row.paper_id),
            title=str(row.title), section_name=str(row.section_name), evidence_text=str(row.text),
            final_rank=len(selected)+1, final_evidence_score=float(row.score), cross_encoder_score=0.0,
            normalized_cross_encoder_score=0.0, normalized_hybrid_score=float(row.score),
            agreement_score=0.0, retrieved_by_both=False, hybrid_rank=int(row.rank),
            hybrid_score=float(row.score), dense_score=float(row.score), dense_rank=int(row.rank),
            chunk_id=str(row.chunk_id), paragraph_id=str(row.paragraph_id), section_id=str(row.section_id),
            figure_table_id=str(row.figure_table_id)))
        if len(selected) == top_n:
            break
    return selected


def build_components(root: Path, backend, device: str):
    encoder = load_embedding_model(DEFAULT_DENSE_MODEL, device=device)
    dense = DenseRetriever.from_artifacts(root / "data/processed/retrieval_v2", model=encoder, require_index=True)
    if dense.manifest.get("model") != DEFAULT_DENSE_MODEL:
        raise ValueError("dense artifact model does not match the frozen model")
    bm25 = PaperScopedBM25Retriever(dense.corpus)
    hybrid = HybridRetriever(bm25, dense, candidate_depth=50)
    adapter = ExistingHybridBackend(hybrid, lambda query: encode_queries([query], encoder, show_progress=False)[0])
    retrieval = RetrievalAgent(adapter, corpus=dense.corpus, candidate_depth=50)
    evidence = EvidenceAgent(CrossEncoderReranker(device=device, batch_size=1), top_n=5)
    return {"encoder": encoder, "dense": dense, "retrieval": retrieval, "evidence": evidence,
            "answer": AnswerAgent(backend), "query": QueryAgent(), "backend": backend}


def _record_base(row, answer, evidence, runtime, status, attempts=None, critic=None):
    citations = [item.to_dict() for item in answer.citations]
    return {"question_id": str(row.question_id), "paper_id": str(row.paper_id), "split": SPLIT,
            "answer": answer.answer, "unanswerable": bool(answer.unanswerable), "final_status": status,
            "citations": citations, "selected_evidence": [item.to_dict() for item in evidence],
            "evidence_count": len(evidence), "runtime_seconds": round(runtime, 6),
            "attempt_history": attempts or [], "critic_verdict": critic}


def infer_configuration(name: str, sample: pd.DataFrame, components) -> list[dict]:
    results = []
    for index, row in enumerate(sample.itertuples(index=False), 1):
        started = time.perf_counter()
        plan, _ = components["query"].plan(str(row.question_id), row.question, str(row.paper_id), SPLIT)
        if name == "dense_single_agent":
            frame = components["dense"].search(plan.retrieval_query, plan.paper_id, SPLIT, 50, plan.question_id)
            corpus = components["dense"].corpus.set_index("document_id")[["title", "section_name", "text"]]
            evidence = _selected_from_frame(frame.join(corpus, on="document_id", validate="many_to_one"))
            answer, _ = components["answer"].answer(row.question, plan, evidence)
            status = "insufficient_evidence" if answer.unanswerable else "accepted"
            record = _record_base(row, answer, evidence, time.perf_counter()-started, status)
        elif name == "evidence_aware_no_critic":
            retrieval, _ = components["retrieval"].retrieve(plan)
            evidence, _ = components["evidence"].select(plan, retrieval)
            answer, _ = components["answer"].answer(row.question, plan, evidence)
            status = "insufficient_evidence" if answer.unanswerable else "accepted"
            record = _record_base(row, answer, evidence, time.perf_counter()-started, status)
        else:
            flow = MultiAgentOrchestrator(components["query"], components["retrieval"], components["evidence"],
                                          components["answer"], CriticAgent(components["backend"], use_llm=True))
            result = flow.run(str(row.question_id), row.question, str(row.paper_id), SPLIT)
            record = {"question_id": result.question_id, "paper_id": result.paper_id, "split": result.split,
                      "answer": result.answer, "unanswerable": result.final_status not in {"accepted", "accepted_revised"},
                      "final_status": result.final_status, "citations": result.citations,
                      "selected_evidence": result.selected_evidence, "evidence_count": len(result.selected_evidence),
                      "runtime_seconds": result.runtime, "attempt_history": result.attempt_history,
                      "critic_verdict": result.critic_verdict}
        results.append(record)
        print(f"{name}: {index}/{len(sample)}", flush=True)
    return results


def prediction_payload(name, records, backend, wall_seconds, peak_gpu_mb):
    return {"schema_version": 1, "configuration": name, "split": SPLIT, "seed": SEED,
            "sample_size": len(records), "model": DEFAULT_MODEL_NAME, "device": "cuda", "dtype": "float16",
            "generation": dict(backend.generation_config), "retrieval": configuration_metadata()[name],
            "wall_runtime_seconds": round(wall_seconds, 6), "peak_gpu_memory_mb": peak_gpu_mb,
            "predictions": records}


def configuration_metadata():
    frozen = {"candidate_depth": 50, "top_evidence_sources": 5, "dense_model": DEFAULT_DENSE_MODEL}
    return {
        "dense_single_agent": {**frozen, "retriever": "paper_scoped_dense", "cross_encoder": None,
                               "evidence_agent": False, "critic_agent": False},
        "evidence_aware_no_critic": {**frozen, "retriever": "paper_scoped_hybrid",
            "cross_encoder": DEFAULT_CROSS_ENCODER_MODEL, "evidence_weights": EVIDENCE_WEIGHTS,
            "evidence_agent": True, "critic_agent": False},
        "full_multi_agent": {**frozen, "retriever": "paper_scoped_hybrid",
            "cross_encoder": DEFAULT_CROSS_ENCODER_MODEL, "evidence_weights": EVIDENCE_WEIGHTS,
            "agents": ["query", "retrieval", "evidence", "answer", "critic"], "maximum_revisions": 1,
            "critic_fallback": "deterministic_accept_after_valid_deterministic_checks"},
    }


def _load_official_evaluator(root: Path):
    path = root / "data/raw/qasper_v0.3/qasper_evaluator.py"
    spec = importlib.util.spec_from_file_location("official_qasper_evaluator_g5", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def evaluate_saved_predictions(root: Path, paths: dict[str, Path], selected_ids: list[str]):
    """This is the sole gold-loading boundary; all prediction paths must already exist."""
    if any(not path.is_file() for path in paths.values()):
        raise RuntimeError("all configurations must be saved before gold evaluation")
    evaluator = _load_official_evaluator(root)
    with (root / "data/raw/qasper_v0.3/qasper-dev-v0.3.json").open(encoding="utf-8") as handle:
        raw_gold = json.load(handle)
    all_gold = evaluator.get_answers_and_evidence(raw_gold, False)
    gold = {qid: all_gold[qid] for qid in selected_ids}
    rows = []
    for name, path in paths.items():
        payload = json.loads(path.read_text(encoding="utf-8")); records = payload["predictions"]
        predicted = {r["question_id"]: {"answer": "Unanswerable" if r["unanswerable"] else r["answer"],
                                         "evidence": []} for r in records}
        official = evaluator.evaluate(gold, predicted)
        statuses = Counter(r["final_status"] for r in records)
        citation_valid = sum(all(c["label"] in {e["citation_label"] for e in r["selected_evidence"]}
                                 for c in r["citations"]) for r in records)
        rows.append({"configuration": name, "question_count": len(records),
            "official_answer_f1": official["Answer F1"], "citation_valid_record_count": citation_valid,
            "citation_valid_record_rate": citation_valid / len(records), "accepted_first_attempt": statuses["accepted"],
            "accepted_after_revision": statuses["accepted_revised"], "rejected": statuses["rejected"],
            "insufficient_evidence_count": statuses["insufficient_evidence"],
            "average_evidence_count": sum(r["evidence_count"] for r in records) / len(records),
            "average_runtime_seconds": sum(r["runtime_seconds"] for r in records) / len(records),
            "total_runtime_seconds": payload["wall_runtime_seconds"], "peak_gpu_memory_mb": payload["peak_gpu_memory_mb"]})
    return pd.DataFrame(rows)


def consolidate_retrieval(root: Path):
    columns = ["method", "k", "hit_rate", "precision", "recall", "evidence_f1", "mrr", "map", "ndcg",
               "evaluated_questions", "excluded_questions"]
    retrieval = pd.read_csv(root / "outputs/tables/retrieval_validation_metrics.csv")[columns]
    reranking = pd.read_csv(root / "outputs/tables/reranking_validation_metrics.csv")[columns]
    retrieval["stage"] = "retrieval"; reranking["stage"] = "reranking"
    result = pd.concat([retrieval, reranking], ignore_index=True)
    result.insert(0, "scope", "full_validation")
    return result[["scope", "stage", *columns]]


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run(root: Path, sample_size=SAMPLE_SIZE, suffix="", mock=False, local_files_only=False):
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    questions_path = root / "data/processed/questions.parquet"
    questions = pd.read_parquet(questions_path)  # metadata table: deliberately no answer/evidence columns
    sample = stable_sample(questions, sample_size)
    tables = root / "outputs/tables"; predictions_dir = root / "outputs/predictions"
    tables.mkdir(parents=True, exist_ok=True); predictions_dir.mkdir(parents=True, exist_ok=True)
    sample_path = tables / f"g5_validation_sample{suffix}.csv"
    sample_artifact(sample, int(questions.split.eq(SPLIT).sum())).to_csv(sample_path, index=False)
    backend = (MockLLMBackend('{"answer":"Insufficient evidence","citation_labels":[],"unanswerable":true}')
               if mock else TransformersLLMBackend(device="cuda", local_files_only=local_files_only))
    components = build_components(root, backend, "cuda")
    paths = {}
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None
    for name in CONFIGURATIONS:
        started = time.perf_counter(); records = infer_configuration(name, sample, components)
        peak = round(torch.cuda.max_memory_allocated() / 1048576, 2) if torch is not None and torch.cuda.is_available() else None
        filename = Path(PREDICTION_FILENAMES[name]).stem + suffix + ".json"
        paths[name] = predictions_dir / filename
        _write_json(paths[name], prediction_payload(name, records, backend, time.perf_counter()-started, peak))
    comparison = evaluate_saved_predictions(root, paths, sample.question_id.astype(str).tolist())
    comparison_path = tables / f"g5_answer_comparison{suffix}.csv"; comparison.to_csv(comparison_path, index=False)
    retrieval = consolidate_retrieval(root)
    retrieval_path = tables / f"g5_retrieval_comparison{suffix}.csv"; retrieval.to_csv(retrieval_path, index=False)
    summary = {"scope": "G5 validation only", "sample": {"seed": SEED, "size": sample_size,
        "validation_pool_size": int(questions.split.eq(SPLIT).sum()), "unique_papers": int(sample.paper_id.nunique()),
        "sample_file": str(sample_path.relative_to(root))}, "configurations": configuration_metadata(),
        "generation_experiment": comparison.to_dict("records"),
        "retrieval_results": {"scope": "full validation; separate from 100-question generation experiment",
                              "table": str(retrieval_path.relative_to(root))},
        "official_evidence_f1_reported": False, "test_split_evaluated": False}
    summary_path = root / f"outputs/summaries/g5_experiment_summary{suffix}.json"; _write_json(summary_path, summary)
    return {"sample": sample_path, "predictions": paths, "comparison": comparison_path,
            "retrieval": retrieval_path, "summary": summary_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if not args.mock and args.sample_size not in {3, SAMPLE_SIZE}:
        raise ValueError("real runs must be the three-question smoke or fixed 100-question experiment")
    print(json.dumps({key: str(value) for key, value in run(args.project_root, args.sample_size, args.suffix,
                                                            args.mock, args.local_files_only).items()}, indent=2))


if __name__ == "__main__":
    main()
