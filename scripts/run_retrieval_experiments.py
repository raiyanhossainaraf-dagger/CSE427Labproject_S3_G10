"""Run paper-scoped BM25, dense, and hybrid retrieval on QASPER validation."""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bm25_retrieval import PaperScopedBM25Retriever
from src.config import DEFAULT_DENSE_MODEL
from src.embeddings import encode_queries, load_embedding_model
from src.hybrid_retrieval import weighted_rrf
from src.retrieval import DenseRetriever
from src.retrieval_metrics import DEFAULT_K_VALUES, evaluate_retrieval


def run_experiment(processed_dir, artifact_dir, limit=None, top_k=20, candidate_depth=100,
                   batch_size=32, rrf_constant=60, bm25_weight=1.0, dense_weight=1.0,
                   device=None, model_name=DEFAULT_DENSE_MODEL):
    """Run validation-only retrieval and return predictions, metrics, and runtime metadata."""
    started = time.perf_counter()
    tables = {name: pd.read_parquet(processed_dir / f"{name}.parquet") for name in
              ("questions", "answers", "evidence", "evidence_mappings", "chunks")}
    questions = tables["questions"][tables["questions"].split.eq("validation")].sort_values(
        ["paper_id", "question_id"], kind="stable").reset_index(drop=True)
    if limit is not None: questions = questions.head(limit).copy()
    if not len(questions): raise ValueError("No validation questions selected")

    model = load_embedding_model(model_name, device=device)
    dense = DenseRetriever.from_artifacts(artifact_dir, model=model, require_index=True)
    if dense.manifest.get("model") != model_name:
        raise ValueError(f"Dense artifact model {dense.manifest.get('model')} != requested {model_name}")
    bm25 = PaperScopedBM25Retriever(dense.corpus)
    print(f"Encoding {len(questions)} validation queries...", flush=True)
    query_embeddings = encode_queries(questions.question.tolist(), model, batch_size=batch_size, show_progress=True)

    predictions = {"bm25": [], "dense": [], "hybrid": []}
    retrieval_started = time.perf_counter()
    for index, row in enumerate(questions.itertuples(index=False)):
        b = bm25.search(row.question, row.paper_id, "validation", max(candidate_depth, top_k), row.question_id)
        d = dense.search_embedding(query_embeddings[index], row.paper_id, "validation", max(candidate_depth, top_k), row.question_id)
        h = weighted_rrf(b, d, top_k, bm25_weight, dense_weight, rrf_constant)
        predictions["bm25"].append(b.head(top_k)); predictions["dense"].append(d.head(top_k)); predictions["hybrid"].append(h)
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(questions):
            print(f"Retrieved {index + 1}/{len(questions)} questions", flush=True)
    predictions = {name: pd.concat(frames, ignore_index=True) for name, frames in predictions.items()}
    retrieval_seconds = time.perf_counter() - retrieval_started

    question_ids = set(questions.question_id.astype(str))
    scoped_questions = tables["questions"][tables["questions"].question_id.astype(str).isin(question_ids)]
    results, diagnostics = {}, {}
    for method, frame in predictions.items():
        results[method] = evaluate_retrieval(frame, scoped_questions, tables["answers"], tables["evidence"],
                                             tables["evidence_mappings"], tables["chunks"],
                                             k_values=DEFAULT_K_VALUES, split="validation")
        diagnostics[method] = {}
        for source_type in ("paragraph", "section", "float"):
            mappings = tables["evidence_mappings"][tables["evidence_mappings"].source_type.eq(source_type)]
            evidence = tables["evidence"][tables["evidence"].evidence_id.isin(set(mappings.evidence_id))]
            diagnostics[method][source_type] = evaluate_retrieval(
                frame, scoped_questions, tables["answers"], evidence, mappings, tables["chunks"],
                k_values=DEFAULT_K_VALUES, split="validation", include_union=False)
    runtime = {"query_count": len(questions), "retrieval_seconds": retrieval_seconds,
               "total_seconds": time.perf_counter() - started, "device": str(getattr(model, "device", device)),
               "model": model_name, "top_k": top_k, "candidate_depth": candidate_depth,
               "rrf_constant": rrf_constant, "bm25_weight": bm25_weight, "dense_weight": dense_weight,
               "query_batch_size": batch_size, "scope": "paper", "split": "validation"}
    return predictions, results, diagnostics, runtime


def save_results(predictions, results, diagnostics, runtime, output_root, suffix="validation"):
    predictions_dir = output_root / "predictions"; tables_dir = output_root / "tables"; summaries_dir = output_root / "summaries"
    for directory in (predictions_dir, tables_dir, summaries_dir): directory.mkdir(parents=True, exist_ok=True)
    for method, frame in predictions.items():
        frame.to_parquet(predictions_dir / f"retrieval_{suffix}_{method}.parquet", index=False)
    rows = []
    for method, result in results.items():
        for k, metrics in result["metrics"].items():
            rows.append({"method": method, "k": int(k), **metrics,
                         "evaluated_questions": result["evaluated_questions"], "excluded_questions": result["excluded_questions"]})
    pd.DataFrame(rows).sort_values(["method", "k"]).to_csv(tables_dir / f"retrieval_{suffix}_metrics.csv", index=False)
    summary = {"configuration": runtime, "results": results, "source_diagnostics": diagnostics}
    (summaries_dir / f"retrieval_{suffix}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-depth", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--suffix", default="validation")
    args = parser.parse_args()
    processed = args.project_root / "data" / "processed"
    payload = run_experiment(processed, processed / "retrieval_v2", args.limit, args.top_k,
                             args.candidate_depth, args.batch_size, device=args.device)
    save_results(*payload, args.project_root / "outputs", args.suffix)
    print(json.dumps(payload[3], indent=2))


if __name__ == "__main__":
    main()
