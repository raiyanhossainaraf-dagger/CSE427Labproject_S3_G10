"""Run G3 cross-encoder and evidence-aware reranking on validation only."""

from __future__ import annotations

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
from src.evidence_scoring import EVIDENCE_WEIGHTS, EvidenceSelector, score_evidence
from src.hybrid_retrieval import weighted_rrf
from src.reranker import DEFAULT_CROSS_ENCODER_MODEL, CrossEncoderReranker
from src.retrieval import DenseRetriever
from src.retrieval_metrics import DEFAULT_K_VALUES, evaluate_retrieval


def _evaluation(frame, tables, questions):
    return evaluate_retrieval(frame, questions, tables["answers"], tables["evidence"],
                              tables["evidence_mappings"], tables["chunks"],
                              k_values=DEFAULT_K_VALUES, split="validation")


def _diagnostics(frame, tables, questions):
    result = {}
    for source_type in ("paragraph", "section", "float"):
        mappings = tables["evidence_mappings"][tables["evidence_mappings"].source_type.eq(source_type)]
        evidence = tables["evidence"][tables["evidence"].evidence_id.isin(set(mappings.evidence_id))]
        result[source_type] = evaluate_retrieval(
            frame, questions, tables["answers"], evidence, mappings, tables["chunks"],
            k_values=DEFAULT_K_VALUES, split="validation", include_union=False)
    return result


def run_experiment(processed_dir: Path, artifact_dir: Path, limit=None, candidate_depth=100,
                   candidate_count=50, retrieval_batch_size=32, reranker_batch_size=32,
                   max_length=512, device=None, model_name=DEFAULT_CROSS_ENCODER_MODEL):
    """Regenerate hybrid candidates and return baseline, CE, fused, and selected evidence."""
    started = time.perf_counter()
    names = ("questions", "answers", "evidence", "evidence_mappings", "chunks")
    tables = {name: pd.read_parquet(processed_dir / f"{name}.parquet") for name in names}
    questions = tables["questions"][tables["questions"].split.eq("validation")].sort_values(
        ["paper_id", "question_id"], kind="stable").reset_index(drop=True)
    if limit is not None:
        questions = questions.head(limit).copy()
    if questions.empty:
        raise ValueError("No validation questions selected")

    embedding_model = load_embedding_model(DEFAULT_DENSE_MODEL, device=device)
    dense = DenseRetriever.from_artifacts(artifact_dir, model=embedding_model, require_index=True)
    if dense.manifest.get("model") != DEFAULT_DENSE_MODEL:
        raise ValueError("Dense artifact model does not match the established G2 model")
    bm25 = PaperScopedBM25Retriever(dense.corpus)
    corpus_lookup = dense.corpus.set_index("document_id")[["title", "section_name", "text"]]
    reranker = CrossEncoderReranker(model_name=model_name, device=device, batch_size=reranker_batch_size,
                                    max_length=max_length)
    query_embeddings = encode_queries(questions.question.tolist(), embedding_model,
                                      batch_size=retrieval_batch_size, show_progress=True)

    hybrid_frames, reranked_frames = [], []
    rerank_started = time.perf_counter()
    for index, question in enumerate(questions.itertuples(index=False)):
        depth = max(candidate_depth, candidate_count)
        bm = bm25.search(question.question, question.paper_id, "validation", depth, question.question_id)
        de = dense.search_embedding(query_embeddings[index], question.paper_id, "validation", depth, question.question_id)
        hybrid = weighted_rrf(bm, de, candidate_count)
        enriched = hybrid.join(corpus_lookup, on="document_id", validate="many_to_one")
        reranked = reranker.rerank(question.question, enriched, candidate_count)
        hybrid_frames.append(enriched); reranked_frames.append(reranked)
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(questions):
            print(f"Reranked {index + 1}/{len(questions)} validation questions", flush=True)
    hybrid = pd.concat(hybrid_frames, ignore_index=True)
    reranked = pd.concat(reranked_frames, ignore_index=True)
    ce = score_evidence(reranked, "cross_encoder")
    fused = score_evidence(reranked, "fused")
    selected = EvidenceSelector(5).select(fused)

    scoped = tables["questions"][tables["questions"].question_id.isin(set(questions.question_id))]
    predictions = {"hybrid": hybrid, "cross_encoder": ce, "evidence_fused": fused}
    results = {name: _evaluation(frame, tables, scoped) for name, frame in predictions.items()}
    diagnostics = {name: _diagnostics(frame, tables, scoped) for name, frame in predictions.items()}
    runtime = {
        "split": "validation", "query_count": len(questions), "candidate_count": candidate_count,
        "candidate_depth": candidate_depth, "dense_model": DEFAULT_DENSE_MODEL,
        "cross_encoder_model": model_name, "device": reranker.device, "max_length": max_length,
        "retrieval_batch_size": retrieval_batch_size, "reranker_batch_size": reranker_batch_size,
        "evidence_weights": EVIDENCE_WEIGHTS, "reranking_seconds": time.perf_counter() - rerank_started,
        "total_seconds": time.perf_counter() - started,
    }
    return predictions, selected, results, diagnostics, runtime


def select_configuration(results):
    """Choose by nDCG@10, then Recall@10 and MRR@10, with stable method tie-break."""
    key = lambda name: (results[name]["metrics"]["10"]["ndcg"],
                        results[name]["metrics"]["10"]["recall"],
                        results[name]["metrics"]["10"]["mrr"])
    return max(sorted(results), key=key)


def save_results(predictions, selected, results, diagnostics, runtime, output_root: Path, suffix="validation"):
    directories = {name: output_root / name for name in ("predictions", "tables", "summaries")}
    for directory in directories.values(): directory.mkdir(parents=True, exist_ok=True)
    # k<=20 is the full evaluation range; retain top-50 candidates only for reproducibility of G3 scoring.
    for method, frame in predictions.items():
        frame.to_parquet(directories["predictions"] / f"reranking_{suffix}_{method}.parquet", index=False)
    selected.to_parquet(directories["predictions"] / f"reranking_{suffix}_selected_evidence_top5.parquet", index=False)
    rows = [{"method": method, "k": int(k), **metrics,
             "evaluated_questions": result["evaluated_questions"], "excluded_questions": result["excluded_questions"]}
            for method, result in results.items() for k, metrics in result["metrics"].items()]
    pd.DataFrame(rows).sort_values(["method", "k"]).to_csv(
        directories["tables"] / f"reranking_{suffix}_metrics.csv", index=False)
    diag_rows = [{"method": method, "source_type": source, "k": int(k), **metrics,
                  "evaluated_questions": result["evaluated_questions"], "excluded_questions": result["excluded_questions"]}
                 for method, sources in diagnostics.items() for source, result in sources.items()
                 for k, metrics in result["metrics"].items()]
    pd.DataFrame(diag_rows).sort_values(["method", "source_type", "k"]).to_csv(
        directories["tables"] / f"reranking_{suffix}_source_diagnostics.csv", index=False)
    selected_method = select_configuration(results)
    summary = {"configuration": runtime, "selection_rule": "max nDCG@10; Recall@10 then MRR@10 tie-breakers",
               "selected_configuration": selected_method, "results": results, "source_diagnostics": diagnostics}
    (directories["summaries"] / f"reranking_{suffix}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return selected_method


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--candidate-depth", type=int, default=100)
    parser.add_argument("--candidate-count", type=int, default=50)
    parser.add_argument("--retrieval-batch-size", type=int, default=32)
    parser.add_argument("--reranker-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--model", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--suffix", default="validation")
    args = parser.parse_args()
    output_root = args.output_root or args.project_root / "outputs"
    payload = run_experiment(args.project_root / "data" / "processed",
                             args.project_root / "data" / "processed" / "retrieval_v2",
                             args.limit, args.candidate_depth, args.candidate_count,
                             args.retrieval_batch_size, args.reranker_batch_size,
                             args.max_length, args.device, args.model)
    selected = save_results(*payload, output_root, args.suffix)
    print(json.dumps({"selected_configuration": selected, **payload[-1]}, indent=2))


if __name__ == "__main__":
    main()
