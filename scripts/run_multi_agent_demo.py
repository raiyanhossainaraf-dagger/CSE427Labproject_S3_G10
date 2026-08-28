"""Run a compact G4B mock or real validation demonstration (never reads gold answers)."""

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

from src.answer_agent import AnswerAgent
from src.bm25_retrieval import PaperScopedBM25Retriever
from src.config import DEFAULT_DENSE_MODEL
from src.critic_agent import CriticAgent
from src.embeddings import encode_queries, load_embedding_model
from src.evidence_agent import EvidenceAgent
from src.hybrid_retrieval import HybridRetriever
from src.llm_backend import MockLLMBackend, TransformersLLMBackend
from src.orchestrator import MultiAgentOrchestrator
from src.query_agent import QueryAgent
from src.reranker import CrossEncoderReranker
from src.retrieval import DenseRetriever
from src.retrieval_agent import ExistingHybridBackend, RetrievalAgent


def build_flow(root: Path, backend, device=None, use_llm_critic=True):
    artifact_dir = root / "data" / "processed" / "retrieval_v2"
    encoder = load_embedding_model(DEFAULT_DENSE_MODEL, device=device)
    dense = DenseRetriever.from_artifacts(artifact_dir, model=encoder, require_index=True)
    bm25 = PaperScopedBM25Retriever(dense.corpus)
    hybrid = HybridRetriever(bm25, dense, candidate_depth=50)
    adapter = ExistingHybridBackend(hybrid, lambda query: encode_queries([query], encoder)[0])
    retrieval = RetrievalAgent(adapter, corpus=dense.corpus, candidate_depth=50)
    evidence = EvidenceAgent(CrossEncoderReranker(device=device, batch_size=1), top_n=5)
    return MultiAgentOrchestrator(QueryAgent(), retrieval, evidence, AnswerAgent(backend),
                                  CriticAgent(backend, use_llm=use_llm_critic))


def summarize(results, backend, started):
    statuses = {}
    verdicts = {}
    valid = 0
    fallback_accepts = 0
    schema_failures = 0
    for result in results:
        statuses[result.final_status] = statuses.get(result.final_status, 0) + 1
        name = result.critic_verdict["verdict"]
        verdicts[name] = verdicts.get(name, 0) + 1
        labels = {item["citation_label"] for item in result.selected_evidence}
        valid += int(all(citation["label"] in labels for citation in result.citations))
        fallback_accepts += int(any(attempt["fallback_used"] and attempt["critic_verdict"] == "accept"
                                    for attempt in result.attempt_history))
        schema_failures += sum(attempt["llm_critic_valid"] is False for attempt in result.attempt_history)
    gpu = {"available": False, "device": "cpu", "peak_memory_mb": None}
    try:
        import torch
        if torch.cuda.is_available():
            gpu = {"available": True, "device": torch.cuda.get_device_name(0),
                   "peak_memory_mb": round(torch.cuda.max_memory_allocated() / 1048576, 2)}
    except ImportError:
        pass
    return {"model_name": backend.model_name, "model_device": getattr(backend, "device", "mock"),
            "model_load_seconds": round(getattr(backend, "load_seconds", 0.0), 3),
            "local_files_only": getattr(backend, "local_files_only", False),
            "generation_configuration": backend.generation_config,
            "question_count": len(results), "status_counts": statuses, "critic_verdict_counts": verdicts,
            "deterministic_fallback_accepts": fallback_accepts, "critic_schema_failures": schema_failures,
            "citation_valid_results": valid, "average_runtime_seconds": round(sum(x.runtime for x in results) / len(results), 3),
            "wall_runtime_seconds": round(time.perf_counter() - started, 3), "hardware": gpu}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mock", "real"), default="mock")
    parser.add_argument("--limit", type=int, default=5, choices=range(5, 11))
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--local-files-only", action="store_true",
                        help="Disable Hugging Face downloads and use only locally cached model files.")
    args = parser.parse_args()
    questions = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "questions.parquet")
    # This table contains question metadata only; answers/evidence labels are deliberately never loaded.
    questions = questions[questions.split.eq("validation")].sort_values(
        ["paper_id", "question_id"], kind="stable").head(args.limit)
    backend = (TransformersLLMBackend(device=args.device, local_files_only=args.local_files_only) if args.mode == "real" else
        MockLLMBackend('{"answer":"Insufficient evidence","citation_labels":[],"unanswerable":true}'))
    flow = build_flow(PROJECT_ROOT, backend, args.device, use_llm_critic=args.mode == "real")
    started = time.perf_counter()
    results = [flow.run(row.question_id, row.question, row.paper_id, "validation")
               for row in questions.itertuples(index=False)]
    payload = {"summary": summarize(results, backend, started), "results": [item.to_dict() for item in results]}
    filename = "g4b_qwen_validation_demo.json" if args.mode == "real" else "g4b_mock_validation_demo.json"
    output = PROJECT_ROOT / "outputs" / "predictions" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"saved": str(output), **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
