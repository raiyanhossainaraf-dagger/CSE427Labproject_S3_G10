"""Build BGE embeddings and a validated versioned FAISS IndexFlatIP."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_DENSE_MODEL, DEFAULT_EMBEDDING_BATCH_SIZE, RETRIEVAL_ARTIFACT_VERSION
from src.embeddings import embedding_dimension, encode_passages, load_embedding_model
from src.retrieval_corpus import corpus_fingerprint
from src.vector_store import create_ip_index, save_faiss_index, validate_embeddings, validate_faiss_index


def build_dense_artifacts(corpus, output_dir, model, model_name, batch_size=32, device="unknown"):
    """Encode in bounded batches, create IndexFlatIP, and save exact row mapping/manifest."""
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    dimension = embedding_dimension(model)
    embedding_path = output_dir / "passage_embeddings.f32.npy"
    embeddings = np.lib.format.open_memmap(embedding_path, mode="w+", dtype=np.float32,
                                           shape=(len(corpus), dimension))
    total_batches = (len(corpus) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(corpus), batch_size), 1):
        end = min(start + batch_size, len(corpus))
        embeddings[start:end] = encode_passages(corpus.embedding_text.iloc[start:end].tolist(), model,
                                                batch_size=batch_size, show_progress=False)
        embeddings.flush()
        if batch_number == 1 or batch_number % 25 == 0 or batch_number == total_batches:
            print(f"Embedded batch {batch_number}/{total_batches} ({end}/{len(corpus)} documents)", flush=True)
    validate_embeddings(embeddings, expected_rows=len(corpus), expected_dimension=dimension)
    index = create_ip_index(embeddings)
    index_path = output_dir / "index_flat_ip.faiss"
    save_faiss_index(index, index_path)
    mapping = corpus[["document_id", "paper_id", "split", "source_type", "source_id", "chunk_id",
                      "paragraph_id", "section_id", "figure_table_id"]].copy()
    mapping.insert(0, "row_id", np.arange(len(mapping), dtype=np.int64))
    mapping.to_parquet(output_dir / "document_map.parquet", index=False)
    validate_faiss_index(index, len(mapping), dimension)
    elapsed = time.perf_counter() - started
    manifest = {"artifact_version": RETRIEVAL_ARTIFACT_VERSION, "model": model_name, "dimension": dimension,
                "row_count": len(corpus), "corpus_fingerprint": corpus_fingerprint(corpus),
                "normalization": "L2", "index_type": "IndexFlatIP", "dtype": "float32",
                "device": device, "batch_size": batch_size, "build_seconds": elapsed}
    (output_dir / "dense_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = PROJECT_ROOT / "data" / "processed" / RETRIEVAL_ARTIFACT_VERSION
    parser.add_argument("--corpus", type=Path, default=default_dir / "corpus.parquet")
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    parser.add_argument("--model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    args = parser.parse_args()
    if args.batch_size < 1: parser.error("--batch-size must be positive")
    corpus = pd.read_parquet(args.corpus)
    if args.limit is not None:
        if args.limit < 1: parser.error("--limit must be positive")
        corpus = corpus.head(args.limit).copy()
    model = load_embedding_model(args.model, device=args.device)
    device = str(getattr(model, "device", args.device or "unknown"))
    manifest = build_dense_artifacts(corpus, args.output_dir, model, args.model, args.batch_size, device)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
