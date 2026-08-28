"""Build the versioned unified retrieval corpus without modifying legacy chunks."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RETRIEVAL_ARTIFACT_VERSION
from src.retrieval_corpus import CORPUS_VERSION, build_retrieval_corpus, corpus_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, help="Deterministic per-source row limit for smoke builds")
    args = parser.parse_args()
    output = args.output_dir or args.processed_dir / RETRIEVAL_ARTIFACT_VERSION
    inputs = {name: pd.read_parquet(args.processed_dir / f"{name}.parquet") for name in
              ("papers", "chunks", "sections", "figures_tables")}
    if args.limit is not None:
        if args.limit < 1: parser.error("--limit must be positive")
        for name in ("chunks", "sections", "figures_tables"):
            inputs[name] = inputs[name].sort_values(inputs[name].columns[0], kind="stable").head(args.limit)
    corpus, invalid = build_retrieval_corpus(**inputs)
    output.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(output / "corpus.parquet", index=False)
    invalid.to_parquet(output / "invalid_documents.parquet", index=False)
    summary = {"corpus_version": CORPUS_VERSION, "row_count": len(corpus), "invalid_count": len(invalid),
               "fingerprint": corpus_fingerprint(corpus),
               "counts": {"|".join(key): int(value) for key, value in corpus.groupby(["split", "source_type"]).size().items()}}
    (output / "corpus_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
