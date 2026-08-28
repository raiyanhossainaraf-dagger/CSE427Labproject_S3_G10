"""Evaluate saved retrieval predictions without building or loading retrieval indexes."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_metrics import DEFAULT_K_VALUES, evaluate_retrieval


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)


def smoke_evaluation():
    questions = pd.DataFrame([{"question_id": "q1", "paper_id": "p1", "split": "test"}])
    answers = pd.DataFrame([{"annotation_id": "a1", "question_id": "q1", "paper_id": "p1", "split": "test"}])
    evidence = pd.DataFrame([{"evidence_id": "e1", "annotation_id": "a1", "question_id": "q1", "paper_id": "p1", "split": "test"}])
    mappings = pd.DataFrame([{"evidence_id": "e1", "paper_id": "p1", "split": "test", "source_type": "paragraph",
                              "paragraph_id": "r1", "section_id": "", "float_id": ""}])
    chunks = pd.DataFrame([{"chunk_id": "c1", "paper_id": "p1", "split": "test", "section_id": "s1", "paragraph_ids": ["r1"]}])
    predictions = pd.DataFrame([{"question_id": "q1", "paper_id": "p1", "rank": 1, "retrieval_score": 1.0, "chunk_id": "c1"}])
    return evaluate_retrieval(predictions, questions, answers, evidence, mappings, chunks, k_values=(1, 3), split="test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, help="CSV or Parquet ranked predictions")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--split", help="Single dataset split to evaluate")
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument("--reference-policy", choices=("best", "union"), default="best")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Run deterministic in-memory smoke evaluation")
    args = parser.parse_args()
    if args.smoke:
        result = smoke_evaluation()
    else:
        if not args.predictions:
            parser.error("--predictions is required unless --smoke is used")
        tables = {name: pd.read_parquet(args.processed_dir / f"{name}.parquet") for name in
                  ("questions", "answers", "evidence", "evidence_mappings", "chunks")}
        result = evaluate_retrieval(_read(args.predictions), tables["questions"], tables["answers"],
                                    tables["evidence"], tables["evidence_mappings"], tables["chunks"],
                                    k_values=args.k, reference_policy=args.reference_policy, split=args.split)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
