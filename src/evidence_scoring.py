"""Transparent per-question evidence scoring and canonical-source selection.

The fixed fused score is a ranking score, not a calibrated probability:
70% min-max normalized cross-encoder relevance, 25% min-max normalized hybrid
strength, and 5% binary BM25/dense retrieval agreement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.hybrid_retrieval import canonical_source

EVIDENCE_WEIGHTS = {"cross_encoder": 0.70, "hybrid": 0.25, "agreement": 0.05}


def minmax_normalize(values: pd.Series) -> pd.Series:
    """Min-max normalize finite scores; a constant finite group safely becomes zero."""
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(0.0, index=values.index, dtype=float)
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    if not finite.any():
        return result
    low, high = numeric[finite].min(), numeric[finite].max()
    if high > low:
        result.loc[finite] = (numeric[finite] - low) / (high - low)
    return result


def score_evidence(candidates: pd.DataFrame, method: str = "fused") -> pd.DataFrame:
    """Score and deterministically rank candidates independently within each question."""
    if method not in {"cross_encoder", "fused"}:
        raise ValueError("method must be 'cross_encoder' or 'fused'")
    if candidates.empty:
        output = candidates.copy()
        for column in ("normalized_cross_encoder_score", "normalized_hybrid_score", "agreement_score",
                       "final_evidence_score", "final_rank", "retrieved_by_both"):
            output[column] = pd.Series(dtype="bool" if column == "retrieved_by_both" else "float64")
        return output
    required = {"question_id", "document_id", "cross_encoder_score", "hybrid_score"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Evidence candidates missing columns: {sorted(missing)}")
    groups = []
    for _, group in candidates.groupby("question_id", sort=True):
        group = group.copy()
        group["normalized_cross_encoder_score"] = minmax_normalize(group["cross_encoder_score"])
        group["normalized_hybrid_score"] = minmax_normalize(group["hybrid_score"])
        both = group.get("bm25_rank", pd.Series(pd.NA, index=group.index)).notna() & group.get(
            "dense_rank", pd.Series(pd.NA, index=group.index)).notna()
        group["retrieved_by_both"] = both.astype(bool)
        group["agreement_score"] = both.astype(float)
        if method == "cross_encoder":
            group["final_evidence_score"] = group["normalized_cross_encoder_score"]
        else:
            group["final_evidence_score"] = (
                EVIDENCE_WEIGHTS["cross_encoder"] * group["normalized_cross_encoder_score"]
                + EVIDENCE_WEIGHTS["hybrid"] * group["normalized_hybrid_score"]
                + EVIDENCE_WEIGHTS["agreement"] * group["agreement_score"])
        group = group.sort_values(["final_evidence_score", "cross_encoder_score", "hybrid_rank", "document_id"],
                                  ascending=[False, False, True, True], kind="stable").reset_index(drop=True)
        group["final_rank"] = np.arange(1, len(group) + 1)
        group["rank"] = group["final_rank"]
        group["score"] = group["final_evidence_score"]
        group["method"] = "cross_encoder" if method == "cross_encoder" else "evidence_fused"
        groups.append(group)
    return pd.concat(groups, ignore_index=True)


class EvidenceSelector:
    """Select canonical-unique evidence records without source-type quotas."""

    def __init__(self, top_n: int = 5):
        if top_n < 1:
            raise ValueError("top_n must be positive")
        self.top_n = top_n

    def select(self, ranked: pd.DataFrame) -> pd.DataFrame:
        if ranked.empty:
            return ranked.copy()
        selected = []
        for _, group in ranked.groupby("question_id", sort=True):
            seen = set()
            ordered = group.sort_values(["final_rank", "document_id"], kind="stable")
            for row in ordered.to_dict("records"):
                key = canonical_source(type("Row", (), row))
                if key in seen:
                    continue
                seen.add(key); selected.append(row)
                if len([x for x in selected if str(x["question_id"]) == str(row["question_id"])]) >= self.top_n:
                    break
        output = pd.DataFrame(selected, columns=ranked.columns)
        if not output.empty:
            output["final_rank"] = output.groupby("question_id", sort=False).cumcount() + 1
            output["rank"] = output["final_rank"]
            # Explicit future Answer Agent contract while retaining the source corpus field.
            output["evidence_text"] = output["text"]
        return output
