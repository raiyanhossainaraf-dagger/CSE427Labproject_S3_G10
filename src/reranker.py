"""Leakage-free cross-encoder reranking of source-deduplicated retrieval results."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def select_device(cuda_available: Optional[Callable[[], bool]] = None) -> str:
    """Select CUDA when available, otherwise CPU (injectable for offline tests)."""
    if cuda_available is None:
        import torch
        cuda_available = torch.cuda.is_available
    return "cuda" if cuda_available() else "cpu"


def document_text(row: object) -> str:
    """Build model input exclusively from title, section heading, and source text."""
    get = row.get if isinstance(row, dict) else lambda key, default="": getattr(row, key, default)
    return f"Title: {get('title', '')}\nSection: {get('section_name', '')}\nPassage: {get('text', '')}"


class CrossEncoderReranker:
    """Sentence Transformers CrossEncoder wrapper with deterministic ranking."""

    def __init__(self, model_name: str = DEFAULT_CROSS_ENCODER_MODEL, device: Optional[str] = None,
                 batch_size: int = 32, max_length: int = 512, model=None):
        if batch_size < 1 or max_length < 1:
            raise ValueError("batch_size and max_length must be positive")
        self.model_name = model_name
        self.device = device or select_device()
        self.batch_size = batch_size
        self.max_length = max_length
        if model is None:
            try:
                from sentence_transformers import CrossEncoder
                model = CrossEncoder(model_name, device=self.device, max_length=max_length)
            except Exception as exc:
                raise RuntimeError(f"Could not load requested cross-encoder '{model_name}' on {self.device}: {exc}") from exc
        self.model = model
        if hasattr(self.model, "model") and hasattr(self.model.model, "eval"):
            self.model.model.eval()

    def rerank(self, question: str, candidates: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
        """Rerank up to top_k candidates and preserve their complete retrieval metadata."""
        if candidates.empty:
            output = candidates.copy()
            output["hybrid_rank"] = pd.Series(dtype="int64")
            output["hybrid_score"] = pd.Series(dtype="float64")
            output["cross_encoder_score"] = pd.Series(dtype="float64")
            output["reranked_rank"] = pd.Series(dtype="int64")
            return output
        required = {"document_id", "rank", "score", "title", "section_name", "text"}
        missing = required - set(candidates.columns)
        if missing:
            raise ValueError(f"Reranking candidates missing columns: {sorted(missing)}")
        frame = candidates.sort_values(["rank", "document_id"], kind="stable").head(top_k).copy()
        frame["hybrid_rank"] = pd.to_numeric(frame["rank"]).astype(int)
        frame["hybrid_score"] = pd.to_numeric(frame.get("fused_score", frame["score"]), errors="coerce")
        pairs = [(str(question), document_text(row)) for row in frame.to_dict("records")]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if len(scores) != len(frame):
            raise ValueError(f"Cross-encoder returned {len(scores)} scores for {len(frame)} pairs")
        frame["cross_encoder_score"] = scores
        frame = frame.sort_values(["cross_encoder_score", "hybrid_rank", "document_id"],
                                  ascending=[False, True, True], kind="stable").reset_index(drop=True)
        frame["reranked_rank"] = np.arange(1, len(frame) + 1)
        frame["rank"] = frame["reranked_rank"]
        frame["score"] = frame["cross_encoder_score"]
        frame["method"] = "cross_encoder"
        return frame
