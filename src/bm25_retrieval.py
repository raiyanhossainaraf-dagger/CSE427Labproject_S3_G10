"""Deterministic BM25 over the unified retrieval corpus, scoped by paper and split."""
import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from src.retrieval import format_predictions

def tokenize(text: object) -> List[str]:
    """Lowercase regex tokenization with no downloaded language resources."""
    return re.findall(r"\w+", str(text).lower(), flags=re.UNICODE)

class BM25Retriever:
    """BM25 retriever compatible with the legacy constructor and corpus dataframes."""
    def __init__(self, corpus_df: pd.DataFrame):
        if "text" not in corpus_df.columns:
            raise ValueError("corpus_df must contain a 'text' column")
        self.corpus_df = corpus_df.reset_index(drop=True).copy()
        self.chunks_df = self.corpus_df
        tokens = [tokenize(text) for text in self.corpus_df["text"]]
        self.bm25 = BM25Okapi(tokens) if len(self.corpus_df) else None

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """Return deterministic results; score ties break by stable document/source ID."""
        if top_k < 1: raise ValueError("top_k must be positive")
        if self.bm25 is None:
            return self.corpus_df.assign(score=pd.Series(dtype=float), rank=pd.Series(dtype=int)).head(0)
        scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float64)
        result = self.corpus_df.copy(); result["score"] = scores
        tie = "document_id" if "document_id" in result else ("chunk_id" if "chunk_id" in result else result.columns[0])
        result = result.sort_values(["score", tie], ascending=[False, True], kind="stable").head(top_k).copy()
        result.insert(0, "rank", np.arange(1, len(result) + 1))
        result["similarity_score"] = result["score"]
        return result

class PaperScopedBM25Retriever:
    """Lazily cache one small BM25 index per requested (split, paper_id)."""
    def __init__(self, corpus_df: pd.DataFrame):
        self.corpus = corpus_df.reset_index(drop=True)
        self._groups = self.corpus.groupby(["split", "paper_id"], sort=False).indices
        self._cache: Dict[Tuple[str, str], BM25Retriever] = {}

    def search(self, query: str, paper_id: str, split: str, top_k: int = 20, question_id: str = "") -> pd.DataFrame:
        key = (str(split), str(paper_id))
        if key not in self._groups:
            return format_predictions(pd.DataFrame(), question_id, paper_id, split, "bm25")
        if key not in self._cache:
            self._cache[key] = BM25Retriever(self.corpus.iloc[self._groups[key]])
        return format_predictions(self._cache[key].search(query, top_k), question_id, paper_id, split, "bm25", bm25=True)

def build_bm25_index(chunks_df: pd.DataFrame) -> BM25Retriever:
    return BM25Retriever(chunks_df)

def bm25_search(retriever: BM25Retriever, query: str, chunks_df=None, top_k: int = 5) -> pd.DataFrame:
    return retriever.search(query, top_k=top_k)

