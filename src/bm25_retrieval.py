import pandas as pd
import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

def tokenize(text: str) -> List[str]:
    """
    Lightweight tokenizer for BM25.
    Converts to lowercase and splits by non-alphanumeric characters while preserving basic terms.
    """
    if not isinstance(text, str):
        text = str(text)
    # Lowercase and split by whitespace
    # Using re.findall to get alphanumeric words, preserving most scientific terms
    return re.findall(r'\w+', text.lower())

class BM25Retriever:
    def __init__(self, chunks_df: pd.DataFrame):
        """
        Initializes the BM25 retriever with a dataframe of chunks.
        """
        if "text" not in chunks_df.columns:
            raise ValueError("chunks_df must contain a 'text' column.")
            
        self.chunks_df = chunks_df
        self.corpus = chunks_df["text"].tolist()
        
        print(f"Tokenizing corpus for BM25 ({len(self.corpus)} documents)...")
        self.tokenized_corpus = [tokenize(doc) for doc in self.corpus]
        
        try:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        except ImportError:
            raise ImportError("rank-bm25 is not installed. Run: pip install rank-bm25")

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """
        Searches the BM25 index for a given query and returns top-k chunks with metadata.
        """
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = pd.Series(scores).nlargest(top_k).index
        
        results = self.chunks_df.iloc[top_indices].copy()
        results["similarity_score"] = [scores[i] for i in top_indices]
        
        # Ensure consistent column order and required fields
        return results

def build_bm25_index(chunks_df: pd.DataFrame) -> BM25Retriever:
    """
    Helper function to build and return a BM25Retriever instance.
    """
    print(f"Building BM25 index for {len(chunks_df)} chunks...")
    return BM25Retriever(chunks_df)

def bm25_search(retriever: BM25Retriever, query: str, chunks_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """
    Helper function for searching using a BM25Retriever instance.
    Included for consistency with other retrieval modules.
    """
    return retriever.search(query, top_k=top_k)
