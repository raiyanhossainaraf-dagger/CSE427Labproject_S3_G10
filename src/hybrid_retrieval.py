import numpy as np
import pandas as pd
from typing import List
from src.bm25_retrieval import BM25Retriever
from src.retrieval import retrieve_relevant_chunks
from sklearn.preprocessing import MinMaxScaler

class HybridRetriever:
    def __init__(self, bm25_retriever: BM25Retriever, dense_model, dense_index, chunks_df: pd.DataFrame):
        self.bm25_retriever = bm25_retriever
        self.dense_model = dense_model
        self.dense_index = dense_index
        self.chunks_df = chunks_df
        self.scaler = MinMaxScaler()

    def search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> pd.DataFrame:
        """
        Performs hybrid search by combining BM25 and Dense retrieval scores.
        alpha: Weight for dense retrieval (0.0 to 1.0).
        """
        # 1. Get BM25 results (get more than top_k for better fusion)
        bm25_results = self.bm25_retriever.search(query, top_k=min(top_k * 5, len(self.chunks_df)))
        
        # 2. Get Dense results
        dense_results = retrieve_relevant_chunks(
            query, 
            self.dense_model, 
            self.dense_index, 
            self.chunks_df, 
            top_k=min(top_k * 5, len(self.chunks_df))
        )

        # 3. Score Fusion
        # Create a mapping of chunk_id to fused score
        # Since FAISS IndexFlatL2 returns distances (lower is better), 
        # we convert to similarity (higher is better) for fusion.
        
        # Normalize scores
        bm25_scores = bm25_results[['chunk_id', 'similarity_score']].set_index('chunk_id')
        dense_scores = dense_results[['chunk_id', 'similarity_score']].set_index('chunk_id')
        
        # For FAISS L2 distance, lower is better. We'll invert it: similarity = 1 / (1 + distance)
        dense_scores['similarity_score'] = 1 / (1 + dense_scores['similarity_score'])
        
        # Scale scores to [0, 1] for fair fusion
        if not bm25_scores.empty:
            bm25_scores['norm_score'] = self.scaler.fit_transform(bm25_scores[['similarity_score']])
        else:
            bm25_scores['norm_score'] = 0.0
            
        if not dense_scores.empty:
            dense_scores['norm_score'] = self.scaler.fit_transform(dense_scores[['similarity_score']])
        else:
            dense_scores['norm_score'] = 0.0

        # Combine
        all_ids = list(set(bm25_scores.index) | set(dense_scores.index))
        fused_results = []
        
        for cid in all_ids:
            b_score = bm25_scores.loc[cid, 'norm_score'] if cid in bm25_scores.index else 0.0
            d_score = dense_scores.loc[cid, 'norm_score'] if cid in dense_scores.index else 0.0
            
            fused_score = alpha * d_score + (1 - alpha) * b_score
            fused_results.append({'chunk_id': cid, 'similarity_score': fused_score})
            
        fused_df = pd.DataFrame(fused_results).sort_values('similarity_score', ascending=False).head(top_k)
        
        # Join with metadata
        final_results = fused_df.merge(self.chunks_df, on='chunk_id')
        
        return final_results
