import numpy as np
import pandas as pd
from typing import List, Dict
from src.embeddings import generate_embeddings
from src.vector_store import search_similar_chunks

def retrieve_relevant_chunks(
    question: str,
    model,
    index,
    chunks_df: pd.DataFrame,
    top_k: int = 5
) -> pd.DataFrame:
    """
    Retrieves the top-k relevant chunks for a given question.
    """
    # Generate embedding for the question
    q_emb = generate_embeddings([question], model)
    
    # Search FAISS index
    distances, indices = search_similar_chunks(q_emb, index, top_k=top_k)
    
    # Get metadata for the retrieved chunks
    retrieved_chunks = chunks_df.iloc[indices].copy()
    retrieved_chunks['similarity_score'] = distances
    
    return retrieved_chunks

def display_retrieval_results(question: str, retrieved_chunks: pd.DataFrame):
    """
    Displays retrieval results in a readable format.
    """
    print(f"\nQuestion: {question}")
    print("-" * 50)
    for i, (_, row) in enumerate(retrieved_chunks.iterrows()):
        print(f"Rank {i+1} | Score: {row['similarity_score']:.4f}")
        print(f"Paper: {row['paper_title']} ({row['paper_id']})")
        print(f"Section: {row['section_name']}")
        print(f"Text: {row['text'][:200]}...")
        print("-" * 30)
