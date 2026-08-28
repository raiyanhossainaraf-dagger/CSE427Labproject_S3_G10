import numpy as np
import pandas as pd
from typing import List, Dict
from src.retrieval_metrics import evaluate_retrieval

def calculate_recall_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """Calculates Recall@K."""
    if not ground_truth_ids:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    hits = len(set(retrieved_k) & set(ground_truth_ids))
    return hits / len(ground_truth_ids)

def calculate_mrr(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
    """Calculates Mean Reciprocal Rank (MRR)."""
    for i, rid in enumerate(retrieved_ids):
        if rid in ground_truth_ids:
            return 1.0 / (i + 1)
    return 0.0

def calculate_precision_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> float:
    """Calculates Precision@K."""
    if k == 0:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    hits = len(set(retrieved_k) & set(ground_truth_ids))
    return hits / k

def evaluate_retriever(retriever_func, questions_df: pd.DataFrame, evidence_df: pd.DataFrame, k_list=[5, 10]):
    """
    Evaluates a retriever function over a set of questions.
    retriever_func: A function that takes a query and returns a list of chunk_ids.
    """
    metrics = {f"Recall@{k}": [] for k in k_list}
    metrics["MRR"] = []
    
    # Group evidence by question_id for ground truth
    # In QASPER, evidence paragraphs are free text, we need to map them back to chunks.
    # However, for Milestone 3 evaluation, we might need a mapping or use a simplified 
    # check (e.g., if retrieved chunk contains the evidence text).
    
    # For now, let's assume we are checking if the paper_id matches and if the evidence text 
    # is present in the retrieved chunk.
    
    # Pre-process ground truth: map question_id to its evidence texts
    gt = evidence_df.groupby('question_id')['evidence_text'].apply(list).to_dict()
    
    for _, row in questions_df.iterrows():
        qid = row['question_id']
        query = row['question']
        
        if qid not in gt:
            continue
            
        gold_texts = gt[qid]
        retrieved_results = retriever_func(query, top_k=max(k_list))
        retrieved_chunks_texts = retrieved_results['text'].tolist()
        
        # Binary check for hits: Does the retrieved chunk contain any of the gold evidence texts?
        # Note: This is an approximation for QASPER since exact chunk mapping is complex.
        hits = []
        for chk_text in retrieved_chunks_texts:
            hit = any(gold in chk_text or chk_text in gold for gold in gold_texts)
            hits.append(hit)
            
        # Calculate Recall@K
        for k in k_list:
            # A hit at K is if any of the first K retrieved chunks is a hit
            k_hits = hits[:k]
            recall = 1.0 if any(k_hits) else 0.0 # Simplified Recall: is at least one evidence item found?
            metrics[f"Recall@{k}"].append(recall)
            
        # Calculate MRR
        mrr = 0.0
        for i, h in enumerate(hits):
            if h:
                mrr = 1.0 / (i + 1)
                break
        metrics["MRR"].append(mrr)
        
    # Average metrics
    avg_metrics = {m: np.mean(vals) for m, vals in metrics.items()}
    return avg_metrics
