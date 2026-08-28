"""Deterministic source-aware weighted Reciprocal Rank Fusion."""
import numpy as np
import pandas as pd
from src.retrieval import PREDICTION_COLUMNS

def canonical_source(row):
    """Map documents to evidence-source identity, deduplicating paragraph chunk variants."""
    if row.source_type == "chunk" and str(row.paragraph_id): return "paragraph", str(row.paragraph_id)
    if row.source_type == "section": return "section", str(row.section_id)
    if row.source_type == "float": return "float", str(row.figure_table_id)
    return "chunk", str(row.source_id)

def weighted_rrf(bm25, dense, top_k=20, bm25_weight=1.0, dense_weight=1.0, rrf_constant=60):
    """Fuse canonical sources using weight/(constant + rank)."""
    if top_k < 1 or rrf_constant < 1 or bm25_weight < 0 or dense_weight < 0: raise ValueError("Invalid RRF configuration")
    method_rows = {}
    for name, frame in (("bm25", bm25), ("dense", dense)):
        chosen = {}
        for row in frame.sort_values(["rank", "document_id"], kind="stable").itertuples(index=False):
            key = canonical_source(row)
            if key not in chosen: chosen[key] = row
        method_rows[name] = chosen
    records = []
    for key in sorted(set(method_rows["bm25"]) | set(method_rows["dense"])):
        b = method_rows["bm25"].get(key); d = method_rows["dense"].get(key)
        fused = (bm25_weight / (rrf_constant + int(b.rank)) if b else 0.0) + (dense_weight / (rrf_constant + int(d.rank)) if d else 0.0)
        representative = min([row for row in (b, d) if row is not None], key=lambda row: (int(row.rank), str(row.document_id)))
        record = representative._asdict()
        record.update({"method": "hybrid", "score": fused, "fused_score": fused,
                       "bm25_score": float(b.score) if b else np.nan, "bm25_rank": int(b.rank) if b else pd.NA,
                       "dense_score": float(d.score) if d else np.nan, "dense_rank": int(d.rank) if d else pd.NA})
        records.append(record)
    output = pd.DataFrame(records)
    if output.empty: return pd.DataFrame(columns=PREDICTION_COLUMNS)
    output = output.sort_values(["fused_score", "document_id"], ascending=[False, True], kind="stable").head(top_k).copy()
    output["rank"] = np.arange(1, len(output) + 1)
    return output[PREDICTION_COLUMNS]

class HybridRetriever:
    def __init__(self, bm25_retriever, dense_retriever, candidate_depth=100, rrf_constant=60, bm25_weight=1.0, dense_weight=1.0):
        self.bm25 = bm25_retriever; self.dense = dense_retriever; self.candidate_depth = candidate_depth
        self.rrf_constant = rrf_constant; self.bm25_weight = bm25_weight; self.dense_weight = dense_weight

    def search_embedding(self, query, query_embedding, paper_id, split, top_k=20, question_id=""):
        b = self.bm25.search(query, paper_id, split, self.candidate_depth, question_id)
        d = self.dense.search_embedding(query_embedding, paper_id, split, self.candidate_depth, question_id)
        return weighted_rrf(b, d, top_k, self.bm25_weight, self.dense_weight, self.rrf_constant)

