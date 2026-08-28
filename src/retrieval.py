"""Dense retrieval and shared T06-compatible prediction formatting."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.embeddings import encode_queries, generate_embeddings
from src.retrieval_corpus import corpus_fingerprint
from src.vector_store import load_faiss_index, search_ip_index, search_similar_chunks, validate_embeddings, validate_faiss_index

PREDICTION_COLUMNS = ["question_id", "paper_id", "split", "method", "rank", "score", "document_id",
 "source_type", "source_id", "chunk_id", "paragraph_id", "section_id", "figure_table_id", "float_id",
 "bm25_score", "bm25_rank", "dense_score", "dense_rank", "fused_score"]

def format_predictions(ranked, question_id, paper_id, split, method, bm25=False, dense=False):
    """Convert ranked corpus rows to the stable T06-compatible prediction schema."""
    if ranked.empty: return pd.DataFrame(columns=PREDICTION_COLUMNS)
    output = ranked.copy()
    output["question_id"] = str(question_id); output["paper_id"] = str(paper_id)
    output["split"] = str(split); output["method"] = method
    output["rank"] = np.arange(1, len(output) + 1); output["score"] = output["score"].astype(float)
    output["float_id"] = output.get("figure_table_id", "")
    output["bm25_score"] = output["score"] if bm25 else np.nan
    output["bm25_rank"] = output["rank"] if bm25 else pd.NA
    output["dense_score"] = output["score"] if dense else np.nan
    output["dense_rank"] = output["rank"] if dense else pd.NA
    output["fused_score"] = np.nan
    for column in PREDICTION_COLUMNS:
        if column not in output:
            output[column] = "" if column.endswith("_id") or column in {"source_type", "source_id"} else np.nan
    return output[PREDICTION_COLUMNS]

class DenseRetriever:
    """Exact cosine retriever over validated G1 embeddings with paper/global modes."""
    def __init__(self, corpus, embeddings, model=None, index=None, document_map=None):
        self.corpus = corpus.reset_index(drop=True); self.embeddings = embeddings; self.model = model; self.index = index
        self.document_map = document_map.reset_index(drop=True) if document_map is not None else None
        validate_embeddings(embeddings, len(corpus))
        self._groups = self.corpus.groupby(["split", "paper_id"], sort=False).indices

    @classmethod
    def from_artifacts(cls, artifact_dir: Path, model=None, require_index=True):
        """Load only after fingerprint, row-map, embedding, and index validation."""
        artifact_dir = Path(artifact_dir)
        required = ["corpus.parquet", "corpus_manifest.json", "dense_manifest.json", "document_map.parquet", "passage_embeddings.f32.npy"]
        if require_index: required.append("index_flat_ip.faiss")
        missing = [name for name in required if not (artifact_dir / name).exists()]
        if missing: raise FileNotFoundError(f"Missing dense retrieval artifacts: {missing}")
        corpus = pd.read_parquet(artifact_dir / "corpus.parquet")
        cm = json.loads((artifact_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
        dm = json.loads((artifact_dir / "dense_manifest.json").read_text(encoding="utf-8"))
        fingerprint = corpus_fingerprint(corpus)
        if fingerprint != cm.get("fingerprint") or fingerprint != dm.get("corpus_fingerprint"):
            raise ValueError("Dense artifacts are stale: corpus fingerprint mismatch")
        embeddings = np.load(artifact_dir / "passage_embeddings.f32.npy", mmap_mode="r")
        validate_embeddings(embeddings, len(corpus), int(dm["dimension"]))
        mapping = pd.read_parquet(artifact_dir / "document_map.parquet")
        if len(mapping) != len(corpus) or mapping.document_id.tolist() != corpus.document_id.tolist():
            raise ValueError("Dense artifact document mapping is stale")
        index = load_faiss_index(artifact_dir / "index_flat_ip.faiss") if require_index else None
        if index is not None: validate_faiss_index(index, len(corpus), embeddings.shape[1])
        instance = cls(corpus, embeddings, model, index, mapping); instance.manifest = dm
        return instance

    def search_embedding(self, query_embedding, paper_id, split, top_k=20, question_id=""):
        key = (str(split), str(paper_id))
        if key not in self._groups: return format_predictions(pd.DataFrame(), question_id, paper_id, split, "dense")
        query = validate_embeddings(np.asarray(query_embedding).reshape(1, -1), expected_dimension=self.embeddings.shape[1])
        rows = np.asarray(self._groups[key], dtype=np.int64)
        ranked = self.corpus.iloc[rows].copy(); ranked["score"] = np.asarray(self.embeddings[rows] @ query[0], dtype=np.float64)
        ranked = ranked.sort_values(["score", "document_id"], ascending=[False, True], kind="stable").head(top_k)
        return format_predictions(ranked, question_id, paper_id, split, "dense", dense=True)

    def search(self, query, paper_id, split, top_k=20, question_id=""):
        if self.model is None: raise ValueError("Dense query model is not loaded")
        return self.search_embedding(encode_queries([query], self.model, show_progress=False)[0], paper_id, split, top_k, question_id)

    def global_search_embedding(self, query_embedding, top_k=20):
        """Optional global FAISS mode; excluded from primary paper-scoped evaluation."""
        if self.index is None or self.document_map is None: raise ValueError("Global FAISS index is not loaded")
        return search_ip_index(self.index, np.asarray(query_embedding).reshape(1, -1), self.document_map, top_k)

def retrieve_relevant_chunks(question, model, index, chunks_df, top_k=5):
    """Preserved legacy dense retrieval API."""
    distances, indices = search_similar_chunks(generate_embeddings([question], model), index, top_k=top_k)
    results = chunks_df.iloc[indices].copy(); results["similarity_score"] = distances
    return results

def display_retrieval_results(question, retrieved_chunks):
    print(f"\nQuestion: {question}")
    for rank, (_, row) in enumerate(retrieved_chunks.iterrows(), 1):
        print(f"Rank {rank} | Score: {row.get('similarity_score', row.get('score', 0)):.4f}")
        print(f"Paper: {row.get('title', row.get('paper_title', ''))} ({row['paper_id']})")
        print(f"Section: {row.get('section_name', '')}\nText: {row.get('text', '')[:200]}...")

