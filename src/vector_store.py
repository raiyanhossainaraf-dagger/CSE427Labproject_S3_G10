try:
    import faiss
except ImportError:
    faiss = None

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List

def create_faiss_index(embeddings: np.ndarray):
    """
    Creates a FAISS CPU index for the given embeddings.
    """
    if faiss is None:
        raise ImportError("faiss-cpu is not installed. Please run: pip install faiss-cpu")
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    print(f"FAISS index created with {index.ntotal} vectors.")
    return index


def validate_embeddings(embeddings: np.ndarray, expected_rows: int | None = None,
                        expected_dimension: int | None = None, atol: float = 1e-3) -> np.ndarray:
    """Validate a finite float matrix whose rows are approximately L2-normalized."""
    values = np.asarray(embeddings)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Embeddings must be a non-empty two-dimensional matrix")
    if expected_rows is not None and values.shape[0] != expected_rows:
        raise ValueError(f"Embedding row count {values.shape[0]} != corpus row count {expected_rows}")
    if expected_dimension is not None and values.shape[1] != expected_dimension:
        raise ValueError(f"Embedding dimension {values.shape[1]} != expected {expected_dimension}")
    if not np.isfinite(values).all():
        raise ValueError("Embeddings contain non-finite values")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1.0, atol=atol):
        raise ValueError("Embeddings are not L2-normalized")
    return np.ascontiguousarray(values, dtype=np.float32)


def create_ip_index(embeddings: np.ndarray):
    """Create an exact IndexFlatIP over normalized float32 vectors (cosine search)."""
    if faiss is None:
        raise ImportError("faiss-cpu is not installed. Please run: pip install faiss-cpu")
    values = validate_embeddings(embeddings)
    index = faiss.IndexFlatIP(values.shape[1])
    index.add(values)
    validate_faiss_index(index, len(values), values.shape[1])
    return index


def validate_faiss_index(index, expected_rows: int, expected_dimension: int) -> None:
    """Validate FAISS count, dimension, and required exact inner-product type."""
    if index.ntotal != expected_rows:
        raise ValueError(f"FAISS ntotal {index.ntotal} != expected {expected_rows}")
    if index.d != expected_dimension:
        raise ValueError(f"FAISS dimension {index.d} != expected {expected_dimension}")
    if faiss is not None and not isinstance(index, faiss.IndexFlatIP):
        raise ValueError("Dense retrieval_v2 index must be faiss.IndexFlatIP")


def search_ip_index(index, query_embeddings: np.ndarray, document_map: pd.DataFrame, top_k: int = 10):
    """Search normalized queries and return scores joined to exact corpus row mappings."""
    queries = validate_embeddings(query_embeddings, expected_dimension=index.d)
    validate_faiss_index(index, len(document_map), index.d)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    count = min(top_k, len(document_map))
    scores, ids = index.search(queries, count)
    if (ids < 0).any() or (ids >= len(document_map)).any():
        raise ValueError("FAISS returned an invalid document row ID")
    results = []
    mapping = document_map.reset_index(drop=True)
    for query_index in range(len(queries)):
        for rank, (row_id, score) in enumerate(zip(ids[query_index], scores[query_index]), 1):
            record = mapping.iloc[int(row_id)].to_dict()
            record.update({"query_index": query_index, "rank": rank, "score": float(score), "row_id": int(row_id)})
            results.append(record)
    return pd.DataFrame(results)

def save_faiss_index(index, output_path: Path):
    """Saves FAISS index to disk."""
    if faiss is None:
        raise ImportError("faiss-cpu is not installed. Please run: pip install faiss-cpu")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))
    print(f"FAISS index saved to {output_path}")

def load_faiss_index(input_path: Path):
    """Loads FAISS index from disk."""
    if faiss is None:
        raise ImportError("faiss-cpu is not installed. Please run: pip install faiss-cpu")
    return faiss.read_index(str(input_path))

def search_similar_chunks(
    query_embedding: np.ndarray, 
    index, 
    top_k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Searches for the top-k most similar chunks in the FAISS index.
    """
    distances, indices = index.search(query_embedding.astype('float32'), top_k)
    return distances[0], indices[0]
