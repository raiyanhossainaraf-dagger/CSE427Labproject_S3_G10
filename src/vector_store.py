try:
    import faiss
except ImportError:
    print("Error: faiss-cpu is not installed.")
    print("Please run: pip install faiss-cpu")
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
