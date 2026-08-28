import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Sequence, Union
try:
    import torch
except ImportError:
    torch = None

# Suppress HF Hub warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

DEFAULT_BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def embedding_dimension(model) -> int:
    """Return model output dimension across Sentence Transformers API versions."""
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())


def load_embedding_model(model_name: str = DEFAULT_BGE_MODEL, device: str | None = None):
    """Loads the sentence-transformer model with GPU support if available."""
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Please run: pip install sentence-transformers"
        )
    
    device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    if device == "cuda" and torch is not None:
        # Additional check to ensure it's not a mock or broken installation
        try:
            torch.cuda.get_device_name(0)
        except Exception:
            device = "cpu"
    
    print(f"Loading embedding model: {model_name} on device: {device}...")
    
    return SentenceTransformer(model_name, device=device)


def encode_passages(texts: Sequence[str], model, batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
    """Encode passages without a query instruction as float32 unit vectors."""
    return _encode(texts, model, batch_size, show_progress)


def encode_queries(queries: Sequence[str], model, batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
    """Encode queries with the exact BGE retrieval instruction as float32 unit vectors."""
    instructed = [BGE_QUERY_INSTRUCTION + str(query) for query in queries]
    return _encode(instructed, model, batch_size, show_progress)


def _encode(texts: Sequence[str], model, batch_size: int, show_progress: bool) -> np.ndarray:
    if model is None:
        raise ValueError("Model is not loaded")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not texts:
        dimension = embedding_dimension(model) if hasattr(model, "get_sentence_embedding_dimension") or hasattr(model, "get_embedding_dimension") else 0
        return np.empty((0, dimension), dtype=np.float32)
    values = model.encode(list(texts), batch_size=batch_size, show_progress_bar=show_progress,
                          convert_to_numpy=True, normalize_embeddings=False)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Encoder returned invalid embeddings")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Encoder returned a zero-length embedding")
    return np.ascontiguousarray(values / norms, dtype=np.float32)

def generate_embeddings(
    texts: List[str], 
    model: Union['SentenceTransformer', None], 
    batch_size: int = 32
) -> np.ndarray:
    """
    Generates embeddings for a list of texts in batches using optimized parameters.
    """
    if model is None:
        raise ValueError("Model is not loaded. Please provide a valid SentenceTransformer model.")
    
    print(f"Generating embeddings for {len(texts)} items (Batch Size: {batch_size})...")
    
    return encode_passages(texts, model, batch_size=batch_size, show_progress=True)

def save_embeddings(embeddings: np.ndarray, output_path: Path):
    """Saves embeddings to a .npy file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Embeddings saved to {output_path}")

def load_embeddings(input_path: Path) -> np.ndarray:
    """Loads embeddings from a .npy file."""
    return np.load(input_path)
