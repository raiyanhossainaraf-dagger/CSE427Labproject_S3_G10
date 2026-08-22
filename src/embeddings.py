import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union
import torch

# Suppress HF Hub warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers is not installed.")
    print("Please run: pip install sentence-transformers")
    SentenceTransformer = None

from tqdm.auto import tqdm

def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Loads the sentence-transformer model with GPU support if available."""
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Please run: pip install sentence-transformers"
        )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # Additional check to ensure it's not a mock or broken installation
        try:
            torch.cuda.get_device_name(0)
        except Exception:
            device = "cpu"
    
    print(f"Loading embedding model: {model_name} on device: {device}...")
    
    return SentenceTransformer(model_name, device=device)

def generate_embeddings(
    texts: List[str], 
    model: Union['SentenceTransformer', None], 
    batch_size: int = 64
) -> np.ndarray:
    """
    Generates embeddings for a list of texts in batches using optimized parameters.
    """
    if model is None:
        raise ValueError("Model is not loaded. Please provide a valid SentenceTransformer model.")
    
    print(f"Generating embeddings for {len(texts)} items (Batch Size: {batch_size})...")
    
    embeddings = model.encode(
        texts, 
        batch_size=batch_size, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings

def save_embeddings(embeddings: np.ndarray, output_path: Path):
    """Saves embeddings to a .npy file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    print(f"Embeddings saved to {output_path}")

def load_embeddings(input_path: Path) -> np.ndarray:
    """Loads embeddings from a .npy file."""
    return np.load(input_path)
