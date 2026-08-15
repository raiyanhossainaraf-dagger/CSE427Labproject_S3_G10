import os
import random
import numpy as np
from pathlib import Path

def set_seed(seed: int = 42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def ensure_dirs(base_dir: Path):
    """Ensures all required project directories exist."""
    dirs = [
        base_dir / "data" / "raw",
        base_dir / "data" / "interim",
        base_dir / "data" / "processed",
        base_dir / "outputs" / "figures",
        base_dir / "outputs" / "tables",
        base_dir / "outputs" / "summaries",
        base_dir / "notebooks",
        base_dir / "src",
        base_dir / "report"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs
