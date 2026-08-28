import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Union

DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_BATCH_SIZE = 32
RETRIEVAL_ARTIFACT_VERSION = "retrieval_v2"

@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    src_dir: Path
    notebooks_dir: Path
    scripts_dir: Path
    tests_dir: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    cache_dir: Path
    outputs_dir: Path
    figures_dir: Path
    tables_dir: Path
    summaries_dir: Path
    predictions_dir: Path
    run_metadata_dir: Path
    report_dir: Path

def resolve_project_root(
    start: Union[str, Path, None] = None,
    explicit_root: Union[str, Path, None] = None,
) -> Path:
    """
    Resolves the canonical project root directory.
    Priority:
    1. explicit_root
    2. Environment variable CSE427_PROJECT_ROOT
    3. Search upward from start
    4. Search upward from current working directory
    5. Fallback to location of this file
    """
    if explicit_root:
        root = Path(explicit_root).resolve()
        if not _is_valid_root(root):
            raise ValueError(f"Invalid explicit project root: {root}")
        return root

    env_root = os.environ.get("CSE427_PROJECT_ROOT")
    if env_root:
        root = Path(env_root).resolve()
        if not _is_valid_root(root):
            raise ValueError(f"Invalid CSE427_PROJECT_ROOT environment variable: {env_root}")
        return root

    search_start = Path(start).resolve() if start else Path.cwd().resolve()
    
    # Search upward
    for parent in [search_start] + list(search_start.parents):
        if _is_valid_root(parent):
            return parent

    # Fallback to src/config.py location
    # This file is expected to be in <root>/src/config.py
    this_file_root = Path(__file__).resolve().parent.parent
    if _is_valid_root(this_file_root):
        return this_file_root

    raise RuntimeError("Could not resolve project root.")

def _is_valid_root(path: Path) -> bool:
    """Checks if a directory is a valid repository root using markers."""
    markers = ["src", "notebooks", "requirements.txt"]
    return all((path / marker).exists() for marker in markers)

def get_project_paths(
    start: Union[str, Path, None] = None,
    explicit_root: Union[str, Path, None] = None,
) -> ProjectPaths:
    """Returns a ProjectPaths object with canonical absolute paths."""
    root = resolve_project_root(start, explicit_root)
    
    return ProjectPaths(
        project_root=root,
        src_dir=root / "src",
        notebooks_dir=root / "notebooks",
        scripts_dir=root / "scripts",
        tests_dir=root / "tests",
        data_dir=root / "data",
        raw_data_dir=root / "data" / "raw",
        processed_data_dir=root / "data" / "processed",
        cache_dir=root / "data" / "cache",
        outputs_dir=root / "outputs",
        figures_dir=root / "outputs" / "figures",
        tables_dir=root / "outputs" / "tables",
        summaries_dir=root / "outputs" / "summaries",
        predictions_dir=root / "outputs" / "predictions",
        run_metadata_dir=root / "outputs" / "run_metadata",
        report_dir=root / "report",
    )

def ensure_project_directories(paths: ProjectPaths) -> None:
    """Idempotently creates all required project directories."""
    dirs_to_create = [
        paths.data_dir,
        paths.raw_data_dir,
        paths.processed_data_dir,
        paths.cache_dir,
        paths.outputs_dir,
        paths.figures_dir,
        paths.tables_dir,
        paths.summaries_dir,
        paths.predictions_dir,
        paths.run_metadata_dir,
        paths.report_dir,
    ]
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)
