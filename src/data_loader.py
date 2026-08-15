import os
import tarfile
import urllib.request
import json
from pathlib import Path
from typing import Dict, Optional

def is_within_directory(directory: Path, target: Path):
    abs_directory = directory.resolve()
    abs_target = target.resolve()
    return abs_directory in abs_target.parents or abs_directory == abs_target

def safe_extract(tar: tarfile.TarFile, path: Path):
    for member in tar.getmembers():
        member_path = (path / member.name).resolve()
        if not is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")
    tar.extractall(path)

def find_qasper_file(output_dir: Path, filename: str) -> Optional[Path]:
    """Recursively search for a file in the output directory."""
    matches = list(output_dir.rglob(filename))
    return matches[0] if matches else None

def validate_archive(archive_path: Path) -> bool:
    """Check if a .tgz archive is valid and readable."""
    if not archive_path.exists():
        return False
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            # Try to list members to verify readability
            tar.getmembers()
            return True
    except (tarfile.TarError, EOFError, IOError):
        return False

def load_qasper_dataset(raw_data_dir: Path) -> Dict:
    """
    Loads the QASPER dataset from local JSON files.
    If files don't exist, it attempts to extract them from archives.
    Returns a dictionary with splits: 'train', 'validation', 'test'.
    """
    output_dir = raw_data_dir / "qasper_v0.3"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapping of expected files
    files_map = {
        "train": "qasper-train-v0.3.json",
        "validation": "qasper-dev-v0.3.json",
        "test": "qasper-test-v0.3.json"
    }
    
    # Check if files already exist
    found_files = {}
    missing_any = False
    for split, filename in files_map.items():
        path = find_qasper_file(output_dir, filename)
        if path:
            found_files[split] = path
        else:
            missing_any = True
    
    if missing_any:
        print("Required QASPER JSON files not found. Attempting extraction...")
        extract_qasper_archives(raw_data_dir, output_dir)
        
        # Re-check after extraction
        for split, filename in files_map.items():
            path = find_qasper_file(output_dir, filename)
            if path:
                found_files[split] = path
            else:
                print(f"Error: Could not locate {filename} after extraction.")
    
    dataset = {}
    for split, filename in files_map.items():
        file_path = found_files.get(split)
        if file_path and file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError(f"Expected dictionary in {file_path}, got {type(data)}")
                dataset[split] = data
            print(f"Loaded {len(data)} papers for split: {split}")
        else:
            print(f"Warning: Could not load split '{split}'.")
            dataset[split] = {}
            
    return dataset

def extract_qasper_archives(raw_data_dir: Path, output_dir: Path):
    """
    Safely extracts train_dev.tgz and test.tgz into output_dir.
    """
    archives = {
        "train_dev.tgz": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
        "test.tgz": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"
    }
    
    for archive, url in archives.items():
        archive_path = raw_data_dir / archive
        
        # Validate existing archive
        if archive_path.exists() and not validate_archive(archive_path):
            print(f"Archive {archive} appears corrupted. It will be redownloaded.")
            archive_path.unlink()
            
        # Download if missing or corrupted
        if not archive_path.exists():
            print(f"Downloading {archive}...")
            urllib.request.urlretrieve(url, archive_path)
            
        # Extract
        print(f"Extracting {archive}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            safe_extract(tar, output_dir)
        print(f"Successfully extracted {archive}")

def download_qasper_fallback(raw_data_dir: Path):
    """
    Ensures QASPER archives are present in raw_data_dir.
    Downloads them only if missing or corrupted.
    """
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    extract_qasper_archives(raw_data_dir, raw_data_dir / "qasper_v0.3")

def get_scireviewgen_info():
    """Returns documentation about SciReviewGen dataset."""
    return {
        "name": "SciReviewGen",
        "repository": "https://github.com/tetsu9923/SciReviewGen",
        "relevance": "Designed for scientific literature-review generation, useful for later milestones.",
        "status": "Disabled by default for Milestone 1."
    }
