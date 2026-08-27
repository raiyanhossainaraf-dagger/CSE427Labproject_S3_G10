import json
import os
import platform
import sys
import argparse
from pathlib import Path
import importlib.metadata
from datetime import datetime, timezone

def get_package_versions(packages):
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions

def main():
    parser = argparse.ArgumentParser(description="Export project environment metadata to JSON.")
    parser.add_argument("--output", type=str, help="Path to save the JSON output.")
    args = parser.parse_args()

    # Direct dependencies to track
    core_packages = [
        "datasets", "pandas", "numpy", "matplotlib", "seaborn", 
        "pyarrow", "tqdm", "scikit-learn", "ipywidgets"
    ]
    retrieval_packages = [
        "rank-bm25", "torch", "transformers", "sentence-transformers", "faiss-cpu"
    ]
    llm_packages = [
        "accelerate", "sentencepiece", "safetensors", "bitsandbytes"
    ]
    dev_packages = ["pytest", "nbformat"]

    all_packages = sorted(list(set(core_packages + retrieval_packages + llm_packages + dev_packages)))

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "operating_system": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "package_versions": get_package_versions(all_packages),
        "cuda_available": False,
        "gpu_name": None
    }

    # Optional torch/CUDA check
    try:
        import torch
        metadata["cuda_available"] = torch.cuda.is_available()
        if metadata["cuda_available"]:
            metadata["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    output_json = json.dumps(metadata, indent=4)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(output_json)
        print(f"Environment metadata exported to {args.output}")
    else:
        print(output_json)

if __name__ == "__main__":
    main()
