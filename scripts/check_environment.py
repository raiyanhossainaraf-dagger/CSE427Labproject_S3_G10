import os
import sys
import argparse
import json
from pathlib import Path

# Try to import from src.config
try:
    # Ensure current directory or its parent is in sys.path to find 'src'
    current_dir = Path.cwd().resolve()
    if (current_dir / "src").exists():
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))
    elif (current_dir.parent / "src").exists():
        if str(current_dir.parent) not in sys.path:
            sys.path.insert(0, str(current_dir.parent))
            
    from src.config import get_project_paths, resolve_project_root
except ImportError:
    print("Error: Could not import src.config. Ensure you are running from the project root or notebooks directory.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Check the project environment.")
    parser.add_argument("--json", action="store_true", help="Output in JSON format.")
    args = parser.parse_args()

    try:
        root = resolve_project_root()
        paths = get_project_paths()
        colab_detected = "google.colab" in sys.modules
        
        # Check required markers
        markers = ["src", "notebooks", "requirements.txt"]
        marker_status = {m: (root / m).exists() for m in markers}
        
        status = "passed" if all(marker_status.values()) else "failed"

        # Check existing directories
        essential_dirs = [paths.src_dir, paths.notebooks_dir, paths.data_dir]
        dir_status = {str(d.relative_to(root)): d.exists() for d in essential_dirs}

        if args.json:
            output = {
                "status": status,
                "python_version": sys.version,
                "current_working_directory": str(Path.cwd().resolve()),
                "project_root": str(root),
                "colab_detected": colab_detected,
                "required_markers": marker_status,
                "paths": {k: str(v) for k, v in paths.__dict__.items()},
                "essential_directories_exist": dir_status
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"Project Root: {root}")
            print(f"Current WWD:  {Path.cwd().resolve()}")
            print(f"Python:       {sys.version.split()[0]}")
            print(f"Colab:        {colab_detected}")
            print("-" * 20)
            print("Markers Check:")
            for m, exists in marker_status.items():
                print(f"  {m:16}: {'[OK]' if exists else '[MISSING]'}")
            print("-" * 20)
            print("Essential Directories:")
            for d, exists in dir_status.items():
                print(f"  {d:16}: {'[OK]' if exists else '[MISSING]'}")
            
        if status != "passed":
            sys.exit(1)

    except Exception as e:
        if args.json:
            print(json.dumps({"status": "error", "message": str(e)}))
        else:
            print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
