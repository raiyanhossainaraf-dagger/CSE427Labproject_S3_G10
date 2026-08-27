import unittest
import os
import sys
from pathlib import Path
import json
import subprocess

class TestDependencyFiles(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parent.parent
        self.req_files = [
            "requirements-core.txt",
            "requirements-retrieval.txt",
            "requirements-llm.txt",
            "requirements-colab.txt",
            "requirements-dev.txt",
            "requirements.txt"
        ]

    def test_requirements_exist(self):
        for f in self.req_files:
            self.assertTrue((self.root / f).exists(), f"{f} missing")

    def test_no_pathlib_in_requirements(self):
        for f in self.req_files:
            content = (self.root / f).read_text()
            self.assertNotIn("pathlib", content.lower(), f"pathlib found in {f}")

    def test_requirement_includes_exist(self):
        for f in self.req_files:
            content = (self.root / f).read_text()
            for line in content.splitlines():
                if line.startswith("-r "):
                    include_file = line.split("-r ")[1].strip()
                    self.assertTrue((self.root / include_file).exists(), f"Included file {include_file} in {f} does not exist")

    def test_no_absolute_paths_in_requirements(self):
        for f in self.req_files:
            content = (self.root / f).read_text()
            # Basic check for Windows/Linux absolute paths
            self.assertFalse(":\\" in content, f"Possible absolute Windows path in {f}")
            self.assertFalse(content.startswith("/"), f"Possible absolute Linux path in {f}")

    def test_core_dependencies_present(self):
        content = (self.root / "requirements-core.txt").read_text()
        core = ["pandas", "numpy", "datasets", "matplotlib", "scikit-learn"]
        for pkg in core:
            self.assertIn(pkg, content, f"{pkg} missing from core requirements")

    def test_retrieval_dependencies_present(self):
        content = (self.root / "requirements-retrieval.txt").read_text()
        retrieval = ["rank-bm25", "torch", "sentence-transformers", "faiss-cpu"]
        for pkg in retrieval:
            self.assertIn(pkg, content, f"{pkg} missing from retrieval requirements")

    def test_root_requirements_points_to_retrieval(self):
        content = (self.root / "requirements.txt").read_text().strip()
        self.assertEqual(content, "-r requirements-retrieval.txt")

    def test_gitignore_exists_and_valid(self):
        gitignore = self.root / ".gitignore"
        self.assertTrue(gitignore.exists())
        content = gitignore.read_text()
        essential = ["__pycache__/", ".venv/", ".idea/", "*.safetensors", "data/cache/"]
        for pattern in essential:
            self.assertIn(pattern, content, f"Essential pattern {pattern} missing from .gitignore")

    def test_gitignore_trackable_files(self):
        # We check that these are NOT in gitignore in a way that would broadly ignore them
        content = (self.root / ".gitignore").read_text()
        self.assertNotIn("*.csv", content)
        self.assertNotIn("*.json", content)
        self.assertNotIn("*.png", content)
        self.assertNotIn("*.ipynb", content)

    def test_export_environment_script(self):
        script_path = self.root / "scripts" / "export_environment.py"
        result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        try:
            data = json.loads(result.stdout)
            self.assertIn("python_version", data)
            self.assertIn("package_versions", data)
        except json.JSONDecodeError:
            self.fail("export_environment.py did not produce valid JSON")

if __name__ == "__main__":
    import sys
    unittest.main()
