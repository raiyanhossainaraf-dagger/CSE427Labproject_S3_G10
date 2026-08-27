import unittest
import os
import shutil
import tempfile
from pathlib import Path
import sys

# Add src to path if needed
current_dir = Path.cwd().resolve()
if (current_dir / "src").exists() and str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.config import resolve_project_root, get_project_paths, ensure_project_directories, ProjectPaths

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        # Create a mock repository structure
        (self.test_dir / "src").mkdir()
        (self.test_dir / "notebooks").mkdir()
        (self.test_dir / "requirements.txt").touch()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_resolve_project_root_explicit(self):
        root = resolve_project_root(explicit_root=self.test_dir)
        self.assertEqual(root, self.test_dir.resolve())

    def test_resolve_project_root_invalid_explicit(self):
        invalid_dir = self.test_dir / "invalid"
        invalid_dir.mkdir()
        with self.assertRaises(ValueError):
            resolve_project_root(explicit_root=invalid_dir)

    def test_resolve_project_root_env_var(self):
        os.environ["CSE427_PROJECT_ROOT"] = str(self.test_dir)
        try:
            root = resolve_project_root()
            self.assertEqual(root, self.test_dir.resolve())
        finally:
            del os.environ["CSE427_PROJECT_ROOT"]

    def test_resolve_project_root_upward(self):
        start_dir = self.test_dir / "notebooks"
        root = resolve_project_root(start=start_dir)
        self.assertEqual(root, self.test_dir.resolve())

    def test_get_project_paths(self):
        paths = get_project_paths(explicit_root=self.test_dir)
        self.assertIsInstance(paths, ProjectPaths)
        self.assertEqual(paths.project_root, self.test_dir.resolve())
        self.assertEqual(paths.src_dir, self.test_dir.resolve() / "src")
        self.assertEqual(paths.data_dir, self.test_dir.resolve() / "data")
        # Check absolute paths
        self.assertTrue(paths.project_root.is_absolute())

    def test_ensure_project_directories(self):
        paths = get_project_paths(explicit_root=self.test_dir)
        # Before ensure
        self.assertFalse(paths.outputs_dir.exists())
        
        ensure_project_directories(paths)
        
        self.assertTrue(paths.outputs_dir.exists())
        self.assertTrue(paths.processed_data_dir.exists())
        self.assertTrue(paths.run_metadata_dir.exists())
        
        # Idempotency check
        ensure_project_directories(paths)
        self.assertTrue(paths.outputs_dir.exists())

    def test_import_no_side_effects(self):
        # We need to test this in a subprocess to be sure, 
        # but we can at least check if resolve doesn't create anything.
        paths = get_project_paths(explicit_root=self.test_dir)
        self.assertFalse((self.test_dir / "data").exists())

    def test_output_dirs_not_in_root(self):
        paths = get_project_paths(explicit_root=self.test_dir)
        # outputs_dir should be <root>/outputs, not <root>
        self.assertNotEqual(paths.outputs_dir, paths.project_root)
        self.assertEqual(paths.outputs_dir.parent, paths.project_root)

if __name__ == "__main__":
    unittest.main()
