from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from ragrep.cli import main


class CLITests(unittest.TestCase):
    def test_stats_flag_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / ".ragrep.db"

            output = StringIO()
            with redirect_stdout(output):
                exit_code = main(["--stats", "--json", "--db-path", str(db_path)])

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["backend"], "sqlite")
            self.assertEqual(payload["total_chunks"], 0)

    def test_check_gpu_flag_alias(self):
        output = StringIO()
        with redirect_stdout(output):
            exit_code = main(["--check-gpu", "--json"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertIn("resolved_device", payload)
        self.assertIn("torch_available", payload)

    def test_index_prints_new_file_paths(self):
        class DummyRAG:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def index(self, path=".", force=False):
                return {
                    "indexed": True,
                    "reason": "new files detected",
                    "root": "/tmp/work",
                    "files": 3,
                    "chunks": 10,
                    "chunks_indexed": 4,
                    "indexed_files": 1,
                    "new_files": ["src/new_file.py"],
                    "updated_files": [],
                    "removed_files": [],
                    "full_rebuild": False,
                }

        output = StringIO()
        with patch("ragrep.cli.RAGrep", DummyRAG):
            with redirect_stdout(output):
                exit_code = main(["index", "."])

        self.assertEqual(exit_code, 0)
        text = output.getvalue()
        self.assertIn("New files indexed:", text)
        self.assertIn("src/new_file.py", text)


if __name__ == "__main__":
    unittest.main()
