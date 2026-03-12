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

    def test_index_prints_added_modified_and_removed_file_paths(self):
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
                    "reason": "new files detected, updated files detected, files removed",
                    "root": "/tmp/work",
                    "files": 3,
                    "chunks": 10,
                    "chunks_indexed": 4,
                    "indexed_files": 2,
                    "new_files": ["src/new_file.py"],
                    "updated_files": ["src/changed_file.py"],
                    "removed_files": ["src/removed_file.py"],
                    "full_rebuild": False,
                }

        output = StringIO()
        with patch("ragrep.cli.RAGrep", DummyRAG):
            with redirect_stdout(output):
                exit_code = main(["index", "."])

        self.assertEqual(exit_code, 0)
        text = output.getvalue()
        self.assertIn("Index updated for /tmp/work: 1 added, 1 modified, 1 removed.", text)
        self.assertIn("Added files:", text)
        self.assertIn("src/new_file.py", text)
        self.assertIn("Modified files:", text)
        self.assertIn("src/changed_file.py", text)
        self.assertIn("Removed files:", text)
        self.assertIn("src/removed_file.py", text)
        self.assertIn(
            "Indexed 2 changed files (4 chunks updated, 10 total): "
            "new files detected, updated files detected, files removed",
            text,
        )

    def test_recall_prints_up_to_date_status_before_results(self):
        class DummyRAG:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return None

            def recall(self, query, limit=20, path=None, auto_index=True):
                return {
                    "query": query,
                    "count": 1,
                    "matches": [
                        {
                            "score": 0.9,
                            "text": "# Schema",
                            "metadata": {"source": "docs/schema.md"},
                        }
                    ],
                    "auto_index": {
                        "indexed": False,
                        "reason": "index is current",
                        "root": "/tmp/work",
                        "files": 2,
                        "chunks": 8,
                        "chunks_indexed": 0,
                        "indexed_files": 0,
                        "new_files": [],
                        "updated_files": [],
                        "removed_files": [],
                        "full_rebuild": False,
                    },
                }

        output = StringIO()
        with patch("ragrep.cli.RAGrep", DummyRAG):
            with redirect_stdout(output):
                exit_code = main(["schema"])

        self.assertEqual(exit_code, 0)
        text = output.getvalue()
        self.assertIn(
            "Index is already up to date for /tmp/work (2 files, 8 chunks): index is current",
            text,
        )
        self.assertIn("Results: 1", text)
        self.assertLess(
            text.index("Index is already up to date for /tmp/work"),
            text.index("Results: 1"),
        )


if __name__ == "__main__":
    unittest.main()
