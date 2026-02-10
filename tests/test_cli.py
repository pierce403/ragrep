from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
