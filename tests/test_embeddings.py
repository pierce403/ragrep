from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ragrep.retrieval.embeddings import default_model_dir, resolve_embedding_model


class EmbeddingConfigTests(unittest.TestCase):
    def test_model_alias_resolution(self):
        self.assertEqual(
            resolve_embedding_model("mxbai-embed-large"),
            "mixedbread-ai/mxbai-embed-large-v1",
        )
        self.assertEqual(resolve_embedding_model("custom/model"), "custom/model")

    def test_model_dir_env_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"RAGREP_MODEL_DIR": temp_dir}, clear=False):
                self.assertEqual(default_model_dir(), Path(temp_dir).resolve())


if __name__ == "__main__":
    unittest.main()
