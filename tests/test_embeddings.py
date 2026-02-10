from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ragrep.retrieval.embeddings import (
    default_model_dir,
    get_runtime_device_info,
    resolve_embedding_model,
    resolve_runtime_device,
)


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

    def test_device_auto_without_torch(self):
        with patch.dict(sys.modules, {"torch": None}):
            self.assertEqual(resolve_runtime_device("auto"), "cpu")

    def test_device_auto_prefers_cuda(self):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(resolve_runtime_device("auto"), "cuda")

    def test_device_auto_uses_mps_when_cuda_missing(self):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(resolve_runtime_device("auto"), "mps")

    def test_explicit_device_is_respected(self):
        self.assertEqual(resolve_runtime_device("cpu"), "cpu")
        self.assertEqual(resolve_runtime_device("cuda:0"), "cuda:0")

    def test_runtime_device_info_without_torch(self):
        with patch.dict(sys.modules, {"torch": None}):
            info = get_runtime_device_info("auto")
            self.assertFalse(info["torch_available"])
            self.assertEqual(info["resolved_device"], "cpu")

    def test_runtime_device_info_with_cuda_inventory(self):
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                device_count=lambda: 2,
                get_device_name=lambda i: f"GPU-{i}",
            ),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            info = get_runtime_device_info("auto")
            self.assertTrue(info["torch_available"])
            self.assertTrue(info["cuda_available"])
            self.assertEqual(info["cuda_device_count"], 2)
            self.assertEqual(info["cuda_devices"], ["GPU-0", "GPU-1"])


if __name__ == "__main__":
    unittest.main()
