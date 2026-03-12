from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import patch

from ragrep.retrieval.embeddings import (
    LocalEmbedder,
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

    def test_local_embedder_uses_local_files_only_when_model_is_cached(self):
        calls = []
        sentence_transformers = ModuleType("sentence_transformers")
        huggingface_hub = ModuleType("huggingface_hub")
        cached_marker = object()

        class FakeSentenceTransformer:
            def __init__(self, model_name, **kwargs):
                calls.append({"model_name": model_name, **kwargs})

        def try_to_load_from_cache(*, repo_id, filename, cache_dir):
            if repo_id == "mixedbread-ai/mxbai-embed-large-v1" and filename == "modules.json":
                return str(Path(cache_dir) / "models--cached" / "modules.json")
            return cached_marker

        sentence_transformers.SentenceTransformer = FakeSentenceTransformer
        huggingface_hub.try_to_load_from_cache = try_to_load_from_cache
        huggingface_hub._CACHED_NO_EXIST = cached_marker

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                sys.modules,
                {
                    "sentence_transformers": sentence_transformers,
                    "huggingface_hub": huggingface_hub,
                },
            ):
                LocalEmbedder(model_dir=temp_dir, device="cpu")

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["local_files_only"])

    def test_local_embedder_allows_download_when_cache_is_missing(self):
        calls = []
        sentence_transformers = ModuleType("sentence_transformers")
        huggingface_hub = ModuleType("huggingface_hub")
        cached_marker = object()

        class FakeSentenceTransformer:
            def __init__(self, model_name, **kwargs):
                calls.append({"model_name": model_name, **kwargs})

        def try_to_load_from_cache(*, repo_id, filename, cache_dir):
            return cached_marker

        sentence_transformers.SentenceTransformer = FakeSentenceTransformer
        huggingface_hub.try_to_load_from_cache = try_to_load_from_cache
        huggingface_hub._CACHED_NO_EXIST = cached_marker

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                sys.modules,
                {
                    "sentence_transformers": sentence_transformers,
                    "huggingface_hub": huggingface_hub,
                },
            ):
                LocalEmbedder(model_dir=temp_dir, device="cpu")

        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0]["local_files_only"])

    def test_local_embedder_retries_without_local_only_when_cached_load_fails(self):
        calls = []
        sentence_transformers = ModuleType("sentence_transformers")
        huggingface_hub = ModuleType("huggingface_hub")
        cached_marker = object()

        class FakeSentenceTransformer:
            def __init__(self, model_name, **kwargs):
                calls.append({"model_name": model_name, **kwargs})
                if kwargs.get("local_files_only"):
                    raise OSError("cache incomplete")

        def try_to_load_from_cache(*, repo_id, filename, cache_dir):
            if filename == "modules.json":
                return str(Path(cache_dir) / "models--cached" / "modules.json")
            return cached_marker

        sentence_transformers.SentenceTransformer = FakeSentenceTransformer
        huggingface_hub.try_to_load_from_cache = try_to_load_from_cache
        huggingface_hub._CACHED_NO_EXIST = cached_marker

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                sys.modules,
                {
                    "sentence_transformers": sentence_transformers,
                    "huggingface_hub": huggingface_hub,
                },
            ):
                LocalEmbedder(model_dir=temp_dir, device="cpu")

        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0]["local_files_only"])
        self.assertFalse(calls[1]["local_files_only"])


if __name__ == "__main__":
    unittest.main()
