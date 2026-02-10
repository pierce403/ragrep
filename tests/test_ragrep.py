from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from ragrep import RAGrep


class FakeEmbedder:
    model = "mxbai-embed-large"

    _vocab = [
        "auth",
        "login",
        "token",
        "database",
        "query",
        "payment",
        "cache",
        "error",
    ]

    def embed_texts(self, texts, batch_size: int = 32):
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str):
        return self._embed(query)

    def _embed(self, text: str):
        lower = text.lower()
        vector = []
        for token in self._vocab:
            vector.append(float(lower.count(token)))
        return vector


class RAGrepTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "auth.py").write_text(
            "def login_user(token):\n    return verify_auth_token(token)\n",
            encoding="utf-8",
        )
        (self.root / "db.py").write_text(
            "def run_query(database):\n    return database.execute('select 1')\n",
            encoding="utf-8",
        )
        self.db_path = self.root / ".ragrep.db"

    def tearDown(self):
        self.tempdir.cleanup()

    def test_index_and_recall(self):
        rag = RAGrep(db_path=str(self.db_path), embedder=FakeEmbedder())
        try:
            index_result = rag.index(str(self.root))
            self.assertTrue(index_result["indexed"])
            self.assertEqual(index_result["files"], 2)

            recall_result = rag.recall("auth login token", limit=2, auto_index=False)
            self.assertGreaterEqual(recall_result["count"], 1)
            top_source = recall_result["matches"][0]["metadata"]["source"]
            self.assertEqual(top_source, "auth.py")
        finally:
            rag.close()

    def test_auto_index_skips_when_unchanged(self):
        rag = RAGrep(db_path=str(self.db_path), embedder=FakeEmbedder())
        try:
            rag.index(str(self.root))
            recall_result = rag.recall("database query", path=str(self.root), auto_index=True)
            self.assertFalse(recall_result["auto_index"]["indexed"])
            self.assertEqual(recall_result["auto_index"]["reason"], "index is current")
        finally:
            rag.close()

    def test_auto_index_uses_indexed_root_when_path_omitted(self):
        rag = RAGrep(db_path=str(self.db_path), embedder=FakeEmbedder())
        try:
            rag.index(str(self.root))
            recall_result = rag.recall("database query", auto_index=True)
            self.assertFalse(recall_result["auto_index"]["indexed"])
            self.assertEqual(recall_result["auto_index"]["reason"], "index is current")
            self.assertEqual(recall_result["auto_index"]["root"], str(self.root))
        finally:
            rag.close()

    def test_auto_index_reindexes_after_file_change(self):
        rag = RAGrep(db_path=str(self.db_path), embedder=FakeEmbedder())
        try:
            rag.index(str(self.root))
            time.sleep(0.001)
            (self.root / "auth.py").write_text(
                "def login_user(token):\n    return verify_auth_token(token)\n\n"
                "def payment_auth(token):\n    return token\n",
                encoding="utf-8",
            )

            recall_result = rag.recall("payment auth", path=str(self.root), auto_index=True)
            self.assertTrue(recall_result["auto_index"]["indexed"])
            self.assertEqual(recall_result["auto_index"]["reason"], "updated files detected")
            self.assertEqual(recall_result["auto_index"]["new_files"], [])
            self.assertEqual(recall_result["auto_index"]["updated_files"], ["auth.py"])
            self.assertGreater(recall_result["auto_index"]["chunks"], recall_result["auto_index"]["chunks_indexed"])
        finally:
            rag.close()

    def test_auto_index_reindexes_after_new_file(self):
        rag = RAGrep(db_path=str(self.db_path), embedder=FakeEmbedder())
        try:
            rag.index(str(self.root))
            time.sleep(0.001)
            (self.root / "new_feature.py").write_text(
                "def new_feature_auth(token):\n    return token\n",
                encoding="utf-8",
            )

            recall_result = rag.recall("new feature auth", path=str(self.root), auto_index=True)
            self.assertTrue(recall_result["auto_index"]["indexed"])
            self.assertEqual(recall_result["auto_index"]["reason"], "new files detected")
            self.assertEqual(recall_result["auto_index"]["new_files"], ["new_feature.py"])
            self.assertEqual(recall_result["auto_index"]["updated_files"], [])
            self.assertEqual(recall_result["auto_index"]["indexed_files"], 1)
        finally:
            rag.close()

    def test_model_cache_changes_do_not_trigger_reindex(self):
        model_dir = self.root / ".ragrep-model-cache"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "cache.json").write_text("{\"version\": 1}\n", encoding="utf-8")

        rag = RAGrep(
            db_path=str(self.db_path),
            model_dir=str(model_dir),
            embedder=FakeEmbedder(),
        )
        try:
            index_result = rag.index(str(self.root))
            self.assertEqual(index_result["files"], 2)

            time.sleep(0.001)
            (model_dir / "cache.json").write_text("{\"version\": 2}\n", encoding="utf-8")
            (model_dir / "metadata.txt").write_text("updated\n", encoding="utf-8")

            recall_result = rag.recall("database query", path=str(self.root), auto_index=True)
            self.assertFalse(recall_result["auto_index"]["indexed"])
            self.assertEqual(recall_result["auto_index"]["reason"], "index is current")
        finally:
            rag.close()

    def test_stats(self):
        rag = RAGrep(db_path=str(self.db_path), embedder=FakeEmbedder())
        try:
            rag.index(str(self.root))
            stats = rag.stats()
            self.assertEqual(stats["embedding_model"], "mxbai-embed-large")
            self.assertEqual(stats["total_files"], 2)
            self.assertGreater(stats["total_chunks"], 0)
        finally:
            rag.close()

    def test_stats_without_embedder_initialization(self):
        rag = RAGrep(db_path=str(self.db_path))
        try:
            stats = rag.stats()
            self.assertEqual(stats["backend"], "sqlite")
            self.assertEqual(stats["total_chunks"], 0)
        finally:
            rag.close()


if __name__ == "__main__":
    unittest.main()
