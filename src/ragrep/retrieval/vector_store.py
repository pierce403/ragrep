"""Vector storage and retrieval functionality."""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging

logger = logging.getLogger(__name__)

try:  # Optional heavy dependency
    import chromadb
    from chromadb.config import Settings

    _HAS_CHROMADB = True
except Exception:  # pragma: no cover - exercised only when dependency missing
    chromadb = None  # type: ignore[assignment]
    Settings = None  # type: ignore[assignment]
    _HAS_CHROMADB = False


def _normalise_persist_directory(persist_directory: str) -> Path:
    """Return a usable path for storing persistent data."""

    path = Path(persist_directory)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


class _BaseBackend:
    """Common interface for vector store backends."""

    backend_name = "unknown"

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def search(self, query: str, n_results: int) -> List[Dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError

    def get_random_chunks(self, n_results: int) -> List[Dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError

    def get_collection_info(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class _ChromadbBackend(_BaseBackend):
    """Vector store backed by ChromaDB when the dependency is available."""

    backend_name = "chromadb"

    def __init__(self, persist_directory: Path) -> None:
        if chromadb is None:  # pragma: no cover - guard for type checkers
            raise RuntimeError("ChromaDB is not available")

        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        if logger.isEnabledFor(logging.DEBUG):
            print("   🔄 Initializing ChromaDB vector database...")
        logger.info("Initializing ChromaDB vector database...")
        if self.persist_directory.exists():
            logger.info(
                "Removing existing database at %s to avoid schema conflicts",
                self.persist_directory,
            )
            shutil.rmtree(self.persist_directory, ignore_errors=True)
            self.persist_directory.mkdir(parents=True, exist_ok=True)

        persist_path = str(self.persist_directory)
        try:
            self.client = chromadb.PersistentClient(
                path=persist_path,
                settings=Settings(anonymized_telemetry=False),
            )
        except (TypeError, AttributeError) as exc:
            logger.warning(
                "ChromaDB initialization with settings failed; falling back: %s",
                exc,
            )
            try:
                self.client = chromadb.PersistentClient(path=persist_path)
            except Exception as exc2:  # pragma: no cover - depends on runtime env
                logger.warning("ChromaDB initialization without settings failed: %s", exc2)
                alt_path = persist_path + "_alt"
                logger.info("Trying alternative path: %s", alt_path)
                try:
                    self.client = chromadb.PersistentClient(path=alt_path)
                except Exception as exc3:
                    logger.error("All ChromaDB initialization methods failed: %s", exc3)
                    try:
                        logger.info("Trying in-memory ChromaDB client as last resort")
                        self.client = chromadb.EphemeralClient()
                        logger.warning("Using in-memory client - data will not persist between runs")
                    except Exception as exc4:  # pragma: no cover - depends on runtime env
                        raise RuntimeError(f"Unable to initialize ChromaDB client: {exc4}") from exc3

        if logger.isEnabledFor(logging.DEBUG):
            print("   🔄 Setting up document collection...")
        logger.info("Setting up document collection...")

        try:
            self.collection = self.client.get_or_create_collection(
                name="ragrep_documents",
                metadata={"description": "RAGrep document collection"},
            )
        except Exception as exc:  # pragma: no cover - depends on runtime env
            logger.warning("Collection creation failed with metadata, trying without: %s", exc)
            try:
                self.collection = self.client.get_or_create_collection(name="ragrep_documents")
            except Exception as exc2:  # pragma: no cover - depends on runtime env
                raise RuntimeError(f"Unable to create collection: {exc2}") from exc

        if logger.isEnabledFor(logging.DEBUG):
            print("   ✅ ChromaDB ready!")

        if hasattr(self.client, "persist_directory"):
            logger.info("Initialized vector store at %s", self.persist_directory)
        else:
            logger.info("Initialized in-memory vector store (data will not persist)")

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            logger.warning("No chunks to add")
            return

        if logger.isEnabledFor(logging.DEBUG):
            print(f"💾 Adding {len(chunks)} chunks to vector database...")
            print("   🔄 ChromaDB is generating embeddings (this may take a moment)...")
        logger.info("Adding %s chunks to vector database...", len(chunks))

        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [chunk["metadata"].get("id", f"chunk_{i}") for i, chunk in enumerate(chunks)]

        self.collection.add(documents=texts, metadatas=metadatas, ids=ids)

        if logger.isEnabledFor(logging.DEBUG):
            print("   🔄 Building search index...")
            print(f"✅ Successfully added {len(chunks)} chunks to vector store")
        logger.info("Successfully added %s chunks to vector store", len(chunks))

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        if logger.isEnabledFor(logging.DEBUG):
            print(f"🔍 Searching vector database for: '{query}'...")
        logger.info("Searching for: %s", query)

        results = self.collection.query(query_texts=[query], n_results=n_results)

        formatted_results: List[Dict[str, Any]] = []
        for i in range(len(results["documents"][0])):
            formatted_results.append(
                {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results.get("distances", [[None]])[0][i],
                }
            )

        logger.info("Found %s similar documents", len(formatted_results))
        return formatted_results

    def get_random_chunks(self, n_results: int = 10) -> List[Dict[str, Any]]:
        all_results = self.collection.get()

        if not all_results["documents"]:
            return []

        chunks: List[Dict[str, Any]] = []
        for i in range(len(all_results["documents"])):
            chunks.append(
                {
                    "text": all_results["documents"][i],
                    "metadata": all_results["metadatas"][i] if all_results["metadatas"] else {},
                    "id": all_results["ids"][i] if all_results["ids"] else f"chunk_{i}",
                }
            )

        random.shuffle(chunks)
        return chunks[:n_results]

    def get_collection_info(self) -> Dict[str, Any]:
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection.name,
            "persist_directory": str(self.persist_directory),
            "backend": self.backend_name,
        }


class _SimpleBackend(_BaseBackend):
    """Fallback vector store that avoids heavy third-party dependencies."""

    backend_name = "simple"

    def __init__(self, persist_directory: Path) -> None:
        self.persist_directory = persist_directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.persist_directory / "index.json"
        self._index: List[Dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenise(text: str) -> Dict[str, int]:
        tokens = re.findall(r"\b\w+\b", text.lower())
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        return counts

    @staticmethod
    def _cosine_similarity(vec_a: Dict[str, int], vec_b: Dict[str, int]) -> float:
        if not vec_a or not vec_b:
            return 0.0

        dot_product = sum(vec_a.get(token, 0) * vec_b.get(token, 0) for token in vec_a.keys())
        norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
        norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _store(self) -> None:
        serialisable = [
            {
                "id": entry["id"],
                "text": entry["text"],
                "metadata": entry["metadata"],
                "token_counts": entry["token_counts"],
            }
            for entry in self._index
        ]
        with self.index_path.open("w", encoding="utf-8") as handle:
            json.dump(serialisable, handle, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        if not self.index_path.exists():
            self._index = []
            return

        try:
            with self.index_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            self._index = [
                {
                    "id": entry.get("id", f"chunk_{idx}"),
                    "text": entry.get("text", ""),
                    "metadata": entry.get("metadata", {}),
                    "token_counts": entry.get("token_counts", {}),
                }
                for idx, entry in enumerate(data)
            ]
        except Exception as exc:  # pragma: no cover - defensive programming
            logger.warning("Failed to load existing index at %s: %s", self.index_path, exc)
            self._index = []

    # ------------------------------------------------------------------
    # Backend API
    # ------------------------------------------------------------------
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            logger.warning("No chunks to add")
            return

        logger.info("Adding %s chunks using simple backend", len(chunks))
        self._index = []
        for chunk in chunks:
            metadata = dict(chunk.get("metadata", {}))
            chunk_id = metadata.get("id") or f"chunk_{len(self._index)}"
            metadata.setdefault("id", chunk_id)
            entry = {
                "id": chunk_id,
                "text": chunk.get("text", ""),
                "metadata": metadata,
                "token_counts": self._tokenise(chunk.get("text", "")),
            }
            self._index.append(entry)

        self._store()
        logger.info("Successfully added %s chunks to simple backend", len(chunks))

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        query_tokens = self._tokenise(query)
        scored: List[Dict[str, Any]] = []

        for entry in self._index:
            similarity = self._cosine_similarity(query_tokens, entry["token_counts"])
            if similarity <= 0:
                continue
            scored.append(
                {
                    "text": entry["text"],
                    "metadata": entry["metadata"],
                    "distance": 1 - similarity,
                }
            )

        scored.sort(key=lambda item: item["distance"])
        return scored[:n_results]

    def get_random_chunks(self, n_results: int = 10) -> List[Dict[str, Any]]:
        if not self._index:
            return []

        population = list(self._index)
        random.shuffle(population)
        sample = population[:n_results]
        return [
            {"text": entry["text"], "metadata": entry["metadata"], "id": entry["id"]}
            for entry in sample
        ]

    def get_collection_info(self) -> Dict[str, Any]:
        return {
            "total_documents": len(self._index),
            "collection_name": "ragrep_documents",
            "persist_directory": str(self.persist_directory),
            "backend": self.backend_name,
        }


class VectorStore:
    """Handles vector storage and similarity search."""

    def __init__(self, persist_directory: str = "./.ragrep.db", *, prefer_chromadb: Optional[bool] = None):
        self.original_persist_directory = persist_directory
        if os.environ.get("GITHUB_ACTIONS") == "true":
            unique_id = uuid.uuid4().hex[:8]
            persist_directory = f"./.ragrep_test_{unique_id}.db"

        resolved_path = _normalise_persist_directory(persist_directory)

        if prefer_chromadb is None:
            env_setting = os.getenv("RAGREP_USE_CHROMADB")
            if env_setting is not None:
                prefer_chromadb = env_setting.lower() in {"1", "true", "yes"}
            else:
                prefer_chromadb = False

        use_chromadb = bool(prefer_chromadb) and _HAS_CHROMADB

        if use_chromadb:
            self._backend: _BaseBackend = _ChromadbBackend(resolved_path)
        else:
            if prefer_chromadb:
                logger.warning("ChromaDB requested but not available; falling back to simple backend")
            self._backend = _SimpleBackend(resolved_path)

        self.persist_directory = str(resolved_path)
        logger.info(
            "Vector store backend selected: %s (persisting to %s)",
            getattr(self._backend, "backend_name", "unknown"),
            self.persist_directory,
        )

    # ------------------------------------------------------------------
    # Public API delegating to backend implementation
    # ------------------------------------------------------------------
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        self._backend.add_documents(chunks)

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        return self._backend.search(query, n_results)

    def get_random_chunks(self, n_results: int = 10) -> List[Dict[str, Any]]:
        return self._backend.get_random_chunks(n_results)

    def get_collection_info(self) -> Dict[str, Any]:
        return self._backend.get_collection_info()
