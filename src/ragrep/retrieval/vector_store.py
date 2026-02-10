"""SQLite-backed vector storage for RAGrep."""

from __future__ import annotations

import json
import math
import shutil
import sqlite3
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _normalise_db_path(db_path: str) -> Path:
    path = Path(db_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _pack_vector(vector: Iterable[float]) -> bytes:
    values = array("f", [float(value) for value in vector])
    return values.tobytes()


def _unpack_vector(payload: bytes) -> array:
    values = array("f")
    values.frombytes(payload)
    return values


def _vector_norm(vector: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


class VectorStore:
    """Persist chunks + embeddings in a local SQLite database file."""

    def __init__(self, db_path: str = "./.ragrep.db") -> None:
        self.db_path = _normalise_db_path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Migrate old directory-based storage automatically.
        if self.db_path.exists() and self.db_path.is_dir():
            legacy_path = self.db_path.with_name(f"{self.db_path.name}.legacy")
            if legacy_path.exists():
                shutil.rmtree(legacy_path)
            self.db_path.rename(legacy_path)

        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    def close(self) -> None:
        self.connection.close()

    def _create_tables(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    embedding_norm REAL NOT NULL,
                    FOREIGN KEY(file_path) REFERENCES files(path) ON DELETE CASCADE
                )
                """
            )
            self.connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_file_path
                ON chunks(file_path)
                """
            )

    def needs_reindex(
        self,
        *,
        root_path: Path,
        files: List[Dict[str, Any]],
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> Tuple[bool, str]:
        metadata = self._get_metadata()

        if "indexed_root" not in metadata:
            return True, "index has not been created yet"

        if metadata.get("indexed_root") != str(root_path):
            return True, "indexed root changed"

        if metadata.get("embedding_model") != embedding_model:
            return True, "embedding model changed"

        if metadata.get("chunk_size") != str(chunk_size):
            return True, "chunk size changed"

        if metadata.get("chunk_overlap") != str(chunk_overlap):
            return True, "chunk overlap changed"

        db_files = {
            row["path"]: (int(row["size"]), int(row["mtime_ns"]))
            for row in self.connection.execute("SELECT path, size, mtime_ns FROM files")
        }
        current_files = {
            entry["path"]: (int(entry["size"]), int(entry["mtime_ns"]))
            for entry in files
        }

        if db_files != current_files:
            return True, "indexed files changed"

        return False, "index is current"

    def replace_index(
        self,
        *,
        root_path: Path,
        files: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk count and embedding count must match")

        with self.connection:
            self.connection.execute("DELETE FROM chunks")
            self.connection.execute("DELETE FROM files")

            if files:
                self.connection.executemany(
                    "INSERT INTO files (path, size, mtime_ns) VALUES (?, ?, ?)",
                    [
                        (
                            entry["path"],
                            int(entry["size"]),
                            int(entry["mtime_ns"]),
                        )
                        for entry in files
                    ],
                )

            if chunks:
                self.connection.executemany(
                    """
                    INSERT INTO chunks (
                        id, file_path, chunk_index, start_char, end_char,
                        text, metadata_json, embedding, embedding_dim, embedding_norm
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk["id"],
                            chunk["file_path"],
                            int(chunk["chunk_index"]),
                            int(chunk["start_char"]),
                            int(chunk["end_char"]),
                            chunk["text"],
                            json.dumps(chunk["metadata"], ensure_ascii=True),
                            _pack_vector(vector),
                            len(vector),
                            _vector_norm(vector),
                        )
                        for chunk, vector in zip(chunks, embeddings)
                    ],
                )

            now = datetime.now(timezone.utc).isoformat()
            self._set_metadata("indexed_root", str(root_path))
            self._set_metadata("embedding_model", embedding_model)
            self._set_metadata("chunk_size", str(chunk_size))
            self._set_metadata("chunk_overlap", str(chunk_overlap))
            self._set_metadata("indexed_at", now)

    def search(self, query_embedding: List[float], limit: int = 20) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []

        query_values = array("f", [float(value) for value in query_embedding])
        query_norm = _vector_norm(query_values)
        if query_norm == 0:
            return []

        rows = self.connection.execute(
            """
            SELECT id, text, metadata_json, embedding, embedding_dim, embedding_norm
            FROM chunks
            """
        ).fetchall()

        matches: List[Dict[str, Any]] = []
        for row in rows:
            if int(row["embedding_dim"]) != len(query_values):
                continue

            vector = _unpack_vector(row["embedding"])
            stored_norm = float(row["embedding_norm"]) if row["embedding_norm"] else 0.0
            if stored_norm == 0:
                continue

            dot = 0.0
            for lhs, rhs in zip(vector, query_values):
                dot += float(lhs) * float(rhs)

            score = dot / (query_norm * stored_norm)
            metadata = json.loads(row["metadata_json"])
            matches.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "metadata": metadata,
                    "score": score,
                    "distance": 1.0 - score,
                }
            )

        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches[:limit]

    def get_collection_info(self) -> Dict[str, Any]:
        return self.get_stats()

    def get_stats(self) -> Dict[str, Any]:
        metadata = self._get_metadata()
        total_chunks = self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        total_files = self.connection.execute("SELECT COUNT(*) FROM files").fetchone()[0]

        return {
            "backend": "sqlite",
            "persist_path": str(self.db_path),
            "total_documents": int(total_chunks),
            "total_chunks": int(total_chunks),
            "total_files": int(total_files),
            "embedding_model": metadata.get("embedding_model"),
            "indexed_root": metadata.get("indexed_root"),
            "indexed_at": metadata.get("indexed_at"),
            "chunk_size": int(metadata["chunk_size"]) if metadata.get("chunk_size") else None,
            "chunk_overlap": int(metadata["chunk_overlap"]) if metadata.get("chunk_overlap") else None,
        }

    def _get_metadata(self) -> Dict[str, str]:
        rows = self.connection.execute("SELECT key, value FROM metadata").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def _set_metadata(self, key: str, value: str) -> None:
        self.connection.execute(
            """
            INSERT INTO metadata (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
