"""Core RAGrep orchestration: index, recall, stats."""

from __future__ import annotations

from typing import Any, Dict, List, Protocol

from .document_processor import DocumentProcessor
from ..retrieval.embeddings import LocalEmbedder
from ..retrieval.vector_store import VectorStore


class _EmbedderProtocol(Protocol):
    model: str

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:  # pragma: no cover
        ...

    def embed_query(self, query: str) -> List[float]:  # pragma: no cover
        ...


class RAGrep:
    """High-level API for semantic indexing and recall."""

    def __init__(
        self,
        db_path: str = "./.ragrep.db",
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "mxbai-embed-large",
        model_dir: str | None = None,
        embedder: _EmbedderProtocol | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.model_dir = model_dir

        self.document_processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = VectorStore(db_path)
        self._embedder: _EmbedderProtocol | None = embedder

    @property
    def embedder(self) -> _EmbedderProtocol:
        """Return an embedding backend, loading it only when needed."""
        if self._embedder is None:
            self._embedder = LocalEmbedder(
                model=self.embedding_model,
                model_dir=self.model_dir,
            )
        return self._embedder

    def index(self, path: str = ".", *, force: bool = False) -> Dict[str, Any]:
        """Index files from a path into the local vector store."""
        chunks, file_records, root_path = self.document_processor.process_path(path)

        if not force:
            needs_reindex, reason = self.vector_store.needs_reindex(
                root_path=root_path,
                files=file_records,
                embedding_model=self.embedding_model,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            if not needs_reindex:
                return {
                    "indexed": False,
                    "reason": reason,
                    "root": str(root_path),
                    "files": len(file_records),
                    "chunks": len(chunks),
                }
        else:
            reason = "forced reindex"

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts) if texts else []

        self.vector_store.replace_index(
            root_path=root_path,
            files=file_records,
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        return {
            "indexed": True,
            "reason": reason,
            "root": str(root_path),
            "files": len(file_records),
            "chunks": len(chunks),
        }

    def recall(
        self,
        query: str,
        *,
        limit: int = 20,
        path: str = ".",
        auto_index: bool = True,
    ) -> Dict[str, Any]:
        """Recall indexed chunks relevant to a query.

        If ``auto_index`` is true, the index is updated only when source files changed.
        """
        index_result: Dict[str, Any] | None = None
        if auto_index:
            index_result = self.index(path, force=False)

        query_embedding = self.embedder.embed_query(query)
        matches = self.vector_store.search(query_embedding, limit=limit)

        return {
            "query": query,
            "matches": matches,
            "count": len(matches),
            "auto_index": index_result,
        }

    def dump(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Backward-compatible alias for ``recall``."""
        result = self.recall(query, limit=limit, auto_index=False)
        return result["matches"]

    def stats(self) -> Dict[str, Any]:
        """Return indexing and storage statistics."""
        return self.vector_store.get_stats()

    def get_stats(self) -> Dict[str, Any]:
        """Backward-compatible alias for ``stats``."""
        return self.stats()

    def close(self) -> None:
        self.vector_store.close()

    def __enter__(self) -> "RAGrep":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# Backward-compatible name kept for callers importing RAGSystem.
RAGSystem = RAGrep
