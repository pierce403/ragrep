"""RAGrep public package API."""

from __future__ import annotations

__version__ = "0.2.0"
__author__ = "RAGrep Team"


def __getattr__(name: str):
    if name in {"RAGrep", "RAGSystem"}:
        from .core.rag_system import RAGrep, RAGSystem

        return RAGrep if name == "RAGrep" else RAGSystem

    if name == "DocumentProcessor":
        from .core.document_processor import DocumentProcessor

        return DocumentProcessor

    if name == "VectorStore":
        from .retrieval.vector_store import VectorStore

        return VectorStore

    if name in {"LocalEmbedder", "OllamaEmbedder"}:
        # Keep OllamaEmbedder alias for backward compatibility.
        from .retrieval.embeddings import LocalEmbedder

        return LocalEmbedder

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "RAGrep",
    "RAGSystem",
    "DocumentProcessor",
    "VectorStore",
    "LocalEmbedder",
    "OllamaEmbedder",
]
