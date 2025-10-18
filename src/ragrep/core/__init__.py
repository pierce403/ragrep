"""Core RAG functionality."""

from __future__ import annotations

# Lazy imports to avoid heavy dependencies during CLI startup
def __getattr__(name):
    """Lazy import of modules to avoid heavy dependencies during CLI startup."""
    if name == "RAGSystem":
        from .rag_system import RAGSystem
        return RAGSystem
    elif name == "DocumentProcessor":
        from .document_processor import DocumentProcessor
        return DocumentProcessor
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["RAGSystem", "DocumentProcessor"]
