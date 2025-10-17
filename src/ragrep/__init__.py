"""
RAGRep - Retrieval-Augmented Generation Tool

A powerful tool for implementing RAG systems that combine document retrieval 
with AI-powered text generation.
"""

__version__ = "0.1.0"
__author__ = "RAGRep Team"

# Lazy imports to avoid heavy dependencies during CLI startup
def __getattr__(name):
    """Lazy import of modules to avoid heavy dependencies during CLI startup."""
    if name == "RAGSystem":
        from .core.rag_system import RAGSystem
        return RAGSystem
    elif name == "DocumentProcessor":
        from .core.document_processor import DocumentProcessor
        return DocumentProcessor
    elif name == "VectorStore":
        from .retrieval.vector_store import VectorStore
        return VectorStore
    elif name == "TextGenerator":
        from .generation.text_generator import TextGenerator
        return TextGenerator
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "RAGSystem",
    "DocumentProcessor", 
    "VectorStore",
    "TextGenerator"
]
