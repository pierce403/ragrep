"""
RAGRep - Retrieval-Augmented Generation Tool

A powerful tool for implementing RAG systems that combine document retrieval 
with AI-powered text generation.
"""

__version__ = "0.1.0"
__author__ = "RAGRep Team"

from .core.rag_system import RAGSystem
from .core.document_processor import DocumentProcessor
from .retrieval.vector_store import VectorStore
from .generation.text_generator import TextGenerator

__all__ = [
    "RAGSystem",
    "DocumentProcessor", 
    "VectorStore",
    "TextGenerator"
]
