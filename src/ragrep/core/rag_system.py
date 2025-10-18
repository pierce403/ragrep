"""Main RAG system that orchestrates retrieval and generation."""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
from .document_processor import DocumentProcessor
from ..retrieval.vector_store import VectorStore
from ..generation.text_generator import TextGenerator
import logging

logger = logging.getLogger(__name__)


class RAGSystem:
    """Main RAG system that combines retrieval and generation."""
    
    def __init__(self,
                 vector_db_path: str = "./.ragrep.db",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 generation_model: str = "microsoft/DialoGPT-medium"):
        """Initialize RAG system.
        
        Args:
            vector_db_path: Path to store the vector database
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            generation_model: Hugging Face model name for text generation
        """
        logger.info("Initializing RAG system components...")
        
        logger.info("Setting up document processor...")
        self.document_processor = DocumentProcessor(chunk_size, chunk_overlap)
        
        logger.info("Setting up vector database...")
        self.vector_store = VectorStore(vector_db_path)
        
        logger.info("Loading language model (this may take a moment on first run)...")
        print("   🔄 Downloading and loading model...")
        self.text_generator = TextGenerator(generation_model)
        print("   ✅ Language model loaded successfully!")
        
        logger.info("RAG system initialized successfully")
    
    def add_documents(self, file_path: str) -> None:
        """Add documents to the knowledge base.
        
        Args:
            file_path: Path to document or directory
        """
        if os.path.isfile(file_path):
            chunks = self.document_processor.process_document(file_path)
        elif os.path.isdir(file_path):
            chunks = self.document_processor.process_directory(file_path)
        else:
            raise ValueError(f"Path {file_path} does not exist")
        
        self.vector_store.add_documents(chunks)
        logger.info(f"Added documents from {file_path}")
    
    def query(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        """Query the RAG system.
        
        Args:
            question: User's question
            n_results: Number of documents to retrieve
            
        Returns:
            Dictionary containing answer and source information
        """
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.search(question, n_results)
        
        # Generate response
        answer = self.text_generator.generate_response(question, retrieved_docs)
        
        # Prepare sources
        sources = []
        for doc in retrieved_docs:
            source_info = {
                'text': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'],
                'metadata': doc['metadata'],
                'relevance_score': 1 - doc.get('distance', 0) if doc.get('distance') else None
            }
            sources.append(source_info)
        
        result = {
            'question': question,
            'answer': answer,
            'sources': sources,
            'num_sources': len(sources)
        }
        
        logger.info(f"Processed query: {question}")
        return result
    
    def dump_chunks(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Dump chunks from the knowledge base matching a query (no LLM processing).
        
        Args:
            query: Search query to find relevant chunks
            limit: Maximum number of chunks to return
            
        Returns:
            List of document chunks with metadata, sorted by relevance
        """
        # Get all chunks from the vector store
        collection_info = self.vector_store.get_collection_info()
        total_chunks = collection_info['total_documents']
        
        if total_chunks == 0:
            logger.info("No chunks found in knowledge base")
            return []
        
        # Use semantic search to find relevant chunks
        # Increase search results to get better coverage, then limit
        search_limit = min(limit * 2, total_chunks)  # Get 2x to have better selection
        chunks = self.vector_store.search(query, n_results=search_limit)
        
        # Limit to requested number
        chunks = chunks[:limit]
        
        logger.info(f"Dumped {len(chunks)} chunks matching query '{query}' from knowledge base")
        return chunks

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics.
        
        Returns:
            Dictionary with system information
        """
        vector_stats = self.vector_store.get_collection_info()
        return {
            'vector_store': vector_stats,
            'generation_model': self.text_generator.model_name,
            'chunk_size': self.document_processor.chunk_size,
            'chunk_overlap': self.document_processor.chunk_overlap
        }
