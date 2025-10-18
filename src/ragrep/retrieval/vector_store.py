"""Vector storage and retrieval functionality."""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """Handles vector storage and similarity search."""
    
    def __init__(self, persist_directory: str = "./.ragrep.db"):
        """Initialize vector store.
        
        Args:
            persist_directory: Directory to persist the vector database
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB
        if logger.isEnabledFor(logging.DEBUG):
            print("   🔄 Initializing ChromaDB vector database...")
        logger.info("Initializing ChromaDB vector database...")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        if logger.isEnabledFor(logging.DEBUG):
            print("   🔄 Setting up document collection...")
        logger.info("Setting up document collection...")
        self.collection = self.client.get_or_create_collection(
            name="ragrep_documents",
            metadata={"description": "RAGRep document collection"}
        )
        
        if logger.isEnabledFor(logging.DEBUG):
            print("   ✅ ChromaDB ready!")
        logger.info(f"Initialized vector store at {persist_directory}")
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Add document chunks to the vector store.
        
        Args:
            chunks: List of document chunks with text and metadata
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
            
        if logger.isEnabledFor(logging.DEBUG):
            print(f"💾 Adding {len(chunks)} chunks to vector database...")
            print("   🔄 ChromaDB is generating embeddings (this may take a moment)...")
        logger.info(f"Adding {len(chunks)} chunks to vector database...")
        
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # This is where ChromaDB does the heavy lifting - generating embeddings
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        if logger.isEnabledFor(logging.DEBUG):
            print("   🔄 Building search index...")
            print(f"✅ Successfully added {len(chunks)} chunks to vector store")
        logger.info(f"Successfully added {len(chunks)} chunks to vector store")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        if logger.isEnabledFor(logging.DEBUG):
            print(f"🔍 Searching vector database for: '{query}'...")
        logger.info(f"Searching for: {query}")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        logger.info(f"Found {len(formatted_results)} similar documents")
        return formatted_results
    
    def get_random_chunks(self, n_results: int = 10) -> List[Dict[str, Any]]:
        """Get random chunks from the collection.
        
        Args:
            n_results: Number of random chunks to return
            
        Returns:
            List of random document chunks with metadata
        """
        # Get all documents and randomly sample
        all_results = self.collection.get()
        
        if not all_results['documents']:
            return []
        
        # Convert to our format and shuffle
        chunks = []
        for i in range(len(all_results['documents'])):
            chunks.append({
                'text': all_results['documents'][i],
                'metadata': all_results['metadatas'][i] if all_results['metadatas'] else {},
                'id': all_results['ids'][i] if all_results['ids'] else f"chunk_{i}"
            })
        
        # Shuffle and take the requested number
        import random
        random.shuffle(chunks)
        return chunks[:n_results]

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            'total_documents': count,
            'collection_name': self.collection.name,
            'persist_directory': self.persist_directory
        }
