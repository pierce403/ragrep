"""Document processing and chunking functionality."""

import os
from typing import List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document loading, preprocessing, and chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize document processor.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def load_document(self, file_path: str) -> str:
        """Load text content from a file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Text content of the document
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            logger.info(f"Loaded document: {file_path}")
            return content
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            raise
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            metadata: Additional metadata to include with each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if metadata is None:
            metadata = {}
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': len(chunks),
                'start_char': start,
                'end_char': end
            })
            
            chunks.append({
                'text': chunk_text,
                'metadata': chunk_metadata
            })
            
            # Move start position, accounting for overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop if chunk_overlap >= chunk_size
            if start >= len(text):
                break
                
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a single document into chunks.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of processed chunks
        """
        text = self.load_document(file_path)
        metadata = {
            'source': file_path,
            'filename': os.path.basename(file_path)
        }
        return self.chunk_text(text, metadata)
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Process all text files in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of all processed chunks from all documents
        """
        all_chunks = []
        directory = Path(directory_path)
        
        # Supported file extensions
        text_extensions = {'.txt', '.md', '.py', '.js', '.html', '.css'}
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in text_extensions:
                try:
                    chunks = self.process_document(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {e}")
                    
        logger.info(f"Processed {len(all_chunks)} chunks from {directory_path}")
        return all_chunks
