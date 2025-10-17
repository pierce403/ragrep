"""Text generation using OpenAI API."""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


class TextGenerator:
    """Handles text generation using OpenAI's GPT models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """Initialize text generator.
        
        Args:
            api_key: OpenAI API key (if None, will use environment variable)
            model: OpenAI model to use for generation
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        logger.info(f"Initialized text generator with model: {model}")
    
    def generate_response(self, query: str, context_documents: List[Dict[str, Any]]) -> str:
        """Generate a response based on query and context documents.
        
        Args:
            query: User's question or query
            context_documents: Retrieved documents to use as context
            
        Returns:
            Generated response text
        """
        # Prepare context from retrieved documents
        context_text = self._prepare_context(context_documents)
        
        # Create the prompt
        prompt = self._create_prompt(query, context_text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Use only the information from the context to answer questions. If the context doesn't contain enough information to answer the question, say so."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            generated_text = response.choices[0].message.content
            logger.info("Generated response successfully")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """Prepare context text from retrieved documents.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.get('metadata', {}).get('source', 'Unknown')
            text = doc.get('text', '')
            context_parts.append(f"Document {i} (Source: {source}):\n{text}\n")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create the prompt for the language model.
        
        Args:
            query: User's question
            context: Context documents
            
        Returns:
            Formatted prompt
        """
        return f"""Based on the following context documents, please answer the question: {query}

Context:
{context}

Answer:"""
