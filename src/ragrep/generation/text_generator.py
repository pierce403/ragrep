"""Text generation using local language models."""

import os
from typing import List, Dict, Any, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)


class TextGenerator:
    """Handles text generation using local language models."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initialize text generator.
        
        Args:
            model_name: Hugging Face model name for text generation
        """
        self.model_name = model_name
        self.device = "cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
        
        try:
            # Load model and tokenizer
            logger.info(f"Downloading and loading model: {model_name}")
            print("      📥 Downloading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("      ✅ Tokenizer loaded")
            logger.info("Tokenizer loaded successfully")
            
            print("      📥 Downloading model weights...")
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            print("      ✅ Model loaded")
            logger.info("Model loaded successfully")
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create text generation pipeline
            print("      ⚙️  Setting up generation pipeline...")
            logger.info("Setting up text generation pipeline...")
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info(f"Text generator ready with model: {model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            # Fallback to a simpler approach
            self.generator = None
            logger.warning("Using fallback text generation (no AI model)")
    
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
        
        if self.generator is None:
            # Fallback: return a simple response based on context
            return self._fallback_response(query, context_documents)
        
        try:
            # Generate response using local model
            response = self.generator(
                prompt,
                max_length=min(len(prompt.split()) + 100, 512),
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            # Extract only the new generated part (after the prompt)
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            logger.info("Generated response successfully")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._fallback_response(query, context_documents)
    
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
    
    def _fallback_response(self, query: str, context_documents: List[Dict[str, Any]]) -> str:
        """Fallback response when no AI model is available.
        
        Args:
            query: User's question
            context_documents: Retrieved documents
            
        Returns:
            Simple response based on context
        """
        if not context_documents:
            return f"I found no relevant information to answer: {query}"
        
        # Simple keyword-based response
        context_text = " ".join([doc['text'] for doc in context_documents])
        query_words = query.lower().split()
        
        # Find sentences that contain query words
        sentences = context_text.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in query_words if len(word) > 3):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return f"Based on the available information: {' '.join(relevant_sentences[:3])}"
        else:
            return f"I found some relevant documents but couldn't extract a direct answer to: {query}"
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create the prompt for the language model.
        
        Args:
            query: User's question
            context: Context documents
            
        Returns:
            Formatted prompt
        """
        return f"""Question: {query}

Context: {context}

Answer:"""
