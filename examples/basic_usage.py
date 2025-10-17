"""Basic usage example for RAGRep."""

import os
import sys
from pathlib import Path

# Add src to path so we can import ragrep
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragrep import RAGSystem


def main():
    """Demonstrate basic RAG functionality."""
    
    # Initialize RAG system
    print("Initializing RAG system...")
    print("Note: This will download a language model on first run (may take a few minutes)")
    rag = RAGSystem()
    
    # Create a sample document
    sample_doc_path = "sample_document.txt"
    with open(sample_doc_path, "w") as f:
        f.write("""
        Machine Learning is a subset of artificial intelligence that focuses on algorithms 
        that can learn from data. It involves training models on datasets to make predictions 
        or decisions without being explicitly programmed for every task.
        
        There are three main types of machine learning:
        1. Supervised Learning: Learning with labeled training data
        2. Unsupervised Learning: Finding patterns in data without labels
        3. Reinforcement Learning: Learning through interaction with an environment
        
        Popular machine learning frameworks include TensorFlow, PyTorch, and Scikit-learn.
        """)
    
    # Add the document to the knowledge base
    print("Adding document to knowledge base...")
    rag.add_documents(sample_doc_path)
    
    # Query the system
    questions = [
        "What is machine learning?",
        "What are the types of machine learning?",
        "What frameworks are mentioned?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = rag.query(question)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['num_sources']} documents")
        print("-" * 50)
    
    # Show system stats
    print("\nSystem Statistics:")
    stats = rag.get_stats()
    print(f"Documents in vector store: {stats['vector_store']['total_documents']}")
    print(f"Generation model: {stats['generation_model']}")
    
    # Clean up
    os.remove(sample_doc_path)
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
