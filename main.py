#!/usr/bin/env python3
"""Main entry point for RAGRep CLI."""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ragrep import RAGSystem


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="RAGRep - Retrieval-Augmented Generation Tool")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add documents command
    add_parser = subparsers.add_parser('add', help='Add documents to knowledge base')
    add_parser.add_argument('path', help='Path to document or directory')
    add_parser.add_argument('--db-path', default='./data/vector_db', help='Vector database path')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the knowledge base')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--db-path', default='./data/vector_db', help='Vector database path')
    query_parser.add_argument('--n-results', type=int, default=5, help='Number of results to retrieve')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show system statistics')
    stats_parser.add_argument('--db-path', default='./data/vector_db', help='Vector database path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup_logging()
    
    try:
        if args.command == 'add':
            rag = RAGSystem(vector_db_path=args.db_path)
            rag.add_documents(args.path)
            print(f"Successfully added documents from {args.path}")
            
        elif args.command == 'query':
            rag = RAGSystem(vector_db_path=args.db_path)
            result = rag.query(args.question, n_results=args.n_results)
            
            print(f"\nQuestion: {result['question']}")
            print(f"Answer: {result['answer']}")
            print(f"\nSources ({result['num_sources']}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['metadata'].get('filename', 'Unknown')}")
                print(f"   {source['text']}")
                print()
                
        elif args.command == 'stats':
            rag = RAGSystem(vector_db_path=args.db_path)
            stats = rag.get_stats()
            
            print("RAG System Statistics:")
            print(f"Documents in vector store: {stats['vector_store']['total_documents']}")
            print(f"Generation model: {stats['generation_model']}")
            print(f"Chunk size: {stats['chunk_size']}")
            print(f"Chunk overlap: {stats['chunk_overlap']}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
