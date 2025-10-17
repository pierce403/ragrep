#!/usr/bin/env python3
"""Command-line interface for RAGRep."""

import os
import sys
import argparse
import logging
from pathlib import Path

from .core.rag_system import RAGSystem


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
    
    # Index documents command
    index_parser = subparsers.add_parser('index', help='Index documents into knowledge base')
    index_parser.add_argument('path', nargs='?', default='.', help='Path to document or directory (default: current directory)')
    index_parser.add_argument('--db-path', default='./.ragrep.db', help='Vector database path')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the knowledge base')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--db-path', default='./.ragrep.db', help='Vector database path')
    query_parser.add_argument('--n-results', type=int, default=5, help='Number of results to retrieve')
    query_parser.add_argument('--model', default='microsoft/DialoGPT-medium', help='Hugging Face model name')
    
    # Dump command
    dump_parser = subparsers.add_parser('dump', help='Dump knowledge base contents matching query (no LLM processing)')
    dump_parser.add_argument('query', help='Query to search for relevant chunks')
    dump_parser.add_argument('--db-path', default='./.ragrep.db', help='Vector database path')
    dump_parser.add_argument('--limit', type=int, default=20, help='Maximum number of chunks to show (default: 20)')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show system statistics')
    stats_parser.add_argument('--db-path', default='./.ragrep.db', help='Vector database path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup_logging()
    
    try:
        if args.command == 'index':
            rag = RAGSystem(vector_db_path=args.db_path)
            rag.add_documents(args.path)
            print(f"Successfully indexed documents from {args.path}")
            
        elif args.command == 'query':
            rag = RAGSystem(vector_db_path=args.db_path, generation_model=args.model)
            result = rag.query(args.question, n_results=args.n_results)
            
            print(f"\nQuestion: {result['question']}")
            print(f"Answer: {result['answer']}")
            print(f"\nSources ({result['num_sources']}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['metadata'].get('filename', 'Unknown')}")
                print(f"   {source['text']}")
                print()
        
        elif args.command == 'dump':
            rag = RAGSystem(vector_db_path=args.db_path)
            chunks = rag.dump_chunks(args.query, limit=args.limit)
            
            print(f"# Knowledge Base Dump for Query: '{args.query}'")
            print(f"# Found {len(chunks)} relevant chunks")
            print("=" * 80)
            
            for i, chunk in enumerate(chunks, 1):
                source = chunk['metadata'].get('source', 'Unknown')
                similarity = chunk.get('distance', 0)
                print(f"\n## Chunk {i} (Similarity: {similarity:.3f})")
                print(f"**Source:** `{source}`")
                print(f"**Content:**")
                print(chunk['text'])
                print("-" * 40)
                
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
