#!/usr/bin/env python3
"""Command-line interface for RAGRep."""

import os
import sys
import argparse
import logging
from pathlib import Path

from .core.rag_system import RAGSystem


def setup_logging(verbose=False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
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
    index_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
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
    
    # Set up logging with verbose mode for index command
    verbose = getattr(args, 'verbose', False)
    setup_logging(verbose=verbose)
    
    try:
        if args.command == 'index':
            print(f"🚀 Starting indexing process...")
            print(f"📁 Target path: {args.path}")
            print(f"🗄️  Database: {args.db_path}")
            print("=" * 60)
            
            rag = RAGSystem(vector_db_path=args.db_path)
            rag.add_documents(args.path)
            
            print("=" * 60)
            print(f"✅ Successfully indexed documents from {args.path}")
            print(f"📊 Use 'ragrep stats' to see database statistics")
            
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
            # Only initialize vector store for dump command
            from .retrieval.vector_store import VectorStore
            vector_store = VectorStore(args.db_path)
            chunks = vector_store.search(args.query, n_results=args.limit)
            
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
            # Only initialize vector store for stats command
            from .retrieval.vector_store import VectorStore
            vector_store = VectorStore(args.db_path)
            collection_info = vector_store.get_collection_info()
            
            print("RAG System Statistics:")
            print(f"Database path: {collection_info['persist_directory']}")
            print(f"Collection name: {collection_info['collection_name']}")
            print(f"Documents in vector store: {collection_info['total_documents']}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
