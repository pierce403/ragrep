#!/usr/bin/env python3
"""Command-line interface for RAGrep."""

from __future__ import annotations

import os
import sys
import argparse
import logging
from pathlib import Path

# Lazy import to avoid heavy dependencies during CLI startup
# from .core.rag_system import RAGSystem


def setup_logging(verbose=False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="RAGrep - Retrieval-Augmented Generation Tool")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
           # Index documents command
           index_parser = subparsers.add_parser('index', help='Index documents into knowledge base')
           index_parser.add_argument('path', nargs='?', default='.', help='Path to document or directory (default: current directory)')
           index_parser.add_argument('--db-path', default='./.ragrep.db', help='Vector database path')
           index_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')

           # Use unique database path in testing environments
           if os.environ.get('GITHUB_ACTIONS') == 'true':
               index_parser.set_defaults(db_path=f'./.ragrep_test_{os.getpid()}.db')
    
    # Dump command
    dump_parser = subparsers.add_parser('dump', help='Dump knowledge base contents matching query (no LLM processing)')
    dump_parser.add_argument('query', help='Query to search for relevant chunks')
    dump_parser.add_argument('--db-path', default='./.ragrep.db', help='Vector database path')
    dump_parser.add_argument('--limit', type=int, default=20, help='Maximum number of chunks to show (default: 20)')
    dump_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')

    # Use unique database path in testing environments
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        dump_parser.set_defaults(db_path=f'./.ragrep_test_{os.getpid()}.db')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show system statistics')
    stats_parser.add_argument('--db-path', default='./.ragrep.db', help='Vector database path')
    stats_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')

    # Use unique database path in testing environments
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        stats_parser.set_defaults(db_path=f'./.ragrep_test_{os.getpid()}.db')
    
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Set up logging with verbose mode
    verbose = getattr(args, 'verbose', False)
    setup_logging(verbose=verbose)
    
    try:
        if args.command == 'index':
            import time
            start_time = time.time()
            
            if verbose:
                print(f"🚀 Starting indexing process... [{time.strftime('%H:%M:%S')}]")
                print(f"📁 Target path: {args.path}")
                print(f"🗄️  Database: {args.db_path}")
                print("=" * 60)
                print("⚙️  Initializing indexing components...")
                print("   📄 Setting up document processor...")
            
            doc_start = time.time()
            # Only load document processor first - no vector store yet!
            from .core.document_processor import DocumentProcessor
            if verbose:
                print(f"   ✅ DocumentProcessor imported [{time.time() - doc_start:.2f}s]")
                print("   📄 Setting up document processor...")
            
            processor_start = time.time()
            document_processor = DocumentProcessor()
            if verbose:
                print(f"   ✅ DocumentProcessor created [{time.time() - processor_start:.2f}s]")
                print(f"✅ Document processing ready! [{time.time() - start_time:.2f}s total]")
                print("=" * 60)
            
            # Process documents first (lightweight)
            process_start = time.time()
            if verbose:
                print(f"📄 Starting document processing... [{time.strftime('%H:%M:%S')}]")
            chunks = document_processor.process_directory(args.path)
            if verbose:
                print(f"✅ Document processing complete [{time.time() - process_start:.2f}s]")
            
            # Only now load vector store when we have documents to process
            vector_start = time.time()
            if verbose:
                print(f"💾 Loading vector database... [{time.strftime('%H:%M:%S')}]")
            from .retrieval.vector_store import VectorStore
            vector_store = VectorStore(args.db_path)
            if verbose:
                print(f"✅ Vector store ready [{time.time() - vector_start:.2f}s]")
            
            # Add documents to vector store
            vector_add_start = time.time()
            if verbose:
                print(f"💾 Adding documents to vector store... [{time.strftime('%H:%M:%S')}]")
            vector_store.add_documents(chunks)
            if verbose:
                print(f"✅ Vector store operations complete [{time.time() - vector_add_start:.2f}s]")
                print("=" * 60)
            
            print(f"✅ Successfully indexed {len(chunks)} chunks from {args.path}")
            if verbose:
                print(f"📊 Use 'ragrep stats' to see database statistics")
            
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
            print("📊 RAG System Statistics:")
            print("=" * 40)
            
            # Check if database exists
            if os.path.exists(args.db_path):
                print(f"🗄️  Database: {args.db_path}")
                try:
                    from .retrieval.vector_store import VectorStore
                    vector_store = VectorStore(args.db_path)
                    collection_info = vector_store.get_collection_info()
                    print(f"📚 Documents in vector store: {collection_info['total_documents']}")
                except Exception as e:
                    print(f"⚠️  Could not read vector store: {e}")
            else:
                print(f"❌ No database found at: {args.db_path}")
            
            # Scan current directory for indexable files
            print("\n📁 Directory Scan:")
            from .core.file_scanner import FileScanner
            scanner = FileScanner()
            scan_results = scanner.scan_directory(".")
            
            print(f"📄 Indexable files found: {scan_results['total_files']}")
            print(f"💾 Total size: {scan_results['total_size']:,} bytes")
            
            if scan_results['files']:
                print(f"\n📋 File breakdown by type:")
                extensions = {}
                for file_info in scan_results['files']:
                    ext = file_info['extension']
                    if ext not in extensions:
                        extensions[ext] = {'count': 0, 'size': 0}
                    extensions[ext]['count'] += 1
                    extensions[ext]['size'] += file_info['size']
                
                for ext, info in sorted(extensions.items()):
                    print(f"   {ext}: {info['count']} files ({info['size']:,} bytes)")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
