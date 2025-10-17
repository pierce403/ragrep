#!/usr/bin/env python3
"""Command-line interface for RAGRep."""

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
    import time
    print(f"🚀 RAGRep CLI starting... [{time.strftime('%H:%M:%S')}]")
    
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
    print(f"📋 Arguments parsed... [{time.strftime('%H:%M:%S')}]")

    if not args.command:
        parser.print_help()
        return

    # Set up logging with verbose mode for index command
    verbose = getattr(args, 'verbose', False)
    print(f"⚙️  Setting up logging... [{time.strftime('%H:%M:%S')}]")
    setup_logging(verbose=verbose)
    print(f"✅ Logging ready... [{time.strftime('%H:%M:%S')}]")
    
    try:
        if args.command == 'index':
            import time
            start_time = time.time()
            
            print(f"🚀 Starting indexing process... [{time.strftime('%H:%M:%S')}]")
            print(f"📁 Target path: {args.path}")
            print(f"🗄️  Database: {args.db_path}")
            print("=" * 60)
            print("⚙️  Initializing indexing components...")
            print("   📄 Setting up document processor...")
            
            doc_start = time.time()
            # Only load document processor first - no vector store yet!
            from .core.document_processor import DocumentProcessor
            print(f"   ✅ DocumentProcessor imported [{time.time() - doc_start:.2f}s]")
            
            print("   📄 Setting up document processor...")
            processor_start = time.time()
            document_processor = DocumentProcessor()
            print(f"   ✅ DocumentProcessor created [{time.time() - processor_start:.2f}s]")
            
            print(f"✅ Document processing ready! [{time.time() - start_time:.2f}s total]")
            print("=" * 60)
            
            # Process documents first (lightweight)
            process_start = time.time()
            print(f"📄 Starting document processing... [{time.strftime('%H:%M:%S')}]")
            chunks = document_processor.process_directory(args.path)
            print(f"✅ Document processing complete [{time.time() - process_start:.2f}s]")
            
            # Only now load vector store when we have documents to process
            vector_start = time.time()
            print(f"💾 Loading vector database... [{time.strftime('%H:%M:%S')}]")
            from .retrieval.vector_store import VectorStore
            vector_store = VectorStore(args.db_path)
            print(f"✅ Vector store ready [{time.time() - vector_start:.2f}s]")
            
            # Add documents to vector store
            vector_add_start = time.time()
            print(f"💾 Adding documents to vector store... [{time.strftime('%H:%M:%S')}]")
            vector_store.add_documents(chunks)
            print(f"✅ Vector store operations complete [{time.time() - vector_add_start:.2f}s]")
            
            print("=" * 60)
            print(f"✅ Successfully indexed documents from {args.path}")
            print(f"📊 Use 'ragrep stats' to see database statistics")
            
        elif args.command == 'query':
            from .core.rag_system import RAGSystem
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
