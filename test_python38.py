#!/usr/bin/env python3
"""Test script to debug Python 3.8 compatibility issues."""

from __future__ import annotations

import sys
print(f"Python version: {sys.version}")

try:
    print("Testing imports...")
    from src.ragrep.core.document_processor import DocumentProcessor
    print("✅ DocumentProcessor imported successfully")
    
    from src.ragrep.retrieval.vector_store import VectorStore
    print("✅ VectorStore imported successfully")
    
    from src.ragrep.core.file_scanner import FileScanner
    print("✅ FileScanner imported successfully")
    
    print("\nTesting basic functionality...")
    
    # Test DocumentProcessor
    processor = DocumentProcessor()
    print("✅ DocumentProcessor created successfully")
    
    # Test FileScanner
    scanner = FileScanner()
    print("✅ FileScanner created successfully")
    
    # Test VectorStore (this might be where the error occurs)
    print("Testing VectorStore creation...")
    vector_store = VectorStore("./test_db")
    print("✅ VectorStore created successfully")
    
    print("\nAll tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
