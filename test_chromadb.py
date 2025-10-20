#!/usr/bin/env python3
"""Test script to debug ChromaDB compatibility issues."""

from __future__ import annotations

import sys
print(f"Python version: {sys.version}")

try:
    print("Testing ChromaDB import...")
    import chromadb
    print(f"ChromaDB version: {chromadb.__version__}")
    
    print("Testing ChromaDB client creation...")
    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path="./test_chromadb",
        settings=Settings(anonymized_telemetry=False)
    )
    print("✅ ChromaDB client created successfully")
    
    print("Testing collection creation...")
    collection = client.get_or_create_collection(
        name="test_collection",
        metadata={"description": "Test collection"}
    )
    print("✅ Collection created successfully")
    
    print("All ChromaDB tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
