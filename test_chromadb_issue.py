#!/usr/bin/env python3
"""Test the specific ChromaDB issue we're seeing."""

import os
import sys
import shutil

def test_chromadb_initialization():
    """Test ChromaDB initialization with different approaches."""
    print("🔍 Testing ChromaDB initialization...")
    
    # Remove existing database to avoid conflicts
    db_path = ".ragrep.db"
    if os.path.exists(db_path):
        print(f"Removing existing database: {db_path}")
        shutil.rmtree(db_path)
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        print(f"ChromaDB version: {chromadb.__version__}")
        
        # Test 1: Basic initialization
        print("\nTest 1: Basic initialization")
        try:
            client1 = chromadb.PersistentClient(path="./test_db_basic")
            print("✅ Basic initialization successful")
        except Exception as e:
            print(f"❌ Basic initialization failed: {e}")
            return False
        
        # Test 2: With settings
        print("\nTest 2: With settings")
        try:
            client2 = chromadb.PersistentClient(
                path="./test_db_settings",
                settings=Settings(anonymized_telemetry=False)
            )
            print("✅ Settings initialization successful")
        except Exception as e:
            print(f"❌ Settings initialization failed: {e}")
            print("This is expected for Python 3.8 - will use fallback")
        
        # Test 3: Collection creation
        print("\nTest 3: Collection creation")
        try:
            collection = client1.get_or_create_collection(
                name="test_collection",
                metadata={"description": "Test collection"}
            )
            print("✅ Collection creation with metadata successful")
        except Exception as e:
            print(f"❌ Collection creation with metadata failed: {e}")
            print("Trying without metadata...")
            try:
                collection = client1.get_or_create_collection(name="test_collection")
                print("✅ Collection creation without metadata successful")
            except Exception as e2:
                print(f"❌ Collection creation without metadata also failed: {e2}")
                return False
        
        # Test 4: Basic operations
        print("\nTest 4: Basic operations")
        try:
            # Add some test data
            collection.add(
                documents=["This is a test document about machine learning."],
                metadatas=[{"source": "test.txt"}],
                ids=["test_1"]
            )
            print("✅ Document addition successful")
            
            # Search
            results = collection.query(
                query_texts=["machine learning"],
                n_results=1
            )
            print(f"✅ Search successful, found {len(results['documents'][0])} results")
            
        except Exception as e:
            print(f"❌ Basic operations failed: {e}")
            return False
        
        print("\n🎉 All ChromaDB tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ ChromaDB import failed: {e}")
        return False

def test_ragrep_vector_store():
    """Test our VectorStore class specifically."""
    print("\n🔍 Testing RAGrep VectorStore class...")
    
    # Remove existing database
    db_path = ".ragrep.db"
    if os.path.exists(db_path):
        print(f"Removing existing database: {db_path}")
        shutil.rmtree(db_path)
    
    try:
        from src.ragrep.retrieval.vector_store import VectorStore
        
        # Test VectorStore creation
        print("Creating VectorStore...")
        vector_store = VectorStore("./test_ragrep_db")
        print("✅ VectorStore created successfully")
        
        # Test adding documents
        print("Testing document addition...")
        test_chunks = [
            {
                "text": "This is a test document about machine learning and AI.",
                "metadata": {"source": "test.txt", "chunk_id": 0}
            }
        ]
        
        vector_store.add_documents(test_chunks)
        print("✅ Documents added successfully")
        
        # Test search
        print("Testing search...")
        results = vector_store.search("machine learning", n_results=1)
        print(f"✅ Search successful, found {len(results)} results")
        
        return True
        
    except Exception as e:
        print(f"❌ VectorStore test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run ChromaDB issue tests."""
    print("🔧 ChromaDB Issue Test Suite")
    print("="*40)
    
    # Check if we're in the right directory
    if not os.path.exists("src/ragrep"):
        print("❌ Error: Please run this script from the ragrep project root directory")
        sys.exit(1)
    
    # Run tests
    tests = [
        ("ChromaDB Initialization", test_chromadb_initialization),
        ("RAGrep VectorStore", test_ragrep_vector_store),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            print(f"✅ {test_name} - PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} - FAILED")
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All ChromaDB tests passed!")
        return 0
    else:
        print("\n⚠️  Some ChromaDB tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
