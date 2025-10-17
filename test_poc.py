#!/usr/bin/env python3
"""Quick test of the RAG POC."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from ragrep import RAGSystem, DocumentProcessor, VectorStore, TextGenerator
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_document_processor():
    """Test document processing."""
    try:
        from ragrep.core.document_processor import DocumentProcessor
        
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        
        # Create test document
        test_doc = "This is a test document. " * 10  # Make it long enough to chunk
        chunks = processor.chunk_text(test_doc, {"test": True})
        
        assert len(chunks) > 1, "Should create multiple chunks"
        assert all("text" in chunk for chunk in chunks), "Each chunk should have text"
        assert all("metadata" in chunk for chunk in chunks), "Each chunk should have metadata"
        
        print("✅ Document processor test passed")
        return True
    except Exception as e:
        print(f"❌ Document processor test failed: {e}")
        return False

def test_vector_store():
    """Test vector store functionality."""
    try:
        from ragrep.retrieval.vector_store import VectorStore
        
        # Use a temporary directory for testing
        test_db_path = "./test_vector_db"
        if os.path.exists(test_db_path):
            import shutil
            shutil.rmtree(test_db_path)
        
        store = VectorStore(test_db_path)
        
        # Test adding documents
        test_chunks = [
            {"text": "This is about machine learning", "metadata": {"topic": "ml"}},
            {"text": "This is about artificial intelligence", "metadata": {"topic": "ai"}}
        ]
        
        store.add_documents(test_chunks)
        
        # Test searching
        results = store.search("machine learning", n_results=1)
        assert len(results) > 0, "Should find results"
        
        # Clean up
        import shutil
        shutil.rmtree(test_db_path)
        
        print("✅ Vector store test passed")
        return True
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing RAG POC...")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_document_processor,
        test_vector_store
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! POC is ready.")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
