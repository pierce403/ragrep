#!/usr/bin/env python3
"""Test Python 3.9 compatibility specifically."""

import os
import sys
import tempfile
import shutil
import subprocess


def test_python39_imports():
    """Test imports with Python 3.9 syntax."""
    print("🔍 Testing Python 3.9 imports...")

    # Test basic imports
    imports = [
        "from __future__ import annotations",
        "from src.ragrep.core.document_processor import DocumentProcessor",
        "from src.ragrep.retrieval.vector_store import VectorStore",
        "from src.ragrep.core.file_scanner import FileScanner",
    ]

    for import_stmt in imports:
        try:
            exec(import_stmt)
            print(f"✅ {import_stmt}")
        except Exception as e:
            print(f"❌ {import_stmt} - {e}")
            return False

    return True


def test_python39_chromadb():
    """Test ChromaDB with Python 3.9 compatibility."""
    print("\n🔍 Testing ChromaDB Python 3.9 compatibility...")

    # Remove existing database to avoid conflicts
    db_path = ".ragrep.db"
    if os.path.exists(db_path):
        print(f"Removing existing database: {db_path}")
        shutil.rmtree(db_path)

    try:
        from src.ragrep.retrieval.vector_store import VectorStore

        # Test VectorStore creation
        print("Creating VectorStore...")
        vector_store = VectorStore("./test_python39_db")
        print("✅ VectorStore created successfully")

        # Test basic operations
        print("Testing basic operations...")

        # Test adding documents
        test_chunks = [
            {
                "text": "This is a test document about machine learning.",
                "metadata": {"source": "test.txt", "chunk_id": 0},
            },
            {
                "text": "This is another test document about artificial intelligence.",
                "metadata": {"source": "test2.txt", "chunk_id": 1},
            },
        ]

        vector_store.add_documents(test_chunks)
        print("✅ Documents added successfully")

        # Test search
        results = vector_store.search("machine learning", n_results=1)
        print(f"✅ Search successful, found {len(results)} results")

        # Test collection info
        info = vector_store.get_collection_info()
        print(f"✅ Collection info: {info['total_documents']} documents")

        return True

    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_python39_cli():
    """Test CLI with Python 3.9 compatibility."""
    print("\n🔍 Testing CLI Python 3.9 compatibility...")

    # Create test files
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        # Create test files
        test_files = {
            "test.txt": "This is a test document about machine learning and AI.",
            "test.py": "# Test Python file\n\ndef test_function():\n    return 'Hello, World!'",
        }

        for filename, content in test_files.items():
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, "w") as f:
                f.write(content)

        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH")
            if existing_pythonpath:
                env["PYTHONPATH"] = f"{original_cwd}{os.pathsep}{existing_pythonpath}"
            else:
                env["PYTHONPATH"] = original_cwd
            # Test CLI commands
            commands = [
                ("python3 -m src.ragrep.cli stats", "Stats command"),
                ("python3 -m src.ragrep.cli index .", "Index command"),
                ("python3 -m src.ragrep.cli dump 'machine learning' --limit 1", "Dump command"),
            ]

            for cmd, desc in commands:
                print(f"Running: {cmd}")
                try:
                    result = subprocess.run(
                        cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        env=env,
                    )
                    if result.returncode == 0:
                        print(f"✅ {desc}")
                    else:
                        print(f"❌ {desc} - {result.stderr.strip()}")
                        return False
                except subprocess.TimeoutExpired:
                    print(f"⏰ {desc} - Timeout")
                    return False
                except Exception as e:
                    print(f"💥 {desc} - {e}")
                    return False

            return True

        finally:
            os.chdir(original_cwd)


def main():
    """Run Python 3.9 compatibility tests."""
    print("🐍 Python 3.9 Compatibility Test Suite")
    print("=" * 50)

    # Check if we're in the right directory
    if not os.path.exists("src/ragrep"):
        print("❌ Error: Please run this script from the ragrep project root directory")
        sys.exit(1)

    # Run tests
    tests = [
        ("Python 3.9 Imports", test_python39_imports),
        ("ChromaDB Compatibility", test_python39_chromadb),
        ("CLI Compatibility", test_python39_cli),
    ]

    passed = 0
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        if test_func():
            print(f"✅ {test_name} - PASSED")
            passed += 1
        else:
            print(f"❌ {test_name} - FAILED")

    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("\n🎉 All Python 3.9 compatibility tests passed!")
        return 0
    else:
        print("\n⚠️  Some Python 3.9 compatibility tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
