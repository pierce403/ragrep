#!/usr/bin/env python3
"""Local test suite to verify functionality before pushing to GitHub."""

import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🧪 Testing: {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 {description} - EXCEPTION: {e}")
        return False

def test_basic_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*60)
    print("🔍 TESTING BASIC IMPORTS")
    print("="*60)

    tests = [
        ("python3 -c 'from src.ragrep.core.document_processor import DocumentProcessor'", "DocumentProcessor import"),
        ("python3 -c 'from src.ragrep.retrieval.vector_store import VectorStore'", "VectorStore import"),
        ("python3 -c 'from src.ragrep.core.file_scanner import FileScanner'", "FileScanner import"),
        ("python3 -c 'from src.ragrep.generation.text_generator import TextGenerator'", "TextGenerator import"),
        ("python3 -c 'from src.ragrep.cli import main'", "CLI import"),
    ]

    passed = 0
    for cmd, desc in tests:
        if run_command(cmd, desc):
            passed += 1

    print(f"\n📊 Import Tests: {passed}/{len(tests)} passed")
    return passed == len(tests)

def test_syntax_validation():
    """Test that all Python files have valid syntax."""
    print("\n" + "="*60)
    print("🔍 TESTING SYNTAX VALIDATION")
    print("="*60)

    # Find all Python files
    import glob
    python_files = glob.glob("src/**/*.py", recursive=True)

    passed = 0
    for py_file in python_files:
        try:
            with open(py_file, 'r') as f:
                content = f.read()
            compile(content, py_file, 'exec')
            print(f"✅ {py_file} - Valid syntax")
            passed += 1
        except SyntaxError as e:
            print(f"❌ {py_file} - Syntax error: {e}")
            return False
        except Exception as e:
            print(f"❌ {py_file} - Error: {e}")
            return False

    print(f"\n📊 Syntax Tests: {passed}/{len(python_files)} passed")
    return passed == len(python_files)

def test_cli_help():
    """Test that CLI help works."""
    print("\n" + "="*60)
    print("🔍 TESTING CLI HELP")
    print("="*60)
    
    tests = [
        ("python3 -m src.ragrep.cli --help", "CLI help command"),
        ("python3 -m src.ragrep.cli index --help", "Index help command"),
        ("python3 -m src.ragrep.cli dump --help", "Dump help command"),
        ("python3 -m src.ragrep.cli stats --help", "Stats help command"),
    ]
    
    passed = 0
    for cmd, desc in tests:
        if run_command(cmd, desc):
            passed += 1
    
    print(f"\n📊 CLI Help Tests: {passed}/{len(tests)} passed")
    return passed == len(tests)

def test_basic_functionality():
    """Test basic functionality with clean environment."""
    print("\n" + "="*60)
    print("🔍 TESTING BASIC FUNCTIONALITY")
    print("="*60)
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create test files
        test_files = {
            "test.txt": "This is a test document about machine learning and artificial intelligence.",
            "test.py": "# Python Code\n\ndef hello():\n    print('Hello, World!')\n    return 'success'",
            "test.md": "# Test Markdown\n\nThis is a **test** document with some *formatting*."
        }
        
        for filename, content in test_files.items():
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)
        
        # Test with absolute path to avoid module import issues
        original_cwd = os.getcwd()
        
        try:
            # Set PYTHONPATH to include the current directory
            env = os.environ.copy()
            env['PYTHONPATH'] = original_cwd
            
            # Test stats command (should work without database)
            if not run_command(f"cd {temp_dir} && PYTHONPATH={original_cwd} python3 -m src.ragrep.cli stats", "Stats command (no database)"):
                return False
            
            # Test index command
            if not run_command(f"cd {temp_dir} && PYTHONPATH={original_cwd} python3 -m src.ragrep.cli index .", "Index command"):
                return False
            
            # Test dump command
            if not run_command(f"cd {temp_dir} && PYTHONPATH={original_cwd} python3 -m src.ragrep.cli dump 'machine learning' --limit 2", "Dump command"):
                return False
            
            # Test stats command (with database)
            if not run_command(f"cd {temp_dir} && PYTHONPATH={original_cwd} python3 -m src.ragrep.cli stats", "Stats command (with database)"):
                return False
            
            return True
            
        finally:
            pass

def test_python_versions():
    """Test with different Python versions if available."""
    print("\n" + "="*60)
    print("🔍 TESTING PYTHON VERSIONS")
    print("="*60)
    
    python_versions = ["python3", "python3.9", "python3.10", "python3.11", "python3.12"]
    available_versions = []
    
    for version in python_versions:
        if run_command(f"{version} --version", f"Check {version} availability"):
            available_versions.append(version)
    
    print(f"\n📊 Available Python versions: {available_versions}")
    
    # Test basic import with each available version
    passed = 0
    for version in available_versions:
        if run_command(f"{version} -c 'from src.ragrep.core.document_processor import DocumentProcessor'", f"Import test with {version}"):
            passed += 1
    
    print(f"\n📊 Python Version Tests: {passed}/{len(available_versions)} passed")
    return passed == len(available_versions)

def test_clean_database():
    """Test with clean database to avoid conflicts."""
    print("\n" + "="*60)
    print("🔍 TESTING CLEAN DATABASE")
    print("="*60)
    
    # Remove any existing database
    db_path = ".ragrep.db"
    if os.path.exists(db_path):
        print(f"Removing existing database: {db_path}")
        shutil.rmtree(db_path)
    
    # Test with clean database
    tests = [
        ("python3 -m src.ragrep.cli stats", "Stats with clean database"),
        ("python3 -m src.ragrep.cli index examples/", "Index with clean database"),
        ("python3 -m src.ragrep.cli dump 'test' --limit 1", "Dump with clean database"),
    ]
    
    passed = 0
    for cmd, desc in tests:
        if run_command(cmd, desc):
            passed += 1
    
    print(f"\n📊 Clean Database Tests: {passed}/{len(tests)} passed")
    return passed == len(tests)

def main():
    """Run all tests."""
    print("🚀 RAGrep Local Test Suite")
    print("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists("src/ragrep"):
        print("❌ Error: Please run this script from the ragrep project root directory")
        sys.exit(1)
    
    # Run all tests
    test_results = []

    test_results.append(("Syntax Validation", test_syntax_validation()))
    test_results.append(("Basic Imports", test_basic_imports()))
    test_results.append(("CLI Help", test_cli_help()))
    test_results.append(("Basic Functionality", test_basic_functionality()))
    test_results.append(("Python Versions", test_python_versions()))
    test_results.append(("Clean Database", test_clean_database()))
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(test_results)} test suites passed")
    
    if passed == len(test_results):
        print("\n🎉 All tests passed! Safe to push to GitHub.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Fix issues before pushing to GitHub.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
