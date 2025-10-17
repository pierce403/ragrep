"""Lightweight file scanning without heavy dependencies."""

import os
from typing import List, Dict, Any
from pathlib import Path
import fnmatch
import logging

logger = logging.getLogger(__name__)


class FileScanner:
    """Lightweight file scanner that doesn't load heavy dependencies."""
    
    def __init__(self):
        """Initialize file scanner."""
        self.ignore_patterns = self._load_gitignore_patterns()
        
    def _load_gitignore_patterns(self) -> List[str]:
        """Load .gitignore patterns from the current directory and parent directories."""
        patterns = []
        current_dir = Path.cwd()

        # Walk up the directory tree to find .gitignore files
        while current_dir != current_dir.parent:
            gitignore_path = current_dir / '.gitignore'
            if gitignore_path.exists():
                try:
                    with open(gitignore_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                patterns.append(line)
                    logger.info(f"Loaded {len(patterns)} ignore patterns from {gitignore_path}")
                except Exception as e:
                    logger.warning(f"Failed to load .gitignore from {gitignore_path}: {e}")
            current_dir = current_dir.parent

        # Add some common default ignore patterns
        default_patterns = [
            ".git/", "__pycache__/", "*.pyc", "*.pyo", "*.pyd",
            "venv/", ".venv/", "env/", ".env/",
            "*.db", "*.sqlite", "*.sqlite3", ".ragrep.db",
            "*.log", "*.tmp", "*.bak",
            "data/", "test_vector_db/"
        ]
        for pattern in default_patterns:
            if pattern not in patterns:
                patterns.append(pattern)

        logger.info(f"Loaded {len(patterns)} ignore patterns")
        return patterns

    def _should_ignore(self, file_path: Path) -> bool:
        """Check if a file should be ignored based on .gitignore patterns."""
        file_str = str(file_path)
        try:
            relative_path = str(file_path.relative_to(Path.cwd()))
        except ValueError:
            # File is not in current directory, use absolute path
            relative_path = file_str

        for pattern in self.ignore_patterns:
            # Handle directory patterns (ending with /)
            if pattern.endswith('/'):
                dir_pattern = pattern.rstrip('/')
                if fnmatch.fnmatch(relative_path, dir_pattern + '/*') or fnmatch.fnmatch(file_str, dir_pattern + '/*'):
                    return True
                if file_path.is_dir() and (fnmatch.fnmatch(relative_path, dir_pattern) or fnmatch.fnmatch(file_str, dir_pattern)):
                    return True
            else:
                # Handle file patterns
                if fnmatch.fnmatch(relative_path, pattern) or fnmatch.fnmatch(file_path.name, pattern):
                    return True
        return False

    def scan_directory(self, directory_path: str) -> Dict[str, Any]:
        """Scan directory for files without processing them.
        
        Args:
            directory_path: Path to directory to scan
            
        Returns:
            Dictionary with scan results
        """
        directory = Path(directory_path)
        
        # Supported file extensions
        text_extensions = {'.txt', '.md', '.py', '.js', '.html', '.css'}
        
        files_found = []
        total_size = 0
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in text_extensions:
                # Check if file should be ignored
                if self._should_ignore(file_path):
                    continue
                    
                try:
                    file_size = file_path.stat().st_size
                    files_found.append({
                        'path': str(file_path),
                        'name': file_path.name,
                        'size': file_size,
                        'extension': file_path.suffix.lower()
                    })
                    total_size += file_size
                except Exception as e:
                    logger.warning(f"Failed to get info for {file_path}: {e}")
        
        return {
            'files': files_found,
            'total_files': len(files_found),
            'total_size': total_size,
            'directory': str(directory)
        }
