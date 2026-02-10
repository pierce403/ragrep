"""Document loading and chunking utilities."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, Dict, Iterable, List


_DEFAULT_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".json",
    ".md",
    ".py",
    ".rb",
    ".rs",
    ".sql",
    ".toml",
    ".ts",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

_DEFAULT_IGNORE_PATTERNS = {
    ".git/",
    "__pycache__/",
    "*.pyc",
    "*.sqlite",
    "*.sqlite3",
    "*.db",
    ".ragrep.db",
    ".ragrep.db.legacy/",
    "venv/",
    ".venv/",
    "node_modules/",
    "dist/",
    "build/",
}


class DocumentProcessor:
    """Read text files from disk and split them into overlapping chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process_path(
        self,
        path: str,
        *,
        extra_ignore_paths: Iterable[Path] | None = None,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Path]:
        files, file_records, scan_root = self.discover_path(
            path,
            extra_ignore_paths=extra_ignore_paths,
        )
        chunks = self.process_files(files, scan_root)
        return chunks, file_records, scan_root

    def discover_path(
        self,
        path: str,
        *,
        extra_ignore_paths: Iterable[Path] | None = None,
    ) -> tuple[List[Path], List[Dict[str, Any]], Path]:
        root = Path(path).resolve()
        if not root.exists():
            raise ValueError(f"Path does not exist: {path}")

        if root.is_file():
            files = [root]
            scan_root = root.parent
        else:
            scan_root = root
            files = self.scan_files(scan_root, extra_ignore_paths=extra_ignore_paths)

        file_records = self.collect_file_records(files, scan_root)
        return files, file_records, scan_root

    def process_files(self, files: Iterable[Path], scan_root: Path) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        for file_path in files:
            relative_path = file_path.relative_to(scan_root).as_posix()
            text = self._load_text(file_path)
            chunks.extend(self._chunk_text(text, relative_path))
        return chunks

    def scan_files(
        self,
        root: Path,
        *,
        extra_ignore_paths: Iterable[Path] | None = None,
    ) -> List[Path]:
        ignore_patterns = self._load_ignore_patterns(root)
        resolved_ignores = [path.expanduser().resolve() for path in (extra_ignore_paths or [])]
        files: List[Path] = []

        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if self._matches_extra_ignore(file_path.resolve(), resolved_ignores):
                continue
            if file_path.suffix.lower() not in _DEFAULT_EXTENSIONS:
                continue

            relative = file_path.relative_to(root).as_posix()
            if self._should_ignore(relative, ignore_patterns):
                continue

            files.append(file_path)

        files.sort()
        return files

    def collect_file_records(self, files: Iterable[Path], root: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []

        for file_path in files:
            stat = file_path.stat()
            records.append(
                {
                    "path": file_path.relative_to(root).as_posix(),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )

        records.sort(key=lambda item: item["path"])
        return records

    def _chunk_text(self, text: str, relative_path: str) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunk_index = len(chunks)
            chunk_id = f"{relative_path}:{chunk_index}:{start}:{end}"

            metadata = {
                "source": relative_path,
                "chunk_index": chunk_index,
                "start_char": start,
                "end_char": end,
            }
            chunks.append(
                {
                    "id": chunk_id,
                    "file_path": relative_path,
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                    "text": chunk_text,
                    "metadata": metadata,
                }
            )

            if end >= len(text):
                break
            start = end - self.chunk_overlap

        return chunks

    @staticmethod
    def _load_text(file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _load_ignore_patterns(root: Path) -> List[str]:
        patterns = set(_DEFAULT_IGNORE_PATTERNS)

        current = root
        while True:
            gitignore = current / ".gitignore"
            if gitignore.exists():
                for line in gitignore.read_text(encoding="utf-8", errors="ignore").splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    patterns.add(stripped)
            if current.parent == current:
                break
            current = current.parent

        return sorted(patterns)

    @staticmethod
    def _should_ignore(relative_path: str, patterns: Iterable[str]) -> bool:
        path = relative_path

        for pattern in patterns:
            if pattern.startswith("!"):
                continue

            normalized = pattern.strip()
            if not normalized:
                continue

            if normalized.endswith("/"):
                directory = normalized.rstrip("/")
                if path == directory or path.startswith(directory + "/"):
                    return True
                continue

            if fnmatch.fnmatch(path, normalized):
                return True
            if fnmatch.fnmatch(Path(path).name, normalized):
                return True

        return False

    @staticmethod
    def _matches_extra_ignore(file_path: Path, ignored_paths: Iterable[Path]) -> bool:
        for ignored in ignored_paths:
            if file_path == ignored:
                return True
            try:
                file_path.relative_to(ignored)
                return True
            except ValueError:
                continue
        return False
