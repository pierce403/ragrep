#!/usr/bin/env python3
"""Command-line interface for RAGrep."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List

from .core.rag_system import RAGrep
from .retrieval.embeddings import get_runtime_device_info


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--db-path",
        default=os.getenv("RAGREP_DB_PATH", "./.ragrep.db"),
        help="Path to local SQLite database",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", "1000")),
        help="Chunk size",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", "200")),
        help="Chunk overlap",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("EMBEDDING_MODEL", "mxbai-embed-large"),
        help="Embedding model name",
    )
    parser.add_argument(
        "--model-dir",
        default=os.getenv("RAGREP_MODEL_DIR"),
        help="Optional directory for downloaded embedding models",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("RAGREP_DEVICE", "auto"),
        help="Embedding device: auto, cpu, cuda, mps, or explicit device (e.g. cuda:0)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser


def _build_recall_parser(prog: str = "ragrep") -> argparse.ArgumentParser:
    parser = _build_common_parser("Recall relevant chunks (auto-indexes when files changed)")
    parser.prog = prog
    parser.add_argument("query", nargs="+", help="Semantic query")
    parser.add_argument(
        "--path",
        default=None,
        help="Directory or file to index when needed (defaults to existing indexed root, else current dir)",
    )
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of results")
    parser.add_argument("--no-auto-index", action="store_true", help="Disable automatic index updates")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    return parser


def _build_index_parser() -> argparse.ArgumentParser:
    parser = _build_common_parser("Index a directory or file")
    parser.add_argument("path", nargs="?", default=".", help="Path to index")
    parser.add_argument("--force", action="store_true", help="Force a full re-index")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    return parser


def _build_stats_parser() -> argparse.ArgumentParser:
    parser = _build_common_parser("Show index statistics")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    return parser


def _build_gpu_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show GPU/device support for embeddings")
    parser.add_argument(
        "--device",
        default=os.getenv("RAGREP_DEVICE", "auto"),
        help="Requested embedding device: auto, cpu, cuda, mps, cuda:0, etc.",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")
    return parser


def _run_gpu_info(args: argparse.Namespace) -> int:
    info = get_runtime_device_info(args.device)
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"Requested: {info['requested_device']}")
        print(f"Resolved: {info['resolved_device']}")
        print(f"PyTorch available: {info['torch_available']}")
        print(f"CUDA available: {info['cuda_available']}")
        print(f"CUDA device count: {info['cuda_device_count']}")
        if info["cuda_devices"]:
            print("CUDA devices:")
            for index, name in enumerate(info["cuda_devices"]):
                print(f"  {index}: {name}")
        print(f"MPS available: {info['mps_available']}")
    return 0


def _run_recall(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)
    query = " ".join(args.query).strip()

    with RAGrep(
        db_path=args.db_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.model,
        model_dir=args.model_dir,
        embedding_device=args.device,
    ) as rag:
        result = rag.recall(
            query,
            limit=args.limit,
            path=args.path,
            auto_index=not args.no_auto_index,
        )

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    index_info = result.get("auto_index")
    if index_info and index_info.get("indexed"):
        print(
            f"Indexed {index_info['files']} files ({index_info['chunks']} chunks): "
            f"{index_info['reason']}"
        )

    matches = result.get("matches", [])
    print(f"Results: {len(matches)}")
    for position, match in enumerate(matches, start=1):
        source = match.get("metadata", {}).get("source", "unknown")
        print(f"{position}. score={match['score']:.4f} source={source}")
        print(match.get("text", "").rstrip())

    return 0


def _run_index(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)

    with RAGrep(
        db_path=args.db_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.model,
        model_dir=args.model_dir,
        embedding_device=args.device,
    ) as rag:
        result = rag.index(path=args.path, force=args.force)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    if result["indexed"]:
        print(f"Indexed {result['files']} files ({result['chunks']} chunks)")
    else:
        print(f"Index unchanged: {result['reason']}")

    return 0


def _run_stats(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)

    with RAGrep(
        db_path=args.db_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.model,
        model_dir=args.model_dir,
        embedding_device=args.device,
    ) as rag:
        result = rag.stats()

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Database: {result['persist_path']}")
        print(f"Indexed root: {result.get('indexed_root')}")
        print(f"Embedding model: {result.get('embedding_model')}")
        print(f"Files: {result['total_files']}")
        print(f"Chunks: {result['total_chunks']}")
        print(f"Indexed at: {result.get('indexed_at')}")

    return 0


def main(argv: List[str] | None = None) -> int:
    args_list = list(argv) if argv is not None else sys.argv[1:]

    if not args_list:
        parser = _build_recall_parser()
        parser.print_help()
        return 0

    try:
        first = args_list[0]
        if first in {"--check-gpu", "--gpu-info"}:
            parser = _build_gpu_parser()
            args = parser.parse_args(args_list[1:])
            return _run_gpu_info(args)

        if first in {"--stats", "-s"}:
            parser = _build_stats_parser()
            args = parser.parse_args(args_list[1:])
            return _run_stats(args)

        if first == "index":
            parser = _build_index_parser()
            args = parser.parse_args(args_list[1:])
            return _run_index(args)

        if first == "stats":
            parser = _build_stats_parser()
            args = parser.parse_args(args_list[1:])
            return _run_stats(args)

        if first == "recall":
            parser = _build_recall_parser("ragrep recall")
            args = parser.parse_args(args_list[1:])
            return _run_recall(args)

        parser = _build_recall_parser()
        args = parser.parse_args(args_list)
        return _run_recall(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
