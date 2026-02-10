"""Basic usage example for RAGrep."""

from __future__ import annotations

from pathlib import Path
import sys

# Add src to path when running from repository checkout.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragrep import RAGrep


def main() -> None:
    rag = RAGrep(db_path="./.ragrep.db")

    # Index the current repository.
    index_result = rag.index(".")
    print(index_result)

    # Recall relevant chunks.
    recall_result = rag.recall("semantic search", limit=5)
    for match in recall_result["matches"]:
        print(f"{match['score']:.4f} {match['metadata'].get('source')}")

    # Print stats.
    print(rag.stats())


if __name__ == "__main__":
    main()
