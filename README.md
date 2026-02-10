# RAGrep

RAGrep is a dead-simple local semantic recall tool for code and text files.

It uses:
- `mxbai-embed-large` embeddings from a local Ollama server
- a single local SQLite database file: `.ragrep.db`

No ChromaDB. No remote API keys.

## Install

```bash
pip install ragrep
```

## Prerequisites

1. Install Ollama: https://ollama.com/download
2. Pull the embedding model:

```bash
ollama pull mxbai-embed-large
```

## CLI Usage

Recall is the default command.

```bash
# Implied recall (auto-indexes when needed)
ragrep "authentication middleware"

# Explicit recall (same behavior)
ragrep recall "authentication middleware"

# Build/update index manually
ragrep index .

# Show stats
ragrep stats
```

Useful flags:

```bash
ragrep "query text" --path . --limit 10 --db-path ./.ragrep.db
ragrep "query text" --json
ragrep index . --force
```

## Python Usage

```python
from ragrep import RAGrep

rag = RAGrep(db_path="./.ragrep.db")
rag.index(".")

result = rag.recall("database transactions", limit=5)
for match in result["matches"]:
    print(match["score"], match["metadata"]["source"])

print(rag.stats())
rag.close()
```

Library methods:
- `index(path=".", force=False)`
- `recall(query, limit=20, path=".", auto_index=True)`
- `stats()`

Backwards-compatible aliases still available:
- `RAGSystem` (alias of `RAGrep`)
- `dump(...)` (alias of `recall(..., auto_index=False)` result list)

## Local Database

RAGrep stores everything in one SQLite file (default `./.ragrep.db`):
- indexed files and mtimes
- chunked source text
- embedding vectors
- index metadata (model, chunk settings, root path)

## Development

```bash
pip install -e .[dev]
pytest
python -m build
twine check dist/*
```

## License

MIT
