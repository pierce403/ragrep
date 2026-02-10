# RAGrep

RAGrep is a dead-simple local semantic recall tool for code and text files.

It uses:
- `mxbai-embed-large` embeddings in-process (no server)
- a single local SQLite database file: `.ragrep.db`

No ChromaDB. No remote API keys.

## Install

```bash
pip install ragrep
```

## Embedding Model Storage

RAGrep downloads the embedding model automatically on first use.

Default model directories:
- Linux: `~/.config/ragrep/models`
- macOS: `~/Library/Application Support/ragrep/models`
- Windows: `%APPDATA%\\ragrep\\models`

Override with:
- env var: `RAGREP_MODEL_DIR`
- CLI flag: `--model-dir`
- Python API: `RAGrep(model_dir="...")`

## GPU Usage

RAGrep can use GPU for embeddings when available.

- Default behavior: `RAGREP_DEVICE=auto` (prefers `cuda`, then `mps`, then `cpu`)
- Override via env: `RAGREP_DEVICE=cpu|cuda|mps|cuda:0`
- Override via CLI: `--device ...`
- Override via Python API: `RAGrep(embedding_device="...")`
- Note: GPU usage requires a GPU-capable PyTorch build in your environment.
- Check runtime GPU support: `ragrep --check-gpu` (or `ragrep --check-gpu --json`)

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

# Stats alias
ragrep --stats
```

When `--path` is omitted, auto-indexing uses the previously indexed root if one exists;
otherwise it uses the current directory.

Useful flags:

```bash
ragrep "query text" --path . --limit 10 --db-path ./.ragrep.db
ragrep "query text" --model-dir ~/.config/ragrep/models --json
ragrep "query text" --device auto
ragrep index . --force
```

## Python Usage

```python
from ragrep import RAGrep

rag = RAGrep(
    db_path="./.ragrep.db",
    embedding_model="mxbai-embed-large",
    embedding_device="auto",
)
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
python -m unittest discover -s tests -p 'test_*.py'
python -m build
twine check dist/*
```

## License

MIT
