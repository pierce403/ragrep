# RAGrep - AI Agent File Navigator

A command-line tool similar to `grep`, but designed for AI agents to index and navigate files/code using semantic search.

## Overview

RAGRep is a specialized command-line tool that enables AI agents to efficiently search and understand codebases through semantic indexing. Unlike traditional `grep` which searches for exact text matches, RAGRep uses vector embeddings to find semantically related content, making it perfect for AI agents that need to understand context and meaning.

## Features

- **Semantic Search**: Find content by meaning, not just exact text matches
- **AI Agent Optimized**: Designed specifically for AI agents to navigate codebases
- **Fully Local**: No external API keys or remote services required
- **Fast Indexing**: Efficiently processes and indexes code files
- **Multiple Formats**: Supports TXT, MD, PY, JS, HTML, CSS and other text formats
- **Git Integration**: Respects `.gitignore` patterns automatically
- **LLM-Ready Output**: Structured output perfect for feeding into other AI models

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ragrep.git
cd ragrep
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the tool:
```bash
pip install -e .
```

## Usage

### Basic Commands

```bash
# Index the current directory for semantic search
ragrep index

# Index a specific directory
ragrep index ./src

# Search for semantically related content
ragrep dump "authentication logic" --limit 10

# Show indexing statistics
ragrep stats
```

### Command Reference

#### `ragrep index [path]`
Index files for semantic search. If no path is provided, indexes the current directory.

```bash
# Index current directory
ragrep index

# Index specific directory
ragrep index ./src

# Index with verbose output
ragrep index -v
```

#### `ragrep dump <query> [options]`
Search for semantically related content and output in LLM-ready format.

```bash
# Basic search
ragrep dump "database connection"

# Limit results
ragrep dump "error handling" --limit 5

# Search with verbose output
ragrep dump "API endpoints" -v
```

**Output Format:**
```
# Knowledge Base Dump for Query: 'database connection'
# Found 3 relevant chunks
================================================================================

## Chunk 1 (Similarity: 0.892)
**Source:** `src/database/connection.py`
**Content:**
"""Database connection management and pooling functionality."""
...

## Chunk 2 (Similarity: 0.756)
**Source:** `src/config/database.py`
**Content:**
"""Database configuration and connection parameters."""
...
```


#### `ragrep stats [options]`
Show statistics about the indexed knowledge base.

```bash
# Basic stats
ragrep stats

# Verbose output
ragrep stats -v
```

**Output:**
```
📊 RAG System Statistics:
========================================
🗄️  Database: ./.ragrep.db
📚 Documents in vector store: 47

📁 Directory Scan:
📄 Indexable files found: 23
💾 Total size: 45,231 bytes
```

## Use Cases for AI Agents

### Code Understanding
```bash
# Understand a specific feature
ragrep dump "user registration flow" --limit 5

# Find related functions
ragrep dump "password hashing" --limit 3

# Explore error handling patterns
ragrep dump "exception handling" --limit 10
```

### Documentation Generation
```bash
# Understand API structure
ragrep dump "API routes" --limit 15

# Find configuration options
ragrep dump "configuration settings" --limit 8

# Explore module organization
ragrep dump "module structure" --limit 10
```

### Debugging Support
```bash
# Find error-related code
ragrep dump "error logging" --limit 5

# Find test cases
ragrep dump "unit tests" --limit 10

# Explore data flow patterns
ragrep dump "data flow" --limit 8
```

## Configuration

The tool can be configured through environment variables:

- `GENERATION_MODEL`: Hugging Face model name for text generation (default: microsoft/DialoGPT-medium)
- `CHUNK_SIZE`: Size of document chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `CUDA_AVAILABLE`: Set to "true" to use GPU acceleration (default: false)

## How It Works

1. **Indexing**: Files are processed and chunked into semantic units
2. **Embedding**: Each chunk is converted to a vector embedding
3. **Storage**: Embeddings are stored in a local SQLite database
4. **Search**: Queries are converted to embeddings and matched against stored chunks
5. **Output**: Results are formatted for easy consumption by AI agents

## Comparison with grep

| Feature | grep | RAGRep |
|---------|------|--------|
| Search Type | Exact text matching | Semantic similarity |
| AI Agent Friendly | Limited | Optimized |
| Context Understanding | None | High |
| Code Navigation | Basic | Advanced |
| Learning Curve | Low | Medium |

## Examples

### Finding Authentication Code
```bash
# Traditional grep approach
grep -r "authenticate" src/

# RAGRep semantic approach
ragrep dump "user authentication system" --limit 5
```

### Understanding Database Layer
```bash
# Find related code
ragrep dump "database models and schemas" --limit 8

# Explore database connections
ragrep dump "database connection" --limit 5
```

### Exploring API Endpoints
```bash
# Find all API-related code
ragrep dump "REST API endpoints" --limit 10

# Explore API structure
ragrep dump "API structure" --limit 8
```

## Development

### Project Structure

```
ragrep/
├── src/
│   └── ragrep/
│       ├── core/          # Core functionality
│       ├── retrieval/     # Vector search
│       ├── generation/    # AI text generation
│       └── cli.py         # Command-line interface
├── examples/             # Usage examples
├── docs/                 # Documentation
└── requirements.txt      # Dependencies
```

### Running Tests

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the examples in the `examples/` directory
- Review the documentation in the `docs/` directory

---

*RAGRep - Making codebases navigable for AI agents through semantic search.*
