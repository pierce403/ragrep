# RAGRep - Retrieval-Augmented Generation Tool

A powerful tool for implementing Retrieval-Augmented Generation (RAG) systems that combines document retrieval with AI-powered text generation.

## Overview

RAGRep is designed to help users build sophisticated RAG applications that can search through documents and generate contextually relevant responses. The tool provides a comprehensive framework for implementing retrieval-augmented generation systems with modern AI capabilities.

## Features

- **Fully Local**: No external API keys or remote services required
- **Document Processing**: Automatic chunking and preprocessing of various document formats
- **Vector Search**: Semantic search using local embedding models
- **AI Generation**: Context-aware text generation using local language models
- **Knowledge Base Inspection**: Semantic search dump command for LLM-ready context extraction
- **Flexible Architecture**: Modular design allowing customization of retrieval and generation components
- **Multiple Formats**: Support for TXT, MD, PY, JS, HTML, CSS and other text formats
- **CLI Interface**: Command-line tool for easy usage
- **Python API**: Programmatic access for integration into other applications

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Git

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

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional):
```bash
cp env.example .env
# Edit .env to customize model and configuration
```

## Usage

### Command Line Interface

```bash
# Index documents in current directory
ragrep index

# Index documents in specific directory
ragrep index ./documents

# Query the knowledge base
ragrep query "What is machine learning?"

# Dump knowledge base contents matching query (no LLM processing)
ragrep dump "machine learning" --limit 10

# Show system statistics
ragrep stats
```

### Knowledge Base Dump

The `dump` command provides semantic search without LLM processing, perfect for feeding context directly into other LLMs:

```bash
# Search for relevant chunks
ragrep dump "vector database" --limit 5

# Output is structured for LLM consumption:
# # Knowledge Base Dump for Query: 'vector database'
# # Found 5 relevant chunks
# ================================================================================
# 
# ## Chunk 1 (Similarity: 1.073)
# **Source:** `src/ragrep/retrieval/vector_store.py`
# **Content:**
# """Vector storage and retrieval functionality."""
# ...
```

### Python API

```python
from ragrep import RAGSystem

# Initialize the RAG system
rag = RAGSystem()

# Add documents
rag.add_documents("./documents")

# Query the system
response = rag.query("What is the main topic of the documents?")
print(response.text)
print(response.sources)
```

### Python API

```python
from ragrep import RAGSystem

# Initialize the RAG system
rag = RAGSystem()

# Add documents
rag.add_documents("./documents")

# Query the system
response = rag.query("What is the main topic of the documents?")
print(response['answer'])
print(f"Sources: {response['num_sources']} documents")
```

## Configuration

The tool can be configured through environment variables:

- `GENERATION_MODEL`: Hugging Face model name for text generation (default: microsoft/DialoGPT-medium)
- `CHUNK_SIZE`: Size of document chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `CUDA_AVAILABLE`: Set to "true" to use GPU acceleration (default: false)

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documents     │───▶│  Preprocessing  │───▶│  Vector Store   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐           │
│   User Query    │───▶│   Retrieval     │◀──────────┘
└─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Generation    │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Response      │
                       └─────────────────┘
```

## Development

### Project Structure

```
ragrep/
├── src/
│   ├── ragrep/
│   │   ├── core/          # Core RAG functionality
│   │   ├── retrieval/     # Document retrieval modules
│   │   ├── generation/    # Text generation modules
│   │   ├── api/          # API endpoints
│   │   └── web/          # Web interface
├── tests/                # Test suite
├── docs/                 # Documentation and research
├── examples/             # Usage examples
└── requirements.txt      # Python dependencies
```

### Running Tests

```bash
pytest tests/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for providing the GPT models
- Hugging Face for the transformer models
- The open-source community for various libraries and tools

## Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the documentation in the `docs/` directory
- Review the examples in the `examples/` directory

---

*RAGRep - Making RAG accessible and powerful for everyone.*