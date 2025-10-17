# RAGRep - Retrieval-Augmented Generation Tool

A powerful tool for implementing Retrieval-Augmented Generation (RAG) systems that combines document retrieval with AI-powered text generation.

## Overview

RAGRep is designed to help users build sophisticated RAG applications that can search through documents and generate contextually relevant responses. The tool provides a comprehensive framework for implementing retrieval-augmented generation systems with modern AI capabilities.

## Features

- **Document Processing**: Automatic chunking and preprocessing of various document formats
- **Vector Search**: Semantic search using state-of-the-art embedding models
- **AI Generation**: Context-aware text generation using large language models
- **Flexible Architecture**: Modular design allowing customization of retrieval and generation components
- **Multiple Formats**: Support for PDF, TXT, MD, and other text formats
- **API Interface**: RESTful API for easy integration with other applications
- **Web Interface**: User-friendly web interface for interactive usage

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

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Usage

### Command Line Interface

```bash
# Process documents and create knowledge base
python -m ragrep index --input-dir ./documents --output-dir ./knowledge_base

# Query the knowledge base
python -m ragrep query "What is machine learning?" --knowledge-base ./knowledge_base

# Start the web interface
python -m ragrep serve --port 8000
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

### Web Interface

1. Start the server:
```bash
python -m ragrep serve
```

2. Open your browser to `http://localhost:8000`

3. Upload documents and start querying!

## Configuration

The tool can be configured through environment variables or a configuration file:

- `OPENAI_API_KEY`: Your OpenAI API key for text generation
- `EMBEDDING_MODEL`: Embedding model to use (default: text-embedding-ada-002)
- `CHUNK_SIZE`: Size of document chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

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