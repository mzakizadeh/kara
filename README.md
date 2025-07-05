# kara-python - Knowledge-Aware Re-embedding Algorithm

[![PyPI version](https://badge.fury.io/py/kara-python.svg)](https://badge.fury.io/py/kara-python)
[![Python Support](https://img.shields.io/pypi/pyversions/kara-python.svg)](https://pypi.org/project/kara-python/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

kara-python is a Python library that implements the Knowledge-Aware Re-embedding Algorithm (KARA) for efficient updates to RAG (Retrieval-Augmented Generation) knowledge bases. It minimizes the number of embedding operations needed when updating document collections by intelligently reusing existing embeddings.

## 🚀 Features

- **Efficient Updates**: Reduce embedding operations by up to 70% when updating documents
- **Intelligent Chunking**: Optimal text splitting and chunk reuse strategies
- **LangChain Integration**: Drop-in replacement for LangChain text splitters
- **Configurable**: Adjustable parameters for different use cases
- **Type Safe**: Full type hints and comprehensive testing

## 📦 Installation

```bash
pip install kara-python
```

### Optional Dependencies

For LangChain integration:

```bash
pip install kara-python[langchain]
```

## 🔧 Quick Start

### Basic Usage

```python
from pykara import KARAUpdater
from pykara.splitters import RecursiveTextSplitter

# Initialize the updater
updater = KARAUpdater(
    splitter=RecursiveTextSplitter(chunk_size=1000, chunk_overlap=0),
    epsilon=0.1  # Cost factor for reusing existing chunks
)

# Initial document processing
initial_docs = ["Your initial document content..."]
chunks = updater.initialize(initial_docs)

# Update with new content
updated_docs = ["Your updated document content..."]
update_result = updater.update(updated_docs)

print(f"Added: {update_result.num_added}")
print(f"Reused: {update_result.num_skipped}")
print(f"Deleted: {update_result.num_deleted}")
print(f"Efficiency: {update_result.efficiency_ratio:.1%}")
```

### LangChain Integration

```python
from pykara.integrations.langchain import KARATextSplitter
from langchain.docstore.document import Document

# Drop-in replacement for LangChain text splitters
splitter = KARATextSplitter(
    chunk_size=1000,
    epsilon=0.1,  # KARA efficiency parameter
    separators=["\n\n", "\n", " ", ""]
)

# Use like any LangChain text splitter
documents = [Document(page_content="Your text here...")]
split_docs = splitter.split_documents(documents)

# Get efficiency metrics for updates
result = splitter.update_text("Updated text...")
print(f"Efficiency: {result.efficiency_ratio:.1%}")
```

## 🧠 Algorithm Overview

The KARA algorithm works by:

1. **Text Splitting**: Breaking documents into splits using configurable separators
2. **Optimal Chunking**: Using dynamic programming to find the optimal way to group splits
3. **Hash-based Matching**: Identifying reusable chunks from previous versions
4. **Cost Optimization**: Minimizing the total cost of embedding operations

## 📊 Performance Benefits

- **Up to 70% reduction** in embedding operations for typical document updates
- **Configurable trade-offs** between embedding cost and chunk reuse
- **Maintains quality** while significantly reducing computational overhead

## 🧪 Examples

Check out the [`examples/`](examples/) directory for more detailed usage examples:

- [`basic_usage.py`](examples/basic_usage.py) - Core KARA functionality
- [`langchain_example.py`](examples/langchain_example.py) - LangChain integration

## 🧪 Testing

Run the test suite:

```bash
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## 📞 Support

- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/kara-python/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/kara-python/discussions)
