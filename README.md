# KARA - Efficient RAG Document Collection Updates

[![CI](https://github.com/mzakizadeh/kara/workflows/CI/badge.svg)](https://github.com/mzakizadeh/kara/actions)
[![PyPI version](https://badge.fury.io/py/kara-toolkit.svg)](https://badge.fury.io/py/kara-toolkit)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
<!-- [![Downloads](https://static.pepy.tech/badge/kara-toolkit)](https://pepy.tech/project/kara-toolkit) -->

> **KARA** stands for **Knowledge-Aware Reembedding Algorithm**. The word "Kara" (کارآ) also means "efficient" in Persian.

KARA is a Python library that efficiently updates document collections by reducing unnecessary embedding operations. When documents change, KARA automatically identifies and reuses existing chunks, minimizing the need for new embeddings.

## Installation

Install KARA with your preferred framework and provider:

```bash
# Core only
pip install kara-toolkit

# For OpenAI + LangChain
pip install "kara-toolkit[openai,langchain]"

# For Hugging Face + LangChain
pip install "kara-toolkit[huggingface,langchain]"

# Install everything
pip install "kara-toolkit[all]"
```

## Key Parameters

| Parameter    | Type  | Default | Description                                                                   |
|--------------|-------|---------|-------------------------------------------------------------------------------|
| `chunk_size` | `int` | `1000`   | Maximum size of each chunk (typically measured in tokens).                    |
| `overlap`    | `int` | `0`     | Number of overlapping units (tokens) between consecutive chunks.              |

## Quick Start

KARA is designed for token-level precision. Here is how to use it with OpenAI's `tiktoken`:

```python
from kara import OpenAITokenChunker, KARAUpdater

# Initialize with token-based chunking
chunker = OpenAITokenChunker(encoding_name="cl100k_base", chunk_size=512)
updater = KARAUpdater(chunker=chunker)

# Process initial documents
result = updater.create_collection(["Your long document text..."])

# Update with modified content - reuses existing token chunks automatically
update_result = updater.update_collection(
    result.new_chunked_doc,
    ["Your updated document text..."]
)

print(f"Efficiency: {update_result.efficiency_ratio:.1%}")
print(f"Tokens reused: {update_result.num_reused * 512} (approx)")
```

## LangChain Integration

KARA provides dedicated factory methods for seamless LangChain integration:

```python
from kara.integrations.langchain import KARATextSplitter
from langchain_core.documents import Document

# Create a token-aware splitter for OpenAI
splitter = KARATextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=512,
    chunk_overlap=50
)

# Use as a standard LangChain splitter
docs = [Document(page_content="Your content...", metadata={"source": "manual.pdf"})]
chunks = splitter.split_documents(docs)

# Later, update with a new version of the document to see efficiency gains
# KARATextSplitter automatically manages state for incremental updates
```

For Hugging Face models:
```python
splitter = KARATextSplitter.from_huggingface_tokenizer(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=256
)
```


## Examples

See [`examples/`](examples/) for complete usage examples.


## How It Works

KARA formulates chunking as a graph optimization problem:
1. Creates a Directed Acyclic Graph (DAG) where nodes are split positions and edges are potential chunks.
2. Assigns costs to edges: reused chunks have a lower cost based on their fill rate, while new chunks have an additional penalty.
3. Uses Dijkstra's algorithm to find the shortest path (lowest cost), which corresponds to the optimal chunking strategy that maximizes reuse.

Typical efficiency gains: 70-90% fewer embedding operations for document updates.


## Limitations

While KARA provides significant efficiency improvements for document collection updates, there are some current limitations to be aware of:

- **Document Version Dependency**: You need to keep the last version of documents to identify reusable chunks. However, you may be able to reconstruct document content using saved chunks in your vector store to reduce storage overhead. When compared to LangChain's indexing solution ([documented here](https://www.langchain.com/blog/syncing-data-sources-to-vector-stores)), which maintains a separate SQL database for chunk hashes while being extremely inefficient, our approach is still superior.

- **Chunking Configuration Changes**: Changing splitting configurations (chunk size, separators) between updates may disrupt the algorithm's ability to reuse chunks effectively.

## Roadmap to 1.0.0

- [x] **Token-Based Optimal Chunking** - Support for OpenAI and Hugging Face tokenizers
- [x] **Overlapping Chunk Support** - Support for overlapping units between chunks
- [ ] **100% Test Coverage** - Complete test suite with full coverage
- [ ] **Performance Benchmarks** - Real-world efficiency testing
- [ ] **Framework Support** - LlamaIndex, Haystack, and others
- [ ] **Complete Documentation** - API reference, guides, and examples

## License

CC BY 4.0 License - see [LICENSE](LICENSE) file for details.
