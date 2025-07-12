# KARA - Efficient RAG Knowledge Base Updates

[![CI](https://github.com/mzakizadeh/kara/workflows/CI/badge.svg)](https://github.com/mzakizadeh/kara/actions)
[![PyPI version](https://badge.fury.io/py/kara-toolkit.svg)](https://badge.fury.io/py/kara-toolkit)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
<!-- [![Downloads](https://static.pepy.tech/badge/kara-toolkit)](https://pepy.tech/project/kara-toolkit) -->

> **KARA** stands for **Knowledge-Aware Reembedding Algorithm**. The word "Kara" (کارآ) also means "efficient" in Persian.

KARA is a Python library that efficiently updates knowledge bases by reducing unnecessary embedding operations. When documents change, KARA automatically identifies and reuses existing chunks, minimizing the need for new embeddings.

## How It Works

KARA formulates chunking as a graph optimization problem:
1. Creates a DAG where nodes are split positions and edges are potential chunks
2. Uses Dijkstra's algorithm to find optimal chunking paths
3. Automatically reuses existing chunks to minimize embedding costs

<!-- Typical efficiency gains: 70-90% fewer embeddings for document updates. -->

## Installation

```bash
pip install kara-toolkit

# With LangChain integration
pip install kara-toolkit[langchain]
```

## Quick Start

```python
from kara import KARAUpdater, RecursiveCharacterChunker

# Initialize
chunker = RecursiveCharacterChunker(chunk_size=500)
updater = KARAUpdater(chunker=chunker, epsilon=0.1)

# Process initial documents
result = updater.create_knowledge_base(["Your document content..."])

# Update with new content - reuses existing chunks automatically
update_result = updater.update_knowledge_base(
    result.new_chunked_doc, 
    ["Updated document content..."]
)

print(f"Efficiency: {update_result.efficiency_ratio:.1%}")
print(f"Chunks reused: {update_result.num_reused}")
```

## LangChain Integration

```python
from kara.integrations.langchain import KARATextSplitter
from langchain_core.documents import Document

# Use as a drop-in replacement for LangChain text splitters
splitter = KARATextSplitter(chunk_size=300, epsilon=0.1)

docs = [Document(page_content="Your content...", metadata={"source": "file.pdf"})]
chunks = splitter.split_documents(docs)
```


## Examples

See [`examples/`](examples/) for complete usage examples.

## License

CC BY 4.0 License - see [LICENSE](LICENSE) file for details.
