# KARA - Knowledge-Aware Re-embedding Algorithm

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![PyPI version](https://badge.fury.io/py/kara-toolkit.svg)](https://badge.fury.io/py/kara-toolkit)
[![Python Support](https://img.shields.io/pypi/pyversions/kara-toolkit.svg)](https://pypi.org/project/kara-toolkit/)

KARA is a Python library for efficient document updates in RAG systems. It minimizes embedding operations by intelligently reusing existing chunks when documents are updated.

## Installation

```bash
pip install kara-toolkit
```

## Quick Start

```python
from kara import KARAUpdater, RecursiveCharacterChunker

# Initialize
updater = KARAUpdater(
    chunker=RecursiveCharacterChunker(chunk_size=1000),
    epsilon=0.1
)

# Process initial documents
updater.initialize(["Your document content..."])

# Update with new content
result = updater.update(["Updated document content..."])
print(f"Efficiency: {result.efficiency_ratio:.1%}")
```

## How It Works

KARA formulates the chunking problem as a DAG (Directed Acyclic Graph) for a single document where each node represents a position in the document splits, and edges represent possible chunks. It then uses Dijkstra's algorithm to find the optimal chunking path.

## Examples

See the [`examples/`](examples/) directory for more usage examples.

## License

CC BY 4.0 License - see [LICENSE](LICENSE) file for details.
