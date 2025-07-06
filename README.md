# KARA - Knowledge-Aware Re-embedding Algorithm

[![PyPI version](https://badge.fury.io/py/kara-py.svg)](https://badge.fury.io/py/kara-py)
[![Python Support](https://img.shields.io/pypi/pyversions/kara-py.svg)](https://pypi.org/project/kara-py/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

KARA is a Python library for efficient document updates in RAG systems. It minimizes embedding operations by intelligently reusing existing chunks when documents are updated.

## Installation

```bash
pip install kara-py
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

KARA splits documents into chunks and uses dynamic programming to find the optimal way to reuse existing chunks when content is updated, reducing embedding operations by up to 70%.

## Examples

See the [`examples/`](examples/) directory for more usage examples.

## License

CC BY 4.0 License - see [LICENSE](LICENSE) file for details.
