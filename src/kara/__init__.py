"""
kara-python - Knowledge-Aware Re-embedding Algorithm

A Python library for efficient updates to RAG knowledge bases,
minimizing embedding operations through intelligent chunk reuse.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core import KARAUpdater, UpdateResult
from .splitters import BaseTextSplitter, RecursiveTextSplitter
from .chunkers import BaseChunker, GreedyChunker, OptimalChunker

__all__ = [
    "KARAUpdater",
    "UpdateResult",
    "BaseTextSplitter",
    "RecursiveTextSplitter",
    "BaseChunker",
    "GreedyChunker",
    "OptimalChunker",
]

# Version information
__version__ = "0.1.0"
