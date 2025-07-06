"""
kara-py - Knowledge-Aware Re-embedding Algorithm

A Python library for efficient updates to RAG knowledge bases,
minimizing embedding operations through intelligent chunk reuse.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core import KARAUpdater, UpdateResult
from .splitters import (
    BaseDocumentChunker,
    FixedSizeCharacterChunker,
    RecursiveCharacterChunker,
    RecursiveTokenChunker,
    SimpleCharacterChunker,
)

__all__ = [
    "KARAUpdater",
    "UpdateResult",
    "BaseDocumentChunker",
    "RecursiveCharacterChunker",
    "SimpleCharacterChunker",
    "FixedSizeCharacterChunker",
    "RecursiveTokenChunker",
]
