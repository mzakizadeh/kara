"""
kara-toolkit - Knowledge-Aware Re-embedding Algorithm

A Python library for efficient updates to RAG knowledge bases,
minimizing embedding operations through intelligent chunk reuse.
"""

try:
    from ._version import __version__
except ImportError:
    # Fallback for development installs without setuptools-scm
    __version__ = "0.0.0+unknown"

__author__ = "Mahdi Zakizadeh"
__email__ = "mzakizadeh.me@gmail.com"

from .core import KARAUpdater, UpdateResult
from .splitters import (
    BaseDocumentChunker,
    RecursiveCharacterChunker,
    RecursiveTokenChunker,
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
