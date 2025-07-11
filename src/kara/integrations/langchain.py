"""
LangChain integration for kara-toolkit.
"""

from typing import Any, List, Optional

try:
    from langchain_text_splitters.base import TextSplitter
except ImportError as e:
    raise ImportError(
        "LangChain is not installed. This module requires LangChain to be installed. "
        "Please install it with: pip install kara-toolkit[langchain]"
    ) from e

from ..core import ChunkedDocument, KARAUpdater
from ..splitters import RecursiveCharacterChunker


class KARATextSplitter(TextSplitter):
    """
    KARA-powered text splitter that inherits from LangChain's TextSplitter.

    This splitter maintains compatibility with LangChain's ecosystem while
    providing efficient chunk updates through the KARA algorithm.
    """

    def __init__(
        self,
        chunk_size: int = 4000,
        separators: Optional[List[str]] = None,
        epsilon: float = 0.01,
        previous_chunks: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the KARA text splitter.

        Args:
            separators: List of separators to use for splitting
            epsilon: Cost factor for reusing existing chunks
            is_separator_regex: Whether separators are regex patterns
            **kwargs: Additional arguments passed to TextSplitter
        """
        super().__init__(**kwargs)

        # Initialize the underlying chunker
        self._kara_chunker = RecursiveCharacterChunker(
            separators=separators or ["\n\n", "\n", " "],
            keep_separator=kwargs.get("keep_separator", True),
            chunk_size=chunk_size,
        )

        # Initialize KARA updater
        self.kara_updater = KARAUpdater(
            chunker=self._kara_chunker,
            epsilon=epsilon,
        )

        # Store current knowledge base
        self._current_knowledge_base: Optional[ChunkedDocument] = None
        if previous_chunks is not None:
            self._current_knowledge_base = ChunkedDocument.from_chunks(
                previous_chunks, self._kara_chunker
            )

    def split_text(self, text: str) -> List[str]:
        """
        Split text using KARA algorithm.

        Args:
            text: Input text to split

        Returns:
            List of text chunks
        """
        if not self._current_knowledge_base:
            # First time - initialize
            result = self.kara_updater.create_knowledge_base([text])
            self._current_knowledge_base = result.new_chunked_doc
        else:
            # Update existing splits
            result = self.kara_updater.update_knowledge_base(self._current_knowledge_base, [text])
            self._current_knowledge_base = result.new_chunked_doc

        # Type guard to ensure we have a valid knowledge base
        if self._current_knowledge_base is None:
            return []

        return self._current_knowledge_base.get_chunk_contents()
