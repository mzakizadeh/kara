"""
LangChain integration for kara-toolkit.
"""

from typing import AbstractSet, Any, Collection, List, Literal, Optional, Union

try:
    from langchain_text_splitters.base import TextSplitter
except ImportError as e:
    raise ImportError(
        "LangChain is not installed. This module requires LangChain to be installed. "
        "Please install it with: pip install kara-toolkit[langchain]"
    ) from e

from ..chunkers import BaseDocumentChunker
from ..core import ChunkedDocument, KARAUpdater, UpdateResult


class KARATextSplitter(TextSplitter):
    """
    KARA-powered text splitter that inherits from LangChain's TextSplitter.

    This splitter maintains compatibility with LangChain's ecosystem while
    providing efficient chunk updates through the KARA algorithm.
    """

    def __init__(
        self,
        chunker: BaseDocumentChunker,
        previous_chunks: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the KARA text splitter.

        Args:
            chunker: Chunker instance to use for splitting
            previous_chunks: Optional list of previous chunks for incremental updates
            **kwargs: Additional arguments passed to TextSplitter
        """
        super().__init__(
            chunk_size=chunker.chunk_size,
            chunk_overlap=chunker.overlap,
            **kwargs,
        )

        self._kara_chunker = chunker
        self._last_result: Optional[UpdateResult] = None

        # Initialize KARA updater
        self.kara_updater = KARAUpdater(
            chunker=self._kara_chunker,
        )

        # Store current knowledge base
        self._current_knowledge_base: Optional[ChunkedDocument] = None
        if previous_chunks is not None:
            self._current_knowledge_base = ChunkedDocument.from_chunks(
                previous_chunks, self._kara_chunker
            )

    @classmethod
    def from_tiktoken_encoder(
        cls,
        encoding_name: str = "cl100k_base",
        model_name: Optional[str] = None,
        allowed_special: Union[Literal["all"], AbstractSet[str], None] = None,
        disallowed_special: Union[Literal["all"], Collection[str], None] = None,
        **kwargs: Any,
    ) -> "KARATextSplitter":
        """
        Create a KARATextSplitter using OpenAITokenChunker.

        Args:
            encoding_name: tiktoken encoding name
            model_name: Optional model name
            allowed_special: Allowed special tokens
            disallowed_special: Disallowed special tokens
            **kwargs: Additional arguments passed to KARATextSplitter

        Returns:
            Initialized KARATextSplitter
        """
        from ..chunkers import OpenAITokenChunker

        chunk_size = kwargs.pop("chunk_size", 1000)
        chunk_overlap = kwargs.pop("chunk_overlap", 0)
        previous_chunks = kwargs.pop("previous_chunks", None)

        chunker = OpenAITokenChunker(
            encoding_name=encoding_name,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )
        return cls(
            chunker=chunker,
            previous_chunks=previous_chunks,
            **kwargs,
        )

    @classmethod
    def from_huggingface_tokenizer(
        cls,
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> "KARATextSplitter":
        """
        Create a KARATextSplitter using HuggingFaceTokenChunker.

        Args:
            tokenizer: HuggingFace tokenizer instance (or model name string for backcompat)
            **kwargs: Additional arguments passed to KARATextSplitter

        Returns:
            Initialized KARATextSplitter
        """
        from ..chunkers import HuggingFaceTokenChunker

        chunk_size = kwargs.pop("chunk_size", 1000)
        chunk_overlap = kwargs.pop("chunk_overlap", 0)
        previous_chunks = kwargs.pop("previous_chunks", None)

        model_name = kwargs.pop("model_name", None)
        if model_name is None:
            if isinstance(tokenizer, str):
                model_name = tokenizer
            else:
                model_name = getattr(tokenizer, "name_or_path", "gpt2")

        chunker = HuggingFaceTokenChunker(
            model_name=model_name,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )
        return cls(
            chunker=chunker,
            previous_chunks=previous_chunks,
            **kwargs,
        )

    @property
    def last_result(self) -> Optional[UpdateResult]:
        """Get the result of the last split operation."""
        return self._last_result

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
            self._last_result = self.kara_updater.create_knowledge_base([text])
            self._current_knowledge_base = self._last_result.new_chunked_doc
        else:
            # Update existing splits
            self._last_result = self.kara_updater.update_knowledge_base(
                self._current_knowledge_base, [text]
            )
            self._current_knowledge_base = self._last_result.new_chunked_doc

        # Type guard to ensure we have a valid knowledge base
        if self._current_knowledge_base is None:
            return []

        return self._current_knowledge_base.get_chunk_contents()
