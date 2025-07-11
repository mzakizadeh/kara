"""
Document chunkers for breaking documents into optimal chunks.
"""

import re
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union


class BaseDocumentChunker(ABC):
    """Abstract base class for document chunkers."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 0):
        """
        Initialize the document chunker.

        Args:
            chunk_size: Maximum size of each chunk
            overlap: Overlap between chunks (for future implementation)
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be zero or positive")
        self.chunk_size = chunk_size
        self.overlap = overlap

    @abstractmethod
    def create_chunks(self, text: str) -> List[str]:
        """Split text into optimally-sized chunks."""
        pass

    @abstractmethod
    def _split_to_units(self, text: str) -> List[str]:
        """Split text into smallest units (e.g., by separators, tokens)."""
        pass

    def _merge_units_greedy(self, units: List[str], max_chunk_size: int) -> List[str]:
        """
        Merge units greedily to create chunks within size limit.

        Args:
            units: List of text units to merge
            max_chunk_size: Maximum size of each chunk

        Returns:
            List of chunks
        """
        if not units:
            return []

        def unit_length(unit: str) -> int:
            return len(unit)

        def chunk_length(chunk_units: List[str]) -> int:
            return sum(len(unit) for unit in chunk_units)

        chunks = []
        current_chunk: List[str] = []

        for unit in units:
            # If adding this unit would exceed the max size, start a new chunk
            if current_chunk and chunk_length(current_chunk) + unit_length(unit) > max_chunk_size:
                # Only add non-empty chunks
                if current_chunk:
                    chunks.append("".join(current_chunk))
                current_chunk = [unit]
            else:
                current_chunk.append(unit)

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks


class RecursiveCharacterChunker(BaseDocumentChunker):
    """
    Recursive character-based chunker that tries multiple separators.

    First splits text into smallest units using separators, then greedily
    merges them into chunks within the size limit.
    """

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        chunk_size: int = 4000,
        # TODO: The algorithm currently does not support overlap
        # This is a placeholder for future implementation
        # overlap: int = 0,
        keep_separator: bool = True,
    ):
        """
        Initialize the recursive character chunker.

        Args:
            separators: List of separators to try, in order of preference
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks (not implemented yet)
            keep_separator: Whether to keep separators in the result
        """
        overlap = 0  # Placeholder for future implementation
        super().__init__(chunk_size=chunk_size, overlap=overlap)
        self.separators = separators or ["\n\n", "\n", " "]
        self.keep_separator = keep_separator

    def create_chunks(self, text: str) -> List[str]:
        """
        Split text into optimally-sized chunks.

        Args:
            text: Input text to split

        Returns:
            List of chunks
        """
        # First, split into smallest units
        units = self._split_to_units(text)

        # Then, greedily merge into chunks
        return self._merge_units_greedy(units, self.chunk_size)

    def _split_to_units(self, text: str) -> List[str]:
        """Split text into smallest units using separators."""
        return self._split_text_with_regex(text, self.separators, self.keep_separator)

    def _split_text_with_regex(
        self,
        text: str,
        separators: Union[str, List[str]],
        keep_separator: bool = False,
    ) -> List[str]:
        """
        Split text using regex with optional separator preservation.

        Args:
            text: Input text
            separators: Separator(s) to use for splitting
            keep_separator: Whether to keep separators in the result

        Returns:
            List of split text units
        """
        if isinstance(separators, list):
            separator_pattern = "|".join(re.escape(sep) for sep in separators)
        elif isinstance(separators, str):
            separator_pattern = re.escape(separators)
        else:
            raise ValueError("The separator must be a string or a list of strings.")

        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result
            _splits = re.split(f"({separator_pattern})", text)
            splits = []

            # Recombine text with separators
            for i in range(0, len(_splits), 2):
                if i + 1 < len(_splits):
                    splits.append(_splits[i] + _splits[i + 1])
                else:
                    splits.append(_splits[i])
        else:
            splits = re.split(separator_pattern, text)

        return [s for s in splits if s]


class RecursiveTokenChunker(BaseDocumentChunker):
    """
    Token-based chunker that splits text into tokens and merges them greedily.

    This demonstrates how the unified chunking approach works for different
    unit types (tokens instead of characters).
    """

    def __init__(
        self,
        tokenizer_function: Callable[[str], List[str]],
        chunk_size: int = 512,
        overlap: int = 0,
    ):
        """
        Initialize the token-based chunker.

        Args:
            chunk_size: Maximum chunk size in tokens
            overlap: Overlap between chunks in tokens
            tokenizer_function: Function to tokenize text
        """
        super().__init__(chunk_size=chunk_size, overlap=overlap)
        self.tokenizer_function = tokenizer_function

    def _default_tokenizer(self, text: str) -> List[str]:
        """Default tokenizer that splits on whitespace."""
        return text.split()

    def create_chunks(self, text: str) -> List[str]:
        """
        Split text into token-based chunks.

        Args:
            text: Input text to split

        Returns:
            List of chunks
        """
        # First, split into tokens
        tokens = self._split_to_units(text)

        # Then, greedily merge into chunks
        return self._merge_tokens_greedy(tokens)

    def _split_to_units(self, text: str) -> List[str]:
        """Split text into token units."""
        return self.tokenizer_function(text)

    def _merge_tokens_greedy(self, tokens: List[str]) -> List[str]:
        """
        Merge tokens greedily to create chunks within token limit.

        Args:
            tokens: List of tokens to merge

        Returns:
            List of chunks
        """
        if not tokens:
            return []

        chunks = []
        current_chunk_tokens: List[str] = []

        for token in tokens:
            # If adding this token would exceed the max size, start a new chunk
            if len(current_chunk_tokens) >= self.chunk_size:
                if current_chunk_tokens:
                    chunks.append(" ".join(current_chunk_tokens))
                current_chunk_tokens = [token]
            else:
                current_chunk_tokens.append(token)

        # Add the last chunk if it's not empty
        if current_chunk_tokens:
            chunks.append(" ".join(current_chunk_tokens))

        return chunks
