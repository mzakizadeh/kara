"""
Document chunkers for breaking documents into optimal chunks.
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Sequence, Union


class BaseDocumentChunker(ABC):
    """Abstract base class for document chunkers."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 0):
        """
        Initialize the document chunker.

        Args:
            chunk_size: Maximum size of each chunk
            overlap: Overlap between chunks in units
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be zero or positive")
        self.chunk_size = chunk_size
        self.overlap = overlap

    @abstractmethod
    def create_chunks(self, text: str) -> List[List[Any]]:
        """Split text into optimally-sized chunks."""
        pass

    @abstractmethod
    def _split_to_units(self, text: str) -> List[Any]:
        """Split text into smallest units (e.g., by separators, tokens)."""
        pass

    def normalize_chunk(self, chunk: Any) -> List[Any]:
        """Normalize a chunk to a list of units."""
        if isinstance(chunk, str):
            return self._split_to_units(chunk)
        if isinstance(chunk, (list, tuple)):
            return list(chunk)
        raise TypeError("Chunk must be a string or a sequence of units.")

    def unit_length(self, unit: Any) -> int:
        """Return the unit length for sizing and chunk limits."""
        if isinstance(unit, str):
            return len(unit)
        return 1

    def serialize_units(self, units: Sequence[Any]) -> bytes:
        """Serialize units to bytes for hashing."""
        if all(isinstance(unit, str) for unit in units):
            return "".join(units).encode("utf-8")
        serialized = json.dumps(list(units), separators=(",", ":"), ensure_ascii=True)
        return serialized.encode("utf-8")

    def render_units(self, units: Sequence[Any]) -> Any:
        """Render units for output or storage."""
        if all(isinstance(unit, str) for unit in units):
            return "".join(units)
        return list(units)

    def _merge_units_greedy(self, units: List[Any], max_chunk_size: int) -> List[List[Any]]:
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

        def chunk_length(chunk_units: List[Any]) -> int:
            return sum(self.unit_length(unit) for unit in chunk_units)

        chunks: List[List[Any]] = []
        overlap_units = self.overlap
        start = 0
        units_count = len(units)

        while start < units_count:
            current_chunk: List[str] = []
            current_length = 0
            end = start

            while end < units_count:
                unit = units[end]
                unit_len = self.unit_length(unit)
                if current_length > 0 and current_length + unit_len > max_chunk_size:
                    break
                current_chunk.append(unit)
                current_length += unit_len
                end += 1

            if not current_chunk:
                break

            chunks.append(current_chunk)

            if end >= units_count:
                break

            if overlap_units > 0:
                start = max(start + 1, end - overlap_units)
            else:
                start = end

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
        overlap: int = 0,
        keep_separator: bool = True,
    ):
        """
        Initialize the recursive character chunker.

        Args:
            separators: List of separators to try, in order of preference
            chunk_size: Maximum chunk size in characters. Defaults to 4000
            overlap: Overlap between chunks in units
            keep_separator: Whether to keep separators in the result
        """
        super().__init__(chunk_size=chunk_size, overlap=overlap)
        self.separators = separators or ["\n\n", "\n", " "]
        self.keep_separator = keep_separator

    def create_chunks(self, text: str) -> List[List[Any]]:
        """
        Split text into optimally-sized chunks.

        Args:
            text: Input text to split

        Returns:
            List of chunks as unit lists
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
        tokenizer_function: Callable[[str], List[Any]],
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

    def create_chunks(self, text: str) -> List[List[Any]]:
        """
        Split text into token-based chunks.

        Args:
            text: Input text to split

        Returns:
            List of chunks as token lists
        """
        # First, split into tokens
        tokens = self._split_to_units(text)

        # Then, greedily merge into chunks
        return self._merge_tokens_greedy(tokens)

    def _split_to_units(self, text: str) -> List[Any]:
        """Split text into token units."""
        return self.tokenizer_function(text)

    def _merge_tokens_greedy(self, tokens: List[Any]) -> List[List[Any]]:
        """
        Merge tokens greedily to create chunks within token limit.

        Args:
            tokens: List of tokens to merge

        Returns:
            List of chunks
        """
        if not tokens:
            return []

        chunks: List[List[str]] = []
        overlap_units = self.overlap
        start = 0
        tokens_count = len(tokens)

        while start < tokens_count:
            end = min(start + self.chunk_size, tokens_count)
            chunk_tokens = tokens[start:end]
            if not chunk_tokens:
                break
            chunks.append(chunk_tokens)

            if end >= tokens_count:
                break

            if overlap_units > 0:
                start = max(start + 1, end - overlap_units)
            else:
                start = end

        return chunks
