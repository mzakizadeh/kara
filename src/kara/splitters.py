"""
Text splitters for breaking documents into sub-chunks.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Union


class BaseTextSplitter(ABC):
    """Abstract base class for text splitters."""

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into sub-chunks."""
        pass


class RecursiveTextSplitter(BaseTextSplitter):
    """
    Recursive text splitter that tries multiple separators.

    Based on the approach from your research notebook.
    """

    def __init__(
        self,
        separators: List[str] = None,
        chunk_size: int = 1000,
        overlap: int = 0,
        keep_separator: bool = True,
    ):
        """
        Initialize the recursive text splitter.

        Args:
            separators: List of separators to try, in order of preference
            chunk_size: Target chunk size (not enforced, used for guidance)
            overlap: Overlap between chunks (not implemented yet)
            keep_separator: Whether to keep separators in the result
        """
        self.separators = separators or ["\n\n", "\n", " ", ""]
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.keep_separator = keep_separator

    def split_text(self, text: str) -> List[str]:
        """
        Split text using the recursive approach.

        Args:
            text: Input text to split

        Returns:
            List of sub-chunks
        """
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
            List of split text chunks
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

        return [s for s in splits if s.strip()]


class SimpleTextSplitter(BaseTextSplitter):
    """Simple text splitter that uses a single separator."""

    def __init__(self, separator: str = "\n", keep_separator: bool = True):
        """
        Initialize the simple text splitter.

        Args:
            separator: Separator to use for splitting
            keep_separator: Whether to keep the separator in the result
        """
        self.separator = separator
        self.keep_separator = keep_separator

    def split_text(self, text: str) -> List[str]:
        """
        Split text using the configured separator.

        Args:
            text: Input text to split

        Returns:
            List of sub-chunks
        """
        if self.keep_separator:
            # Split and keep separator
            parts = text.split(self.separator)
            if len(parts) > 1:
                # Add separator back to all parts except the last one
                result = []
                for _i, part in enumerate(parts[:-1]):
                    result.append(part + self.separator)
                result.append(parts[-1])  # Last part without separator
                return [s for s in result if s.strip()]
            else:
                return parts
        else:
            return [s for s in text.split(self.separator) if s.strip()]


class CharacterTextSplitter(BaseTextSplitter):
    """Character-based text splitter."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 0):
        """
        Initialize the character text splitter.

        Args:
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text: str) -> List[str]:
        """
        Split text into character-based chunks.

        Args:
            text: Input text to split

        Returns:
            List of sub-chunks
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.overlap

        return chunks
