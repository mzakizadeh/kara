"""
Document chunkers for breaking documents into optimal chunks.
"""

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from collections.abc import Set as AbstractSet
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
)

T = TypeVar("T")


class BaseDocumentChunker(ABC, Generic[T]):
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
    def create_chunks(self, text: str) -> list[list[T]]:
        """Split text into optimally-sized chunks."""
        pass

    @abstractmethod
    def _split_to_units(self, text: str) -> list[T]:
        """Split text into smallest units (e.g., by separators, tokens)."""
        pass

    def normalize_chunk(self, chunk: Any) -> list[T]:
        """Normalize a chunk to a list of units."""
        if isinstance(chunk, str):
            return self._split_to_units(chunk)
        if isinstance(chunk, (list, tuple)):
            return list(chunk)
        raise TypeError("Chunk must be a string or a sequence of units.")

    def unit_length(self, unit: T) -> int:
        """Return the unit length for sizing and chunk limits."""
        if isinstance(unit, str):
            return len(unit)
        return 1

    def serialize_units(self, units: Sequence[T]) -> bytes:
        """Serialize units to bytes for hashing."""
        if all(isinstance(unit, str) for unit in units):
            return "".join(units).encode("utf-8")  # type: ignore
        serialized = json.dumps(list(units), separators=(",", ":"), ensure_ascii=True)
        return serialized.encode("utf-8")

    def render_units(self, units: Sequence[T]) -> Any:
        """Render units for output or storage."""
        if all(isinstance(unit, str) for unit in units):
            return "".join(units)  # type: ignore
        return list(units)

    def _merge_units_greedy(self, units: list[T], max_chunk_size: int) -> list[list[T]]:
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

        def chunk_length(chunk_units: list[Any]) -> int:
            return sum(self.unit_length(unit) for unit in chunk_units)

        chunks: list[list[Any]] = []
        overlap_units = self.overlap
        start = 0
        units_count = len(units)

        while start < units_count:
            current_chunk: list[Any] = []
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


class CharacterChunker(BaseDocumentChunker[str]):
    """
    Recursive character-based chunker that tries multiple separators.

    First splits text into smallest units using separators, then greedily
    merges them into chunks within the size limit.
    """

    def __init__(
        self,
        separators: Optional[list[str]] = None,
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

    def create_chunks(self, text: str) -> list[list[str]]:
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

    def _split_to_units(self, text: str) -> list[str]:
        """Split text into smallest units using separators."""
        return self._split_text_with_regex(text, self.separators, self.keep_separator)

    def _split_text_with_regex(
        self,
        text: str,
        separators: Union[str, list[str]],
        keep_separator: bool = False,
    ) -> list[str]:
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


class TokenChunker(BaseDocumentChunker[int]):
    """
    Token-based chunker that splits text into tokens and merges them greedily.

    This demonstrates how the unified chunking approach works for different
    unit types (tokens instead of characters).
    """

    def __init__(
        self,
        tokenizer_function: Optional[Callable[[str], list[int]]] = None,
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

    def create_chunks(self, text: str) -> list[list[int]]:
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
        return self._merge_units_greedy(tokens, self.chunk_size)

    def _split_to_units(self, text: str) -> list[int]:
        """Split text into token units."""
        if self.tokenizer_function is None:
            raise NotImplementedError(
                "_split_to_units must be implemented by subclasses or a tokenizer_function provided"
            )
        return self.tokenizer_function(text)

    def unit_length(self, unit: int) -> int:
        """
        Return the unit length for sizing and chunk limits.

        For token-based chunking, each token counts as 1 unit,
        regardless of its representation (e.g., string length).
        """
        return 1


class OpenAITokenChunker(TokenChunker):
    """Token chunker using OpenAI's tiktoken encodings."""

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        chunk_size: int = 1000,
        overlap: int = 0,
        allowed_special: Optional[Union[Literal["all"], AbstractSet[str]]] = None,
        disallowed_special: Optional[Union[Literal["all"], Collection[str]]] = None,
    ):
        """
        Initialize the OpenAI token chunker.

        Args:
            encoding_name: tiktoken encoding name to use
            chunk_size: Maximum size of each chunk in tokens
            overlap: Overlap between chunks in tokens
            allowed_special: Allowed special tokens
            disallowed_special: Disallowed special tokens
        """
        super().__init__(chunk_size=chunk_size, overlap=overlap)
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "tiktoken is required for OpenAITokenChunker. "
                "Install with: pip install kara-toolkit[openai]"
            ) from exc

        self.encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)
        self.allowed_special = allowed_special
        self.disallowed_special = disallowed_special

    def _split_to_units(self, text: str) -> list[int]:
        """Split text into token IDs using tiktoken."""
        # Only pass special token arguments if they are explicitly set to non-None values
        kwargs: dict[str, Any] = {}
        if self.allowed_special is not None:
            kwargs["allowed_special"] = self.allowed_special
        if self.disallowed_special is not None:
            kwargs["disallowed_special"] = self.disallowed_special

        return self._encoding.encode(text, **kwargs)

    def render_units(self, units: Sequence[Any]) -> str:
        """Render token units by decoding them as a sequence."""
        if all(isinstance(unit, int) for unit in units):
            return str(self._encoding.decode(list(units)))
        return str(super().render_units(units))


class HuggingFaceTokenChunker(TokenChunker):
    """Token chunker using Hugging Face tokenizers."""

    def __init__(
        self,
        model_name: str,
        chunk_size: int = 1000,
        overlap: int = 0,
    ):
        """
        Initialize the Hugging Face token chunker.

        Args:
            model_name: Hugging Face model name to load
            chunk_size: Maximum size of each chunk in tokens
            overlap: Overlap between chunks in tokens
        """
        super().__init__(chunk_size=chunk_size, overlap=overlap)
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for HuggingFaceTokenChunker. "
                "Install with: pip install kara-toolkit[huggingface]"
            ) from exc

        self.model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _split_to_units(self, text: str) -> list[int]:
        """Split text into token IDs using a Hugging Face tokenizer."""
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def render_units(self, units: Sequence[Any]) -> str:
        """Render token units by decoding them as a sequence."""
        if all(isinstance(unit, int) for unit in units):
            return str(self._tokenizer.decode(list(units), clean_up_tokenization_spaces=False))
        return str(super().render_units(units))
