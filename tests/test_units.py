"""
Unit tests for core KARA data structures and utilities.
These tests focus on testing individual classes and methods in isolation.
For integration testing and scenario-based testing, see test_data_driven.py.
"""

from typing import List
from unittest.mock import MagicMock, patch

import pytest

from kara.chunkers import (
    CharacterChunker,
    HuggingFaceTokenChunker,
    OpenAITokenChunker,
    TokenChunker,
)
from kara.core import ChunkData, ChunkedDocument


class TestChunkData:
    """Tests for ChunkData class."""

    def test_from_splits(self) -> None:
        """Test creating ChunkData from splits."""
        splits = ["Hello", " ", "World"]
        chunk = ChunkData.from_splits(splits)

        assert chunk.content == "Hello World"
        assert chunk.splits == ["Hello", " ", "World"]
        assert chunk.hash is not None

    def test_hash_consistency(self) -> None:
        """Test that identical content produces identical hashes."""
        chunk1 = ChunkData.from_splits(["Hello", " ", "World"])
        chunk2 = ChunkData.from_splits(["Hello", " ", "World"])

        assert chunk1.content == chunk2.content
        assert chunk1.hash == chunk2.hash


class TestChunkedDocument:
    """Tests for ChunkedDocument class."""

    def test_get_chunk_hashes(self) -> None:
        """Test getting chunk hashes."""
        chunk1 = ChunkData.from_splits(["Hello"])
        chunk2 = ChunkData.from_splits(["World"])

        doc = ChunkedDocument(chunks=[chunk1, chunk2])
        hashes = doc.get_chunk_hashes()

        assert len(hashes) == 2
        assert chunk1.hash in hashes
        assert chunk2.hash in hashes

    def test_get_chunk_contents(self) -> None:
        """Test getting chunk contents."""
        chunk1 = ChunkData.from_splits(["Hello"])
        chunk2 = ChunkData.from_splits(["World"])

        doc: ChunkedDocument[str] = ChunkedDocument(chunks=[chunk1, chunk2])
        contents = doc.get_chunk_contents()

        assert contents == ["Hello", "World"]

    def test_empty_document(self) -> None:
        """Test creating an empty chunked document."""
        doc: ChunkedDocument[str] = ChunkedDocument(chunks=[])

        assert doc.get_chunk_hashes() == set()
        assert doc.get_chunk_contents() == []


class TestCharacterChunker:
    """Tests for CharacterChunker."""

    def test_basic_chunking(self, sample_text: str) -> None:
        """Test basic text chunking."""
        chunker = CharacterChunker(separators=["\n\n", "\n", " "])
        result = chunker.create_chunks(sample_text)

        assert len(result) > 0
        assert all(isinstance(chunk, list) for chunk in result)

    def test_keep_separator(self, sample_text: str) -> None:
        """Test chunking with separator preservation."""
        chunker = CharacterChunker(separators=["\n"], keep_separator=True)
        result = chunker.create_chunks(sample_text)

        # Count newlines in original vs result
        original_newlines = sample_text.count("\n")
        result_newlines = sum("".join(chunk).count("\n") for chunk in result)

        # Should preserve most newlines (some might be at the end)
        assert result_newlines >= original_newlines - 1

    def test_no_separator(self, sample_text: str) -> None:
        """Test chunking without separator preservation."""
        chunker = CharacterChunker(separators=["\n"], keep_separator=False)
        result = chunker.create_chunks(sample_text)

        # Should have removed newlines
        assert all("\n" not in "".join(chunk) for chunk in result)

    def test_empty_text(self) -> None:
        """Test chunking empty text."""
        chunker = CharacterChunker()
        result = chunker.create_chunks("")

        assert result == []

    def test_single_separator(self) -> None:
        """Test chunking with single separator."""
        chunker = CharacterChunker(separators=["\n"])
        result = chunker.create_chunks("line1\nline2\nline3")

        assert len(result) >= 1
        assert all(isinstance(chunk, list) for chunk in result)


class TestTokenChunker:
    """Tests for TokenChunker."""

    def test_basic_token_chunking(self) -> None:
        """Test that token-based chunking respects the chunk_size."""

        def mock_tokenizer(text: str) -> List[int]:
            return [ord(c) for c in text.replace(" ", "")]

        # text has 7 words, but mock_tokenizer returns length of text without spaces
        text = "a b c d e f g"
        chunker = TokenChunker(tokenizer_function=mock_tokenizer, chunk_size=3)

        chunks = chunker.create_chunks(text)

        assert len(chunks) == 3  # 7 / 3 = 2.33 -> 3 chunks
        for chunk in chunks:
            assert len(chunk) <= 3

    def test_exact_multiple_of_chunk_size(self) -> None:
        """Tokens equal to a multiple of chunk_size should divide evenly."""

        def mock_tokenizer(text: str) -> List[int]:
            return [1] * len(text.split())

        text = "tok1 tok2 tok3 tok4 tok5 tok6"
        chunker = TokenChunker(tokenizer_function=mock_tokenizer, chunk_size=3)

        chunks = chunker.create_chunks(text)

        assert len(chunks) == 2
        assert all(len(chunk) == 3 for chunk in chunks)

    def test_empty_text(self) -> None:
        """Test that empty input returns no chunks."""

        def mock_tokenizer(text: str) -> List[int]:
            return []

        chunker = TokenChunker(tokenizer_function=mock_tokenizer, chunk_size=2)
        assert chunker.create_chunks("") == []


class TestOpenAITokenChunker:
    """Tests for OpenAITokenChunker."""

    @patch("tiktoken.get_encoding")
    def test_initialization(self, mock_get_encoding: MagicMock) -> None:
        """Test initialization of OpenAITokenChunker."""
        mock_encoding = MagicMock()
        mock_get_encoding.return_value = mock_encoding

        chunker = OpenAITokenChunker(encoding_name="cl100k_base", chunk_size=10, overlap=2)

        assert chunker.encoding_name == "cl100k_base"
        assert chunker.chunk_size == 10
        assert chunker.overlap == 2
        mock_get_encoding.assert_called_once_with("cl100k_base")

    @patch("tiktoken.get_encoding")
    def test_split_to_units(self, mock_get_encoding: MagicMock) -> None:
        """Test splitting text into token units."""
        mock_encoding = MagicMock()
        mock_get_encoding.return_value = mock_encoding
        # Mock encoding.encode to return some token IDs
        mock_encoding.encode.return_value = [1, 2, 3]

        chunker = OpenAITokenChunker()
        units = chunker._split_to_units("one two three")

        assert units == [1, 2, 3]
        mock_encoding.encode.assert_called_once_with("one two three")

    @patch("tiktoken.get_encoding")
    def test_create_chunks(self, mock_get_encoding: MagicMock) -> None:
        """Test creating chunks with overlap."""
        mock_encoding = MagicMock()
        mock_get_encoding.return_value = mock_encoding
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]

        # chunk_size=3, overlap=1
        # Chunk 1: [1, 2, 3]
        # Chunk 2: [3, 4, 5]
        chunker = OpenAITokenChunker(chunk_size=3, overlap=1)
        chunks = chunker.create_chunks("1 2 3 4 5")

        assert len(chunks) == 2
        assert chunks[0] == [1, 2, 3]
        assert chunks[1] == [3, 4, 5]

    @patch("tiktoken.get_encoding")
    def test_render_units(self, mock_get_encoding: MagicMock) -> None:
        """Test rendering token units."""
        mock_encoding = MagicMock()
        mock_get_encoding.return_value = mock_encoding
        mock_encoding.decode.return_value = "one two"

        chunker = OpenAITokenChunker()
        content = chunker.render_units([1, 2])

        assert content == "one two"
        mock_encoding.decode.assert_called_once_with([1, 2])

    @patch("tiktoken.get_encoding")
    def test_unit_length(self, mock_get_encoding: MagicMock) -> None:
        """Test that each token unit has length 1."""
        chunker = OpenAITokenChunker()
        assert chunker.unit_length(123) == 1

    def test_tiktoken_not_installed(self) -> None:
        """Test that ImportError is raised when tiktoken is not installed."""
        with patch.dict("sys.modules", {"tiktoken": None}):
            with pytest.raises(ImportError, match="tiktoken is required"):
                OpenAITokenChunker()


class TestHuggingFaceTokenChunker:
    """Tests for HuggingFaceTokenChunker."""

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_initialization(self, mock_from_pretrained: MagicMock) -> None:
        """Test initialization of HuggingFaceTokenChunker."""
        mock_tokenizer = MagicMock()
        mock_from_pretrained.return_value = mock_tokenizer

        chunker = HuggingFaceTokenChunker(model_name="gpt2", chunk_size=100, overlap=10)

        assert chunker.model_name == "gpt2"
        assert chunker.chunk_size == 100
        assert chunker.overlap == 10
        mock_from_pretrained.assert_called_once_with("gpt2")

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_split_to_units(self, mock_from_pretrained: MagicMock) -> None:
        """Test splitting text into token units."""
        mock_tokenizer = MagicMock()
        mock_from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.encode.return_value = [10, 20, 30]

        chunker = HuggingFaceTokenChunker(model_name="test")
        units = chunker._split_to_units("abc")

        assert units == [10, 20, 30]
        mock_tokenizer.encode.assert_called_once_with("abc", add_special_tokens=False)

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_create_chunks(self, mock_from_pretrained: MagicMock) -> None:
        """Test creating chunks with overlap."""
        mock_tokenizer = MagicMock()
        mock_from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.encode.return_value = [1, 2, 3, 4]

        # chunk_size=2, overlap=1
        # Chunk 1: [1, 2]
        # Chunk 2: [2, 3]
        # Chunk 3: [3, 4]
        chunker = HuggingFaceTokenChunker(model_name="test", chunk_size=2, overlap=1)
        chunks = chunker.create_chunks("1 2 3 4")

        assert len(chunks) == 3
        assert chunks[0] == [1, 2]
        assert chunks[1] == [2, 3]
        assert chunks[2] == [3, 4]

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_render_units(self, mock_from_pretrained: MagicMock) -> None:
        """Test rendering token units."""
        mock_tokenizer = MagicMock()
        mock_from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.decode.return_value = "1 2"

        chunker = HuggingFaceTokenChunker(model_name="test")
        content = chunker.render_units([1, 2])

        assert content == "1 2"
        mock_tokenizer.decode.assert_called_once_with([1, 2], clean_up_tokenization_spaces=False)

    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_unit_length(self, mock_from_pretrained: MagicMock) -> None:
        """Test that each token unit has length 1."""
        chunker = HuggingFaceTokenChunker(model_name="test")
        assert chunker.unit_length(123) == 1

    def test_transformers_not_installed(self) -> None:
        """Test that ImportError is raised when transformers is not installed."""
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError, match="transformers is required"):
                HuggingFaceTokenChunker(model_name="test")
