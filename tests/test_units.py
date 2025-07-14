"""
Unit tests for core KARA data structures and utilities.
These tests focus on testing individual classes and methods in isolation.
For integration testing and scenario-based testing, see test_data_driven.py.
"""

from kara.core import ChunkData, ChunkedDocument
from kara.splitters import RecursiveCharacterChunker, RecursiveTokenChunker


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

        doc = ChunkedDocument(chunks=[chunk1, chunk2])
        contents = doc.get_chunk_contents()

        assert contents == ["Hello", "World"]

    def test_empty_document(self) -> None:
        """Test creating an empty chunked document."""
        doc = ChunkedDocument(chunks=[])

        assert doc.get_chunk_hashes() == set()
        assert doc.get_chunk_contents() == []


class TestRecursiveCharacterChunker:
    """Tests for RecursiveCharacterChunker."""

    def test_basic_chunking(self, sample_text: str) -> None:
        """Test basic text chunking."""
        chunker = RecursiveCharacterChunker(separators=["\n\n", "\n", " "])
        result = chunker.create_chunks(sample_text)

        assert len(result) > 0
        assert all(isinstance(chunk, str) for chunk in result)

    def test_keep_separator(self, sample_text: str) -> None:
        """Test chunking with separator preservation."""
        chunker = RecursiveCharacterChunker(separators=["\n"], keep_separator=True)
        result = chunker.create_chunks(sample_text)

        # Count newlines in original vs result
        original_newlines = sample_text.count("\n")
        result_newlines = sum(chunk.count("\n") for chunk in result)

        # Should preserve most newlines (some might be at the end)
        assert result_newlines >= original_newlines - 1

    def test_no_separator(self, sample_text: str) -> None:
        """Test chunking without separator preservation."""
        chunker = RecursiveCharacterChunker(separators=["\n"], keep_separator=False)
        result = chunker.create_chunks(sample_text)

        # Should have removed newlines
        assert all("\n" not in chunk for chunk in result)

    def test_empty_text(self) -> None:
        """Test chunking empty text."""
        chunker = RecursiveCharacterChunker()
        result = chunker.create_chunks("")

        assert result == []

    def test_single_separator(self) -> None:
        """Test chunking with single separator."""
        chunker = RecursiveCharacterChunker(separators=["\n"])
        result = chunker.create_chunks("line1\nline2\nline3")

        assert len(result) >= 1
        assert all(isinstance(chunk, str) for chunk in result)


class TestRecursiveTokenChunker:
    """Tests for RecursiveTokenChunker."""

    def test_basic_token_chunking(self) -> None:
        """Test that token-based chunking respects the chunk_size."""
        text = "one two three four five six seven"
        chunker = RecursiveTokenChunker(tokenizer_function=str.split, chunk_size=3)

        chunks = chunker.create_chunks(text)

        # Expect ceil(7/3) = 3 chunks
        assert len(chunks) == 3
        # Each chunk should contain at most 3 tokens
        for chunk in chunks:
            assert len(chunk.split()) <= 3

    def test_exact_multiple_of_chunk_size(self) -> None:
        """Tokens equal to a multiple of chunk_size should divide evenly."""
        text = "tok1 tok2 tok3 tok4 tok5 tok6"
        chunker = RecursiveTokenChunker(tokenizer_function=str.split, chunk_size=3)

        chunks = chunker.create_chunks(text)

        assert len(chunks) == 2
        assert all(len(chunk.split()) == 3 for chunk in chunks)

    def test_empty_text(self) -> None:
        """Test that empty input returns no chunks."""
        chunker = RecursiveTokenChunker(tokenizer_function=str.split, chunk_size=2)

        assert chunker.create_chunks("") == []
