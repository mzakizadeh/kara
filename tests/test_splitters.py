"""
Tests for document chunkers.
"""

from kara.splitters import (
    FixedSizeCharacterChunker,
    RecursiveCharacterChunker,
    SimpleCharacterChunker,
)


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


class TestSimpleCharacterChunker:
    """Tests for SimpleCharacterChunker."""

    def test_basic_chunking(self) -> None:
        """Test basic text chunking."""
        chunker = SimpleCharacterChunker(separator="\n")
        result = chunker.create_chunks("line1\nline2\nline3")

        assert len(result) >= 1
        assert all(isinstance(chunk, str) for chunk in result)

    def test_no_separator_preservation(self) -> None:
        """Test chunking without separator preservation."""
        chunker = SimpleCharacterChunker(separator="\n", keep_separator=False)
        result = chunker.create_chunks("line1\nline2\nline3")

        assert len(result) >= 1
        assert all("\n" not in chunk for chunk in result)

    def test_custom_separator(self) -> None:
        """Test with custom separator."""
        chunker = SimpleCharacterChunker(separator="|", keep_separator=True)
        result = chunker.create_chunks("part1|part2|part3")

        assert len(result) >= 1
        assert all(isinstance(chunk, str) for chunk in result)


class TestFixedSizeCharacterChunker:
    """Tests for FixedSizeCharacterChunker."""

    def test_basic_chunking(self) -> None:
        """Test basic character chunking."""
        chunker = FixedSizeCharacterChunker(chunk_size=10, overlap=0)
        result = chunker.create_chunks("hello world test string")

        assert len(result) > 1
        assert all(len(chunk) <= 10 for chunk in result)

    def test_with_overlap(self) -> None:
        """Test character chunking with overlap."""
        chunker = FixedSizeCharacterChunker(chunk_size=10, overlap=3)
        result = chunker.create_chunks("hello world test string")

        assert len(result) > 1
        assert all(len(chunk) <= 10 for chunk in result)

    def test_short_text(self) -> None:
        """Test with text shorter than chunk size."""
        chunker = FixedSizeCharacterChunker(chunk_size=100, overlap=0)
        result = chunker.create_chunks("short text")

        assert len(result) == 1
        assert result[0] == "short text"

    def test_exact_chunk_size(self) -> None:
        """Test with text exactly matching chunk size."""
        chunker = FixedSizeCharacterChunker(chunk_size=10, overlap=0)
        result = chunker.create_chunks("1234567890")

        assert len(result) == 1
        assert result[0] == "1234567890"
