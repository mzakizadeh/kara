"""
Tests for document chunkers.
"""

from kara.splitters import RecursiveCharacterChunker


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
