"""
Tests for text splitters.
"""

import pytest

from kara.splitters import (
    CharacterTextSplitter,
    RecursiveTextSplitter,
    SimpleTextSplitter,
)


class TestRecursiveTextSplitter:
    """Tests for RecursiveTextSplitter."""

    def test_basic_splitting(self, sample_text):
        """Test basic text splitting."""
        splitter = RecursiveTextSplitter(separators=["\n\n", "\n", " "])
        result = splitter.split_text(sample_text)

        assert len(result) > 0
        assert all(isinstance(chunk, str) for chunk in result)

    def test_keep_separator(self, sample_text):
        """Test splitting with separator preservation."""
        splitter = RecursiveTextSplitter(separators=["\n"], keep_separator=True)
        result = splitter.split_text(sample_text)

        # Count newlines in original vs result
        original_newlines = sample_text.count("\n")
        result_newlines = sum(chunk.count("\n") for chunk in result)

        # Should preserve most newlines (some might be at the end)
        assert result_newlines >= original_newlines - 1

    def test_no_separator(self, sample_text):
        """Test splitting without separator preservation."""
        splitter = RecursiveTextSplitter(separators=["\n"], keep_separator=False)
        result = splitter.split_text(sample_text)

        # Should have removed newlines
        assert all("\n" not in chunk for chunk in result)

    def test_empty_text(self):
        """Test splitting empty text."""
        splitter = RecursiveTextSplitter()
        result = splitter.split_text("")

        assert result == []

    def test_single_separator(self):
        """Test with single separator."""
        splitter = RecursiveTextSplitter(separators=[" "])
        result = splitter.split_text("hello world test")

        assert len(result) == 3
        assert "hello " in result
        assert "world " in result
        assert "test" in result


class TestSimpleTextSplitter:
    """Tests for SimpleTextSplitter."""

    def test_basic_splitting(self):
        """Test basic text splitting."""
        splitter = SimpleTextSplitter(separator="\n")
        result = splitter.split_text("line1\nline2\nline3")

        assert len(result) == 3
        assert "line1\n" in result
        assert "line2\n" in result
        assert "line3" in result

    def test_no_separator_preservation(self):
        """Test splitting without separator preservation."""
        splitter = SimpleTextSplitter(separator="\n", keep_separator=False)
        result = splitter.split_text("line1\nline2\nline3")

        assert len(result) == 3
        assert result == ["line1", "line2", "line3"]

    def test_custom_separator(self):
        """Test with custom separator."""
        splitter = SimpleTextSplitter(separator="|", keep_separator=True)
        result = splitter.split_text("part1|part2|part3")

        assert len(result) == 3
        assert "part1|" in result
        assert "part2|" in result
        assert "part3" in result


class TestCharacterTextSplitter:
    """Tests for CharacterTextSplitter."""

    def test_basic_splitting(self):
        """Test basic character splitting."""
        splitter = CharacterTextSplitter(chunk_size=10, overlap=0)
        result = splitter.split_text("hello world test string")

        assert len(result) > 1
        assert all(len(chunk) <= 10 for chunk in result)

    def test_with_overlap(self):
        """Test character splitting with overlap."""
        splitter = CharacterTextSplitter(chunk_size=10, overlap=3)
        result = splitter.split_text("hello world test string")

        assert len(result) > 1
        assert all(len(chunk) <= 10 for chunk in result)

    def test_short_text(self):
        """Test with text shorter than chunk size."""
        splitter = CharacterTextSplitter(chunk_size=100, overlap=0)
        result = splitter.split_text("short text")

        assert len(result) == 1
        assert result[0] == "short text"

    def test_exact_chunk_size(self):
        """Test with text exactly matching chunk size."""
        splitter = CharacterTextSplitter(chunk_size=10, overlap=0)
        result = splitter.split_text("1234567890")

        assert len(result) == 1
        assert result[0] == "1234567890"
