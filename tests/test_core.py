"""
Tests for the core KARA algorithm.
"""

from kara.core import KARAUpdater, UpdateResult
from kara.splitters import RecursiveTextSplitter


class TestUpdateResult:
    """Tests for UpdateResult class."""

    def test_addition(self):
        """Test adding two UpdateResult objects."""
        result1 = UpdateResult(num_added=5, num_updated=2, num_skipped=10, num_deleted=3)
        result2 = UpdateResult(num_added=3, num_updated=1, num_skipped=7, num_deleted=2)

        combined = result1 + result2

        assert combined.num_added == 8
        assert combined.num_updated == 3
        assert combined.num_skipped == 17
        assert combined.num_deleted == 5

    def test_total_operations(self):
        """Test total operations calculation."""
        result = UpdateResult(num_added=5, num_updated=2, num_skipped=10, num_deleted=3)

        assert result.total_operations == 10  # added + updated + deleted

    def test_efficiency_ratio(self):
        """Test efficiency ratio calculation."""
        result = UpdateResult(num_added=5, num_updated=2, num_skipped=10, num_deleted=3)

        expected_ratio = 10 / 20  # skipped / total
        assert result.efficiency_ratio == expected_ratio

    def test_efficiency_ratio_empty(self):
        """Test efficiency ratio with empty result."""
        result = UpdateResult()

        assert result.efficiency_ratio == 0.0


class TestKARAUpdater:
    """Tests for KARAUpdater class."""

    def test_initialization(self):
        """Test basic initialization."""
        splitter = RecursiveTextSplitter()
        updater = KARAUpdater(splitter=splitter)

        assert updater.splitter == splitter
        assert updater.epsilon == 0.01
        assert updater.max_chunk_size == 1000

    def test_initialize_documents(self, sample_text):
        """Test initializing with documents."""
        splitter = RecursiveTextSplitter(separators=["\n"], keep_separator=True)
        updater = KARAUpdater(splitter=splitter, max_chunk_size=100)

        chunks = updater.initialize([sample_text])

        assert len(chunks) > 0
        assert all(isinstance(chunk, list) for chunk in chunks)
        assert all(isinstance(split, str) for chunk in chunks for split in chunk)

    def test_update_documents(self, sample_text):
        """Test updating documents."""
        splitter = RecursiveTextSplitter(separators=["\n"], keep_separator=True)
        updater = KARAUpdater(splitter=splitter, max_chunk_size=100)

        # Initialize
        updater.initialize([sample_text])

        # Update with slightly modified text
        updated_text = sample_text.replace("sample", "example")
        result = updater.update([updated_text])

        assert isinstance(result, UpdateResult)
        assert result.num_added >= 0
        assert result.num_skipped >= 0
        assert result.num_deleted >= 0

    def test_epsilon_effect(self, sample_text):
        """Test the effect of epsilon parameter."""
        splitter = RecursiveTextSplitter(separators=["\n"], keep_separator=True)

        # Test with low epsilon (prefer reusing)
        updater_low = KARAUpdater(splitter=splitter, epsilon=0.001, max_chunk_size=100)
        updater_low.initialize([sample_text])
        result_low = updater_low.update([sample_text])  # Same text

        # Test with high epsilon (prefer new chunks)
        updater_high = KARAUpdater(splitter=splitter, epsilon=0.999, max_chunk_size=100)
        updater_high.initialize([sample_text])
        result_high = updater_high.update([sample_text])  # Same text

        # Low epsilon should reuse more chunks
        assert result_low.num_skipped >= result_high.num_skipped

    def test_empty_documents(self):
        """Test with empty documents."""
        splitter = RecursiveTextSplitter()
        updater = KARAUpdater(splitter=splitter)

        # Initialize with empty
        chunks = updater.initialize([])
        assert chunks == []

        # Update with empty
        result = updater.update([])
        assert isinstance(result, UpdateResult)
        assert result.num_added == 0
        assert result.num_updated == 0
        assert result.num_skipped == 0
        assert result.num_deleted == 0

    def test_get_current_chunks(self, sample_text):
        """Test getting current chunks."""
        splitter = RecursiveTextSplitter(separators=["\n"], keep_separator=True)
        updater = KARAUpdater(splitter=splitter, max_chunk_size=100)

        chunks = updater.initialize([sample_text])
        current_chunks = updater.get_current_chunks()

        assert chunks == current_chunks
        assert chunks is not current_chunks  # Should be a copy

    def test_wikipedia_scenario(self, wikipedia_style_text, updated_wikipedia_text):
        """Test with Wikipedia-style text update scenario."""
        splitter = RecursiveTextSplitter(separators=["\n"], keep_separator=True)
        updater = KARAUpdater(splitter=splitter, epsilon=0.1, max_chunk_size=200)

        # Initialize with original text
        updater.initialize([wikipedia_style_text])

        # Update with modified text
        result = updater.update([updated_wikipedia_text])

        # Should have some reused chunks and some new ones
        assert result.num_skipped > 0  # Some content remained the same
        assert result.num_added > 0  # Some content was added

        # Efficiency should be reasonable
        assert result.efficiency_ratio > 0.1  # At least 10% reuse
