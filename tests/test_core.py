"""
Tests for the core KARA algorithm.
"""

from kara.core import ChunkData, ChunkedDocument, KARAUpdater, UpdateResult
from kara.splitters import RecursiveCharacterChunker


class TestChunkData:
    """Tests for ChunkData class."""

    def test_from_splits(self) -> None:
        """Test creating ChunkData from splits."""
        splits = ["Hello ", "world", "!"]
        chunk_data = ChunkData.from_splits(splits)

        assert chunk_data.content == "Hello world!"
        assert chunk_data.splits == splits
        assert chunk_data.hash is not None
        assert isinstance(chunk_data.hash, str)

    def test_hash_consistency(self) -> None:
        """Test that hash is consistent for same content."""
        splits1 = ["Hello ", "world"]
        splits2 = ["Hello ", "world"]

        chunk1 = ChunkData.from_splits(splits1)
        chunk2 = ChunkData.from_splits(splits2)

        assert chunk1.hash == chunk2.hash


class TestKnowledgeBase:
    """Tests for KnowledgeBase class."""

    def test_get_chunk_hashes(self) -> None:
        """Test getting chunk hashes."""
        chunk1 = ChunkData.from_splits(["Hello"])
        chunk2 = ChunkData.from_splits(["World"])

        kb = ChunkedDocument(chunks=[chunk1, chunk2])
        hashes = kb.get_chunk_hashes()

        assert len(hashes) == 2
        assert chunk1.hash in hashes
        assert chunk2.hash in hashes

    def test_get_chunk_contents(self) -> None:
        """Test getting chunk contents."""
        chunk1 = ChunkData.from_splits(["Hello"])
        chunk2 = ChunkData.from_splits(["World"])

        kb = ChunkedDocument(chunks=[chunk1, chunk2])
        contents = kb.get_chunk_contents()

        assert contents == ["Hello", "World"]


class TestUpdateResult:
    """Tests for UpdateResult class."""

    def test_addition(self) -> None:
        """Test adding two UpdateResult objects."""
        result1 = UpdateResult(num_added=5, num_updated=2, num_skipped=10, num_deleted=3)
        result2 = UpdateResult(num_added=3, num_updated=1, num_skipped=7, num_deleted=2)

        combined = result1 + result2

        assert combined.num_added == 8
        assert combined.num_updated == 3
        assert combined.num_skipped == 17
        assert combined.num_deleted == 5

    def test_total_operations(self) -> None:
        """Test total operations calculation."""
        result = UpdateResult(num_added=5, num_updated=2, num_skipped=10, num_deleted=3)

        assert result.total_operations == 10  # added + updated + deleted

    def test_efficiency_ratio(self) -> None:
        """Test efficiency ratio calculation."""
        result = UpdateResult(num_added=5, num_updated=2, num_skipped=10, num_deleted=3)

        expected_ratio = 10 / 20  # skipped / total
        assert result.efficiency_ratio == expected_ratio

    def test_efficiency_ratio_empty(self) -> None:
        """Test efficiency ratio with empty result."""
        result = UpdateResult()

        assert result.efficiency_ratio == 0.0


class TestKARAUpdater:
    """Tests for KARAUpdater class."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        chunker = RecursiveCharacterChunker()
        updater = KARAUpdater(chunker=chunker)

        assert updater.chunker == chunker
        assert updater.epsilon == 0.01
        assert updater.max_chunk_size == 1000

    def test_create_knowledge_base(self, sample_text: str) -> None:
        """Test creating knowledge base with documents."""
        chunker = RecursiveCharacterChunker(separators=["\n"], keep_separator=True)
        updater = KARAUpdater(chunker=chunker)

        create_result = updater.create_knowledge_base([sample_text])
        assert isinstance(create_result, UpdateResult)
        kb = create_result.new_chunked_doc
        assert kb is not None  # Type guard

        assert isinstance(kb, ChunkedDocument)
        assert len(kb.chunks) > 0
        assert all(isinstance(chunk, ChunkData) for chunk in kb.chunks)

    def test_update_knowledge_base(self, sample_text: str) -> None:
        """Test updating knowledge base."""
        chunker = RecursiveCharacterChunker(separators=["\n"], keep_separator=True)
        updater = KARAUpdater(chunker=chunker)

        # Create initial knowledge base
        create_result = updater.create_knowledge_base([sample_text])
        kb = create_result.new_chunked_doc
        assert kb is not None  # Type guard

        # Update with slightly modified text
        updated_text = sample_text.replace("sample", "example")
        result = updater.update_knowledge_base(kb, [updated_text])

        assert isinstance(result, UpdateResult)
        assert result.num_added >= 0
        assert result.num_skipped >= 0
        assert result.num_deleted >= 0
        assert result.new_chunked_doc is not None
        assert isinstance(result.new_chunked_doc, ChunkedDocument)

    def test_epsilon_effect(self, sample_text: str) -> None:
        """Test the effect of epsilon parameter."""
        chunker = RecursiveCharacterChunker(separators=["\n"], keep_separator=True)

        # Test with low epsilon (prefer reusing)
        updater_low = KARAUpdater(chunker=chunker, epsilon=0.001)
        create_result_low = updater_low.create_knowledge_base([sample_text])
        kb_low = create_result_low.new_chunked_doc
        assert kb_low is not None  # Type guard
        result_low = updater_low.update_knowledge_base(kb_low, [sample_text])  # Same text

        # Test with high epsilon (prefer new chunks)
        updater_high = KARAUpdater(chunker=chunker, epsilon=0.999)
        create_result_high = updater_high.create_knowledge_base([sample_text])
        kb_high = create_result_high.new_chunked_doc
        assert kb_high is not None  # Type guard
        result_high = updater_high.update_knowledge_base(kb_high, [sample_text])  # Same text

        # Low epsilon should reuse more chunks
        assert result_low.num_skipped >= result_high.num_skipped

    def test_empty_documents(self) -> None:
        """Test with empty documents."""
        chunker = RecursiveCharacterChunker()
        updater = KARAUpdater(chunker=chunker)

        # Create with empty
        create_result = updater.create_knowledge_base([])
        assert isinstance(create_result, UpdateResult)
        assert create_result.new_chunked_doc is not None
        kb = create_result.new_chunked_doc
        assert len(kb.chunks) == 0

        # Update with empty
        result = updater.update_knowledge_base(kb, [])
        assert isinstance(result, UpdateResult)
        assert result.num_added == 0
        assert result.num_updated == 0
        assert result.num_skipped == 0
        assert result.num_deleted == 0

    def test_stateless_operation(self, sample_text: str) -> None:
        """Test that the updater is stateless."""
        chunker = RecursiveCharacterChunker(separators=["\n"], keep_separator=True)
        updater = KARAUpdater(chunker=chunker)

        # Create two separate knowledge bases
        create_result1 = updater.create_knowledge_base([sample_text])
        create_result2 = updater.create_knowledge_base([sample_text])
        kb1 = create_result1.new_chunked_doc
        kb2 = create_result2.new_chunked_doc
        assert kb1 is not None  # Type guard
        assert kb2 is not None  # Type guard

        # They should be independent
        assert kb1.chunks[0].hash == kb2.chunks[0].hash  # Same content = same hash
        assert kb1 is not kb2  # Different objects

        # Update one shouldn't affect the other
        updated_text = sample_text + " Additional content."
        result1 = updater.update_knowledge_base(kb1, [updated_text])
        result2 = updater.update_knowledge_base(kb2, [sample_text])  # Same as original

        # kb2 update should have high reuse since it's the same content
        assert result2.num_skipped > result1.num_skipped

    def test_wikipedia_scenario(
        self, wikipedia_style_text: str, updated_wikipedia_text: str
    ) -> None:
        """Test with Wikipedia-style text update scenario."""
        chunker = RecursiveCharacterChunker(separators=["\n"], keep_separator=True)
        updater = KARAUpdater(chunker=chunker, epsilon=0.1)

        # Create initial knowledge base
        create_result = updater.create_knowledge_base([wikipedia_style_text])
        kb = create_result.new_chunked_doc
        assert kb is not None  # Type guard

        # Update with modified text
        result = updater.update_knowledge_base(kb, [updated_wikipedia_text])

        # The Wikipedia update is extensive, so we expect mostly new content
        # But the algorithm should handle the update correctly
        assert result.num_added > 0  # Some content was added
        assert result.num_deleted > 0  # Some content was deleted

        # The update should be processed successfully
        assert isinstance(result.efficiency_ratio, float)
        assert result.efficiency_ratio >= 0.0  # Should be non-negative
        assert result.new_chunked_doc is not None
