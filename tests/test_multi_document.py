"""
Tests for multi-document support in KARA.
"""

import pytest

from src.kara.core import ChunkedDocument, KARAUpdater
from src.kara.splitters import RecursiveCharacterChunker


class TestMultiDocumentSupport:
    """Test multi-document functionality in KARA."""

    @pytest.fixture
    def chunker(self):
        """Create a test chunker."""
        return RecursiveCharacterChunker(chunk_size=100, separators=["\n\n", "\n", " ", ""])

    @pytest.fixture
    def updater(self, chunker):
        """Create a KARA updater."""
        return KARAUpdater(chunker=chunker, epsilon=0.01)

    def test_create_empty_knowledge_base(self, updater):
        """Test creating knowledge base with empty document list."""
        result = updater.create_knowledge_base([])

        assert result.num_added == 0
        assert result.num_updated == 0
        assert result.num_skipped == 0
        assert result.num_deleted == 0
        assert len(result.new_chunked_doc.chunks) == 0
        assert len(result.new_chunked_doc.get_document_ids()) == 0

    def test_create_single_document(self, updater):
        """Test creating knowledge base with single document."""
        documents = ["This is a test document with some content."]
        result = updater.create_knowledge_base(documents)

        assert result.num_added > 0
        assert len(result.new_chunked_doc.chunks) > 0
        assert result.new_chunked_doc.get_document_ids() == {0}

        # All chunks should belong to document 0
        for chunk in result.new_chunked_doc.chunks:
            assert chunk.document_id == 0

    def test_create_multiple_documents(self, updater):
        """Test creating knowledge base with multiple documents."""
        documents = [
            "First document with some content about AI.",
            "Second document discussing machine learning algorithms.",
            "Third document about natural language processing.",
        ]
        result = updater.create_knowledge_base(documents)

        assert result.num_added > 0
        assert len(result.new_chunked_doc.chunks) > 0
        assert result.new_chunked_doc.get_document_ids() == {0, 1, 2}

        # Each document should have at least one chunk
        for doc_id in range(3):
            chunks = result.new_chunked_doc.get_chunks_by_document(doc_id)
            assert len(chunks) > 0

            # All chunks should have correct document_id
            for chunk in chunks:
                assert chunk.document_id == doc_id

    def test_update_empty_to_documents(self, updater):
        """Test updating from empty knowledge base to multiple documents."""
        # Start with empty KB
        empty_kb = ChunkedDocument(chunks=[])

        documents = ["New document one.", "New document two."]

        result = updater.update_knowledge_base(empty_kb, documents)

        assert result.num_added > 0
        assert result.num_skipped == 0
        assert result.num_deleted == 0
        assert len(result.new_chunked_doc.get_document_ids()) == 2

    def test_update_with_unchanged_document(self, updater):
        """Test updating where one document remains unchanged."""
        # Create initial KB
        initial_docs = ["Unchanged document content.", "Original second document."]
        initial_result = updater.create_knowledge_base(initial_docs)

        # Update with first doc unchanged, second doc modified
        updated_docs = [
            "Unchanged document content.",  # Same as before
            "Modified second document with new content.",  # Different
        ]

        update_result = updater.update_knowledge_base(initial_result.new_chunked_doc, updated_docs)

        # Should have some reused chunks from first document
        assert update_result.num_skipped > 0
        # Should have new chunks from modified second document
        assert update_result.num_added > 0
        # Some old chunks should be deleted
        assert update_result.num_deleted > 0

    def test_update_with_document_addition(self, updater):
        """Test adding a new document to existing knowledge base."""
        # Create initial KB with 2 documents
        initial_docs = ["First document.", "Second document."]
        initial_result = updater.create_knowledge_base(initial_docs)

        # Update with same 2 documents plus a new one
        updated_docs = [
            "First document.",  # Unchanged
            "Second document.",  # Unchanged
            "Brand new third document.",  # Added
        ]

        update_result = updater.update_knowledge_base(initial_result.new_chunked_doc, updated_docs)

        # Should reuse existing chunks and add new ones
        assert update_result.num_skipped > 0  # Reused from first 2 docs
        assert update_result.num_added > 0  # New chunks from 3rd doc
        assert len(update_result.new_chunked_doc.get_document_ids()) == 3

    def test_update_with_document_removal(self, updater):
        """Test removing a document from knowledge base."""
        # Create initial KB with 3 documents
        initial_docs = ["First document.", "Second document.", "Third document."]
        initial_result = updater.create_knowledge_base(initial_docs)

        # Update with only first 2 documents (remove third)
        updated_docs = [
            "First document.",  # Unchanged
            "Second document.",  # Unchanged
        ]

        update_result = updater.update_knowledge_base(initial_result.new_chunked_doc, updated_docs)

        # Should reuse chunks from first 2 docs and delete chunks from 3rd
        assert update_result.num_skipped > 0  # Reused from first 2 docs
        assert update_result.num_deleted > 0  # Deleted from 3rd doc
        assert len(update_result.new_chunked_doc.get_document_ids()) == 2

    def test_efficiency_calculation(self, updater):
        """Test efficiency ratio calculation."""
        # Create initial KB
        initial_docs = ["Document one.", "Document two."]
        initial_result = updater.create_knowledge_base(initial_docs)

        # Update with one unchanged, one modified
        updated_docs = [
            "Document one.",  # Unchanged - should be reused
            "Modified document two with extra content.",  # Changed
        ]

        update_result = updater.update_knowledge_base(initial_result.new_chunked_doc, updated_docs)

        # Efficiency ratio should be > 0 since some chunks are reused
        assert update_result.efficiency_ratio > 0
        assert update_result.efficiency_ratio <= 1.0

    def test_chunk_data_document_id_assignment(self, updater):
        """Test that ChunkData objects get correct document_id."""
        documents = ["Doc zero content.", "Doc one content.", "Doc two content."]

        result = updater.create_knowledge_base(documents)

        # Group chunks by document_id
        chunks_by_doc = {}
        for chunk in result.new_chunked_doc.chunks:
            doc_id = chunk.document_id
            if doc_id not in chunks_by_doc:
                chunks_by_doc[doc_id] = []
            chunks_by_doc[doc_id].append(chunk)

        # Should have chunks for documents 0, 1, 2
        assert set(chunks_by_doc.keys()) == {0, 1, 2}

        # Each document should have at least one chunk
        for doc_id in [0, 1, 2]:
            assert len(chunks_by_doc[doc_id]) > 0

    def test_get_chunks_by_document(self, updater):
        """Test the get_chunks_by_document method."""
        documents = ["First document content goes here.", "Second document has different content."]

        result = updater.create_knowledge_base(documents)
        kb = result.new_chunked_doc

        # Test getting chunks for each document
        doc0_chunks = kb.get_chunks_by_document(0)
        doc1_chunks = kb.get_chunks_by_document(1)

        assert len(doc0_chunks) > 0
        assert len(doc1_chunks) > 0

        # All chunks should have correct document_id
        for chunk in doc0_chunks:
            assert chunk.document_id == 0
        for chunk in doc1_chunks:
            assert chunk.document_id == 1

        # No overlap between document chunks
        doc0_hashes = {chunk.hash for chunk in doc0_chunks}
        doc1_hashes = {chunk.hash for chunk in doc1_chunks}
        assert doc0_hashes.isdisjoint(doc1_hashes)
