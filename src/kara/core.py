"""
Core KARA algorithm implementation.
"""

import hashlib
import heapq
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .splitters import BaseDocumentChunker


@dataclass
class ChunkData:
    """Represents a chunk with its content and metadata."""

    content: str
    splits: List[str]
    hash: str
    document_id: Optional[int] = None

    @classmethod
    def from_splits(cls, splits: List[str], document_id: Optional[int] = None) -> "ChunkData":
        """Create ChunkData from splits."""
        content = "".join(splits)
        hash_value = hashlib.md5(content.encode("utf-8")).hexdigest()
        return cls(content=content, splits=splits, hash=hash_value, document_id=document_id)


@dataclass
class ChunkedDocument:
    """Represents the current state of the knowledge base."""

    chunks: List[ChunkData]

    def get_chunk_hashes(self) -> Set[str]:
        """Get all chunk hashes in the knowledge base."""
        return {chunk.hash for chunk in self.chunks}

    def get_chunks_by_document(self, document_id: int) -> List[ChunkData]:
        """Get all chunks belonging to a specific document."""
        return [chunk for chunk in self.chunks if chunk.document_id == document_id]

    def get_document_ids(self) -> Set[int]:
        """Get all unique document IDs in the knowledge base."""
        return {chunk.document_id for chunk in self.chunks if chunk.document_id is not None}

    def get_chunk_contents(self) -> List[str]:
        """Get all chunk contents as strings."""
        return [chunk.content for chunk in self.chunks]

    @classmethod
    def from_chunks(
        cls, chunks: List[str], chunker: BaseDocumentChunker, document_id: Optional[int] = None
    ) -> "ChunkedDocument":
        """
        Create a ChunkedDocument from a list of texts using the specified separators.
        Args:
            texts: List of text strings to chunk
            separators: List of separators to use for chunking
            document_id: Optional document identifier

        Returns:
            ChunkedDocument with chunks created
        """
        warnings.warn(
            (
                "If the separator list is not the same as that used for separating previous "
                "chunks, the algorithm might fail."
            ),
            UserWarning,
            stacklevel=2,
        )
        result = []
        for text in chunks:
            assert len(text) <= chunker.chunk_size, (
                "Text length exceeds the maximum chunk size defined in the chunker."
                f" Text length: {len(text)}, Max chunk size: {chunker.chunk_size}"
            )
            splits = chunker._split_to_units(text)
            result.append(ChunkData.from_splits(splits, document_id))
        return cls(chunks=result)


@dataclass
class UpdateResult:
    """Result of a KARA update operation."""

    num_added: int = 0
    num_reused: int = 0
    num_deleted: int = 0
    new_chunked_doc: Optional["ChunkedDocument"] = None

    def __add__(self, other: "UpdateResult") -> "UpdateResult":
        """Add two UpdateResult objects."""
        return UpdateResult(
            num_added=self.num_added + other.num_added,
            num_reused=self.num_reused + other.num_reused,
            num_deleted=self.num_deleted + other.num_deleted,
        )

    @property
    def total_operations(self) -> int:
        """Total number of operations performed."""
        return self.num_added + self.num_deleted

    @property
    def efficiency_ratio(self) -> float:
        """Ratio of skipped operations to total operations."""
        total_chunks = len(self.new_chunked_doc.chunks) if self.new_chunked_doc else 0
        return self.num_reused / total_chunks if total_chunks > 0 else 0.0


class KARAUpdater:
    """
    Knowledge-Aware Re-embedding Algorithm updater.

    Efficiently updates document chunks by minimizing embedding operations
    through intelligent reuse of existing chunks.
    """

    def __init__(
        self,
        chunker: BaseDocumentChunker,
        epsilon: float = 0.01,
    ):
        """
        Initialize the KARA updater.

        Args:
            chunker: Document chunker for breaking documents into optimal chunks
            epsilon: Cost factor for reusing existing chunks (0 < epsilon < 1)
        """
        self.chunker = chunker
        self.epsilon = epsilon
        self.max_chunk_size = getattr(chunker, "chunk_size", 1000)

    def _get_chunker_config(self) -> Dict[str, Any]:
        """
        Get configuration parameters from the chunker.

        Returns:
            Dictionary with chunker configuration
        """
        config = {
            "chunk_size": getattr(self.chunker, "chunk_size", None),
            "separators": getattr(self.chunker, "separators", None),
            "keep_separator": getattr(self.chunker, "keep_separator", None),
            "chunker_type": type(self.chunker).__name__,
        }
        return config

    def create_knowledge_base(self, documents: List[str]) -> UpdateResult:
        """
        Create a new knowledge base from documents.

        Args:
            documents: List of document texts

        Returns:
            UpdateResult with initial chunks
        """
        if not documents:
            return UpdateResult(
                num_added=0,
                new_chunked_doc=ChunkedDocument(chunks=[]),
            )

        all_chunks = []
        total_added = 0

        for doc_id, document in enumerate(documents):
            chunk_strings = self.chunker.create_chunks(document)

            for chunk_str in chunk_strings:
                # Split each chunk back into its component units
                splits = self.chunker._split_to_units(chunk_str)
                all_chunks.append(ChunkData.from_splits(splits, doc_id))
                total_added += 1

        return UpdateResult(
            num_added=total_added,
            new_chunked_doc=ChunkedDocument(chunks=all_chunks),
        )

    def update_knowledge_base(
        self, current_kb: ChunkedDocument, documents: List[str]
    ) -> UpdateResult:
        """
        Update the knowledge base with new documents.

        Args:
            current_kb: Current knowledge base state
            documents: List of updated document texts

        Returns:
            UpdateResult with statistics and new knowledge base
        """
        if not documents:
            return UpdateResult(
                num_deleted=len(current_kb.chunks),
                new_chunked_doc=ChunkedDocument(chunks=[]),
            )

        # Process each document separately and combine results
        all_new_chunks: List[ChunkData] = []
        combined_result = UpdateResult()
        old_chunk_hashes = current_kb.get_chunk_hashes()
        used_hashes: Set[str] = set()

        for doc_id, document in enumerate(documents):
            new_splits = self.chunker._split_to_units(document)
            doc_result = self._update_chunks_for_document(
                current_kb, new_splits, doc_id, old_chunk_hashes
            )

            assert doc_result.new_chunked_doc is not None
            all_new_chunks.extend(doc_result.new_chunked_doc.chunks)
            combined_result.num_added += doc_result.num_added
            combined_result.num_reused += doc_result.num_reused

            # Track which hashes are used across all documents
            for chunk in doc_result.new_chunked_doc.chunks:
                if chunk.hash in old_chunk_hashes:
                    used_hashes.add(chunk.hash)

        # Count deleted chunks that are not reused by any document
        for chunk_hash in old_chunk_hashes:
            if chunk_hash not in used_hashes:
                combined_result.num_deleted += 1

        # Create the final chunked document
        combined_result.new_chunked_doc = ChunkedDocument(chunks=all_new_chunks)

        return combined_result

    def _update_chunks_for_document(
        self,
        current_kb: ChunkedDocument,
        new_splits: List[str],
        document_id: int,
        old_chunk_hashes: Set[str],
    ) -> UpdateResult:
        """
        Update chunks for a single document using the KARA algorithm.

        Args:
            current_kb: Current knowledge base state
            new_splits: New splits to process for this document
            document_id: ID of the document being processed
            old_chunk_hashes: Set of existing chunk hashes

        Returns:
            UpdateResult with new chunks and statistics for this document
        """
        N = len(new_splits)
        if N == 0:
            return UpdateResult(
                num_deleted=0,  # Will be calculated at the end
                new_chunked_doc=ChunkedDocument(chunks=[]),
            )

        # Build graph of possible chunks for this document
        edges: List[List[Tuple[int, float, List[str], str]]] = [[] for _ in range(N + 1)]

        for i in range(N):
            current_length = 0
            chunk_splits = []

            for j in range(i + 1, N + 1):
                if j <= N:
                    split = new_splits[j - 1]
                    chunk_splits.append(split)
                    current_length += len(split)

                    # A single split cannot exceed the max chunk size
                    # TODO: handle the edge case in which all splits are larger than max_chunk_size
                    assert len(split) <= self.max_chunk_size, (
                        f"Split length {len(split)} exceeds max chunk size {self.max_chunk_size}."
                    )

                if current_length > self.max_chunk_size:
                    break

                chunk_str = "".join(chunk_splits)
                chunk_hash = hashlib.md5(chunk_str.encode("utf-8")).hexdigest()

                if chunk_hash in old_chunk_hashes:
                    cost = self.epsilon
                else:
                    cost = 1.0

                edges[i].append((j, cost, chunk_splits.copy(), chunk_hash))

        # Find optimal path using Dijkstra's algorithm
        min_cost = [float("inf")] * (N + 1)
        min_cost[0] = 0
        previous_node: List[Optional[int]] = [None] * (N + 1)
        previous_edge: List[Optional[Tuple[int, float, List[str], str]]] = [None] * (N + 1)

        heap: List[Tuple[float, int]] = [(0, 0)]

        while heap:
            cost_u, u = heapq.heappop(heap)
            if cost_u > min_cost[u]:
                continue

            for v, edge_cost, chunk_splits, chunk_hash in edges[u]:
                new_cost = min_cost[u] + edge_cost
                if new_cost < min_cost[v]:
                    min_cost[v] = new_cost
                    previous_node[v] = u
                    previous_edge[v] = (v, edge_cost, chunk_splits, chunk_hash)
                    heapq.heappush(heap, (new_cost, v))

        # Reconstruct the solution for this document
        new_chunks: List[ChunkData] = []
        result = UpdateResult()

        node = N
        while node > 0:
            edge = previous_edge[node]
            if edge is None:
                break

            _, edge_cost, chunk_splits, chunk_hash = edge
            chunk_data = ChunkData.from_splits(chunk_splits, document_id)
            new_chunks.insert(0, chunk_data)

            if chunk_hash in old_chunk_hashes:
                result.num_reused += 1
            else:
                result.num_added += 1

            prev_node = previous_node[node]
            if prev_node is None:
                break
            node = prev_node

        result.new_chunked_doc = ChunkedDocument(chunks=new_chunks)
        return result

    def _update_chunks(self, current_kb: ChunkedDocument, new_splits: List[str]) -> UpdateResult:
        """
        Update chunks using the KARA algorithm for backward compatibility.
        This method handles single document updates.

        Args:
            current_kb: Current knowledge base state
            new_splits: New splits to process

        Returns:
            UpdateResult with new chunks and statistics
        """
        old_chunk_hashes = current_kb.get_chunk_hashes()

        # Use the new multi-document method with document_id = 0
        doc_result = self._update_chunks_for_document(current_kb, new_splits, 0, old_chunk_hashes)

        # Count deleted chunks that are not reused
        used_hashes: Set[str] = set()
        assert doc_result.new_chunked_doc is not None
        for chunk in doc_result.new_chunked_doc.chunks:
            if chunk.hash in old_chunk_hashes:
                used_hashes.add(chunk.hash)

        # Count deleted chunks
        for chunk_hash in old_chunk_hashes:
            if chunk_hash not in used_hashes:
                doc_result.num_deleted += 1

        return doc_result
