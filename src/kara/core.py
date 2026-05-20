"""
Core KARA algorithm implementation.
"""

import hashlib
import heapq
import json
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .chunkers import BaseDocumentChunker


@dataclass
class ChunkData:
    """Represents a chunk with its content and metadata."""

    content: Any
    splits: List[Any]
    hash: str
    document_id: Optional[int] = None

    @classmethod
    def from_splits(
        cls,
        splits: Sequence[Any],
        document_id: Optional[int] = None,
        serializer: Optional[Callable[[Sequence[Any]], bytes]] = None,
        renderer: Optional[Callable[[Sequence[Any]], Any]] = None,
    ) -> "ChunkData":
        """Create ChunkData from splits."""
        content: Any
        if renderer is None:
            if all(isinstance(unit, str) for unit in splits):
                content = "".join(splits)
            else:
                content = list(splits)
        else:
            content = renderer(splits)

        if serializer is None:
            if all(isinstance(unit, str) for unit in splits):
                serialized = "".join(splits).encode("utf-8")
            else:
                serialized = json.dumps(
                    list(splits), separators=(",", ":"), ensure_ascii=True
                ).encode("utf-8")
        else:
            serialized = serializer(splits)

        hash_value = hashlib.md5(serialized).hexdigest()
        return cls(content=content, splits=list(splits), hash=hash_value, document_id=document_id)


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

    def get_chunk_contents(self) -> List[Any]:
        """Get all chunk contents."""
        return [chunk.content for chunk in self.chunks]

    @classmethod
    def from_chunks(
        cls, chunks: List[Any], chunker: BaseDocumentChunker, document_id: Optional[int] = None
    ) -> "ChunkedDocument":
        """Create a :class:`ChunkedDocument` from pre-split chunks.

        Args:
            chunks: List of text chunks to include
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
        for chunk in chunks:
            splits = chunker.normalize_chunk(chunk)
            chunk_length = sum(chunker.unit_length(unit) for unit in splits)
            assert chunk_length <= chunker.chunk_size, (
                "Chunk length exceeds the maximum chunk size defined in the chunker."
                f" Chunk length: {chunk_length}, Max chunk size: {chunker.chunk_size}"
            )
            result.append(
                ChunkData.from_splits(
                    splits,
                    document_id,
                    serializer=chunker.serialize_units,
                    renderer=chunker.render_units,
                )
            )
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
    ):
        """
        Initialize the KARA updater.

        Args:
            chunker: Document chunker for breaking documents into optimal chunks
        """
        self.chunker: BaseDocumentChunker = chunker
        self.max_chunk_size: int = chunker.chunk_size

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
            chunk_list = self.chunker.create_chunks(document)

            for chunk in chunk_list:
                splits = self.chunker.normalize_chunk(chunk)
                all_chunks.append(
                    ChunkData.from_splits(
                        splits,
                        doc_id,
                        serializer=self.chunker.serialize_units,
                        renderer=self.chunker.render_units,
                    )
                )
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
        old_chunk_counts: Dict[str, int] = {}
        for chunk in current_kb.chunks:
            old_chunk_counts[chunk.hash] = old_chunk_counts.get(chunk.hash, 0) + 1

        used_counts: Dict[str, int] = {}

        for doc_id, document in enumerate(documents):
            new_splits = self.chunker._split_to_units(document)
            doc_result = self._update_chunks_for_document(
                current_kb, new_splits, doc_id, set(old_chunk_counts.keys())
            )

            assert doc_result.new_chunked_doc is not None
            all_new_chunks.extend(doc_result.new_chunked_doc.chunks)

            # Track which hashes are used across all documents
            for chunk in doc_result.new_chunked_doc.chunks:
                used_counts[chunk.hash] = used_counts.get(chunk.hash, 0) + 1

        # Calculate added and reused chunks based on inventory
        combined_result.num_added = 0
        combined_result.num_reused = 0
        for chunk_hash, count in used_counts.items():
            old_count = old_chunk_counts.get(chunk_hash, 0)
            reused = min(count, old_count)
            combined_result.num_reused += reused
            combined_result.num_added += count - reused

        # Count deleted chunks considering duplicate hashes
        for chunk_hash, count in old_chunk_counts.items():
            reused_count = used_counts.get(chunk_hash, 0)
            if reused_count < count:
                combined_result.num_deleted += count - reused_count

        # Create the final chunked document
        combined_result.new_chunked_doc = ChunkedDocument(chunks=all_new_chunks)

        return combined_result

    def _update_chunks_for_document(
        self,
        current_kb: ChunkedDocument,
        new_splits: List[Any],
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
        edges: List[List[Tuple[int, float, List[Any], str]]] = [[] for _ in range(N + 1)]

        max_chunk_size = self.max_chunk_size
        max_chunk_size_float = float(max_chunk_size)
        overlap_units = self.chunker.overlap
        unit_length = self.chunker.unit_length

        for i in range(N):
            current_length = 0
            chunk_splits = []

            for j in range(i + 1, N + 1):
                if j <= N:
                    split = new_splits[j - 1]
                    chunk_splits.append(split)
                    current_length += unit_length(split)

                    # A single split cannot exceed the max chunk size
                    # TODO: handle the edge case in which all splits are larger than max_chunk_size
                    if unit_length(split) > max_chunk_size:
                        raise ValueError(
                            f"Split length {unit_length(split)} exceeds max chunk size "
                            f"{max_chunk_size}."
                        )

                if current_length > max_chunk_size:
                    break

                serialized = self.chunker.serialize_units(chunk_splits)
                chunk_hash = hashlib.md5(serialized).hexdigest()

                fill_rate = current_length / max_chunk_size_float
                penalty = (1 - fill_rate) ** 2

                if chunk_hash in old_chunk_hashes:
                    cost = penalty
                else:
                    cost = 1.0 + penalty

                if j == N:
                    next_node = N
                else:
                    next_node = max(i + 1, j - overlap_units)

                edges[i].append((next_node, cost, chunk_splits.copy(), chunk_hash))

        # Find optimal path using Dijkstra's algorithm with edge count tie-breaking
        int_inf: int = sys.maxsize

        min_cost = [float("inf")] * (N + 1)
        min_num_edges = [int_inf] * (N + 1)
        min_cost[0] = 0
        min_num_edges[0] = 0
        previous_node: List[Optional[int]] = [None] * (N + 1)
        previous_edge: List[Optional[Tuple[int, float, List[Any], str]]] = [None] * (N + 1)

        heap: List[Tuple[float, int, int]] = [(0, 0, 0)]  # (cost, edge_count, node)

        while heap:
            cost_u, edges_count_u, u = heapq.heappop(heap)
            if cost_u > min_cost[u] or (cost_u == min_cost[u] and edges_count_u > min_num_edges[u]):
                continue

            for v, edge_cost, chunk_splits, chunk_hash in edges[u]:
                new_cost = min_cost[u] + edge_cost
                new_num_edges = min_num_edges[u] + 1

                if new_cost < min_cost[v] or (
                    new_cost == min_cost[v] and new_num_edges < min_num_edges[v]
                ):
                    min_cost[v] = new_cost
                    min_num_edges[v] = new_num_edges
                    previous_node[v] = u
                    previous_edge[v] = (v, edge_cost, chunk_splits, chunk_hash)
                    heap_item: Tuple[float, int, int] = (new_cost, new_num_edges, v)
                    heapq.heappush(heap, heap_item)

        # Reconstruct the solution for this document
        new_chunks: List[ChunkData] = []
        result = UpdateResult()

        node = N
        while node > 0:
            edge = previous_edge[node]
            if edge is None:
                break

            _, edge_cost, chunk_splits, chunk_hash = edge
            chunk_data = ChunkData.from_splits(
                chunk_splits,
                document_id,
                serializer=self.chunker.serialize_units,
                renderer=self.chunker.render_units,
            )
            new_chunks.insert(0, chunk_data)

            prev_node = previous_node[node]
            if prev_node is None:
                break
            node = prev_node

        result.new_chunked_doc = ChunkedDocument(chunks=new_chunks)
        return result

    def _update_chunks(self, current_kb: ChunkedDocument, new_splits: List[Any]) -> UpdateResult:
        """
        Update chunks using the KARA algorithm for backward compatibility.
        This method handles single document updates.

        Args:
            current_kb: Current knowledge base state
            new_splits: New splits to process

        Returns:
            UpdateResult with new chunks and statistics
        """
        old_chunk_counts: Dict[str, int] = {}
        for chunk in current_kb.chunks:
            old_chunk_counts[chunk.hash] = old_chunk_counts.get(chunk.hash, 0) + 1

        # Use the new multi-document method with document_id = 0
        doc_result = self._update_chunks_for_document(
            current_kb, new_splits, 0, set(old_chunk_counts.keys())
        )

        # Count used chunks
        used_counts: Dict[str, int] = {}
        assert doc_result.new_chunked_doc is not None
        for chunk in doc_result.new_chunked_doc.chunks:
            used_counts[chunk.hash] = used_counts.get(chunk.hash, 0) + 1

        # Calculate added and reused chunks based on inventory
        doc_result.num_added = 0
        doc_result.num_reused = 0
        for chunk_hash, count in used_counts.items():
            old_count = old_chunk_counts.get(chunk_hash, 0)
            reused = min(count, old_count)
            doc_result.num_reused += reused
            doc_result.num_added += count - reused

        # Count deleted chunks
        doc_result.num_deleted = 0
        for chunk_hash, count in old_chunk_counts.items():
            reused_count = used_counts.get(chunk_hash, 0)
            if reused_count < count:
                doc_result.num_deleted += count - reused_count

        return doc_result
