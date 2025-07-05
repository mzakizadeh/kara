"""
Core KARA algorithm implementation.
"""

import hashlib
import heapq
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

from .splitters import BaseTextSplitter
from .chunkers import BaseChunker


@dataclass
class UpdateResult:
    """Result of a KARA update operation."""

    num_added: int = 0
    num_updated: int = 0
    num_skipped: int = 0
    num_deleted: int = 0

    def __add__(self, other: "UpdateResult") -> "UpdateResult":
        """Add two UpdateResult objects."""
        return UpdateResult(
            num_added=self.num_added + other.num_added,
            num_updated=self.num_updated + other.num_updated,
            num_skipped=self.num_skipped + other.num_skipped,
            num_deleted=self.num_deleted + other.num_deleted,
        )

    @property
    def total_operations(self) -> int:
        """Total number of operations performed."""
        return self.num_added + self.num_updated + self.num_deleted

    @property
    def efficiency_ratio(self) -> float:
        """Ratio of skipped operations to total operations."""
        total = (
            self.num_added
            + self.num_updated
            + self.num_skipped
            + self.num_deleted
        )
        return self.num_skipped / total if total > 0 else 0.0


class KARAUpdater:
    """
    Knowledge-Aware Rebedding Algorithm updater.

    Efficiently updates document chunks by minimizing embedding operations
    through intelligent reuse of existing chunks.
    """

    def __init__(
        self,
        splitter: BaseTextSplitter,
        chunker: Optional[BaseChunker] = None,
        epsilon: float = 0.01,
        max_chunk_size: int = 1000,
    ):
        """
        Initialize the KARA updater.

        Args:
            splitter: Text splitter for breaking documents into splits
            chunker: Optional custom chunker (defaults to optimal chunker)
            epsilon: Cost factor for reusing existing chunks (0 < epsilon < 1)
            max_chunk_size: Maximum size of a chunk in characters
        """
        self.splitter = splitter
        self.chunker = chunker
        self.epsilon = epsilon
        self.max_chunk_size = max_chunk_size
        self._current_chunks: List[List[str]] = []
        self._chunk_hashes: Dict[str, List[str]] = {}

    def initialize(self, documents: List[str]) -> List[List[str]]:
        """
        Initialize the knowledge base with documents.

        Args:
            documents: List of document texts

        Returns:
            Initial chunks as lists of splits
        """
        if not documents:
            return []

        # For now, process only the first document
        # TODO: Extend to handle multiple documents
        document = documents[0]
        splits = self.splitter.split_text(document)

        if self.chunker:
            chunks = self.chunker.create_chunks(splits, self.max_chunk_size)
        else:
            chunks = self._greedy_merge_chunks(splits)

        self._current_chunks = chunks
        self._build_chunk_hash_map()

        return chunks

    def update(self, documents: List[str]) -> UpdateResult:
        """
        Update the knowledge base with new documents.

        Args:
            documents: List of updated document texts

        Returns:
            UpdateResult with statistics about the update operation
        """
        if not documents:
            return UpdateResult()

        # For now, process only the first document
        # TODO: Extend to handle multiple documents
        document = documents[0]
        new_splits = self.splitter.split_text(document)

        new_chunks, result = self._update_chunks(new_splits)

        self._current_chunks = new_chunks
        self._build_chunk_hash_map()

        return result

    def get_current_chunks(self) -> List[List[str]]:
        """Get the current chunks."""
        return self._current_chunks.copy()

    def _greedy_merge_chunks(self, splits: List[str]) -> List[List[str]]:
        """Greedily merge splits into chunks."""

        def chunk_length(chunk: List[str]) -> int:
            return sum(len(sub) for sub in chunk)

        result = []
        current_chunk = []

        for split in splits:
            if chunk_length(current_chunk) + len(split) > self.max_chunk_size:
                if current_chunk:
                    result.append(current_chunk)
                current_chunk = [split]
            else:
                current_chunk.append(split)

        if current_chunk:
            result.append(current_chunk)

        return result

    def _build_chunk_hash_map(self):
        """Build a hash map of current chunks."""
        self._chunk_hashes = {}
        for chunk in self._current_chunks:
            chunk_str = "".join(chunk)
            chunk_hash = hashlib.md5(chunk_str.encode("utf-8")).hexdigest()
            self._chunk_hashes[chunk_hash] = chunk

    def _update_chunks(
        self, new_splits: List[str]
    ) -> Tuple[List[List[str]], UpdateResult]:
        """
        Update chunks using the KARA algorithm.

        Args:
            new_splits: New splits to process

        Returns:
            Tuple of (new_chunks, update_result)
        """
        old_chunk_hashes = set(self._chunk_hashes.keys())

        N = len(new_splits)
        if N == 0:
            return [], UpdateResult(num_deleted=len(old_chunk_hashes))

        # Build graph of possible chunks
        edges = [[] for _ in range(N + 1)]

        for i in range(N):
            current_length = 0
            chunk_splits = []

            for j in range(i + 1, N + 1):
                if j <= N:
                    split = new_splits[j - 1]
                    chunk_splits.append(split)
                    current_length += len(split)

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
        previous_node = [None] * (N + 1)
        previous_edge = [None] * (N + 1)

        heap = [(0, 0)]

        while heap:
            cost_u, u = heapq.heappop(heap)
            if cost_u > min_cost[u]:
                continue

            for v, edge_cost, chunk_splits, chunk_hash in edges[u]:
                new_cost = min_cost[u] + edge_cost
                if new_cost < min_cost[v]:
                    min_cost[v] = new_cost
                    previous_node[v] = u
                    previous_edge[v] = (chunk_splits, chunk_hash, edge_cost)
                    heapq.heappush(heap, (new_cost, v))

        # Reconstruct the solution
        new_chunks = []
        result = UpdateResult()
        used_hashes = set()

        node = N
        while node > 0:
            edge = previous_edge[node]
            if edge is None:
                break

            chunk_splits, chunk_hash, edge_cost = edge
            new_chunks.insert(0, chunk_splits)

            if chunk_hash in old_chunk_hashes:
                result.num_skipped += 1
                used_hashes.add(chunk_hash)
            else:
                result.num_added += 1

            node = previous_node[node]

        # Count deleted chunks
        for chunk_hash in old_chunk_hashes:
            if chunk_hash not in used_hashes:
                result.num_deleted += 1

        return new_chunks, result
