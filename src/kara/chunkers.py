"""
Chunkers for grouping sub-chunks into optimal chunks.
"""

from abc import ABC, abstractmethod
from typing import List


class BaseChunker(ABC):
    """Abstract base class for chunkers."""

    @abstractmethod
    def create_chunks(self, splits: List[str], max_chunk_size: int) -> List[List[str]]:
        """Create chunks from splits."""
        pass


class GreedyChunker(BaseChunker):
    """
    Greedy chunker that combines sub-chunks until the size limit is reached.
    """

    def create_chunks(self, splits: List[str], max_chunk_size: int) -> List[List[str]]:
        """
        Create chunks using a greedy approach.

        Args:
            splits: List of splits to group
            max_chunk_size: Maximum size of a chunk in characters

        Returns:
            List of chunks, where each chunk is a list of splits
        """

        def chunk_length(chunk: List[str]) -> int:
            return sum(len(sub) for sub in chunk)

        result = []
        current_chunk: List[str] = []

        for split in splits:
            if chunk_length(current_chunk) + len(split) > max_chunk_size:
                if current_chunk:
                    result.append(current_chunk)
                current_chunk = [split]
            else:
                current_chunk.append(split)

        if current_chunk:
            result.append(current_chunk)

        return result


class OptimalChunker(BaseChunker):
    """
    Optimal chunker that uses dynamic programming to find the best chunking.

    This is a placeholder for a more sophisticated chunking algorithm
    that could consider factors like semantic boundaries, chunk balance, etc.
    """

    def __init__(self, balance_factor: float = 0.1):
        """
        Initialize the optimal chunker.

        Args:
            balance_factor: Factor to balance chunk sizes (0.0 = no balancing)
        """
        self.balance_factor = balance_factor

    def create_chunks(self, splits: List[str], max_chunk_size: int) -> List[List[str]]:
        """
        Create chunks using optimal approach.

        For now, this falls back to greedy chunking.
        Future versions could implement true optimal chunking.

        Args:
            splits: List of splits to group
            max_chunk_size: Maximum size of a chunk in characters

        Returns:
            List of chunks, where each chunk is a list of splits
        """
        # TODO: Implement true optimal chunking algorithm
        # For now, use greedy approach
        greedy_chunker = GreedyChunker()
        return greedy_chunker.create_chunks(splits, max_chunk_size)


class BalancedChunker(BaseChunker):
    """
    Balanced chunker that tries to create more evenly sized chunks.
    """

    def __init__(self, target_utilization: float = 0.8):
        """
        Initialize the balanced chunker.

        Args:
            target_utilization: Target utilization of chunk capacity (0.0 to 1.0)
        """
        self.target_utilization = target_utilization

    def create_chunks(self, splits: List[str], max_chunk_size: int) -> List[List[str]]:
        """
        Create balanced chunks.

        Args:
            splits: List of splits to group
            max_chunk_size: Maximum size of a chunk in characters

        Returns:
            List of chunks, where each chunk is a list of splits
        """
        target_size = int(max_chunk_size * self.target_utilization)

        def chunk_length(chunk: List[str]) -> int:
            return sum(len(sub) for sub in chunk)

        result = []
        current_chunk: List[str] = []

        for split in splits:
            current_length = chunk_length(current_chunk)

            # If adding this split would exceed the max size, start a new chunk
            if current_length + len(split) > max_chunk_size:
                if current_chunk:
                    result.append(current_chunk)
                current_chunk = [split]
            # If we're past the target size, consider starting a new chunk
            elif current_length >= target_size:
                # Look ahead to see if the next split would fit better in a new chunk
                if len(split) < target_size // 2:
                    result.append(current_chunk)
                    current_chunk = [split]
                else:
                    current_chunk.append(split)
            else:
                current_chunk.append(split)

        if current_chunk:
            result.append(current_chunk)

        return result
