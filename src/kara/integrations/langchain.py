"""
LangChain integration for kara-python.
"""

from typing import Any, Dict, List, Optional

try:
    from langchain.indexes import SQLRecordManager
    from langchain_core.documents.base import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.indexing import index
    from langchain_core.vectorstores import VectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_text_splitters.base import TextSplitter

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Document = Any
    RecursiveCharacterTextSplitter = Any
    TextSplitter = Any
    VectorStore = Any
    Embeddings = Any
    SQLRecordManager = Any

from ..core import KARAUpdater, UpdateResult
from ..splitters import BaseTextSplitter


class KARATextSplitter(TextSplitter):
    """
    KARA-powered text splitter that inherits from LangChain's TextSplitter.

    This splitter maintains compatibility with LangChain's ecosystem while
    providing efficient chunk updates through the KARA algorithm.
    """

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        epsilon: float = 0.01,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize the KARA text splitter.

        Args:
            separators: List of separators to use for splitting
            epsilon: Cost factor for reusing existing chunks
            is_separator_regex: Whether separators are regex patterns
            **kwargs: Additional arguments passed to TextSplitter
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not installed. Install it with: pip install langchain")

        super().__init__(**kwargs)

        # Initialize the underlying splitter
        from ..splitters import RecursiveTextSplitter

        self._kara_splitter = RecursiveTextSplitter(
            separators=separators or ["\n\n", "\n", " ", ""],
            keep_separator=self._keep_separator,
        )

        # Initialize KARA updater
        self.kara_updater = KARAUpdater(
            splitter=self._kara_splitter,
            epsilon=epsilon,
            max_chunk_size=self._chunk_size,
        )

        self._is_initialized = False
        self._current_splits: List[List[str]] = []

    def split_text(self, text: str) -> List[str]:
        """
        Split text using KARA algorithm.

        Args:
            text: Input text to split

        Returns:
            List of text chunks
        """
        if not self._is_initialized:
            # First time - initialize
            splits = self.kara_updater.initialize([text])
            self._is_initialized = True
        else:
            # Update existing splits
            self.kara_updater.update([text])
            splits = self.kara_updater.get_current_chunks()

        self._current_splits = splits

        # Convert splits back to text chunks
        chunks = []
        for split_group in splits:
            chunk_text = "".join(split_group)
            if self._strip_whitespace:
                chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    def update_text(self, text: str) -> UpdateResult:
        """
        Update the splitter with new text and return efficiency metrics.

        Args:
            text: New text to process

        Returns:
            UpdateResult with efficiency statistics
        """
        if not self._is_initialized:
            self.kara_updater.initialize([text])
            self._is_initialized = True
            return UpdateResult(num_added=len(self._current_splits))
        else:
            result = self.kara_updater.update([text])
            self._current_splits = self.kara_updater.get_current_chunks()
            return result

    def get_efficiency_stats(self) -> Dict[str, Any]:
        """
        Get efficiency statistics for the last update.

        Returns:
            Dictionary with efficiency metrics
        """
        if not hasattr(self, "_last_result"):
            return {
                "efficiency_ratio": 0.0,
                "total_chunks": len(self._current_splits),
            }

        return {
            "efficiency_ratio": self._last_result.efficiency_ratio,
            "total_chunks": len(self._current_splits),
            "num_added": self._last_result.num_added,
            "num_skipped": self._last_result.num_skipped,
            "num_deleted": self._last_result.num_deleted,
        }


class LangChainTextSplitter(BaseTextSplitter):
    """Adapter for LangChain text splitters."""

    def __init__(self, langchain_splitter):
        """
        Initialize with a LangChain text splitter.

        Args:
            langchain_splitter: A LangChain text splitter instance
        """
        self.langchain_splitter = langchain_splitter

    def split_text(self, text: str) -> List[str]:
        """Split text using the LangChain splitter."""
        return self.langchain_splitter.split_text(text)


class LangChainKARAUpdater:
    """
    KARA updater integrated with LangChain components.

    This class provides high-level integration with LangChain's document
    processing pipeline while using KARA for efficient updates.
    """

    def __init__(
        self,
        vectorstore_class: type,
        embeddings: Any,  # Using Any to avoid type errors
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        epsilon: float = 0.01,
        record_manager_url: Optional[str] = None,
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize the LangChain KARA updater.

        Args:
            vectorstore_class: LangChain vectorstore class (e.g., FAISS, Chroma)
            embeddings: LangChain embeddings instance
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            epsilon: Cost factor for reusing existing chunks
            record_manager_url: URL for the record manager database
            separators: Optional custom separators for text splitting
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is not installed. Install it with: pip install langchain")

        self.vectorstore_class = vectorstore_class
        self.embeddings = embeddings
        self.epsilon = epsilon

        # Initialize KARA text splitter
        self.text_splitter = KARATextSplitter(
            separators=separators,
            epsilon=epsilon,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Initialize record manager
        if record_manager_url is None:
            record_manager_url = "sqlite:///record_manager_cache.db"

        self.record_manager = SQLRecordManager("kara_namespace", db_url=record_manager_url)
        self.record_manager.create_schema()

        self.vectorstore: Optional[Any] = None  # Using Any to avoid type errors

    def initialize_documents(
        self, documents: List[Any]
    ) -> Dict[str, int]:  # Using Any to avoid type errors
        """
        Initialize the vector store with documents.

        Args:
            documents: List of LangChain documents

        Returns:
            Dictionary with initialization statistics
        """
        if not documents:
            return {
                "num_added": 0,
                "num_updated": 0,
                "num_skipped": 0,
                "num_deleted": 0,
            }

        # Split documents using KARA text splitter
        split_documents = self.text_splitter.split_documents(documents)

        # Initialize vector store
        self.vectorstore = self.vectorstore_class.from_documents(split_documents, self.embeddings)

        return {
            "num_added": len(split_documents),
            "num_updated": 0,
            "num_skipped": 0,
            "num_deleted": 0,
        }

    def update_documents(
        self, documents: List[Any]
    ) -> Dict[str, int]:  # Using Any to avoid type errors
        """
        Update the vector store with new documents.

        Args:
            documents: List of updated LangChain documents

        Returns:
            Dictionary with update statistics
        """
        if not documents:
            return {
                "num_added": 0,
                "num_updated": 0,
                "num_skipped": 0,
                "num_deleted": 0,
            }

        if self.vectorstore is None:
            return self.initialize_documents(documents)

        # Split documents using KARA text splitter (which handles updates efficiently)
        split_documents = self.text_splitter.split_documents(documents)

        # Update vector store using LangChain's index function
        index_result = index(
            split_documents,
            self.record_manager,
            self.vectorstore,
            cleanup="full",
            source_id_key="source",
        )

        return {
            "num_added": index_result.get("num_added", 0),
            "num_updated": index_result.get("num_updated", 0),
            "num_skipped": index_result.get("num_skipped", 0),
            "num_deleted": index_result.get("num_deleted", 0),
        }

    def get_vectorstore(
        self,
    ) -> Optional[Any]:  # Using Any to avoid type errors
        """Get the current vector store."""
        return self.vectorstore

    def get_text_splitter(self) -> KARATextSplitter:
        """Get the KARA text splitter instance."""
        return self.text_splitter

    def similarity_search(
        self, query: str, k: int = 4
    ) -> List[Any]:  # Using Any to avoid type errors
        """
        Perform similarity search on the vector store.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of similar documents
        """
        if self.vectorstore is None:
            return []

        return self.vectorstore.similarity_search(query, k=k)
