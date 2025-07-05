"""
LangChain integration example for kara-python.

This example demonstrates how to use kara-python with LangChain
for efficient RAG knowledge base updates using the KARATextSplitter.
"""

try:
    from langchain_core.documents.base import Document
    from langchain_core.embeddings import FakeEmbeddings
    from langchain_core.vectorstores import FAISS

    from kara.integrations.langchain import (
        KARATextSplitter,
        LangChainKARAUpdater,
    )

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


def main():
    """Demonstrate LangChain integration with kara-python."""
    if not DEPENDENCIES_AVAILABLE:
        print("Error: Missing dependencies. Install with:")
        print("pip install kara-python[langchain]")
        return

    print("kara-python + LangChain Integration Example")
    print("=" * 40)

    # Sample documents
    original_text = """
    Python Programming Basics

    Python is a high-level programming language.
    It's known for its simplicity and readability.

    Key Features:
    - Easy to learn syntax
    - Extensive standard library
    - Large ecosystem of third-party packages
    - Cross-platform compatibility

    Common uses include web development, data science, 
    automation, and artificial intelligence.
    """

    updated_text = """
    Python Programming Basics

    Python is a high-level, interpreted programming language.
    It's known for its simplicity, readability, and versatility.

    Key Features:
    - Easy to learn syntax with English-like keywords
    - Extensive standard library ("batteries included")
    - Large ecosystem of third-party packages (PyPI)
    - Cross-platform compatibility (Windows, Mac, Linux)
    - Strong community support

    Common uses include web development, data science, 
    automation, artificial intelligence, and machine learning.

    Popular frameworks include Django, Flask, pandas, 
    NumPy, and TensorFlow.
    """

    # Example 1: Using KARATextSplitter directly (LangChain-compatible)
    print("1. Using KARATextSplitter (LangChain-compatible)")
    print("-" * 50)

    kara_splitter = KARATextSplitter(
        chunk_size=200, epsilon=0.1, separators=["\n\n", "\n", " ", ""]
    )

    # Initial split
    initial_chunks = kara_splitter.split_text(original_text)
    print(f"   Initial chunks: {len(initial_chunks)}")

    # Update with new text
    update_result = kara_splitter.update_text(updated_text)
    updated_chunks = kara_splitter.split_text(updated_text)

    print(f"   Updated chunks: {len(updated_chunks)}")
    print(f"   Efficiency: {update_result.efficiency_ratio:.1%}")
    print(f"   Added: {update_result.num_added}, Reused: {update_result.num_skipped}")

    # Example 2: Using with LangChain Documents
    print("\n2. Using with LangChain Documents")
    print("-" * 50)

    original_docs = [
        Document(
            page_content=original_text,
            metadata={"source": "python_guide.txt", "version": "1.0"},
        )
    ]

    updated_docs = [
        Document(
            page_content=updated_text,
            metadata={"source": "python_guide.txt", "version": "2.0"},
        )
    ]

    # Split documents
    split_docs = kara_splitter.split_documents(original_docs)
    print(f"   Split documents: {len(split_docs)}")

    # Example 3: Full integration with vector store
    print("\n3. Full integration with vector store")
    print("-" * 50)

    updater = LangChainKARAUpdater(
        vectorstore_class=FAISS,
        embeddings=FakeEmbeddings(size=100),
        chunk_size=200,
        epsilon=0.1,
    )

    # Initialize
    init_result = updater.initialize_documents(original_docs)
    print(f"   Initialized: {init_result}")

    # Update
    update_result = updater.update_documents(updated_docs)
    print(f"   Updated: {update_result}")

    # Test similarity search
    vectorstore = updater.get_vectorstore()
    if vectorstore:
        results = updater.similarity_search("Python features", k=2)
        print(f"   Search results: {len(results)} documents")

    # Get efficiency stats from the text splitter
    text_splitter = updater.get_text_splitter()
    stats = text_splitter.get_efficiency_stats()
    print(f"   Efficiency stats: {stats}")

    print("\nBenefits of KARATextSplitter:")
    print("- Drop-in replacement for LangChain text splitters")
    print("- Maintains full LangChain compatibility")
    print("- Efficient chunk updates with KARA algorithm")
    print("- Easy integration with existing LangChain workflows")
    print("- Preserves document metadata and structure")


if __name__ == "__main__":
    main()
