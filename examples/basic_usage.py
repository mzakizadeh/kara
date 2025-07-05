"""
Basic usage example of kara-python.

This example demonstrates the core functionality of the KARA algorithm
for efficient document chunk updates.
"""

from kara import KARAUpdater, RecursiveTextSplitter


def main():
    """Demonstrate basic kara-python usage."""
    print("kara-python Basic Usage Example")
    print("=" * 40)

    # Create text splitter
    splitter = RecursiveTextSplitter(
        chunk_size=100, chunk_overlap=20, separators=["\n\n", "\n", " ", ""]
    )

    # Initialize KARA updater
    updater = KARAUpdater(
        splitter=splitter,
        epsilon=0.1,  # Cost factor for reusing chunks
    )

    # Initial documents
    initial_docs = [
        "The quick brown fox jumps over the lazy dog. This is a sample document.",
        "Python is a programming language. It is widely used for data science.",
        "Machine learning is a subset of artificial intelligence. It learns from data.",
    ]

    print(f"Processing {len(initial_docs)} initial documents...")

    # Initialize with documents
    initial_chunks = updater.initialize(initial_docs)
    print(f"Created {len(initial_chunks)} initial chunks")

    # Updated documents (simulating document changes)
    updated_docs = [
        "The quick brown fox jumps over the lazy dog. This is a sample document with updates.",
        "Python is a programming language. It is widely used for data science and web development.",
        "Machine learning is a subset of artificial intelligence. ",
        "It learns from data to make predictions.",
    ]

    print(f"\nUpdating with {len(updated_docs)} modified documents...")

    # Update with modified documents
    update_result = updater.update(updated_docs)

    # Display results
    print("\nUpdate Results:")
    print(f"  Added: {update_result.num_added} chunks")
    print(f"  Reused: {update_result.num_skipped} chunks")
    print(f"  Deleted: {update_result.num_deleted} chunks")
    print(f"  Efficiency: {update_result.efficiency_ratio:.1%}")

    # Show chunk details
    print(f"\nFinal chunks ({len(update_result.chunks)}):")
    for i, chunk in enumerate(update_result.chunks):
        print(f"  {i + 1}. {chunk.content[:50]}...")

    # Demonstrate the savings
    total_chunks = len(update_result.chunks)
    embedding_operations = update_result.num_added
    print("\nEfficiency Summary:")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Embedding operations needed: {embedding_operations}")
    print(f"  Operations saved: {total_chunks - embedding_operations}")
    print(f"  Efficiency ratio: {update_result.efficiency_ratio:.1%}")


if __name__ == "__main__":
    main()
