"""
Basic usage example of kara-toolkit.

This example demonstrates the core functionality of the KARA algorithm
for efficient document chunk updates using the KARAUpdater class.
"""

from kara import KARAUpdater, RecursiveCharacterChunker


def main() -> None:
    """Demonstrate KARA's efficient document updating."""

    print("KARA Algorithm Efficiency Demo")
    print("=" * 30)

    # Initial document (same as LangChain example)
    original_doc = (
        "LangChain is an open-source framework built in Python that helps developers create "
        "applications powered by large language models (LLMs). It allows seamless integration "
        "between LLMs and external data sources like APIs, files, and databases. With LangChain, "
        "developers can build dynamic workflows where a language model not only generates text but "
        "also interacts with tools and environments. This makes it ideal for creating advanced "
        "chatbots, agents, and AI systems that go beyond static prompting. LangChain provides both "
        "low-level components for custom logic and high-level abstractions for rapid prototyping, "
        "making it a versatile toolkit for AI application development.\n\n"
        "Python is the primary language used with LangChain due to its rich ecosystem and "
        "simplicity. Python's popularity in AI and data science makes it a natural fit for "
        "building with LangChain. Libraries like pydantic, asyncio, and openai integrate smoothly "
        "with LangChain, enabling developers to quickly build robust, scalable applications. "
        "Because LangChain supports modularity, developers can extend it using Python's vast "
        "collection of libraries. Whether you're building an autonomous agent or a document QA "
        "tool, Python and LangChain together offer a powerful combination that lowers the barrier "
        "for building intelligent, interactive systems."
    )

    # Updated document (added content in the middle)
    updated_doc = (
        "LangChain is an open-source framework built in Python that helps developers create "
        "applications powered by large language models (LLMs). It allows seamless integration "
        "between LLMs and external data sources like APIs, files, and databases. With LangChain, "
        "developers can build dynamic workflows where a language model not only generates text but "
        "also interacts with tools and environments. Developers can define step-by-step workflows "
        "in which an LLM can retrieve data, call APIs, and act based on context. This flexibility "
        "allows LangChain to support everything from basic assistants to complex, multi-step "
        "agents capable of reasoning and memory retention.\n\n"
        "Python is the primary language used with LangChain due to its rich ecosystem and "
        "simplicity. Python's popularity in AI and data science makes it a natural fit for "
        "building with LangChain. Libraries like pydantic, asyncio, and openai integrate smoothly "
        "with LangChain, enabling developers to quickly build robust, scalable applications. "
        "Because LangChain supports modularity, developers can extend it using Python's vast "
        "collection of libraries. Whether you're building an autonomous agent or a document QA "
        "tool, Python and LangChain together offer a powerful combination that lowers the barrier "
        "for building intelligent, interactive systems."
    )

    # Initialize KARA with character-based chunking
    chunker = RecursiveCharacterChunker(chunk_size=300, separators=["\n\n", "\n", " "])
    updater = KARAUpdater(chunker=chunker, epsilon=0.1)

    # Process original document
    print("1. Processing original document:")
    initial_result = updater.create_knowledge_base([original_doc])
    assert initial_result.new_chunked_doc is not None
    original_chunks = [chunk.content for chunk in initial_result.new_chunked_doc.chunks]
    print(f"   Created {len(original_chunks)} chunks")
    for i, chunk in enumerate(original_chunks, 1):
        print(f"   Chunk {i}: '{chunk.strip()[:60]}...'")

    # Process updated document (KARA reuses existing chunks)
    print("\n2. Processing updated document:")
    result = updater.update_knowledge_base(initial_result.new_chunked_doc, [updated_doc])
    assert result.new_chunked_doc is not None
    updated_chunks = [chunk.content for chunk in result.new_chunked_doc.chunks]
    print(f"   Result: {len(updated_chunks)} chunks")
    for i, chunk in enumerate(updated_chunks, 1):
        # Check if this chunk existed before
        is_reused = chunk in original_chunks
        status = " (reused)" if is_reused else " (new)"
        print(f"   Chunk {i}: '{chunk.strip()[:60]}...'{status}")

    # Show efficiency
    reused_count = result.num_reused
    total_chunks = len(updated_chunks)
    efficiency_pct = result.efficiency_ratio
    print(f"\n3. Efficiency: {reused_count}/{total_chunks} chunks reused ({efficiency_pct:.0%})")
    print(f"   - Chunks added: {result.num_added}")
    print(f"   - Chunks reused: {result.num_reused}")
    print(f"   - Chunks deleted: {result.num_deleted}")

    print("\n4. KARA Benefits:")
    print("   - Minimizes re-embedding operations")
    print("   - Preserves existing chunk IDs for vector databases")
    print("   - Efficient for incremental document updates")
    print("   - Works with any chunking strategy")

    print("\n5. Multi-Document Support:")
    # Multiple documents with some overlap
    doc1 = (
        "LangChain is an open-source framework built in Python that helps developers create "
        "applications powered by large language models (LLMs). It provides tools for chaining "
        "together different components to build complex AI applications."
    )

    doc2 = (
        "Python is the primary language used with LangChain due to its rich ecosystem and "
        "simplicity. The language's extensive libraries make it ideal for AI development."
    )

    doc3 = (
        "Vector databases are essential for RAG applications. They store embeddings and enable "
        "semantic search capabilities for retrieving relevant context."
    )

    print("   Processing multiple documents:")
    multi_result = updater.create_knowledge_base([doc1, doc2, doc3])
    assert multi_result.new_chunked_doc is not None
    multi_chunks = [chunk.content for chunk in multi_result.new_chunked_doc.chunks]
    print(f"   Created {len(multi_chunks)} chunks from {3} documents")

    # Update with modified documents
    doc1_updated = doc1 + " It supports various integrations and extensions."
    doc2_updated = doc2  # Unchanged
    doc4_new = "Embeddings convert text into numerical vectors that capture semantic meaning."

    print("   Updating with modified and new documents:")
    multi_update_result = updater.update_knowledge_base(
        multi_result.new_chunked_doc, [doc1_updated, doc2_updated, doc4_new]
    )
    assert multi_update_result.new_chunked_doc is not None

    print("   - Original documents: 3")
    print("   - Updated documents: 3 (1 modified, 1 unchanged, 1 new)")
    print(f"   - Chunks reused: {multi_update_result.num_reused}")
    print(f"   - Chunks added: {multi_update_result.num_added}")
    print(f"   - Chunks deleted: {multi_update_result.num_deleted}")
    print(f"   - Multi-doc efficiency: {multi_update_result.efficiency_ratio:.0%}")


if __name__ == "__main__":
    main()
