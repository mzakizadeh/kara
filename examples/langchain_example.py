"""
LangChain integration example for kara-toolkit.

This example demonstrates how to use kara-toolkit with LangChain
for efficient RAG knowledge base updates using the KARATextSplitter.
"""

try:
    from langchain_core.documents.base import Document

    from kara.integrations.langchain import KARATextSplitter
except ImportError as e:
    raise ImportError(
        "LangChain is not installed. This module requires LangChain to be installed. "
        "Please install it with: pip install kara-toolkit[langchain]"
    ) from e


def main() -> None:
    """Demonstrate KARA's efficient document updating."""

    print("KARA Algorithm Efficiency Demo")
    print("=" * 30)

    # Initial document
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
        "simplicity. Python’s popularity in AI and data science makes it a natural fit for "
        "building with LangChain. Libraries like pydantic, asyncio, and openai integrate smoothly "
        "with LangChain, enabling developers to quickly build robust, scalable applications. "
        "Because LangChain supports modularity, developers can extend it using Python’s vast "
        "collection of libraries. Whether you're building an autonomous agent or a document QA "
        "tool, Python and LangChain together offer a powerful combination that lowers the barrier "
        "for building intelligent, interactive systems."
    )

    # Updated document (added content at the end)
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
        "simplicity. Python’s popularity in AI and data science makes it a natural fit for "
        "building with LangChain. Libraries like pydantic, asyncio, and openai integrate smoothly "
        "with LangChain, enabling developers to quickly build robust, scalable applications. "
        "Because LangChain supports modularity, developers can extend it using Python’s vast "
        "collection of libraries. Whether you're building an autonomous agent or a document QA "
        "tool, Python and LangChain together offer a powerful combination that lowers the barrier "
        "for building intelligent, interactive systems."
    )

    # Initialize KARA splitter
    splitter = KARATextSplitter(chunk_size=50, epsilon=0.1)

    # Process original document
    print("1. Processing original document:")
    original_chunks = splitter.split_text(original_doc)
    print(f"   Created {len(original_chunks)} chunks")
    for i, chunk in enumerate(original_chunks, 1):
        print(f"   Chunk {i}: '{chunk.strip()}'")

    # Process updated document (KARA reuses existing chunks)
    print("\n2. Processing updated document:")
    updated_chunks = splitter.split_text(updated_doc)
    print(f"   Result: {len(updated_chunks)} chunks")
    for i, chunk in enumerate(updated_chunks, 1):
        # Check if this chunk existed before
        is_reused = chunk in original_chunks
        status = " (reused)" if is_reused else " (new)"
        print(f"   Chunk {i}: '{chunk.strip()}'{status}")

    # Show efficiency
    reused_count = sum(1 for chunk in updated_chunks if chunk in original_chunks)
    efficiency = (reused_count / len(updated_chunks)) * 100
    print(
        f"\n3. Efficiency: {reused_count}/{len(updated_chunks)} chunks reused ({efficiency:.0f}%)"
    )

    # Demonstrate with LangChain Documents
    print("\n4. Using with LangChain Documents:")
    docs = [Document(page_content=updated_doc, metadata={"version": "2.0"})]
    chunked_docs = splitter.split_documents(docs)
    print(f"   Split into {len(chunked_docs)} Document objects")
    print(f"   Each preserves metadata: {chunked_docs[0].metadata}")

    print("\nKARA minimizes re-embedding by reusing unchanged chunks!")


if __name__ == "__main__":
    main()
