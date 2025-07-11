"""
Basic usage example of kara-toolkit.

This example demonstrates the core functionality of the KARA algorithm
for efficient document chunk updates, including both character-based
and token-based chunking strategies.
"""

from kara import KARAUpdater, RecursiveCharacterChunker, RecursiveTokenChunker


def demo_character_chunking() -> None:
    """Demonstrate character-based chunking with KARA."""
    print("=== Character-based Chunking ===")

    # Create document chunker
    chunker = RecursiveCharacterChunker(chunk_size=100, separators=["\n\n", "\n", " "])

    # Initialize KARA updater
    updater = KARAUpdater(
        chunker=chunker,
        epsilon=0.1,  # Cost factor for reusing chunks
    )

    # Sample document
    original_text = """This is a sample document for the KARA algorithm demonstration.
It contains multiple sentences and paragraphs to show how the chunking works.

The KARA algorithm efficiently updates document chunks by reusing existing chunks
when possible, which minimizes the need for re-embedding operations.

This makes it particularly useful for RAG (Retrieval-Augmented Generation) systems
where documents are frequently updated."""

    print(f"Original text length: {len(original_text)} characters")

    # Initialize with the original document
    initial_result = updater.create_knowledge_base([original_text])
    assert initial_result.new_chunked_doc is not None
    initial_chunks = [chunk.content for chunk in initial_result.new_chunked_doc.chunks]
    print(f"Initial chunks created: {len(initial_chunks)}")

    for i, chunk in enumerate(initial_chunks, 1):
        print(f"Chunk {i} ({len(chunk)} chars): {chunk[:50]}...")

    # Update with modified document
    updated_text = (
        original_text
        + """

This is a new paragraph added to the document.
The KARA algorithm will efficiently handle this update by reusing existing chunks
and only creating new chunks for the added content."""
    )

    print(f"\nUpdated text length: {len(updated_text)} characters")

    result = updater.update_knowledge_base(initial_result.new_chunked_doc, [updated_text])
    print("\nUpdate results:")
    print(f"- Chunks added: {result.num_added}")
    print(f"- Chunks reused: {result.num_reused}")
    print(f"- Chunks deleted: {result.num_deleted}")
    print(f"- Efficiency ratio: {result.efficiency_ratio:.2%}")


def demo_token_chunking() -> None:
    """Demonstrate token-based chunking with KARA."""
    print("\n=== Token-based Chunking ===")

    # Create token-based chunker
    chunker = RecursiveTokenChunker(
        chunk_size=25,  # 25 tokens per chunk
        overlap=5,
        tokenizer_function=lambda text: text.split(),  # Simple whitespace tokenizer
    )

    # Initialize KARA updater
    updater = KARAUpdater(
        chunker=chunker,
        epsilon=0.1,
    )

    # Sample document with technical content
    original_text = """Machine learning algorithms have revolutionized the field of artificial
intelligence. These algorithms can learn patterns from data without being explicitly programmed for
every scenario. Deep learning, a subset of machine learning, uses neural networks with multiple
layers toprocess information. Natural language processing combines computational linguistics with
machine learning techniques."""

    print(f"Original text: {original_text}")
    token_count = len(original_text.split())
    print(f"Token count: {token_count}")

    # Initialize with the original document
    initial_result = updater.create_knowledge_base([original_text])
    assert initial_result.new_chunked_doc is not None
    initial_chunks = [chunk.content for chunk in initial_result.new_chunked_doc.chunks]
    print(f"Initial chunks created: {len(initial_chunks)}")

    for i, chunk in enumerate(initial_chunks, 1):
        chunk_tokens = len(chunk.split())
        print(f"Chunk {i} ({chunk_tokens} tokens): {chunk}")

    # Update with additional content
    updated_text = (
        original_text
        + """ Computer vision is another important field that enables machines to interpret visual
information. Reinforcement learning teaches agents to make decisions through trial and error
interactions with environments."""
    )

    print(f"\nUpdated text: {updated_text}")
    updated_token_count = len(updated_text.split())
    print(f"Updated token count: {updated_token_count}")

    result = updater.update_knowledge_base(initial_result.new_chunked_doc, [updated_text])
    print("\nUpdate results:")
    print(f"- Chunks added: {result.num_added}")
    print(f"- Chunks reused: {result.num_reused}")
    print(f"- Chunks deleted: {result.num_deleted}")
    print(f"- Efficiency ratio: {result.efficiency_ratio:.2%}")


def demo_custom_chunker() -> None:
    """Demonstrate creating a custom sentence-based chunker."""
    print("\n=== Custom Sentence-based Chunker ===")

    from typing import List

    from kara import BaseDocumentChunker

    class SentenceChunker(BaseDocumentChunker):
        """Custom chunker that groups sentences together."""

        def create_chunks(self, text: str) -> List[str]:
            """Create chunks by grouping sentences."""
            sentences = self._split_to_units(text)
            chunks = self._merge_units_greedy(sentences, self.chunk_size)
            return chunks

        def _split_to_units(self, text: str) -> List[str]:
            """Split text into sentences."""
            import re

            sentences = re.split(r"[.!?]+", text)
            return [s.strip() + "." for s in sentences if s.strip()]

    # Create custom chunker (group 2 sentences per chunk)
    chunker = SentenceChunker(chunk_size=2)
    updater = KARAUpdater(chunker=chunker, epsilon=0.1)

    text = """Artificial intelligence is transforming many industries. Machine learning enables
computers to learn from data. Natural language processing helps machines understand human language.
Computer vision allows machines to interpret visual information. These technologies are creating new
possibilities for automation and innovation."""

    print(f"Original text: {text}")

    result = updater.create_knowledge_base([text])
    assert result.new_chunked_doc is not None
    chunks = [chunk.content for chunk in result.new_chunked_doc.chunks]
    print(f"Sentence-based chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk}")


def demo_kara_with_different_chunkers() -> None:
    """Demonstrate using KARA with different chunkers for comparison."""
    print("\n=== KARA with Different Chunkers Comparison ===")

    # Character-based KARA
    char_chunker = RecursiveCharacterChunker(chunk_size=50)
    char_updater = KARAUpdater(chunker=char_chunker, epsilon=0.1)

    # Token-based KARA
    token_chunker = RecursiveTokenChunker(
        chunk_size=10, tokenizer_function=lambda text: text.split()
    )
    token_updater = KARAUpdater(chunker=token_chunker, epsilon=0.1)

    original_text = "This is the original document. It has some content."
    updated_text = "This is the original document. It has some content. This is new content added."

    print(f"Original text: {original_text}")
    print(f"Updated text: {updated_text}")

    print("\nCharacter-based KARA:")
    char_result_initial = char_updater.create_knowledge_base([original_text])
    assert char_result_initial.new_chunked_doc is not None
    char_chunks = [chunk.content for chunk in char_result_initial.new_chunked_doc.chunks]
    print(f"Initial chunks: {len(char_chunks)}")

    char_result = char_updater.update_knowledge_base(
        char_result_initial.new_chunked_doc, [updated_text]
    )
    print(
        f"Update result: {char_result.num_added} added, "
        f"{char_result.num_reused} skipped, {char_result.num_deleted} deleted"
    )

    print("\nToken-based KARA:")
    token_result_initial = token_updater.create_knowledge_base([original_text])
    assert token_result_initial.new_chunked_doc is not None
    token_chunks = [chunk.content for chunk in token_result_initial.new_chunked_doc.chunks]
    print(f"Initial chunks: {len(token_chunks)}")

    token_result = token_updater.update_knowledge_base(
        token_result_initial.new_chunked_doc, [updated_text]
    )
    print(
        f"Update result: {token_result.num_added} added, "
        f"{token_result.num_reused} skipped, {token_result.num_deleted} deleted"
    )


def demo_advanced_chunking_strategies() -> None:
    """Demonstrate advanced chunking strategies and the unified approach."""
    print("\n=== Advanced Chunking Strategies ===")

    # Demonstrate recursive character chunking with different separators
    print("1. Recursive Character Chunking with Custom Separators")
    chunker = RecursiveCharacterChunker(
        chunk_size=100, separators=["\n\n", "\n", " "], keep_separator=True
    )

    text = """This is a sample document.
It has multiple paragraphs.

This is the second paragraph.
It demonstrates how the chunker works with different separators.

The chunker will try to split at paragraph boundaries first,
then at sentence boundaries, then at word boundaries."""

    chunks = chunker.create_chunks(text)
    print(f"Text length: {len(text)} characters, Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i} ({len(chunk)} chars): {chunk[:50]}...")

    # Demonstrate token-based chunking
    print("\n2. Token-based Chunking with Custom Tokenizer")
    token_chunker = RecursiveTokenChunker(
        chunk_size=15, tokenizer_function=lambda text: text.split()
    )

    text = """Natural language processing is a field that focuses on making human language
comprehensible to computers. It involves computational techniques for analyzing and
representing naturally occurring texts at one or more levels of linguistic analysis
for the purpose of achieving human-like language processing for a range of tasks or applications."""

    chunks = token_chunker.create_chunks(text)
    print(f"Text: {text}")
    print(f"Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        token_count = len(chunk.split())
        print(f"  Chunk {i} ({token_count} tokens): {chunk}")


def main() -> None:
    """Demonstrate comprehensive kara-toolkit usage with different chunking strategies."""
    print("kara-toolkit Comprehensive Usage Example")
    print("=" * 50)

    demo_character_chunking()
    demo_token_chunking()
    demo_custom_chunker()
    demo_kara_with_different_chunkers()
    demo_advanced_chunking_strategies()

    print("\n" + "=" * 50)
    print("Summary:")
    print("- Character-based chunking: Good for preserving text structure")
    print("- Token-based chunking: Good for language model compatibility")
    print("- Custom chunkers: Flexible for domain-specific requirements")
    print("- KARA algorithm: Efficient updates with chunk reuse")
    print("- Unified approach: Single extensible interface for all chunking strategies")


if __name__ == "__main__":
    main()
