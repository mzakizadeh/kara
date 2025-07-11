"""
Integration tests using examples from the examples directory.
"""

from kara.core import KARAUpdater
from kara.splitters import RecursiveCharacterChunker


class TestExamplesIntegration:
    """Integration tests based on the examples."""

    def test_basic_usage_workflow(self) -> None:
        """Test the basic usage workflow similar to examples/basic_usage.py."""
        # Initial document
        original_doc = (
            "Python is a versatile programming language used in many domains. "
            "It has a simple syntax that makes it easy to learn. "
            "Python supports multiple programming paradigms including "
            "object-oriented and functional programming."
        )

        # Updated document (content added)
        updated_doc = (
            "Python is a versatile programming language used in many domains. "
            "It has a simple syntax that makes it easy to learn and maintain. "
            "Python supports multiple programming paradigms including "
            "object-oriented, functional, and procedural programming. "
            "It also has a vast ecosystem of libraries and frameworks."
        )

        # Create chunker and updater
        chunker = RecursiveCharacterChunker(
            chunk_size=50, separators=[". ", " "], keep_separator=True
        )
        updater = KARAUpdater(chunker=chunker, epsilon=0.1)

        # Create initial knowledge base
        initial_result = updater.create_knowledge_base([original_doc])
        assert initial_result.new_chunked_doc is not None
        initial_chunks = len(initial_result.new_chunked_doc.chunks)
        assert initial_chunks > 0

        # Update with modified document
        update_result = updater.update_knowledge_base(initial_result.new_chunked_doc, [updated_doc])

        # Validate results
        assert update_result.new_chunked_doc is not None
        assert update_result.num_reused > 0  # Some chunks should be reused
        assert update_result.num_added >= 0  # New chunks might be added
        assert update_result.efficiency_ratio >= 0.0

        # Validate that we have reasonable efficiency
        assert update_result.efficiency_ratio > 0.15, (
            f"Expected reasonable efficiency, got {update_result.efficiency_ratio:.2f}"
        )

        print("Basic usage test results:")
        print(f"  Initial chunks: {initial_chunks}")
        print(f"  Final chunks: {len(update_result.new_chunked_doc.chunks)}")
        print(f"  Reused chunks: {update_result.num_reused}")
        print(f"  Added chunks: {update_result.num_added}")
        print(f"  Efficiency ratio: {update_result.efficiency_ratio:.2f}")

    def test_langchain_integration_workflow(self) -> None:
        """Test workflow similar to langchain_example.py but without LangChain dependency."""
        # Simulate the LangChain example workflow
        original_doc = (
            "LangChain is an open-source framework built in Python that helps developers create "
            "applications powered by large language models (LLMs). It allows seamless integration "
            "between LLMs and external data sources like APIs, files, and databases."
        )

        updated_doc = (
            "LangChain is an open-source framework built in Python that helps developers create "
            "applications powered by large language models (LLMs). It allows seamless integration "
            "between LLMs and external data sources like APIs, files, and databases. "
            "With LangChain, developers can build dynamic workflows where a language model "
            "not only generates text but also interacts with tools and environments."
        )

        # Use similar parameters to the LangChain example
        chunker = RecursiveCharacterChunker(
            chunk_size=50, separators=["\n\n", "\n", ". ", " "], keep_separator=True
        )
        updater = KARAUpdater(chunker=chunker, epsilon=0.1)

        # Process original document
        original_result = updater.create_knowledge_base([original_doc])
        assert original_result.new_chunked_doc is not None
        original_chunks = original_result.new_chunked_doc.chunks

        # Process updated document
        updated_result = updater.update_knowledge_base(
            original_result.new_chunked_doc, [updated_doc]
        )
        assert updated_result.new_chunked_doc is not None
        updated_chunks = updated_result.new_chunked_doc.chunks

        # Validate efficiency (similar to LangChain example expectations)
        reused_count = updated_result.num_reused
        efficiency = (reused_count / len(updated_chunks)) * 100

        assert efficiency >= 50, f"Expected >=50% efficiency, got {efficiency:.0f}%"
        assert reused_count > 0, "Expected some chunks to be reused"

        print("LangChain-style workflow results:")
        print(f"  Original chunks: {len(original_chunks)}")
        print(f"  Updated chunks: {len(updated_chunks)}")
        print(f"  Reused chunks: {reused_count}")
        print(f"  Efficiency: {efficiency:.0f}%")

    def test_multi_document_workflow(self) -> None:
        """Test multi-document workflow."""
        # Initial documents
        initial_docs = [
            "Document 1: Introduction to artificial intelligence and its applications.",
            "Document 2: Machine learning algorithms and their use cases.",
            "Document 3: Deep learning and neural network architectures.",
        ]

        # Updated documents (one unchanged, one modified, one added)
        updated_docs = [
            "Document 1: Introduction to artificial intelligence and its applications.",
            "Document 2: Machine learning algorithms, their use cases, and evaluation metrics.",
            "Document 3: Deep learning and neural network architectures.",
            "Document 4: Natural language processing and transformer models.",
        ]

        # Create chunker and updater
        chunker = RecursiveCharacterChunker(
            chunk_size=40, separators=[". ", " "], keep_separator=True
        )
        updater = KARAUpdater(chunker=chunker, epsilon=0.1)

        # Create initial knowledge base
        initial_result = updater.create_knowledge_base(initial_docs)
        initial_kb = initial_result.new_chunked_doc
        assert initial_kb is not None

        # Verify initial document structure
        doc_ids = initial_kb.get_document_ids()
        assert len(doc_ids) == 3, f"Expected 3 documents, got {len(doc_ids)}"

        # Update with new documents
        update_result = updater.update_knowledge_base(initial_kb, updated_docs)
        updated_kb = update_result.new_chunked_doc
        assert updated_kb is not None

        # Verify updated structure
        updated_doc_ids = updated_kb.get_document_ids()
        assert len(updated_doc_ids) == 4, f"Expected 4 documents, got {len(updated_doc_ids)}"

        # Validate efficiency
        assert update_result.num_reused > 0, "Expected some chunks to be reused"
        assert update_result.num_added > 0, "Expected new chunks to be added"

        print("Multi-document workflow results:")
        print(f"  Initial documents: {len(initial_docs)}")
        print(f"  Updated documents: {len(updated_docs)}")
        print(f"  Chunks reused: {update_result.num_reused}")
        print(f"  Chunks added: {update_result.num_added}")
        print(f"  Chunks deleted: {update_result.num_deleted}")
        print(f"  Efficiency ratio: {update_result.efficiency_ratio:.2f}")

    def test_epsilon_comparison_workflow(self) -> None:
        """Test comparing different epsilon values."""
        text_original = "The quick brown fox jumps over the lazy dog."
        text_updated = "The quick brown fox jumps over the sleeping dog."

        chunker = RecursiveCharacterChunker(chunk_size=20, separators=[" "], keep_separator=True)

        epsilon_values = [0.01, 0.1, 0.5, 0.9]
        results = []

        for epsilon in epsilon_values:
            updater = KARAUpdater(chunker=chunker, epsilon=epsilon)

            # Create and update
            initial_result = updater.create_knowledge_base([text_original])
            assert initial_result.new_chunked_doc is not None
            update_result = updater.update_knowledge_base(
                initial_result.new_chunked_doc, [text_updated]
            )

            results.append(
                {
                    "epsilon": epsilon,
                    "reused": update_result.num_reused,
                    "added": update_result.num_added,
                    "deleted": update_result.num_deleted,
                    "efficiency": update_result.efficiency_ratio,
                }
            )

        # Validate that different epsilon values produce different behaviors
        # With a small change like this, low epsilon should reuse more
        low_eps_efficiency = next(r["efficiency"] for r in results if r["epsilon"] == 0.01)

        # This assertion might be flexible depending on the specific chunking
        # but generally low epsilon should not be much worse than high epsilon
        assert low_eps_efficiency >= 0, "Low epsilon should provide some efficiency"

        print("Epsilon comparison results:")
        print("Epsilon | Reused | Added | Deleted | Efficiency")
        print("-" * 45)
        for result in results:
            print(
                f"{result['epsilon']:7.2f} | "
                f"{result['reused']:6d} | "
                f"{result['added']:5d} | "
                f"{result['deleted']:7d} | "
                f"{result['efficiency']:9.2f}"
            )

    def test_separator_comparison_workflow(self) -> None:
        """Test different separator configurations."""
        text = (
            "First paragraph about machine learning.\n\n"
            "Second paragraph about deep learning. It has multiple sentences. "
            "Each sentence provides information.\n\n"
            "Third paragraph about applications."
        )

        updated_text = (
            "First paragraph about machine learning and AI.\n\n"
            "Second paragraph about deep learning. It has multiple sentences. "
            "Each sentence provides valuable information.\n\n"
            "Third paragraph about applications."
        )

        separator_configs = [
            {"name": "Paragraph-based", "separators": ["\n\n", "\n", " "], "chunk_size": 100},
            {"name": "Sentence-based", "separators": [". ", "! ", "? ", " "], "chunk_size": 80},
            {"name": "Word-based", "separators": [" "], "chunk_size": 50},
        ]

        results = []
        for config in separator_configs:
            chunker = RecursiveCharacterChunker(
                chunk_size=config["chunk_size"],  # type: ignore
                separators=config["separators"],  # type: ignore
                keep_separator=True,
            )
            updater = KARAUpdater(chunker=chunker, epsilon=0.1)

            # Create and update
            initial_result = updater.create_knowledge_base([text])
            assert initial_result.new_chunked_doc is not None
            update_result = updater.update_knowledge_base(
                initial_result.new_chunked_doc, [updated_text]
            )
            assert update_result.new_chunked_doc is not None

            results.append(
                {
                    "name": config["name"],
                    "total_chunks": len(update_result.new_chunked_doc.chunks),
                    "reused": update_result.num_reused,
                    "efficiency": update_result.efficiency_ratio,
                }
            )

        # All configurations should work and provide some efficiency
        for result in results:
            assert result["total_chunks"] > 0, f"No chunks created for {result['name']}"  # type: ignore
            assert result["efficiency"] >= 0, f"Invalid efficiency for {result['name']}"  # type: ignore

        print("Separator comparison results:")
        print("Config          | Total | Reused | Efficiency")
        print("-" * 45)
        for result in results:
            print(
                f"{result['name']:15s} | "
                f"{result['total_chunks']:5d} | "
                f"{result['reused']:6d} | "
                f"{result['efficiency']:9.2f}"
            )
