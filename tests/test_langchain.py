from typing import Any

import pytest

from kara.chunkers import CharacterChunker, OpenAITokenChunker
from kara.integrations.langchain import KARATextSplitter


def test_kara_text_splitter_basic() -> None:
    """Test basic functionality of KARATextSplitter."""
    text = "This is a test sentence. This is another test sentence."
    chunker = CharacterChunker(chunk_size=50, overlap=0, separators=[". "])
    splitter = KARATextSplitter(chunker=chunker)

    # First split
    chunks = splitter.split_text(text)
    assert len(chunks) > 0
    assert "This is a test sentence." in chunks[0]

    # Update split
    updated_text = "This is a test sentence. This is a NEW test sentence."
    updated_chunks = splitter.split_text(updated_text)
    assert len(updated_chunks) > 0
    assert "This is a NEW test sentence." in updated_chunks[1]


def test_kara_text_splitter_with_custom_chunker() -> None:
    """Test KARATextSplitter with a custom chunker."""
    custom_chunker = CharacterChunker(chunk_size=10, separators=[" "])
    splitter = KARATextSplitter(chunker=custom_chunker)

    assert splitter._kara_chunker == custom_chunker

    text = "Word word word"
    chunks = splitter.split_text(text)
    assert len(chunks) > 1


def test_kara_text_splitter_from_tiktoken() -> None:
    """Test creating KARATextSplitter from tiktoken encoder."""
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        pytest.skip("tiktoken not installed")

    splitter = KARATextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=10, chunk_overlap=2
    )

    assert isinstance(splitter._kara_chunker, OpenAITokenChunker)
    assert splitter._kara_chunker.chunk_size == 10
    assert splitter._kara_chunker.overlap == 2

    text = "This is a test sentence for tokenization."
    chunks = splitter.split_text(text)
    assert len(chunks) > 0


def test_kara_text_splitter_from_huggingface(mocker: Any) -> None:
    """Test creating KARATextSplitter from Hugging Face tokenizer with mocking."""
    try:
        from transformers import AutoTokenizer  # noqa: F401
    except ImportError:
        pytest.skip("transformers not installed")

    import sys
    from unittest.mock import MagicMock

    # Mock transformers module if not present
    mock_transformers = MagicMock()
    sys.modules["transformers"] = mock_transformers

    mock_instance = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_instance
    mock_instance.encode.return_value = [1, 2, 3]
    mock_instance.decode.side_effect = lambda x, **kwargs: f"token_{x[0]}"

    splitter = KARATextSplitter.from_huggingface_tokenizer(
        model_name="mock-model", chunk_size=10, chunk_overlap=2
    )

    from kara.chunkers import HuggingFaceTokenChunker

    assert isinstance(splitter._kara_chunker, HuggingFaceTokenChunker)
    assert splitter._kara_chunker.chunk_size == 10
    assert splitter._kara_chunker.overlap == 2

    text = "mock text"
    chunks = splitter.split_text(text)
    assert len(chunks) > 0

    # Cleanup
    del sys.modules["transformers"]


def test_kara_text_splitter_previous_chunks() -> None:
    """Test KARATextSplitter with previous chunks."""
    previous_chunks = ["Chunk 1", "Chunk 2"]
    chunker = CharacterChunker(chunk_size=10)
    with pytest.warns(UserWarning, match="If the separator list is not the same"):
        splitter = KARATextSplitter(chunker=chunker, previous_chunks=previous_chunks)

    assert splitter._current_knowledge_base is not None
    assert len(splitter._current_knowledge_base.chunks) == 2

    # New text
    text = "Chunk 1 and some new content"
    chunks = splitter.split_text(text)
    assert len(chunks) > 0
    # Should reuse Chunk 1 if it matches exactly (depends on chunker)
