"""
Automated unit and integration tests for RAG Chatbot module.
Tests real ChromaDB vector database connection, collection count,
BAAI/bge-small-en-v1.5 retrieval, grounded answer generation, and citations.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chatbot.rag_chatbot import CoffeeRAGChatbot, resolve_vector_db_path


def test_vector_db_path_and_collection():
    """Verify that vector_db exists and contains the expected Chroma database."""
    db_path = resolve_vector_db_path()
    assert db_path.is_dir(), f"Vector DB directory not found at {db_path}"
    assert (db_path / "chroma.sqlite3").is_file(), "chroma.sqlite3 database file missing"


def test_rag_chatbot_initialization():
    """Verify CoffeeRAGChatbot loads client, collection, and embedding model."""
    bot = CoffeeRAGChatbot()
    assert bot.is_ready() is True, "RAG Chatbot is not ready"
    assert bot.collection is not None
    count = bot.collection.count()
    assert count >= 600, f"Expected at least 600 chunks, found {count}"


def test_real_retrieval_coffee_rust():
    """Verify real vector retrieval against coffee leaf rust query."""
    bot = CoffeeRAGChatbot()
    chunks = bot.retrieve_chunks("What causes coffee leaf rust and what are its symptoms?", k=3)
    assert len(chunks) > 0, "No chunks retrieved for valid coffee query"
    assert "text" in chunks[0]
    assert "source" in chunks[0]
    assert len(chunks[0]["text"]) > 20

    # Verify query relevance - at least one chunk should mention rust, disease, or fungus
    combined = " ".join([c["text"].lower() for c in chunks])
    assert any(w in combined for w in ["rust", "hemileia", "fungus", "disease", "coffee"])


def test_grounded_answer_generation():
    """Verify grounded answer synthesis includes explanation and agricultural advisory."""
    bot = CoffeeRAGChatbot()
    query = "How to prevent and manage Coffee Berry Disease in Ethiopia?"
    res = bot.answer_question(query, lang="en")

    assert "answer" in res
    assert "sources" in res
    assert len(res["sources"]) > 0
    assert len(res["answer"]) > 50

    # Verify responsible advisory statement is included
    assert "advisory" in res["answer"].lower() or "extension" in res["answer"].lower()


def test_leaf_context_integration():
    """Verify that providing computer vision context guides retrieval and answer."""
    bot = CoffeeRAGChatbot()
    fake_leaf_context = {
        "success": True,
        "predicted_class": "Rust",
        "confidence": 94.5,
        "display_name": "Coffee Leaf Rust"
    }

    res = bot.answer_question(
        question="What treatments are recommended for my affected plant?",
        leaf_context=fake_leaf_context,
        lang="en"
    )

    assert res["leaf_context"] is not None
    assert "Rust" in res["answer"]
    assert len(res["sources"]) > 0
