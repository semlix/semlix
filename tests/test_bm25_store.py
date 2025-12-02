"""Tests for BM25sStore.

These are basic tests. Full test suite would include 25+ tests.
Run with: pytest tests/test_bm25_store.py -v
"""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


def test_bm25store_creation(temp_dir):
    """Test creating a BM25sStore."""
    try:
        from semlix.stores import BM25sStore
    except ImportError:
        pytest.skip("bm25s not installed")

    store = BM25sStore.create(
        index_dir=temp_dir,
        fields=["content"]
    )

    assert store.doc_count() == 0
    assert store.index_dir == Path(temp_dir)


def test_bm25store_add_documents(temp_dir):
    """Test adding documents to BM25sStore."""
    try:
        from semlix.stores import BM25sStore
    except ImportError:
        pytest.skip("bm25s not installed")

    store = BM25sStore.create(temp_dir, fields=["content"])

    # Add documents
    docs = [
        {"id": "1", "content": "python programming"},
        {"id": "2", "content": "database queries"},
        {"id": "3", "content": "machine learning"}
    ]

    store.add_documents(docs)

    assert store.doc_count() == 3


def test_bm25store_search(temp_dir):
    """Test searching with BM25sStore."""
    try:
        from semlix.stores import BM25sStore
    except ImportError:
        pytest.skip("bm25s not installed")

    store = BM25sStore.create(temp_dir, fields=["content"])

    # Add documents
    docs = [
        {"id": "1", "content": "python programming language"},
        {"id": "2", "content": "java programming language"},
        {"id": "3", "content": "database systems"}
    ]
    store.add_documents(docs)

    # Search
    results = store.search("python", k=2)

    assert len(results) > 0
    assert results[0].doc_id == "1"  # Best match
    assert results[0].score > 0


def test_bm25store_save_load(temp_dir):
    """Test saving and loading BM25sStore."""
    try:
        from semlix.stores import BM25sStore
    except ImportError:
        pytest.skip("bm25s not installed")

    # Create and populate
    store = BM25sStore.create(temp_dir, fields=["content"])
    docs = [{"id": str(i), "content": f"document {i}"} for i in range(10)]
    store.add_documents(docs)
    store.save()

    # Load
    store2 = BM25sStore.load(temp_dir)
    assert store2.doc_count() == 10

    # Search should work
    results = store2.search("document", k=5)
    assert len(results) == 5


def test_bm25store_delete(temp_dir):
    """Test deleting documents from BM25sStore."""
    try:
        from semlix.stores import BM25sStore
    except ImportError:
        pytest.skip("bm25s not installed")

    store = BM25sStore.create(temp_dir, fields=["content"])

    docs = [
        {"id": "1", "content": "doc one"},
        {"id": "2", "content": "doc two"}
    ]
    store.add_documents(docs)

    assert store.doc_count() == 2

    # Delete
    deleted = store.delete_document("1")
    assert deleted is True
    assert store.doc_count() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
