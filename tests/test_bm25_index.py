"""Tests for BM25Index.

These are basic integration tests for the Index protocol implementation.
Run with: pytest tests/test_bm25_index.py -v
"""

import pytest
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


def test_bm25index_creation(temp_dir):
    """Test creating a BM25Index."""
    try:
        from semlix.bm25 import create_bm25_index
        from semlix.fields import Schema, TEXT, ID
    except ImportError:
        pytest.skip("bm25s not installed")

    schema = Schema(id=ID(stored=True), content=TEXT(stored=True))
    ix = create_bm25_index(temp_dir, schema)

    assert ix.is_empty()
    assert ix.doc_count() == 0


def test_bm25index_add_documents(temp_dir):
    """Test adding documents via writer."""
    try:
        from semlix.bm25 import create_bm25_index
        from semlix.fields import Schema, TEXT, ID
    except ImportError:
        pytest.skip("bm25s not installed")

    schema = Schema(id=ID(stored=True), content=TEXT(stored=True))
    ix = create_bm25_index(temp_dir, schema)

    with ix.writer() as writer:
        writer.add_document(id="1", content="python programming")
        writer.add_document(id="2", content="database systems")

    assert ix.doc_count() == 2


def test_bm25index_search(temp_dir):
    """Test searching via searcher."""
    try:
        from semlix.bm25 import create_bm25_index
        from semlix.fields import Schema, TEXT, ID
        from semlix.qparser import QueryParser
    except ImportError:
        pytest.skip("bm25s or semlix components not available")

    schema = Schema(id=ID(stored=True), content=TEXT(stored=True))
    ix = create_bm25_index(temp_dir, schema)

    with ix.writer() as writer:
        writer.add_document(id="1", content="python programming language")
        writer.add_document(id="2", content="java programming language")
        writer.add_document(id="3", content="database queries")

    with ix.searcher() as searcher:
        qp = QueryParser("content", ix.schema)
        query = qp.parse("python")

        results = searcher.search(query, limit=2)

        assert len(results) > 0
        # First result should be the python document
        assert "1" in results[0]["id"]


def test_bm25index_reader(temp_dir):
    """Test reader functionality."""
    try:
        from semlix.bm25 import create_bm25_index
        from semlix.fields import Schema, TEXT, ID
    except ImportError:
        pytest.skip("bm25s not installed")

    schema = Schema(id=ID(stored=True), content=TEXT(stored=True))
    ix = create_bm25_index(temp_dir, schema)

    with ix.writer() as writer:
        writer.add_document(id="1", content="test document")

    with ix.reader() as reader:
        assert reader.doc_count() == 1
        fields = reader.stored_fields(0)
        assert fields["id"] == "1"


def test_bm25index_open(temp_dir):
    """Test opening existing index."""
    try:
        from semlix.bm25 import create_bm25_index, open_bm25_index
        from semlix.fields import Schema, TEXT, ID
    except ImportError:
        pytest.skip("bm25s not installed")

    # Create
    schema = Schema(id=ID(stored=True), content=TEXT(stored=True))
    ix1 = create_bm25_index(temp_dir, schema)

    with ix1.writer() as writer:
        writer.add_document(id="1", content="test")

    ix1.close()

    # Reopen
    ix2 = open_bm25_index(temp_dir)
    assert ix2.doc_count() == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
