"""Tests for PgVectorStore.

These tests require a running PostgreSQL instance with pgvector extension.
Set the TEST_POSTGRES_URL environment variable to run these tests.

Example:
    export TEST_POSTGRES_URL="postgresql://user:pass@localhost:5432/testdb"
    pytest tests/test_pgvector_store.py
"""

import os
import pytest
import numpy as np

# Skip all tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    not os.getenv("TEST_POSTGRES_URL"),
    reason="TEST_POSTGRES_URL not set. Set it to run PostgreSQL tests."
)


@pytest.fixture(scope="module")
def postgres_url():
    """Get PostgreSQL connection URL from environment."""
    return os.getenv("TEST_POSTGRES_URL")


@pytest.fixture(scope="module")
def setup_database(postgres_url):
    """Setup database schema before tests."""
    try:
        import psycopg2
        from pgvector.psycopg2 import register_vector
    except ImportError:
        pytest.skip("psycopg2 or pgvector not installed")

    # Read schema file
    import pathlib
    schema_path = pathlib.Path(__file__).parent.parent / "src" / "semlix" / "semantic" / "stores" / "schema.sql"

    with open(schema_path) as f:
        schema_sql = f.read()

    # Create schema
    conn = psycopg2.connect(postgres_url)
    register_vector(conn)
    try:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()
    finally:
        conn.close()

    yield

    # Cleanup after all tests
    conn = psycopg2.connect(postgres_url)
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS semlix_vectors CASCADE")
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def store(postgres_url, setup_database):
    """Create PgVectorStore instance."""
    from semlix.semantic.stores import PgVectorStore

    store = PgVectorStore(
        connection_string=postgres_url,
        dimension=384,
        distance_metric="cosine"
    )

    yield store

    # Cleanup after each test
    try:
        import psycopg2
        conn = psycopg2.connect(postgres_url)
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE semlix_vectors")
        conn.commit()
        conn.close()
    except Exception:
        pass

    store.close()


def test_initialization(store):
    """Test store initialization."""
    assert store.dimension == 384
    assert store.driver_name == "pgvector"
    assert store.distance_metric == "cosine"
    assert store.supports_transactions is True
    assert store.supports_filtering is True


def test_count_empty(store):
    """Test count on empty store."""
    assert store.count == 0


def test_add_single_vector(store):
    """Test adding a single vector."""
    embedding = np.random.randn(384).astype(np.float32)
    store.add(["doc1"], embedding.reshape(1, -1))

    assert store.count == 1


def test_add_multiple_vectors(store):
    """Test adding multiple vectors."""
    embeddings = np.random.randn(10, 384).astype(np.float32)
    doc_ids = [f"doc{i}" for i in range(10)]

    store.add(doc_ids, embeddings)

    assert store.count == 10


def test_add_with_metadata(store):
    """Test adding vectors with metadata."""
    embeddings = np.random.randn(5, 384).astype(np.float32)
    doc_ids = [f"doc{i}" for i in range(5)]
    metadata = [{"category": "tech", "index": i} for i in range(5)]

    store.add(doc_ids, embeddings, metadata)

    assert store.count == 5


def test_add_upsert(store):
    """Test upserting (updating existing document)."""
    embedding1 = np.random.randn(384).astype(np.float32)
    embedding2 = np.random.randn(384).astype(np.float32)

    # Add first time
    store.add(["doc1"], embedding1.reshape(1, -1))
    assert store.count == 1

    # Add again with different embedding (upsert)
    store.add(["doc1"], embedding2.reshape(1, -1))
    assert store.count == 1  # Count should stay the same


def test_search_basic(store):
    """Test basic similarity search."""
    embeddings = np.random.randn(10, 384).astype(np.float32)
    doc_ids = [f"doc{i}" for i in range(10)]

    store.add(doc_ids, embeddings)

    # Search with first embedding (should return itself as top result)
    results = store.search(embeddings[0], k=5)

    assert len(results) == 5
    assert results[0].doc_id == "doc0"
    assert results[0].score > 0.99  # Should be very similar to itself


def test_search_with_filter_ids(store):
    """Test search with doc_id filter."""
    embeddings = np.random.randn(10, 384).astype(np.float32)
    doc_ids = [f"doc{i}" for i in range(10)]

    store.add(doc_ids, embeddings)

    # Search only within specific doc_ids
    filter_ids = ["doc0", "doc1", "doc2"]
    results = store.search(embeddings[0], k=5, filter_ids=filter_ids)

    assert len(results) <= 3  # Should only return from filtered set
    assert all(r.doc_id in filter_ids for r in results)


def test_search_with_metadata_filter(store):
    """Test search with metadata filter."""
    embeddings = np.random.randn(10, 384).astype(np.float32)
    doc_ids = [f"doc{i}" for i in range(10)]
    metadata = [
        {"category": "tech" if i < 5 else "science"}
        for i in range(10)
    ]

    store.add(doc_ids, embeddings, metadata)

    # Search only tech category
    query = embeddings[0]
    results = store.search_with_filter(
        query, k=10,
        metadata_filter={"category": "tech"}
    )

    assert len(results) <= 5  # Only tech docs
    assert all("category" in r.metadata for r in results)


def test_delete(store):
    """Test deleting vectors."""
    embeddings = np.random.randn(10, 384).astype(np.float32)
    doc_ids = [f"doc{i}" for i in range(10)]

    store.add(doc_ids, embeddings)
    assert store.count == 10

    # Delete 3 documents
    deleted = store.delete(["doc0", "doc1", "doc2"])
    assert deleted == 3
    assert store.count == 7


def test_delete_nonexistent(store):
    """Test deleting non-existent documents."""
    embeddings = np.random.randn(5, 384).astype(np.float32)
    doc_ids = [f"doc{i}" for i in range(5)]

    store.add(doc_ids, embeddings)

    # Try to delete non-existent docs
    deleted = store.delete(["doc100", "doc101"])
    assert deleted == 0
    assert store.count == 5


def test_create_index_hnsw(store):
    """Test creating HNSW index."""
    # Add some data first
    embeddings = np.random.randn(100, 384).astype(np.float32)
    doc_ids = [f"doc{i}" for i in range(100)]
    store.add(doc_ids, embeddings)

    # Create index
    store.create_index(index_type="hnsw", m=16, ef_construction=64)

    # Should still be able to search
    results = store.search(embeddings[0], k=5)
    assert len(results) == 5


def test_create_index_ivfflat(store):
    """Test creating IVFFlat index."""
    # Add some data first
    embeddings = np.random.randn(100, 384).astype(np.float32)
    doc_ids = [f"doc{i}" for i in range(100)]
    store.add(doc_ids, embeddings)

    # Create index
    store.create_index(index_type="ivfflat", lists=10)

    # Should still be able to search
    results = store.search(embeddings[0], k=5)
    assert len(results) == 5


def test_dimension_mismatch(store):
    """Test error on dimension mismatch."""
    wrong_embedding = np.random.randn(512).astype(np.float32)

    with pytest.raises(ValueError, match="dimension"):
        store.add(["doc1"], wrong_embedding.reshape(1, -1))


def test_invalid_distance_metric():
    """Test error on invalid distance metric."""
    from semlix.semantic.stores import PgVectorStore

    with pytest.raises(ValueError, match="Unknown distance metric"):
        PgVectorStore(
            connection_string="postgresql://localhost/test",
            dimension=384,
            distance_metric="invalid"
        )


def test_context_manager(postgres_url, setup_database):
    """Test using store as context manager."""
    from semlix.semantic.stores import PgVectorStore

    with PgVectorStore(postgres_url, dimension=384) as store:
        embeddings = np.random.randn(5, 384).astype(np.float32)
        doc_ids = [f"doc{i}" for i in range(5)]
        store.add(doc_ids, embeddings)
        assert store.count == 5

    # Pool should be closed after exiting context


def test_save_raises_error(store):
    """Test that save() raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="database-backed"):
        store.save("some_path.pkl")


def test_load_raises_error():
    """Test that load() raises NotImplementedError."""
    from semlix.semantic.stores import PgVectorStore

    with pytest.raises(NotImplementedError, match="database-backed"):
        PgVectorStore.load("some_path.pkl")


def test_repr(store):
    """Test string representation."""
    repr_str = repr(store)
    assert "PgVectorStore" in repr_str
    assert "dimension=384" in repr_str
    assert "cosine" in repr_str


# Integration test
def test_full_workflow(store):
    """Test complete add -> search -> delete workflow."""
    # Create embeddings
    np.random.seed(42)
    embeddings = np.random.randn(50, 384).astype(np.float32)
    doc_ids = [f"doc{i}" for i in range(50)]
    metadata = [
        {"category": "tech" if i % 2 == 0 else "science", "id": i}
        for i in range(50)
    ]

    # Add to store
    store.add(doc_ids, embeddings, metadata)
    assert store.count == 50

    # Create index for performance
    store.create_index(index_type="hnsw")

    # Search
    query = embeddings[10]
    results = store.search(query, k=10)

    assert len(results) == 10
    assert results[0].doc_id == "doc10"
    assert results[0].score > 0.99

    # Search with metadata filter
    tech_results = store.search_with_filter(
        query, k=10,
        metadata_filter={"category": "tech"}
    )
    assert len(tech_results) <= 25  # Half are tech
    assert all(r.metadata["category"] == "tech" for r in tech_results)

    # Delete some documents
    to_delete = [f"doc{i}" for i in range(10)]
    deleted = store.delete(to_delete)
    assert deleted == 10
    assert store.count == 40

    # Search again (deleted docs shouldn't appear)
    results2 = store.search(embeddings[20], k=10)
    assert all(r.doc_id not in to_delete for r in results2)
