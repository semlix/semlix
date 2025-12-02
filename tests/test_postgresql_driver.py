"""Tests for PostgreSQL storage driver.

These tests require a running PostgreSQL instance.
Set the TEST_POSTGRES_URL environment variable to run these tests.

Example:
    export TEST_POSTGRES_URL="postgresql://user:pass@localhost:5432/testdb"
    pytest tests/test_postgresql_driver.py
"""

import os
import pytest

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
    except ImportError:
        pytest.skip("psycopg2 not installed")

    # Read schema file
    import pathlib
    schema_path = pathlib.Path(__file__).parent.parent / "src" / "semlix" / "filedb" / "drivers" / "schema_lexical.sql"

    with open(schema_path) as f:
        schema_sql = f.read()

    # Create schema
    conn = psycopg2.connect(postgres_url)
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
            cur.execute("DROP TABLE IF EXISTS semlix_postings CASCADE")
            cur.execute("DROP TABLE IF EXISTS semlix_terms CASCADE")
            cur.execute("DROP TABLE IF EXISTS semlix_documents CASCADE")
            cur.execute("DROP TABLE IF EXISTS semlix_index_meta CASCADE")
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def driver(postgres_url, setup_database):
    """Create PostgreSQLDriver instance."""
    from semlix.filedb.drivers import PostgreSQLDriver

    driver = PostgreSQLDriver(
        connection_string=postgres_url,
        index_name="test_index"
    )

    yield driver

    # Cleanup after each test
    try:
        driver.destroy()
    except Exception:
        pass

    driver.close()


def test_initialization(driver):
    """Test driver initialization."""
    assert driver.driver_name == "postgresql"
    assert driver.index_name == "test_index"


def test_create(driver):
    """Test schema creation."""
    # create() should work (schema already created by fixture)
    driver.create()
    assert True  # No exception means success


def test_save_and_load_schema(driver):
    """Test saving and loading schema metadata."""
    schema_dict = {
        "fields": {
            "id": {"type": "ID", "stored": True},
            "content": {"type": "TEXT", "stored": True}
        }
    }

    # Save schema
    driver.save_schema(schema_dict)

    # Load schema
    loaded = driver.load_schema()
    assert loaded is not None
    assert loaded["fields"]["id"]["type"] == "ID"
    assert loaded["fields"]["content"]["type"] == "TEXT"


def test_add_document(driver):
    """Test adding a document."""
    driver.add_document(
        doc_id="doc1",
        doc_number=0,
        stored_fields={"id": "doc1", "content": "Hello world"},
        field_lengths={"content": 2}
    )

    # Verify document was added
    doc = driver.get_document("doc1")
    assert doc is not None
    assert doc["doc_id"] == "doc1"
    assert doc["stored_fields"]["content"] == "Hello world"


def test_add_multiple_documents(driver):
    """Test adding multiple documents."""
    for i in range(5):
        driver.add_document(
            doc_id=f"doc{i}",
            doc_number=i,
            stored_fields={"id": f"doc{i}", "content": f"Content {i}"}
        )

    assert driver.get_doc_count() == 5


def test_update_document(driver):
    """Test updating an existing document."""
    # Add document
    driver.add_document(
        doc_id="doc1",
        doc_number=0,
        stored_fields={"id": "doc1", "content": "Original"}
    )

    # Update document
    driver.add_document(
        doc_id="doc1",
        doc_number=0,
        stored_fields={"id": "doc1", "content": "Updated"}
    )

    # Verify update
    doc = driver.get_document("doc1")
    assert doc["stored_fields"]["content"] == "Updated"
    assert driver.get_doc_count() == 1  # Should still be 1


def test_add_term(driver):
    """Test adding a term posting."""
    # Add document first
    driver.add_document(
        doc_id="doc1",
        doc_number=0,
        stored_fields={"id": "doc1", "content": "hello world"}
    )

    # Add term
    driver.add_term(
        field_name="content",
        term="hello",
        doc_id="doc1",
        frequency=1,
        positions=[0]
    )

    # Verify term was added
    term_info = driver.get_term_info("content", "hello")
    assert term_info is not None
    assert term_info["doc_frequency"] == 1


def test_search_term(driver):
    """Test searching for a term."""
    # Add documents
    driver.add_document(
        doc_id="doc1",
        doc_number=0,
        stored_fields={"id": "doc1", "content": "Python programming"}
    )
    driver.add_document(
        doc_id="doc2",
        doc_number=1,
        stored_fields={"id": "doc2", "content": "Python is great"}
    )

    # Add terms
    driver.add_term("content", "python", "doc1", 1, [0])
    driver.add_term("content", "python", "doc2", 1, [0])

    # Search
    results = driver.search_term("content", "python")

    assert len(results) == 2
    doc_ids = [r["doc_id"] for r in results]
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids


def test_term_frequencies(driver):
    """Test that term frequencies are tracked correctly."""
    # Add document
    driver.add_document(
        doc_id="doc1",
        doc_number=0,
        stored_fields={"id": "doc1", "content": "test"}
    )

    # Add term
    driver.add_term("content", "test", "doc1", 3, [0, 5, 10])

    # Check term info
    term_info = driver.get_term_info("content", "test")
    assert term_info["doc_frequency"] == 1
    assert term_info["collection_frequency"] == 3


def test_delete_document(driver):
    """Test deleting a document."""
    # Add documents
    driver.add_document(
        doc_id="doc1",
        doc_number=0,
        stored_fields={"id": "doc1", "content": "test"}
    )
    driver.add_document(
        doc_id="doc2",
        doc_number=1,
        stored_fields={"id": "doc2", "content": "test"}
    )

    assert driver.get_doc_count() == 2

    # Delete one document
    deleted = driver.delete_document("doc1")
    assert deleted == 1
    assert driver.get_doc_count() == 1

    # Verify deletion
    doc = driver.get_document("doc1")
    assert doc is None


def test_delete_nonexistent_document(driver):
    """Test deleting a document that doesn't exist."""
    deleted = driver.delete_document("nonexistent")
    assert deleted == 0


def test_context_manager(postgres_url, setup_database):
    """Test using driver as context manager."""
    from semlix.filedb.drivers import PostgreSQLDriver

    with PostgreSQLDriver(postgres_url, "context_test") as driver:
        driver.add_document(
            doc_id="doc1",
            doc_number=0,
            stored_fields={"id": "doc1"}
        )
        assert driver.get_doc_count() == 1

    # Pool should be closed after exiting context


def test_destroy(driver):
    """Test destroying an index."""
    # Add some data
    driver.add_document(
        doc_id="doc1",
        doc_number=0,
        stored_fields={"id": "doc1"}
    )
    driver.add_term("content", "test", "doc1", 1)

    # Destroy
    driver.destroy()

    # Verify everything is gone
    assert driver.get_doc_count() == 0
    assert driver.get_term_info("content", "test") is None


def test_repr(driver):
    """Test string representation."""
    repr_str = repr(driver)
    assert "PostgreSQLDriver" in repr_str
    assert "test_index" in repr_str
