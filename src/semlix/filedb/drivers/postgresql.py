"""PostgreSQL storage driver for Semlix lexical indexes.

This is an experimental driver that demonstrates database-backed storage
for Semlix indexes. It stores documents, terms, and postings in PostgreSQL
tables instead of binary files.

Status: Proof of Concept
- Basic document storage ✓
- Term indexing ✓
- Simple search ✓
- Full BM25 implementation ⏳
- Complete Whoosh compatibility ⏳

Requirements:
    pip install psycopg2-binary

Example:
    >>> from semlix.filedb.drivers import PostgreSQLDriver
    >>> from semlix.fields import Schema, TEXT, ID
    >>>
    >>> # Create driver
    >>> driver = PostgreSQLDriver(
    ...     connection_string="postgresql://localhost/semlix",
    ...     index_name="my_index"
    ... )
    >>> driver.create()
    >>>
    >>> # Store documents
    >>> driver.add_document(
    ...     doc_id="doc1",
    ...     doc_number=0,
    ...     stored_fields={"id": "doc1", "content": "Hello world"}
    ... )
    >>>
    >>> # Index terms
    >>> driver.add_term(
    ...     field_name="content",
    ...     term="hello",
    ...     doc_id="doc1",
    ...     frequency=1
    ... )
"""

from __future__ import annotations

import json
import pickle
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

class PostgreSQLDriver:
    """PostgreSQL backend for Semlix lexical index storage.

    This driver provides database-backed storage for document metadata,
    terms, and postings. It uses PostgreSQL tables instead of binary files.

    Attributes:
        connection_string: PostgreSQL connection string
        index_name: Name of the index
        pool_size: Maximum connections in pool
    """

    def __init__(
        self,
        connection_string: str,
        index_name: str = "MAIN",
        pool_size: int = 10
    ):
        """Initialize PostgreSQL driver.

        Args:
            connection_string: PostgreSQL DSN
                (e.g., "postgresql://user:pass@localhost:5432/dbname")
            index_name: Name of the index (for multi-index support)
            pool_size: Maximum connections in connection pool
        """
        try:
            import psycopg2
            from psycopg2.pool import ThreadedConnectionPool
            from psycopg2.extras import Json, RealDictCursor
        except ImportError:
            raise ImportError(
                "PostgreSQLDriver requires psycopg2. "
                "Install with: pip install psycopg2-binary"
            )

        self._psycopg2 = psycopg2
        self._ThreadedConnectionPool = ThreadedConnectionPool
        self._Json = Json
        self._RealDictCursor = RealDictCursor

        self.connection_string = connection_string
        self.index_name = index_name
        self.pool_size = pool_size
        self._pool = None

    @property
    def driver_name(self) -> str:
        """Return driver identifier."""
        return "postgresql"

    def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            self._pool = self._ThreadedConnectionPool(
                minconn=1,
                maxconn=self.pool_size,
                dsn=self.connection_string
            )
        return self._pool

    @contextmanager
    def _get_connection(self):
        """Get connection from pool.

        Yields:
            Database connection
        """
        pool = self._get_pool()
        conn = pool.getconn()
        try:
            yield conn
        finally:
            pool.putconn(conn)

    def create(self) -> "PostgreSQLDriver":
        """Initialize database schema.

        Creates all necessary tables for lexical index storage.

        Returns:
            Self for chaining

        Example:
            >>> driver = PostgreSQLDriver(...).create()
        """
        import pathlib

        # Read schema file
        schema_path = pathlib.Path(__file__).parent / "schema_lexical.sql"
        with open(schema_path) as f:
            schema_sql = f.read()

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(schema_sql)
            conn.commit()

        return self

    def destroy(self) -> None:
        """Drop all tables for this index.

        Warning: This will delete all data!
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Delete from all tables for this index
                cur.execute(
                    "DELETE FROM semlix_postings WHERE index_name = %s",
                    (self.index_name,)
                )
                cur.execute(
                    "DELETE FROM semlix_terms WHERE index_name = %s",
                    (self.index_name,)
                )
                cur.execute(
                    "DELETE FROM semlix_documents WHERE index_name = %s",
                    (self.index_name,)
                )
                cur.execute(
                    "DELETE FROM semlix_index_meta WHERE index_name = %s",
                    (self.index_name,)
                )
            conn.commit()

    def save_schema(self, schema_dict: Dict[str, Any]) -> None:
        """Save index schema metadata.

        Args:
            schema_dict: Serialized schema dictionary
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO semlix_index_meta (index_name, schema_data, generation)
                    VALUES (%s, %s, 0)
                    ON CONFLICT (index_name) DO UPDATE
                    SET schema_data = EXCLUDED.schema_data,
                        updated_at = NOW()
                """, (self.index_name, self._Json(schema_dict)))
            conn.commit()

    def load_schema(self) -> Optional[Dict[str, Any]]:
        """Load index schema metadata.

        Returns:
            Schema dictionary or None if not found
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=self._RealDictCursor) as cur:
                cur.execute("""
                    SELECT schema_data FROM semlix_index_meta
                    WHERE index_name = %s
                """, (self.index_name,))

                row = cur.fetchone()
                return row['schema_data'] if row else None

    def add_document(
        self,
        doc_id: str,
        doc_number: int,
        stored_fields: Dict[str, Any],
        field_lengths: Optional[Dict[str, int]] = None,
        transaction=None
    ) -> None:
        """Add or update a document.

        Args:
            doc_id: Unique document identifier
            doc_number: Internal document number (from Whoosh)
            stored_fields: Dictionary of stored field values
            field_lengths: Dictionary of field lengths for BM25
            transaction: Optional existing transaction

        Example:
            >>> driver.add_document(
            ...     doc_id="doc1",
            ...     doc_number=0,
            ...     stored_fields={"id": "doc1", "content": "Hello"},
            ...     field_lengths={"content": 1}
            ... )
        """
        if field_lengths is None:
            field_lengths = {}

        conn = transaction if transaction else self._get_connection()

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO semlix_documents
                    (index_name, doc_id, doc_number, stored_fields, field_lengths)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (index_name, doc_id) DO UPDATE
                    SET doc_number = EXCLUDED.doc_number,
                        stored_fields = EXCLUDED.stored_fields,
                        field_lengths = EXCLUDED.field_lengths,
                        updated_at = NOW()
                """, (
                    self.index_name,
                    doc_id,
                    doc_number,
                    self._Json(stored_fields),
                    self._Json(field_lengths)
                ))

            if not transaction:
                conn.commit()
        finally:
            if not transaction and hasattr(conn, '__exit__'):
                pass  # Context manager will handle

    def add_term(
        self,
        field_name: str,
        term: str,
        doc_id: str,
        frequency: int = 1,
        positions: Optional[List[int]] = None,
        transaction=None
    ) -> None:
        """Add a term posting.

        Args:
            field_name: Name of the field
            term: The term text
            doc_id: Document containing the term
            frequency: Number of occurrences in document
            positions: List of term positions (optional)
            transaction: Optional existing transaction

        Example:
            >>> driver.add_term(
            ...     field_name="content",
            ...     term="hello",
            ...     doc_id="doc1",
            ...     frequency=2,
            ...     positions=[0, 5]
            ... )
        """
        conn = transaction if transaction else self._get_connection()

        try:
            with conn.cursor() as cur:
                # Insert or get term_id
                cur.execute("""
                    INSERT INTO semlix_terms (index_name, field_name, term)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (index_name, field_name, term) DO NOTHING
                    RETURNING term_id
                """, (self.index_name, field_name, term))

                result = cur.fetchone()
                if result:
                    term_id = result[0]
                else:
                    # Term already exists, get its ID
                    cur.execute("""
                        SELECT term_id FROM semlix_terms
                        WHERE index_name = %s AND field_name = %s AND term = %s
                    """, (self.index_name, field_name, term))
                    term_id = cur.fetchone()[0]

                # Insert posting
                cur.execute("""
                    INSERT INTO semlix_postings
                    (term_id, index_name, doc_id, frequency, positions)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (term_id, index_name, doc_id) DO UPDATE
                    SET frequency = EXCLUDED.frequency,
                        positions = EXCLUDED.positions
                """, (term_id, self.index_name, doc_id, frequency, positions or []))

            if not transaction:
                conn.commit()
        finally:
            if not transaction and hasattr(conn, '__exit__'):
                pass

    def search_term(
        self,
        field_name: str,
        term: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search for documents containing a term.

        Args:
            field_name: Field to search in
            term: Term to search for
            limit: Maximum number of results

        Returns:
            List of documents with their stored fields and term frequency

        Example:
            >>> results = driver.search_term("content", "hello", limit=10)
            >>> for doc in results:
            ...     print(doc['doc_id'], doc['frequency'])
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=self._RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        d.doc_id,
                        d.stored_fields,
                        p.frequency,
                        p.positions
                    FROM semlix_postings p
                    JOIN semlix_documents d
                      ON d.index_name = p.index_name
                      AND d.doc_id = p.doc_id
                    JOIN semlix_terms t
                      ON t.term_id = p.term_id
                    WHERE t.index_name = %s
                      AND t.field_name = %s
                      AND t.term = %s
                    ORDER BY p.frequency DESC
                    LIMIT %s
                """, (self.index_name, field_name, term, limit))

                return [dict(row) for row in cur.fetchall()]

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID.

        Args:
            doc_id: Document identifier

        Returns:
            Document data or None if not found
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=self._RealDictCursor) as cur:
                cur.execute("""
                    SELECT doc_id, doc_number, stored_fields, field_lengths
                    FROM semlix_documents
                    WHERE index_name = %s AND doc_id = %s
                """, (self.index_name, doc_id))

                row = cur.fetchone()
                return dict(row) if row else None

    def get_doc_count(self) -> int:
        """Get total document count.

        Returns:
            Number of documents in index
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM semlix_documents
                    WHERE index_name = %s
                """, (self.index_name,))

                return cur.fetchone()[0]

    def get_term_info(self, field_name: str, term: str) -> Optional[Dict[str, Any]]:
        """Get term statistics.

        Args:
            field_name: Field name
            term: Term text

        Returns:
            Dict with doc_frequency and collection_frequency
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=self._RealDictCursor) as cur:
                cur.execute("""
                    SELECT term_id, doc_frequency, collection_frequency
                    FROM semlix_terms
                    WHERE index_name = %s AND field_name = %s AND term = %s
                """, (self.index_name, field_name, term))

                row = cur.fetchone()
                return dict(row) if row else None

    def delete_document(self, doc_id: str) -> int:
        """Delete a document and its postings.

        Args:
            doc_id: Document to delete

        Returns:
            Number of documents deleted (0 or 1)
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Delete document (CASCADE will delete postings)
                cur.execute("""
                    DELETE FROM semlix_documents
                    WHERE index_name = %s AND doc_id = %s
                """, (self.index_name, doc_id))
                deleted = cur.rowcount
            conn.commit()
            return deleted

    def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            self._pool.closeall()
            self._pool = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        return f"PostgreSQLDriver(index={self.index_name!r})"
