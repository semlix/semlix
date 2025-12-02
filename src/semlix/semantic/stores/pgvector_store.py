"""PostgreSQL + pgvector backend for semantic search.

This module provides a production-ready vector store implementation using
PostgreSQL with the pgvector extension. Suitable for datasets of any size
with ACID guarantees and advanced filtering capabilities.

Requires:
    - PostgreSQL 12+ with pgvector extension
    - pip install psycopg2-binary pgvector

Example:
    >>> from semlix.semantic.stores import PgVectorStore
    >>>
    >>> # Initialize store
    >>> store = PgVectorStore(
    ...     connection_string="postgresql://user:pass@localhost/mydb",
    ...     dimension=384,
    ...     distance_metric="cosine"
    ... )
    >>>
    >>> # Create index for fast search
    >>> store.create_index(index_type="hnsw", m=16, ef_construction=64)
    >>>
    >>> # Add vectors
    >>> import numpy as np
    >>> embeddings = np.random.randn(100, 384).astype(np.float32)
    >>> doc_ids = [f"doc{i}" for i in range(100)]
    >>> store.add(doc_ids, embeddings)
    >>>
    >>> # Search
    >>> query = np.random.randn(384).astype(np.float32)
    >>> results = store.search(query, k=10)
    >>>
    >>> # Search with metadata filter
    >>> results = store.search_with_filter(
    ...     query, k=10,
    ...     metadata_filter={"category": "tech"}
    ... )
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .base import VectorSearchResult

# Distance metric type
DistanceMetric = Literal["cosine", "l2", "inner_product"]


class PgVectorStore:
    """PostgreSQL + pgvector backend for semantic search.

    Uses the pgvector extension and pgvector-python library for native
    vector type support and efficient similarity search.

    Features:
        - ACID transactions
        - Connection pooling
        - Multiple distance metrics (cosine, L2, inner product)
        - HNSW and IVFFlat indexing
        - Metadata filtering with JSONB
        - Batch operations

    Attributes:
        dimension: Dimensionality of stored vectors
        count: Number of vectors in the store
        driver_name: Always "pgvector"
        supports_transactions: Always True
        supports_filtering: Always True
    """

    def __init__(
        self,
        connection_string: str,
        dimension: int,
        table_name: str = "semlix_vectors",
        pool_size: int = 10,
        distance_metric: DistanceMetric = "cosine"
    ):
        """Initialize PgVector store.

        Args:
            connection_string: PostgreSQL connection string
                (e.g., "postgresql://user:pass@localhost:5432/dbname")
            dimension: Vector dimensionality (must match embeddings)
            table_name: Table name for storing vectors
            pool_size: Maximum connections in pool
            distance_metric: Distance metric for similarity search
                - "cosine": Cosine distance (1 - cosine similarity)
                - "l2": Euclidean distance
                - "inner_product": Negative inner product (for max similarity)

        Raises:
            ImportError: If psycopg2 or pgvector is not installed
            ValueError: If distance_metric is invalid
        """
        # Import dependencies
        try:
            import psycopg2
            from psycopg2.pool import ThreadedConnectionPool
            from pgvector.psycopg2 import register_vector
        except ImportError as e:
            raise ImportError(
                "PgVectorStore requires psycopg2 and pgvector. "
                "Install with: pip install psycopg2-binary pgvector"
            ) from e

        self._psycopg2 = psycopg2
        self._register_vector = register_vector
        self._ThreadedConnectionPool = ThreadedConnectionPool

        if dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {dimension}")

        self.connection_string = connection_string
        self._dimension = dimension
        self.table_name = table_name
        self.pool_size = pool_size
        self.distance_metric = distance_metric
        self._pool = None

        # Map distance metrics to pgvector operators
        self._distance_ops = {
            "cosine": "<=>",      # Cosine distance
            "l2": "<->",          # L2 distance
            "inner_product": "<#>", # Negative inner product
        }

        if distance_metric not in self._distance_ops:
            raise ValueError(
                f"Unknown distance metric: {distance_metric}. "
                f"Choose from: {list(self._distance_ops.keys())}"
            )

    @property
    def driver_name(self) -> str:
        """Return driver identifier."""
        return "pgvector"

    @property
    def dimension(self) -> int:
        """Return the dimensionality of stored vectors."""
        return self._dimension

    @property
    def count(self) -> int:
        """Return number of vectors in the store."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                return cur.fetchone()[0]

    @property
    def supports_transactions(self) -> bool:
        """Return whether this driver supports transactions."""
        return True

    @property
    def supports_filtering(self) -> bool:
        """Return whether this driver supports metadata filtering."""
        return True

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
        """Get connection from pool with vector types registered.

        Yields:
            Database connection with pgvector types registered
        """
        pool = self._get_pool()
        conn = pool.getconn()
        try:
            # Register vector types for this connection
            self._register_vector(conn)
            yield conn
        finally:
            pool.putconn(conn)

    def add(
        self,
        doc_ids: List[str],
        embeddings: NDArray[np.float32],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add vectors to the store.

        Vectors are stored with their document IDs and optional metadata.
        If a document ID already exists, it will be updated (upsert).

        Args:
            doc_ids: Unique identifiers for each document
            embeddings: Array of shape (n, dimension) containing vectors
            metadata: Optional metadata for each document (must be JSON-serializable)

        Raises:
            ValueError: If embedding dimensions don't match store dimension
            ValueError: If number of doc_ids doesn't match number of embeddings

        Example:
            >>> embeddings = np.random.randn(10, 384).astype(np.float32)
            >>> doc_ids = [f"doc{i}" for i in range(10)]
            >>> metadata = [{"category": "tech", "lang": "en"}] * 10
            >>> store.add(doc_ids, embeddings, metadata)
        """
        from psycopg2.extras import Json, execute_values

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[1] != self._dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} doesn't match "
                f"store dimension {self._dimension}"
            )

        if len(doc_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Number of doc_ids ({len(doc_ids)}) doesn't match "
                f"number of embeddings ({embeddings.shape[0]})"
            )

        if metadata is None:
            metadata = [{} for _ in doc_ids]
        elif len(metadata) != len(doc_ids):
            raise ValueError(
                f"Number of metadata entries ({len(metadata)}) doesn't match "
                f"number of doc_ids ({len(doc_ids)})"
            )

        # Normalize embeddings for cosine similarity
        if self.distance_metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-9)

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Batch insert using execute_values for performance
                values = [
                    (doc_id, emb.tolist(), Json(meta))
                    for doc_id, emb, meta in zip(doc_ids, embeddings, metadata)
                ]

                execute_values(
                    cur,
                    f"""
                    INSERT INTO {self.table_name} (doc_id, embedding, metadata)
                    VALUES %s
                    ON CONFLICT (doc_id) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        created_at = NOW()
                    """,
                    values,
                    template="(%s, %s::vector, %s)"
                )
            conn.commit()

    def search(
        self,
        query_embedding: NDArray[np.float32],
        k: int = 10,
        filter_ids: Optional[List[str]] = None
    ) -> List[VectorSearchResult]:
        """Search for similar vectors using configured distance metric.

        Args:
            query_embedding: Query vector of shape (dimension,)
            k: Number of results to return
            filter_ids: Optional list of doc_ids to restrict search to

        Returns:
            List of VectorSearchResult sorted by descending similarity score

        Raises:
            ValueError: If query dimension doesn't match store dimension

        Example:
            >>> query = np.random.randn(384).astype(np.float32)
            >>> results = store.search(query, k=5)
            >>> for r in results:
            ...     print(f"{r.doc_id}: {r.score:.4f}")
        """
        from psycopg2.extras import RealDictCursor

        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        if query_embedding.ndim > 1:
            query_embedding = query_embedding.flatten()

        if len(query_embedding) != self._dimension:
            raise ValueError(
                f"Query dimension {len(query_embedding)} doesn't match "
                f"store dimension {self._dimension}"
            )

        # Normalize query for cosine similarity
        if self.distance_metric == "cosine":
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm

        # Get distance operator
        distance_op = self._distance_ops[self.distance_metric]

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if filter_ids:
                    # Search with doc_id filter
                    cur.execute(f"""
                        SELECT
                            doc_id,
                            embedding {distance_op} %s::vector as distance,
                            metadata
                        FROM {self.table_name}
                        WHERE doc_id = ANY(%s)
                        ORDER BY embedding {distance_op} %s::vector
                        LIMIT %s
                    """, (query_embedding.tolist(), filter_ids,
                          query_embedding.tolist(), k))
                else:
                    # Full search
                    cur.execute(f"""
                        SELECT
                            doc_id,
                            embedding {distance_op} %s::vector as distance,
                            metadata
                        FROM {self.table_name}
                        ORDER BY embedding {distance_op} %s::vector
                        LIMIT %s
                    """, (query_embedding.tolist(), query_embedding.tolist(), k))

                results = []
                for row in cur.fetchall():
                    # Convert distance to similarity score
                    distance = float(row['distance'])
                    score = self._distance_to_score(distance)

                    results.append(VectorSearchResult(
                        doc_id=row['doc_id'],
                        score=score,
                        metadata=row['metadata'] or {}
                    ))

                return results

    def search_with_filter(
        self,
        query_embedding: NDArray[np.float32],
        k: int,
        metadata_filter: Dict[str, Any]
    ) -> List[VectorSearchResult]:
        """Search with metadata filtering using JSONB operators.

        Args:
            query_embedding: Query vector of shape (dimension,)
            k: Number of results to return
            metadata_filter: Dictionary of metadata key-value pairs to filter by

        Returns:
            List of VectorSearchResult matching the filter, sorted by similarity

        Example:
            >>> results = store.search_with_filter(
            ...     query,
            ...     k=10,
            ...     metadata_filter={"category": "tech", "lang": "en"}
            ... )
        """
        from psycopg2.extras import RealDictCursor

        query_embedding = np.asarray(query_embedding, dtype=np.float32).flatten()

        if self.distance_metric == "cosine":
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm

        # Build JSONB filter conditions
        filter_conditions = []
        filter_values = [query_embedding.tolist()]

        for key, value in metadata_filter.items():
            filter_conditions.append("metadata->>%s = %s")
            filter_values.extend([key, str(value)])

        where_clause = " AND ".join(filter_conditions)
        distance_op = self._distance_ops[self.distance_metric]

        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = f"""
                    SELECT
                        doc_id,
                        embedding {distance_op} %s::vector as distance,
                        metadata
                    FROM {self.table_name}
                    WHERE {where_clause}
                    ORDER BY embedding {distance_op} %s::vector
                    LIMIT %s
                """

                params = filter_values + [query_embedding.tolist(), k]
                cur.execute(query, params)

                results = []
                for row in cur.fetchall():
                    distance = float(row['distance'])
                    score = self._distance_to_score(distance)

                    results.append(VectorSearchResult(
                        doc_id=row['doc_id'],
                        score=score,
                        metadata=row['metadata'] or {}
                    ))

                return results

    def delete(self, doc_ids: List[str]) -> int:
        """Delete vectors by document ID.

        Args:
            doc_ids: List of document IDs to delete

        Returns:
            Number of vectors deleted

        Example:
            >>> deleted = store.delete(["doc1", "doc2", "doc3"])
            >>> print(f"Deleted {deleted} vectors")
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    DELETE FROM {self.table_name}
                    WHERE doc_id = ANY(%s)
                """, (doc_ids,))
                deleted = cur.rowcount
            conn.commit()
            return deleted

    def create_index(
        self,
        index_type: Literal["hnsw", "ivfflat"] = "hnsw",
        m: int = 16,
        ef_construction: int = 64,
        lists: int = 100
    ) -> None:
        """Create vector index for fast similarity search.

        Args:
            index_type: Type of index to create
                - "hnsw": Hierarchical Navigable Small World (recommended)
                - "ivfflat": Inverted File with Flat compression
            m: HNSW parameter - connections per layer (default: 16)
            ef_construction: HNSW parameter - build-time search depth (default: 64)
            lists: IVFFlat parameter - number of clusters (default: 100)

        Note:
            - HNSW generally provides better performance but uses more memory
            - IVFFlat requires periodic VACUUM for optimal performance
            - Index creation can be slow for large datasets

        Example:
            >>> # Create HNSW index (recommended)
            >>> store.create_index(index_type="hnsw", m=16, ef_construction=64)
            >>>
            >>> # Create IVFFlat index
            >>> store.create_index(index_type="ivfflat", lists=100)
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Determine operator class based on distance metric
                if self.distance_metric == "cosine":
                    ops = "vector_cosine_ops"
                elif self.distance_metric == "l2":
                    ops = "vector_l2_ops"
                else:  # inner_product
                    ops = "vector_ip_ops"

                index_name = f"idx_{self.table_name}_embedding_{index_type}"

                if index_type == "hnsw":
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {self.table_name}
                        USING hnsw (embedding {ops})
                        WITH (m = {m}, ef_construction = {ef_construction})
                    """)
                elif index_type == "ivfflat":
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON {self.table_name}
                        USING ivfflat (embedding {ops})
                        WITH (lists = {lists})
                    """)
                else:
                    raise ValueError(
                        f"Unknown index type: {index_type}. "
                        f"Choose 'hnsw' or 'ivfflat'"
                    )

            conn.commit()

    def _distance_to_score(self, distance: float) -> float:
        """Convert distance to similarity score.

        Args:
            distance: Distance value from pgvector

        Returns:
            Similarity score (higher is more similar)
        """
        if self.distance_metric == "cosine":
            # Cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: 1 = identical, -1 = opposite
            return 1.0 - distance
        elif self.distance_metric == "inner_product":
            # Negative inner product, negate to get positive similarity
            return -distance
        else:  # l2
            # L2 distance: 0 = identical, larger = more different
            # Convert to similarity score
            return 1.0 / (1.0 + distance)

    def save(self, path: Union[str, Any]) -> None:
        """Not applicable for database-backed stores.

        Raises:
            NotImplementedError: Always, as database stores don't support file saving
        """
        raise NotImplementedError(
            "PgVectorStore is database-backed and doesn't support file saving. "
            "Use PostgreSQL backup tools (pg_dump, pg_basebackup) instead."
        )

    @classmethod
    def load(cls, path: Union[str, Any]) -> "PgVectorStore":
        """Not applicable for database-backed stores.

        Raises:
            NotImplementedError: Always, as database stores don't support file loading
        """
        raise NotImplementedError(
            "PgVectorStore is database-backed and doesn't support file loading. "
            "Use standard initialization with connection_string instead."
        )

    def close(self) -> None:
        """Close connection pool and release all resources.

        Example:
            >>> store = PgVectorStore(...)
            >>> try:
            ...     # Use store
            ...     store.add(...)
            ... finally:
            ...     store.close()
        """
        if self._pool:
            self._pool.closeall()
            self._pool = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection pool is closed."""
        self.close()

    def __repr__(self) -> str:
        return (
            f"PgVectorStore(dimension={self._dimension}, "
            f"count={self.count}, "
            f"metric={self.distance_metric})"
        )
