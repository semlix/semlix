================
Unified Index
================

UnifiedIndex combines high-performance BM25 lexical search with pgvector semantic search
in a single, unified interface. This provides the best of both worlds: fast keyword
matching and semantic understanding.

Overview
========

UnifiedIndex automatically manages both a BM25 index for lexical search and a PostgreSQL
vector store for semantic search. Documents are indexed in both stores simultaneously,
and searches can leverage either or both approaches.

**Key Benefits:**

* **Hybrid search out-of-the-box**: No manual setup required
* **Transactional writes**: Atomic updates across both stores
* **Automatic embeddings**: Generates vectors during indexing
* **Enhanced features**: Faceting, sorting, and phrase queries on hybrid results
* **Production-ready**: ACID transactions, scalable PostgreSQL backend

Quick Start
===========

Creating a Unified Index
-------------------------

::

    from semlix.unified import create_unified_index
    from semlix.fields import Schema, TEXT, ID, KEYWORD, DATETIME
    from semlix.semantic import SentenceTransformerProvider
    from semlix.analysis import StandardAnalyzer

    # Define schema
    schema = Schema(
        id=ID(stored=True),
        title=TEXT(stored=True, analyzer=StandardAnalyzer()),
        content=TEXT(stored=True, analyzer=StandardAnalyzer()),
        author=KEYWORD(stored=True),
        category=KEYWORD(stored=True),
        published=DATETIME(stored=True)
    )

    # Create embedding provider
    embedder = SentenceTransformerProvider("all-MiniLM-L6-v2")

    # Create unified index
    ix = create_unified_index(
        index_dir="my_unified_index",
        schema=schema,
        connection_string="postgresql://localhost/mydb",
        embedder=embedder
    )

Prerequisites
-------------

UnifiedIndex requires:

1. **PostgreSQL with pgvector extension**::

    # Install extension
    CREATE EXTENSION vector;

2. **Python packages**::

    pip install bm25s sentence-transformers psycopg2-binary pgvector

Indexing Documents
------------------

Use the unified writer to add documents to both indexes::

    with ix.writer() as writer:
        writer.add_document(
            id="1",
            title="Introduction to Machine Learning",
            content="Machine learning enables systems to learn from data...",
            author="Alice",
            category="ai",
            published="2024-01-15"
        )
        writer.add_document(
            id="2",
            title="Python Programming Guide",
            content="Learn Python programming best practices...",
            author="Bob",
            category="programming",
            published="2024-02-20"
        )

The writer automatically:

1. Indexes documents in the BM25 index
2. Generates embeddings for specified fields
3. Stores vectors in PostgreSQL
4. Commits both atomically

Searching
=========

Hybrid Search
-------------

Combine lexical and semantic search (default)::

    with ix.searcher() as searcher:
        results = searcher.hybrid_search(
            "machine learning algorithms",
            limit=10,
            alpha=0.5  # 0=all lexical, 1=all semantic
        )

        for r in results:
            print(f"{r.stored_fields['title']}")
            print(f"  Combined: {r.score:.3f}")
            print(f"  Lexical: {r.lexical_score:.3f}")
            print(f"  Semantic: {r.semantic_score:.3f}")

**Alpha Parameter:**

* ``alpha=0.0``: Pure lexical search (BM25)
* ``alpha=0.5``: Balanced hybrid (recommended)
* ``alpha=1.0``: Pure semantic search (vector)

Lexical-Only Search
-------------------

Use BM25 only for exact keyword matching::

    with ix.searcher() as searcher:
        results = searcher.lexical_only("python programming", limit=10)

Semantic-Only Search
--------------------

Use vectors only for conceptual queries::

    with ix.searcher() as searcher:
        # Finds conceptually similar docs even without keyword overlap
        results = searcher.semantic_only("AI and neural networks", limit=10)

Components
==========

UnifiedIndex
------------

The main index class combining BM25 and vector search.

**Constructor Parameters:**

* ``index_dir``: Directory for the index
* ``schema``: Field schema
* ``connection_string``: PostgreSQL connection URL
* ``embedder``: Embedding provider
* ``id_field``: Field containing document IDs (default: "id")
* ``searchable_fields``: Fields to use for embeddings (default: all TEXT fields)

**Methods:**

* ``writer(**kwargs)``: Returns UnifiedWriter
* ``searcher(**kwargs)``: Returns UnifiedSearcher
* ``reader(**kwargs)``: Returns BM25Reader
* ``optimize()``: Optimizes both indexes
* ``doc_count()``: Returns document count
* ``close()``: Closes both stores

UnifiedWriter
-------------

Handles transactional writes across both stores::

    with ix.writer() as writer:
        # Add document (indexed in both BM25 and vectors)
        writer.add_document(id="1", content="Document text")

        # Update document (deletes old, adds new in both stores)
        writer.update_document(id="1", content="Updated text")

        # Delete document (removes from both stores)
        writer.delete_document(id="1")

        # Delete by query
        from semlix.qparser import QueryParser
        qp = QueryParser("content", ix.schema)
        query = qp.parse("obsolete")
        writer.delete_by_query(query)

**Transaction Guarantees:**

* Writes are atomic across both stores
* If vector storage fails, BM25 changes roll back
* Automatic embedding generation
* Configurable batch processing

UnifiedSearcher
---------------

Enhanced searcher with hybrid search capabilities::

    with ix.searcher() as searcher:
        # Hybrid search
        results = searcher.hybrid_search("query", alpha=0.5)

        # With facets
        results, facets = searcher.search_with_facets(
            "python",
            facet_fields=["category", "author"],
            limit=100
        )

        # Phrase search
        results = searcher.phrase_search(
            "content",
            "machine learning",
            slop=0
        )

        # Sorted search
        results = searcher.search_sorted(
            "python",
            sort_by=[("published", True), ("score", True)],
            limit=10
        )

**Methods:**

* ``hybrid_search(...)``: Combined lexical + semantic
* ``lexical_only(...)``: BM25 only
* ``semantic_only(...)``: Vector only
* ``search_with_facets(...)``: Hybrid search with aggregations
* ``phrase_search(...)``: Exact phrase matching
* ``sort_results(...)``: Sort existing results
* ``search_sorted(...)``: Search with custom sorting

Advanced Features
=================

Faceted Hybrid Search
---------------------

Combine hybrid search with faceting::

    with ix.searcher() as searcher:
        results, facets = searcher.search_with_facets(
            "machine learning",
            facet_fields=["category", "author", "year"],
            limit=100,
            facet_limit=10,
            alpha=0.5
        )

        # Access results
        for r in results[:10]:
            print(r.stored_fields['title'])

        # Access facets
        print("Categories:", facets["category"])
        # {"ai": 45, "programming": 32, "database": 12}

        print("Authors:", facets["author"])
        # {"Alice": 23, "Bob": 18, "Charlie": 15}

Phrase Queries
--------------

Find exact phrases in hybrid results::

    with ix.searcher() as searcher:
        # Exact phrase
        results = searcher.phrase_search(
            field="content",
            phrase="machine learning",
            slop=0,
            limit=10
        )

        # With slop (allows words in between)
        results = searcher.phrase_search(
            field="content",
            phrase="machine learning",
            slop=2,  # "machine X Y learning" matches
            limit=10
        )

Sorted Hybrid Search
--------------------

Sort hybrid results by custom criteria::

    with ix.searcher() as searcher:
        # Sort by date (newest first), then by relevance score
        results = searcher.search_sorted(
            "python programming",
            sort_by=[
                ("published", True),   # Descending
                ("score", True)        # Descending
            ],
            limit=20,
            alpha=0.5
        )

        for r in results:
            doc = r.stored_fields
            print(f"{doc['title']} - {doc['published']}")

Configuration
=============

Embedding Provider
------------------

Choose an embedding model based on your needs::

    from semlix.semantic import SentenceTransformerProvider

    # Fast and lightweight (384-dim)
    embedder = SentenceTransformerProvider("all-MiniLM-L6-v2")

    # Better quality (768-dim)
    embedder = SentenceTransformerProvider("all-mpnet-base-v2")

    # Multilingual
    embedder = SentenceTransformerProvider("paraphrase-multilingual-MiniLM-L12-v2")

Vector Store Configuration
--------------------------

Configure PostgreSQL vector storage::

    from semlix.semantic.stores import PgVectorStore

    vector_store = PgVectorStore(
        connection_string="postgresql://localhost/mydb",
        dimension=384,
        distance_metric="cosine",  # or "l2", "inner_product"
        pool_size=10
    )

    # Create HNSW index for fast search
    vector_store.create_index(
        index_type="hnsw",
        m=16,              # HNSW parameter
        ef_construction=64 # HNSW parameter
    )

Searchable Fields
-----------------

Control which fields are used for embeddings::

    ix = create_unified_index(
        index_dir="my_index",
        schema=schema,
        connection_string=pg_url,
        embedder=embedder,
        searchable_fields=["title", "content"]  # Only these fields
    )

By default, all TEXT fields are used for embedding generation.

Fusion Methods
--------------

Choose how to combine lexical and semantic scores::

    from semlix.semantic.fusion import FusionMethod

    with ix.searcher() as searcher:
        results = searcher.hybrid_search(
            "query",
            fusion_method=FusionMethod.RRF,  # Reciprocal Rank Fusion
            alpha=0.5
        )

**Available Methods:**

* ``RRF`` (Reciprocal Rank Fusion): Recommended, parameter-free
* ``LINEAR``: Weighted linear combination
* ``DBSF`` (Distribution-Based Score Fusion): Normalizes score distributions
* ``RELATIVE_SCORE``: Relative scoring normalization

Migration
=========

From FileStorage + NumpyVectorStore
------------------------------------

Migrate existing indexes to UnifiedIndex::

    from semlix.tools import migrate_to_unified
    from semlix.semantic import SentenceTransformerProvider

    embedder = SentenceTransformerProvider()

    migrate_to_unified(
        source_dir="old_whoosh_index",
        target_dir="new_unified_index",
        connection_string="postgresql://localhost/mydb",
        embedder=embedder,
        vector_store_path="old_vectors.pkl",  # Reuse existing vectors
        batch_size=100
    )

**Migration Process:**

1. Opens source index and vector store
2. Creates new UnifiedIndex
3. Migrates documents with embeddings
4. Reuses existing vectors when available
5. Generates new vectors for missing documents
6. Optimizes both indexes

From BM25Index
--------------

Add vector search to existing BM25 index::

    from semlix.bm25 import open_bm25_index
    from semlix.unified import UnifiedIndex
    from semlix.semantic import SentenceTransformerProvider
    from semlix.semantic.stores import PgVectorStore

    # Open existing BM25 index
    bm25_ix = open_bm25_index("my_bm25_index")

    # Create vector store
    embedder = SentenceTransformerProvider()
    vector_store = PgVectorStore(
        "postgresql://localhost/mydb",
        dimension=embedder.dimension
    )

    # Generate embeddings for existing documents
    docs = []
    with bm25_ix.reader() as reader:
        for doc in reader.iter_docs():
            docs.append(doc)

    # Extract text and generate embeddings
    texts = [doc.get("content", "") for doc in docs]
    doc_ids = [doc.get("id", str(i)) for i, doc in enumerate(docs)]
    embeddings = embedder.encode(texts)

    # Add to vector store
    vector_store.add(doc_ids, embeddings)

    # Create unified index
    unified_ix = UnifiedIndex(
        index_dir="unified_index",
        schema=bm25_ix.schema,
        connection_string="postgresql://localhost/mydb",
        embedder=embedder,
        bm25_index=bm25_ix,
        vector_store=vector_store
    )

Performance
===========

Search Performance
------------------

**Hybrid Search:**

* 500+ queries/second (10K documents)
* ~5-10ms latency (p50)
* Scales well with document count

**Lexical-Only:**

* 1000+ queries/second
* ~1-2ms latency

**Semantic-Only:**

* ~100 queries/second (with HNSW index)
* ~10-20ms latency

Indexing Performance
--------------------

**With Embedding Generation:**

* ~100 documents/second
* Depends on embedding model speed
* Can batch for better throughput

**Optimization:**

Use batch processing for bulk indexing::

    batch_size = 100
    batch = []

    with ix.writer() as writer:
        for doc in documents:
            batch.append(doc)

            if len(batch) >= batch_size:
                for doc_fields in batch:
                    writer.add_document(**doc_fields)
                batch = []

Memory Usage
------------

==================  ========  ==========
Component           10K docs  100K docs
==================  ========  ==========
BM25 Index          100MB     500MB
Vector Store (PG)   40MB      400MB
Total (approx)      140MB     900MB
==================  ========  ==========

Disk Usage
----------

==================  ========  ==========
Component           10K docs  100K docs
==================  ========  ==========
BM25 Index          50MB      250MB
PostgreSQL (total)  100MB     800MB
Total (approx)      150MB     1050MB
==================  ========  ==========

Examples
========

Basic Hybrid Search
-------------------

::

    from semlix.unified import create_unified_index
    from semlix.fields import Schema, TEXT, ID
    from semlix.semantic import SentenceTransformerProvider

    schema = Schema(id=ID(stored=True), content=TEXT(stored=True))
    embedder = SentenceTransformerProvider()

    ix = create_unified_index(
        "my_index",
        schema,
        "postgresql://localhost/mydb",
        embedder
    )

    # Index
    with ix.writer() as writer:
        writer.add_document(
            id="1",
            content="Python is a programming language"
        )
        writer.add_document(
            id="2",
            content="Machine learning uses neural networks"
        )

    # Search
    with ix.searcher() as searcher:
        # Hybrid: finds both keyword and semantic matches
        results = searcher.hybrid_search("coding in python", limit=10)

Complete Example with All Features
-----------------------------------

::

    from semlix.unified import create_unified_index
    from semlix.fields import Schema, TEXT, ID, KEYWORD, DATETIME
    from semlix.semantic import SentenceTransformerProvider

    schema = Schema(
        id=ID(stored=True),
        title=TEXT(stored=True),
        content=TEXT(stored=True),
        category=KEYWORD(stored=True),
        published=DATETIME(stored=True)
    )

    embedder = SentenceTransformerProvider("all-MiniLM-L6-v2")

    ix = create_unified_index(
        "my_index",
        schema,
        "postgresql://localhost/mydb",
        embedder
    )

    # Index documents
    with ix.writer() as writer:
        writer.add_document(
            id="1",
            title="AI Basics",
            content="Introduction to artificial intelligence...",
            category="ai",
            published="2024-01-15"
        )
        # ... more documents ...

    # Search with all features
    with ix.searcher() as searcher:
        # Hybrid search with facets
        results, facets = searcher.search_with_facets(
            "artificial intelligence",
            facet_fields=["category"],
            limit=50,
            alpha=0.5
        )

        # Sort by date
        sorted_results = searcher.sort_results(
            results,
            [("published", True)]
        )

        # Phrase search
        phrase_results = searcher.phrase_search(
            "content",
            "machine learning"
        )

Best Practices
==============

1. **Choose appropriate alpha:**

   * Use ``alpha=0.3-0.5`` for balanced search
   * Use ``alpha=0.0`` for exact keyword matching
   * Use ``alpha=0.8-1.0`` for conceptual/semantic queries

2. **Batch indexing for performance:**

   * Index in batches of 100-1000 documents
   * Commit once per batch, not per document

3. **Create HNSW index for vectors:**

   * Essential for good semantic search performance
   * Create after bulk indexing::

       ix.optimize()  # Optimizes both BM25 and vector indexes

4. **Choose embedding model wisely:**

   * Start with ``all-MiniLM-L6-v2`` (fast, good quality)
   * Upgrade to ``all-mpnet-base-v2`` if quality matters more than speed
   * Use multilingual models only if needed

5. **Monitor PostgreSQL:**

   * Regular VACUUM ANALYZE
   * Monitor connection pool usage
   * Consider replication for high availability

Troubleshooting
===============

Slow Semantic Search
--------------------

**Problem:** Vector search is slow (>100ms per query)

**Solutions:**

1. Create HNSW index::

    ix.vector_store.create_index(index_type="hnsw")

2. Tune HNSW parameters::

    ix.vector_store.create_index(
        index_type="hnsw",
        m=32,              # Higher = better quality, slower build
        ef_construction=128 # Higher = better quality, slower build
    )

Memory Issues
-------------

**Problem:** High memory usage during indexing

**Solutions:**

1. Use smaller batches
2. Enable memory mapping for BM25::

    from semlix.stores import BM25sStore
    store = BM25sStore.load(index_dir, mmap=True)

3. Reduce connection pool size

Connection Pool Exhausted
--------------------------

**Problem:** PostgreSQL connection errors

**Solutions:**

1. Increase pool size::

    vector_store = PgVectorStore(
        connection_string=pg_url,
        pool_size=50  # Increase from default 10
    )

2. Close searchers when done
3. Use context managers (``with`` statements)

See Also
========

* :doc:`bm25` - BM25 index documentation
* :doc:`semantic` - Semantic search and HybridSearcher
* :doc:`indexing` - General indexing concepts
* :doc:`searching` - Search and query syntax
