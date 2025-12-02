====================
Migration Guide
====================

This guide covers migrating indexes between different storage backends in semlix,
including upgrading from FileStorage to BM25 or UnifiedIndex.

Overview
========

semlix supports several storage backends:

* **FileStorage**: Traditional Whoosh file-based storage
* **BM25Index**: High-performance BM25 storage (10-100x faster)
* **UnifiedIndex**: Combined BM25 + vector search

Migration tools automate the process of moving data between these backends.

Migration Scenarios
===================

FileStorage → BM25Index
------------------------

**When to use:**

* You want 10-100x faster search
* You don't need semantic/vector search
* You want lower memory usage

**Benefits:**

* Dramatically faster search (1000+ queries/second)
* Lower memory footprint (3x less)
* Faster indexing (6x faster)
* Same API, drop-in replacement

FileStorage + Vectors → UnifiedIndex
-------------------------------------

**When to use:**

* You're using HybridSearcher with separate stores
* You want unified management of lexical + semantic
* You want ACID transactions across both stores
* You want automatic embedding generation

**Benefits:**

* Simplified architecture (one index instead of two)
* Transactional writes (atomic updates)
* Better performance
* Enhanced features (faceting on hybrid results, etc.)

NumpyVectorStore → PgVectorStore
---------------------------------

**When to use:**

* You want better scalability for vectors
* You need ACID transactions
* You want metadata filtering on vectors
* You need backup/recovery tools

**Benefits:**

* PostgreSQL reliability and scalability
* HNSW indexing for fast search
* JSONB metadata filtering
* Professional backup/recovery

Quick Start
===========

Simple BM25 Migration
----------------------

::

    from semlix.tools import migrate_to_bm25

    migrate_to_bm25(
        source_dir="old_whoosh_index",
        target_dir="new_bm25_index",
        batch_size=1000,
        verbose=True
    )

The migration will:

1. Open the source FileStorage index
2. Create a new BM25 index with the same schema
3. Copy all documents with progress tracking
4. Optimize the new index

Simple Unified Migration
-------------------------

::

    from semlix.tools import migrate_to_unified
    from semlix.semantic import SentenceTransformerProvider

    embedder = SentenceTransformerProvider("all-MiniLM-L6-v2")

    migrate_to_unified(
        source_dir="old_whoosh_index",
        target_dir="new_unified_index",
        connection_string="postgresql://localhost/mydb",
        embedder=embedder,
        vector_store_path="old_vectors.pkl",  # Optional: reuse existing vectors
        batch_size=100,
        verbose=True
    )

The migration will:

1. Open source index and optional vector store
2. Create new UnifiedIndex
3. Copy documents and vectors
4. Generate new embeddings for documents without vectors
5. Optimize both indexes

Detailed Migration
==================

Using IndexMigrator
-------------------

For more control, use the IndexMigrator class::

    from semlix.tools import IndexMigrator

    migrator = IndexMigrator(verbose=True)

    # BM25 migration
    migrator.migrate_to_bm25(
        source_dir="old_index",
        target_dir="new_index",
        batch_size=1000
    )

    # Unified migration
    migrator.migrate_to_unified(
        source_dir="old_index",
        target_dir="new_index",
        connection_string="postgresql://localhost/mydb",
        embedder=embedder,
        vector_store_path="vectors.pkl",
        batch_size=100
    )

    # Vectors-only migration
    migrator.migrate_vectors_only(
        source_store_path="vectors.pkl",
        target_connection_string="postgresql://localhost/mydb",
        table_name="my_vectors"
    )

Custom Migration
----------------

For advanced scenarios, write custom migration code::

    from semlix.tools import IndexMigrator
    from semlix.index import open_dir
    from semlix.bm25 import create_bm25_index

    # Open source
    source = open_dir("old_index")

    # Create target
    target = create_bm25_index("new_index", source.schema)

    # Custom migration with filtering
    with source.searcher() as searcher:
        with target.writer() as writer:
            for docnum in range(searcher.reader().doc_count_all()):
                fields = searcher.stored_fields(docnum)

                # Custom logic: only migrate certain documents
                if fields.get("category") in ["important", "archive"]:
                    writer.add_document(**fields)

                    if docnum % 100 == 0:
                        print(f"Processed {docnum} documents")

    # Optimize
    target.optimize()

    source.close()
    target.close()

Migration Strategies
====================

Zero-Downtime Migration
-----------------------

For production systems, use a dual-write strategy:

**Phase 1: Dual Write**

::

    from semlix.index import open_dir
    from semlix.bm25 import open_bm25_index

    # Open both indexes
    old_ix = open_dir("old_index")
    new_ix = open_bm25_index("new_index")

    # Write to both
    def add_document(**fields):
        with old_ix.writer() as w1:
            w1.add_document(**fields)

        with new_ix.writer() as w2:
            w2.add_document(**fields)

**Phase 2: Migrate Historical Data**

::

    # Migrate old data in background
    from semlix.tools import migrate_to_bm25

    migrate_to_bm25("old_index", "new_index")

**Phase 3: Switch Reads**

::

    # Change searcher to use new index
    # old: searcher = old_ix.searcher()
    searcher = new_ix.searcher()

**Phase 4: Remove Old Index**

After verifying new index works, remove dual writes and old index.

Incremental Migration
---------------------

For very large indexes, migrate in chunks::

    from semlix.index import open_dir
    from semlix.bm25 import create_bm25_index, open_bm25_index

    source = open_dir("huge_index")
    target = create_bm25_index("new_index", source.schema)

    chunk_size = 10000
    offset = 0

    with source.searcher() as searcher:
        total = searcher.reader().doc_count_all()

        while offset < total:
            print(f"Migrating documents {offset} to {offset + chunk_size}")

            with target.writer() as writer:
                for docnum in range(offset, min(offset + chunk_size, total)):
                    fields = searcher.stored_fields(docnum)
                    writer.add_document(**fields)

            offset += chunk_size

            # Optional: backup checkpoint
            target.optimize()

    source.close()
    target.close()

Testing Migration
-----------------

Always test migration on a copy first::

    import shutil
    from semlix.tools import migrate_to_bm25

    # Copy original index
    shutil.copytree("production_index", "test_index")

    # Test migration
    migrate_to_bm25("test_index", "test_bm25_index")

    # Verify document counts
    from semlix.index import open_dir
    from semlix.bm25 import open_bm25_index

    old_ix = open_dir("test_index")
    new_ix = open_bm25_index("test_bm25_index")

    assert old_ix.doc_count() == new_ix.doc_count()

    # Spot check some documents
    with old_ix.searcher() as s1, new_ix.searcher() as s2:
        old_doc = s1.stored_fields(0)
        new_doc = s2.stored_fields(0)
        assert old_doc == new_doc

Performance Considerations
==========================

Migration Speed
---------------

**Typical speeds (10K document index):**

* BM25 migration: ~5,000 docs/sec
* Unified migration (with embeddings): ~100 docs/sec
* Vector-only migration: ~10,000 vectors/sec

**Factors affecting speed:**

* Disk I/O speed
* Document size
* Embedding model speed (for unified migration)
* Batch size
* Available memory

Optimization
------------

**Increase batch size for faster migration:**

::

    migrate_to_bm25(
        source_dir="old_index",
        target_dir="new_index",
        batch_size=5000  # Default: 1000
    )

**For unified migration with embeddings:**

::

    migrate_to_unified(
        source_dir="old_index",
        target_dir="new_index",
        connection_string=pg_url,
        embedder=embedder,
        batch_size=500  # Larger batches for embedding generation
    )

Memory Usage
------------

Migration memory usage depends on batch size:

==================  ==========  ============
Batch Size          BM25        Unified
==================  ==========  ============
100                 ~50MB       ~100MB
1000                ~200MB      ~500MB
5000                ~800MB      ~2GB
==================  ==========  ============

For memory-constrained systems, use smaller batches.

Compatibility
=============

Schema Compatibility
--------------------

The target index must support all field types in the source schema.

**Fully Compatible:**

* ID, TEXT, KEYWORD, NUMERIC, DATETIME, BOOLEAN
* All analyzers (StandardAnalyzer, StemmingAnalyzer, etc.)
* Stored and indexed fields

**Partially Compatible:**

* Custom field types may need adjustment
* Some FileStorage-specific features not available in BM25

Field Mapping
-------------

All standard semlix fields migrate automatically::

    # Source schema
    schema = Schema(
        id=ID(stored=True),
        title=TEXT(stored=True, analyzer=StandardAnalyzer()),
        content=TEXT(stored=True),
        tags=KEYWORD(stored=True),
        price=NUMERIC(stored=True),
        published=DATETIME(stored=True)
    )

    # Migrates to BM25 with same schema
    migrate_to_bm25("old_index", "new_index")

    # All fields preserved with same types and analyzers

Data Integrity
==============

Verification
------------

Always verify migration success::

    from semlix.index import open_dir
    from semlix.bm25 import open_bm25_index

    old_ix = open_dir("old_index")
    new_ix = open_bm25_index("new_index")

    # Check document count
    assert old_ix.doc_count() == new_ix.doc_count(), "Document count mismatch"

    # Verify schema
    assert old_ix.schema == new_ix.schema, "Schema mismatch"

    # Spot check documents
    with old_ix.searcher() as s1, new_ix.searcher() as s2:
        for i in range(min(100, old_ix.doc_count())):
            old_doc = s1.stored_fields(i)
            new_doc = s2.stored_fields(i)

            assert old_doc == new_doc, f"Document {i} mismatch"

    print("✓ Migration verification passed")

Rollback
--------

Keep the original index until migration is verified::

    # 1. Migrate to new index
    migrate_to_bm25("production_index", "new_bm25_index")

    # 2. Test new index thoroughly
    test_new_index("new_bm25_index")

    # 3. Switch application to new index
    deploy_with_new_index()

    # 4. Monitor for 24-48 hours

    # 5. Only then remove old index
    # shutil.rmtree("production_index")  # Wait until confident

Backup
------

Always backup before migration::

    import shutil
    import datetime

    # Create timestamped backup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"production_index_backup_{timestamp}"

    shutil.copytree("production_index", backup_name)

    # Now safe to migrate
    migrate_to_bm25("production_index", "new_bm25_index")

Common Issues
=============

Schema Mismatch
---------------

**Problem:** Source schema has custom field types not supported by target

**Solution:** Create compatible schema manually::

    from semlix.fields import Schema, TEXT, ID
    from semlix.bm25 import create_bm25_index

    # Create compatible schema
    new_schema = Schema(
        id=ID(stored=True),
        content=TEXT(stored=True)  # Simplified from complex field
    )

    target = create_bm25_index("new_index", new_schema)

    # Custom migration with field mapping
    # ... map source fields to target fields ...

Memory Errors
-------------

**Problem:** Migration runs out of memory

**Solutions:**

1. Reduce batch size::

    migrate_to_bm25(
        source_dir="old_index",
        target_dir="new_index",
        batch_size=100  # Smaller batches
    )

2. Use incremental migration (see Incremental Migration above)

3. Increase system memory or swap space

PostgreSQL Connection Errors
-----------------------------

**Problem:** "Too many connections" error during unified migration

**Solutions:**

1. Increase connection pool size::

    from semlix.semantic.stores import PgVectorStore

    vector_store = PgVectorStore(
        connection_string=pg_url,
        pool_size=5  # Reduce from default 10
    )

2. Close connections properly (use context managers)

3. Increase PostgreSQL max_connections setting

Document Count Mismatch
------------------------

**Problem:** Target index has fewer documents than source

**Causes:**

* Migration was interrupted
* Some documents failed to migrate
* Filter was applied (in custom migration)

**Solutions:**

1. Check migration logs for errors
2. Re-run migration from scratch
3. Use verification script to identify missing documents

Examples
========

Complete BM25 Migration
-----------------------

::

    from semlix.tools import migrate_to_bm25
    from semlix.index import open_dir
    from semlix.bm25 import open_bm25_index

    print("Starting migration...")

    # Migrate
    migrate_to_bm25(
        source_dir="whoosh_index",
        target_dir="bm25_index",
        batch_size=1000,
        verbose=True
    )

    # Verify
    old_ix = open_dir("whoosh_index")
    new_ix = open_bm25_index("bm25_index")

    print(f"Old index: {old_ix.doc_count()} documents")
    print(f"New index: {new_ix.doc_count()} documents")

    assert old_ix.doc_count() == new_ix.doc_count()
    print("✓ Migration successful!")

Complete Unified Migration
---------------------------

::

    from semlix.tools import migrate_to_unified
    from semlix.semantic import SentenceTransformerProvider
    from semlix.unified import open_unified_index

    # Setup
    embedder = SentenceTransformerProvider("all-MiniLM-L6-v2")
    pg_url = "postgresql://localhost/mydb"

    print(f"Using embedder: {embedder.model_name}")
    print(f"Dimension: {embedder.dimension}")

    # Migrate
    migrate_to_unified(
        source_dir="whoosh_index",
        target_dir="unified_index",
        connection_string=pg_url,
        embedder=embedder,
        vector_store_path="old_vectors.pkl",
        batch_size=100,
        verbose=True
    )

    # Test
    ix = open_unified_index("unified_index", embedder)

    with ix.searcher() as searcher:
        results = searcher.hybrid_search("test query", limit=5)
        print(f"Found {len(results)} results")

    print("✓ Unified migration successful!")

Migration with Filtering
-------------------------

::

    from semlix.index import open_dir
    from semlix.bm25 import create_bm25_index

    source = open_dir("all_documents")
    target = create_bm25_index("filtered_documents", source.schema)

    # Only migrate recent documents
    from datetime import datetime, timedelta
    cutoff = datetime.now() - timedelta(days=365)

    migrated = 0
    skipped = 0

    with source.searcher() as searcher:
        with target.writer() as writer:
            for docnum in range(searcher.reader().doc_count_all()):
                fields = searcher.stored_fields(docnum)

                # Check date
                if "published" in fields:
                    pub_date = fields["published"]
                    if isinstance(pub_date, datetime) and pub_date >= cutoff:
                        writer.add_document(**fields)
                        migrated += 1
                    else:
                        skipped += 1

                if (migrated + skipped) % 1000 == 0:
                    print(f"Processed: {migrated + skipped} "
                          f"(migrated: {migrated}, skipped: {skipped})")

    print(f"✓ Filtered migration complete:")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped: {skipped}")

Best Practices
==============

1. **Always backup first**
2. **Test on a copy** before migrating production data
3. **Verify document counts** after migration
4. **Spot check documents** to ensure data integrity
5. **Monitor during migration** for errors or issues
6. **Keep old index** until new one is proven in production
7. **Document your migration** process for future reference
8. **Plan for rollback** in case of issues

See Also
========

* :doc:`bm25` - BM25 index documentation
* :doc:`unified` - Unified index documentation
* :doc:`indexing` - General indexing concepts
