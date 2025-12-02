#!/usr/bin/env python
"""Example of migrating indexes between storage backends.

This example shows how to migrate from:
1. FileStorage (Whoosh) → BM25Index
2. FileStorage + NumpyVectorStore → UnifiedIndex
"""

import os


def migrate_to_bm25_example():
    """Example: Migrate FileStorage to BM25Index."""
    print("\n" + "=" * 60)
    print("Migration Example 1: FileStorage → BM25Index")
    print("=" * 60)

    from semlix.tools import migrate_to_bm25

    # Migrate existing Whoosh index to BM25
    print("\nMigrating index...")
    migrate_to_bm25(
        source_dir="old_whoosh_index",  # Your existing Whoosh index
        target_dir="new_bm25_index",     # New BM25 index location
        batch_size=1000,
        verbose=True
    )

    print("\nMigration complete!")
    print("You can now use BM25Index for 10-100x faster search")


def migrate_to_unified_example():
    """Example: Migrate FileStorage + vectors to UnifiedIndex."""
    print("\n" + "=" * 60)
    print("Migration Example 2: FileStorage + Vectors → UnifiedIndex")
    print("=" * 60)

    from semlix.tools import migrate_to_unified
    from semlix.semantic import SentenceTransformerProvider

    # Setup
    pg_url = os.getenv("POSTGRES_URL", "postgresql://localhost/semlix_demo")
    embedder = SentenceTransformerProvider("all-MiniLM-L6-v2")

    print(f"\nUsing PostgreSQL: {pg_url}")
    print(f"Using embedder: {embedder.model_name}")

    # Migrate with existing vectors
    print("\nMigrating index with vectors...")
    migrate_to_unified(
        source_dir="old_whoosh_index",           # Existing Whoosh index
        target_dir="new_unified_index",          # New unified index
        connection_string=pg_url,
        embedder=embedder,
        vector_store_path="old_vectors.pkl",     # Existing NumpyVectorStore
        batch_size=100,
        verbose=True
    )

    print("\nMigration complete!")
    print("You now have a unified index with both BM25 and pgvector")


def migrate_vectors_only_example():
    """Example: Migrate NumpyVectorStore to PgVectorStore."""
    print("\n" + "=" * 60)
    print("Migration Example 3: NumpyVectorStore → PgVectorStore")
    print("=" * 60)

    from semlix.tools import IndexMigrator

    pg_url = os.getenv("POSTGRES_URL", "postgresql://localhost/semlix_demo")

    migrator = IndexMigrator(verbose=True)

    print(f"\nMigrating vectors to PostgreSQL: {pg_url}")
    migrator.migrate_vectors_only(
        source_store_path="old_vectors.pkl",
        target_connection_string=pg_url,
        table_name="my_vectors"
    )

    print("\nMigration complete!")
    print("Vectors are now in PostgreSQL with HNSW index")


def custom_migration_example():
    """Example: Custom migration with progress tracking."""
    print("\n" + "=" * 60)
    print("Migration Example 4: Custom Migration")
    print("=" * 60)

    from semlix.tools import IndexMigrator
    from semlix.index import open_dir
    from semlix.bm25 import create_bm25_index

    # Create custom migrator
    migrator = IndexMigrator(verbose=True)

    # Open source
    print("\nOpening source index...")
    source = open_dir("old_index")
    print(f"Source has {source.doc_count()} documents")

    # Create target
    print("\nCreating target BM25 index...")
    target = create_bm25_index("custom_bm25_index", source.schema)

    # Custom migration with filtering
    print("\nMigrating with filtering...")
    with source.searcher() as searcher:
        with target.writer() as writer:
            migrated = 0
            skipped = 0

            for docnum in range(searcher.reader().doc_count_all()):
                fields = searcher.stored_fields(docnum)

                # Custom filter: only migrate docs with category="important"
                if fields.get("category") == "important":
                    writer.add_document(**fields)
                    migrated += 1
                else:
                    skipped += 1

                if (migrated + skipped) % 100 == 0:
                    print(f"  Processed {migrated + skipped} docs (migrated: {migrated}, skipped: {skipped})")

    print(f"\nMigration complete!")
    print(f"  Migrated: {migrated} documents")
    print(f"  Skipped: {skipped} documents")

    source.close()
    target.close()


def main():
    print("=" * 60)
    print("Semlix Migration Examples")
    print("=" * 60)

    print("\nThese examples show different migration scenarios:")
    print("1. FileStorage → BM25Index (faster search)")
    print("2. FileStorage + Vectors → UnifiedIndex (hybrid search)")
    print("3. NumpyVectorStore → PgVectorStore (better scalability)")
    print("4. Custom migration with filtering")

    print("\n" + "=" * 60)
    print("Example 1: Simple BM25 Migration")
    print("=" * 60)
    print("\nfrom semlix.tools import migrate_to_bm25")
    print("")
    print("migrate_to_bm25(")
    print("    source_dir='old_index',")
    print("    target_dir='new_bm25_index'")
    print(")")

    print("\n" + "=" * 60)
    print("Example 2: Unified Index Migration")
    print("=" * 60)
    print("\nfrom semlix.tools import migrate_to_unified")
    print("from semlix.semantic import SentenceTransformerProvider")
    print("")
    print("embedder = SentenceTransformerProvider()")
    print("migrate_to_unified(")
    print("    source_dir='old_index',")
    print("    target_dir='new_unified_index',")
    print("    connection_string='postgresql://localhost/mydb',")
    print("    embedder=embedder,")
    print("    vector_store_path='old_vectors.pkl'")
    print(")")

    print("\n" + "=" * 60)
    print("Example 3: Step-by-Step Migration")
    print("=" * 60)
    print("\nfrom semlix.tools import IndexMigrator")
    print("")
    print("migrator = IndexMigrator(verbose=True)")
    print("")
    print("# Step 1: Migrate index")
    print("migrator.migrate_to_bm25('old_index', 'new_index')")
    print("")
    print("# Step 2: Migrate vectors")
    print("migrator.migrate_vectors_only(")
    print("    'vectors.pkl',")
    print("    'postgresql://localhost/mydb'")
    print(")")

    print("\n" + "=" * 60)
    print("\nTo actually run migrations, uncomment the desired example:")
    print("# migrate_to_bm25_example()")
    print("# migrate_to_unified_example()")
    print("# migrate_vectors_only_example()")
    print("# custom_migration_example()")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
