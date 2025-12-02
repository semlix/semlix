"""Example usage of PostgreSQLDriver for lexical index storage.

This example demonstrates the basic PostgreSQL driver for storing
Semlix lexical indexes in a database instead of files.

Status: Proof of Concept
- Shows document storage
- Shows term indexing
- Shows simple search
- Full Whoosh integration pending

Requirements:
    pip install psycopg2-binary

Setup:
    1. Create PostgreSQL database
    2. Run schema: psql -d mydb -f src/semlix/filedb/drivers/schema_lexical.sql
    3. Update CONNECTION_STRING below
    4. Run: python examples/postgresql_driver_example.py
"""

# Configuration - UPDATE THIS
CONNECTION_STRING = "postgresql://postgres:password@localhost:5432/semlix_test"

def main():
    print("=" * 80)
    print("PostgreSQL Storage Driver Example")
    print("=" * 80)

    # 1. Import and create driver
    print("\n1. Creating PostgreSQL driver...")
    from semlix.filedb.drivers import PostgreSQLDriver

    driver = PostgreSQLDriver(
        connection_string=CONNECTION_STRING,
        index_name="example_index"
    )

    print(f"   ✓ Driver created: {driver}")

    # 2. Initialize schema
    print("\n2. Initializing database schema...")
    driver.create()
    print("   ✓ Schema created")

    # 3. Save index schema metadata
    print("\n3. Saving index schema metadata...")
    schema_dict = {
        "fields": {
            "id": {"type": "ID", "stored": True, "unique": True},
            "title": {"type": "TEXT", "stored": True},
            "content": {"type": "TEXT", "stored": True}
        }
    }
    driver.save_schema(schema_dict)
    print("   ✓ Schema metadata saved")

    # 4. Add documents
    print("\n4. Adding documents...")

    documents = [
        {
            "doc_id": "doc1",
            "doc_number": 0,
            "stored_fields": {
                "id": "doc1",
                "title": "Python Programming",
                "content": "Python is a high-level programming language"
            },
            "field_lengths": {"title": 2, "content": 6}
        },
        {
            "doc_id": "doc2",
            "doc_number": 1,
            "stored_fields": {
                "id": "doc2",
                "title": "Machine Learning",
                "content": "Machine learning is a subset of artificial intelligence"
            },
            "field_lengths": {"title": 2, "content": 7}
        },
        {
            "doc_id": "doc3",
            "doc_number": 2,
            "stored_fields": {
                "id": "doc3",
                "title": "PostgreSQL Database",
                "content": "PostgreSQL is an advanced relational database system"
            },
            "field_lengths": {"title": 2, "content": 6}
        },
    ]

    for doc in documents:
        driver.add_document(**doc)

    print(f"   ✓ Added {len(documents)} documents")
    print(f"   Total docs in index: {driver.get_doc_count()}")

    # 5. Index terms from documents
    print("\n5. Indexing terms...")

    # Simulate tokenization and indexing
    # In real implementation, this would be done by Whoosh analyzer
    terms_to_index = [
        # Document 1 terms
        ("content", "python", "doc1", 2, [0, 3]),
        ("content", "programming", "doc1", 1, [5]),
        ("content", "language", "doc1", 1, [6]),
        ("title", "python", "doc1", 1, [0]),

        # Document 2 terms
        ("content", "machine", "doc2", 1, [0]),
        ("content", "learning", "doc2", 1, [1]),
        ("content", "intelligence", "doc2", 1, [6]),
        ("title", "machine", "doc2", 1, [0]),

        # Document 3 terms
        ("content", "postgresql", "doc3", 1, [0]),
        ("content", "database", "doc3", 1, [4]),
        ("title", "postgresql", "doc3", 1, [0]),
        ("title", "database", "doc3", 1, [1]),
    ]

    for field, term, doc_id, freq, positions in terms_to_index:
        driver.add_term(field, term, doc_id, freq, positions)

    print(f"   ✓ Indexed {len(terms_to_index)} term postings")

    # 6. Search for terms
    print("\n6. Searching for terms...")

    # Search for "python"
    print("\n   Searching for 'python' in content:")
    results = driver.search_term("content", "python", limit=10)

    for result in results:
        print(f"      • {result['doc_id']}")
        print(f"        Title: {result['stored_fields']['title']}")
        print(f"        Frequency: {result['frequency']}")
        print(f"        Positions: {result['positions']}")

    # Search for "database"
    print("\n   Searching for 'database' in content:")
    results = driver.search_term("content", "database", limit=10)

    for result in results:
        print(f"      • {result['doc_id']}")
        print(f"        Title: {result['stored_fields']['title']}")
        print(f"        Frequency: {result['frequency']}")

    # 7. Get document by ID
    print("\n7. Getting document by ID...")
    doc = driver.get_document("doc2")

    if doc:
        print(f"   Document: {doc['doc_id']}")
        print(f"   Fields: {doc['stored_fields']}")
        print(f"   Field lengths: {doc['field_lengths']}")

    # 8. Get term statistics
    print("\n8. Getting term statistics...")
    term_info = driver.get_term_info("content", "python")

    if term_info:
        print(f"   Term: 'python'")
        print(f"   Document frequency: {term_info['doc_frequency']}")
        print(f"   Collection frequency: {term_info['collection_frequency']}")

    # 9. Load schema
    print("\n9. Loading schema metadata...")
    loaded_schema = driver.load_schema()
    print(f"   Schema fields: {list(loaded_schema['fields'].keys())}")

    # 10. Delete a document
    print("\n10. Deleting a document...")
    deleted = driver.delete_document("doc3")
    print(f"   ✓ Deleted {deleted} document(s)")
    print(f"   Remaining docs: {driver.get_doc_count()}")

    # 11. Verify deletion
    print("\n11. Verifying deletion...")
    results = driver.search_term("title", "postgresql", limit=10)
    print(f"   Results for 'postgresql': {len(results)} (should be 0)")

    # 12. Cleanup
    print("\n12. Cleaning up...")
    driver.destroy()
    driver.close()
    print("   ✓ Index destroyed and driver closed")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nNote: This is a proof-of-concept driver.")
    print("Full Whoosh/Semlix integration is planned for future sprints.")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"\nError: Missing dependency - {e}")
        print("\nInstall required packages:")
        print("  pip install psycopg2-binary")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("  1. PostgreSQL is running")
        print("  2. Database exists")
        print("  3. Schema is created (run schema_lexical.sql)")
        print("  4. CONNECTION_STRING is correct")
        raise
