#!/usr/bin/env python
"""Complete example of using UnifiedIndex for hybrid search.

This example demonstrates:
1. Creating a UnifiedIndex (BM25 + pgvector)
2. Indexing documents with automatic embedding generation
3. Hybrid search (combining lexical and semantic)
4. Advanced features (faceting, sorting, phrase queries)

Prerequisites:
- PostgreSQL with pgvector extension
- bm25s and sentence-transformers installed
"""

import os


def main():
    print("=" * 70)
    print("UnifiedIndex Hybrid Search Example")
    print("=" * 70)

    # Check for PostgreSQL URL
    pg_url = os.getenv("POSTGRES_URL", "postgresql://postgres:password@localhost/semlix_demo")
    print(f"\nUsing PostgreSQL: {pg_url}")

    from semlix.unified import create_unified_index
    from semlix.fields import Schema, TEXT, ID, KEYWORD, DATETIME
    from semlix.semantic import SentenceTransformerProvider
    from semlix.analysis import StandardAnalyzer

    # 1. Create schema
    print("\n1. Creating schema...")
    schema = Schema(
        id=ID(stored=True),
        title=TEXT(stored=True, analyzer=StandardAnalyzer()),
        content=TEXT(stored=True, analyzer=StandardAnalyzer()),
        author=KEYWORD(stored=True),
        category=KEYWORD(stored=True),
        published=DATETIME(stored=True)
    )

    # 2. Create embedding provider
    print("\n2. Loading embedding model...")
    embedder = SentenceTransformerProvider("all-MiniLM-L6-v2")
    print(f"   Model: {embedder.model_name}, Dimension: {embedder.dimension}")

    # 3. Create unified index
    print("\n3. Creating unified index...")
    ix = create_unified_index(
        index_dir="unified_example_index",
        schema=schema,
        connection_string=pg_url,
        embedder=embedder
    )

    # 4. Index documents
    print("\n4. Indexing documents...")
    documents = [
        {
            "id": "1",
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "author": "Alice",
            "category": "ai",
            "published": "2024-01-15"
        },
        {
            "id": "2",
            "title": "Deep Learning Neural Networks",
            "content": "Neural networks are the foundation of deep learning, inspired by biological neurons.",
            "author": "Bob",
            "category": "ai",
            "published": "2024-02-20"
        },
        {
            "id": "3",
            "title": "Python Programming Best Practices",
            "content": "Writing clean, maintainable Python code requires following established best practices.",
            "author": "Alice",
            "category": "programming",
            "published": "2024-03-10"
        },
        {
            "id": "4",
            "title": "Database Design Principles",
            "content": "Good database design is crucial for application performance and data integrity.",
            "author": "Charlie",
            "category": "database",
            "published": "2024-01-25"
        },
        {
            "id": "5",
            "title": "Natural Language Processing Basics",
            "content": "NLP enables computers to understand, interpret, and generate human language.",
            "author": "Bob",
            "category": "ai",
            "published": "2024-04-05"
        }
    ]

    with ix.writer() as writer:
        for doc in documents:
            writer.add_document(**doc)

    print(f"   Indexed {ix.doc_count()} documents")

    # 5. Lexical-only search (BM25)
    print("\n5. Lexical search (BM25 only)...")
    with ix.searcher() as searcher:
        results = searcher.lexical_only("machine learning", limit=3)

        print(f"   Found {len(results)} results:")
        for r in results:
            print(f"   - {r.stored_fields['title']}")
            print(f"     Score: {r.score:.3f}, Lexical: {r.lexical_score:.3f}")

    # 6. Semantic-only search (vector)
    print("\n6. Semantic search (vector only)...")
    with ix.searcher() as searcher:
        # Search for concept, not exact words
        results = searcher.semantic_only("AI and neural networks", limit=3)

        print(f"   Found {len(results)} results:")
        for r in results:
            print(f"   - {r.stored_fields['title']}")
            print(f"     Score: {r.score:.3f}, Semantic: {r.semantic_score:.3f}")

    # 7. Hybrid search (best of both)
    print("\n7. Hybrid search (BM25 + vector, alpha=0.5)...")
    with ix.searcher() as searcher:
        results = searcher.hybrid_search(
            "learning algorithms and AI",
            limit=3,
            alpha=0.5  # Equal weight for lexical and semantic
        )

        print(f"   Found {len(results)} results:")
        for r in results:
            print(f"   - {r.stored_fields['title']}")
            print(f"     Combined: {r.score:.3f}, Lexical: {r.lexical_score}, Semantic: {r.semantic_score:.3f}")

    # 8. Hybrid search with facets
    print("\n8. Hybrid search with faceting...")
    with ix.searcher() as searcher:
        results, facets = searcher.search_with_facets(
            "programming",
            facet_fields=["category", "author"],
            limit=10
        )

        print(f"   Found {len(results)} results")
        print("   Category facets:")
        for category, count in facets["category"].items():
            print(f"   - {category}: {count}")
        print("   Author facets:")
        for author, count in facets["author"].items():
            print(f"   - {author}: {count}")

    # 9. Phrase search
    print("\n9. Phrase search...")
    with ix.searcher() as searcher:
        results = searcher.phrase_search(
            "content",
            "machine learning",
            slop=0  # Exact phrase
        )

        print(f"   Found {len(results)} exact phrase matches:")
        for r in results:
            print(f"   - {r.stored_fields['title']}")

    # 10. Sorted search
    print("\n10. Search with custom sorting...")
    with ix.searcher() as searcher:
        results = searcher.search_sorted(
            "ai",
            sort_by=[("published", True), ("score", True)],  # Sort by date desc, then score
            limit=5
        )

        print(f"   Results sorted by published date (newest first):")
        for r in results:
            doc = r.stored_fields
            print(f"   - {doc['title']} ({doc.get('published', 'N/A')})")

    # 11. Update a document
    print("\n11. Updating a document...")
    with ix.writer() as writer:
        writer.update_document(
            id="1",
            title="Introduction to Machine Learning (Updated)",
            content="Machine learning is a powerful subset of AI with many applications.",
            author="Alice",
            category="ai",
            published="2024-05-01"
        )

    print(f"   Updated document. Index now has {ix.doc_count()} documents")

    # 12. Search again to see update
    print("\n12. Searching after update...")
    with ix.searcher() as searcher:
        results = searcher.hybrid_search("machine learning applications", limit=2)

        print(f"   Results:")
        for r in results:
            print(f"   - {r.stored_fields['title']}")

    # Cleanup
    ix.close()

    print("\n" + "=" * 70)
    print("Example complete!")
    print("\nTo run this example:")
    print("1. Install dependencies: pip install bm25s sentence-transformers psycopg2-binary pgvector")
    print("2. Setup PostgreSQL with pgvector extension")
    print(f"3. Set POSTGRES_URL environment variable (current: {pg_url})")
    print("4. Run: python unified_index_example.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
