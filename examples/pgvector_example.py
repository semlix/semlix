"""Example usage of PgVectorStore with Semlix.

This example demonstrates:
1. Setting up PgVectorStore
2. Adding vectors with metadata
3. Creating indexes
4. Searching with and without filters
5. Using with HybridSearcher

Requirements:
    pip install psycopg2-binary pgvector sentence-transformers

Setup:
    1. Create PostgreSQL database
    2. Run schema: psql -d mydb -f src/semlix/semantic/stores/schema.sql
    3. Update CONNECTION_STRING below
    4. Run: python examples/pgvector_example.py
"""

import numpy as np
from pathlib import Path

# Configuration - UPDATE THESE
CONNECTION_STRING = "postgresql://postgres:password@localhost:5432/semlix_test"
DIMENSION = 384  # For all-MiniLM-L6-v2

# Sample documents
DOCUMENTS = [
    {
        "id": "doc1",
        "content": "Python is a high-level programming language with dynamic semantics",
        "category": "programming",
        "lang": "en"
    },
    {
        "id": "doc2",
        "content": "Machine learning is a subset of artificial intelligence",
        "category": "ai",
        "lang": "en"
    },
    {
        "id": "doc3",
        "content": "PostgreSQL is an advanced open source relational database",
        "category": "database",
        "lang": "en"
    },
    {
        "id": "doc4",
        "content": "Docker containers provide lightweight virtualization",
        "category": "devops",
        "lang": "en"
    },
    {
        "id": "doc5",
        "content": "React is a JavaScript library for building user interfaces",
        "category": "programming",
        "lang": "en"
    },
    {
        "id": "doc6",
        "content": "Neural networks are computing systems inspired by biological brains",
        "category": "ai",
        "lang": "en"
    },
    {
        "id": "doc7",
        "content": "Git is a distributed version control system",
        "category": "devops",
        "lang": "en"
    },
    {
        "id": "doc8",
        "content": "MongoDB is a document-oriented NoSQL database",
        "category": "database",
        "lang": "en"
    },
]


def main():
    print("=" * 80)
    print("PgVectorStore Example with Semlix")
    print("=" * 80)

    # 1. Import and initialize
    print("\n1. Initializing PgVectorStore...")
    from semlix.semantic.stores import PgVectorStore
    from semlix.semantic import SentenceTransformerProvider

    store = PgVectorStore(
        connection_string=CONNECTION_STRING,
        dimension=DIMENSION,
        distance_metric="cosine"
    )

    print(f"   ✓ Store created (dimension={store.dimension}, metric={store.distance_metric})")

    # 2. Generate embeddings
    print("\n2. Generating embeddings with SentenceTransformer...")
    embedder = SentenceTransformerProvider("all-MiniLM-L6-v2")

    texts = [doc["content"] for doc in DOCUMENTS]
    embeddings = embedder.encode(texts, show_progress=False)

    doc_ids = [doc["id"] for doc in DOCUMENTS]
    metadata = [
        {"category": doc["category"], "lang": doc["lang"]}
        for doc in DOCUMENTS
    ]

    print(f"   ✓ Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    # 3. Add to store
    print("\n3. Adding vectors to PostgreSQL...")
    store.add(doc_ids, embeddings, metadata)
    print(f"   ✓ Added {store.count} vectors to store")

    # 4. Create index
    print("\n4. Creating HNSW index for fast search...")
    store.create_index(index_type="hnsw", m=16, ef_construction=64)
    print("   ✓ HNSW index created")

    # 5. Basic search
    print("\n5. Basic similarity search...")
    query_text = "programming languages and frameworks"
    query_embedding = embedder.encode([query_text])[0]

    results = store.search(query_embedding, k=3)

    print(f"   Query: '{query_text}'")
    print(f"   Top {len(results)} results:")
    for i, result in enumerate(results, 1):
        doc = next(d for d in DOCUMENTS if d["id"] == result.doc_id)
        print(f"      {i}. {result.doc_id} (score: {result.score:.4f})")
        print(f"         '{doc['content']}'")
        print(f"         Metadata: {result.metadata}")

    # 6. Search with metadata filter
    print("\n6. Search with metadata filter (category='ai')...")
    ai_results = store.search_with_filter(
        query_embedding,
        k=5,
        metadata_filter={"category": "ai"}
    )

    print(f"   Query: '{query_text}' (filtered to AI category)")
    print(f"   Found {len(ai_results)} AI-related results:")
    for i, result in enumerate(ai_results, 1):
        doc = next(d for d in DOCUMENTS if d["id"] == result.doc_id)
        print(f"      {i}. {result.doc_id} (score: {result.score:.4f})")
        print(f"         '{doc['content']}'")

    # 7. Search with doc_id filter
    print("\n7. Search within specific documents...")
    filter_ids = ["doc1", "doc3", "doc5"]
    filtered_results = store.search(
        query_embedding,
        k=3,
        filter_ids=filter_ids
    )

    print(f"   Query: '{query_text}' (only docs: {filter_ids})")
    print(f"   Results:")
    for i, result in enumerate(filtered_results, 1):
        doc = next(d for d in DOCUMENTS if d["id"] == result.doc_id)
        print(f"      {i}. {result.doc_id} (score: {result.score:.4f})")
        print(f"         '{doc['content']}'")

    # 8. Update document (upsert)
    print("\n8. Updating a document (upsert)...")
    new_text = "Python is the best programming language for data science and AI"
    new_embedding = embedder.encode([new_text])[0]

    store.add(
        ["doc1"],
        new_embedding.reshape(1, -1),
        [{"category": "programming", "lang": "en", "updated": True}]
    )

    print(f"   ✓ Updated doc1")
    print(f"   Store still has {store.count} vectors (upserted, not added)")

    # 9. Delete documents
    print("\n9. Deleting documents...")
    to_delete = ["doc7", "doc8"]
    deleted = store.delete(to_delete)

    print(f"   ✓ Deleted {deleted} documents: {to_delete}")
    print(f"   Store now has {store.count} vectors")

    # 10. Different distance metrics
    print("\n10. Comparing distance metrics...")

    # L2 distance
    store_l2 = PgVectorStore(
        connection_string=CONNECTION_STRING,
        dimension=DIMENSION,
        distance_metric="l2",
        table_name="semlix_vectors_l2"
    )

    # Create table for L2 metric
    try:
        import psycopg2
        conn = psycopg2.connect(CONNECTION_STRING)
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS semlix_vectors_l2 (
                    doc_id TEXT PRIMARY KEY,
                    embedding vector(384) NOT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
        conn.commit()
        conn.close()

        store_l2.add(doc_ids[:6], embeddings[:6], metadata[:6])
        results_l2 = store_l2.search(query_embedding, k=3)

        print("   L2 Distance results:")
        for i, result in enumerate(results_l2, 1):
            print(f"      {i}. {result.doc_id} (score: {result.score:.4f})")

        # Cleanup
        conn = psycopg2.connect(CONNECTION_STRING)
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS semlix_vectors_l2")
        conn.commit()
        conn.close()

    except Exception as e:
        print(f"   Note: Could not test L2 metric: {e}")

    # 11. Context manager
    print("\n11. Using context manager...")
    with PgVectorStore(CONNECTION_STRING, DIMENSION) as temp_store:
        temp_store.add(["temp1"], embeddings[0].reshape(1, -1))
        print(f"   ✓ Added vector in context manager (count: {temp_store.count})")
    print("   ✓ Store automatically closed after context")

    # Cleanup
    print("\n12. Cleaning up...")
    remaining_ids = [doc["id"] for doc in DOCUMENTS if doc["id"] not in to_delete]
    store.delete(remaining_ids)
    store.close()
    print(f"   ✓ Deleted all vectors and closed store")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"\nError: Missing dependency - {e}")
        print("\nInstall required packages:")
        print("  pip install psycopg2-binary pgvector sentence-transformers")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("  1. PostgreSQL is running")
        print("  2. Database exists")
        print("  3. Schema is created (run schema.sql)")
        print("  4. CONNECTION_STRING is correct")
        raise
