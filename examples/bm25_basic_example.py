#!/usr/bin/env python
"""Basic example of using BM25Index.

This example shows how to:
1. Create a BM25 index
2. Add documents
3. Search with BM25 scoring
4. Use advanced features (phrase queries, faceting)
"""

from semlix.bm25 import create_bm25_index, open_bm25_index
from semlix.fields import Schema, TEXT, ID, KEYWORD
from semlix.analysis import StandardAnalyzer


def main():
    print("=" * 60)
    print("BM25Index Basic Example")
    print("=" * 60)

    # 1. Create schema
    print("\n1. Creating schema...")
    schema = Schema(
        id=ID(stored=True),
        title=TEXT(stored=True, analyzer=StandardAnalyzer()),
        content=TEXT(stored=True, analyzer=StandardAnalyzer()),
        category=KEYWORD(stored=True)
    )

    # 2. Create index
    print("\n2. Creating BM25 index...")
    ix = create_bm25_index("bm25_example_index", schema)

    # 3. Add documents
    print("\n3. Adding documents...")
    with ix.writer() as writer:
        writer.add_document(
            id="1",
            title="Introduction to Python",
            content="Python is a high-level programming language. It is easy to learn and widely used.",
            category="tutorial"
        )
        writer.add_document(
            id="2",
            title="Python Data Science",
            content="Python is the most popular language for data science and machine learning.",
            category="guide"
        )
        writer.add_document(
            id="3",
            title="SQL Database Tutorial",
            content="SQL is used for managing relational databases. Learn SQL basics here.",
            category="tutorial"
        )
        writer.add_document(
            id="4",
            title="Advanced Python Techniques",
            content="Learn advanced Python programming techniques including decorators and generators.",
            category="advanced"
        )
        writer.add_document(
            id="5",
            title="Python Web Development",
            content="Build web applications with Python using Flask or Django frameworks.",
            category="guide"
        )

    print(f"   Added {ix.doc_count()} documents")

    # 4. Basic search
    print("\n4. Basic BM25 search...")
    with ix.searcher() as searcher:
        from semlix.qparser import QueryParser

        qp = QueryParser("content", ix.schema)
        query = qp.parse("python programming")

        results = searcher.search(query, limit=3)

        print(f"   Found {len(results)} results for 'python programming':")
        for hit in results:
            print(f"   - {hit['title']} (score: {hit.score:.3f})")

    # 5. Phrase query
    print("\n5. Phrase query...")
    from semlix.bm25 import PhraseQuery

    pq = PhraseQuery("content", ["data", "science"])
    phrase_results = pq.search(ix, limit=5)

    print(f"   Found {len(phrase_results)} results for phrase 'data science':")
    for result in phrase_results:
        print(f"   - {result['fields']['title']} (score: {result['score']:.3f})")

    # 6. Faceting
    print("\n6. Faceting by category...")
    from semlix.bm25 import Facets

    with ix.searcher() as searcher:
        qp = QueryParser("content", ix.schema)
        query = qp.parse("python")
        results = searcher.search(query, limit=10)

        facets = Facets(ix)
        category_counts = facets.count_by_field(results, "category")

        print("   Category counts:")
        for category, count in category_counts.items():
            print(f"   - {category}: {count}")

    # 7. Reopen and search
    print("\n7. Reopening index...")
    ix.close()

    ix2 = open_bm25_index("bm25_example_index")
    print(f"   Reopened index with {ix2.doc_count()} documents")

    with ix2.searcher() as searcher:
        qp = QueryParser("title", ix2.schema)
        query = qp.parse("tutorial")
        results = searcher.search(query, limit=5)

        print(f"   Search results for 'tutorial' in title:")
        for hit in results:
            print(f"   - {hit['title']}")

    ix2.close()

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
