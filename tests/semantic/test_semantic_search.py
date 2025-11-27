#!/usr/bin/env python3
"""Test script for semantic search functionality using docs from the docs folder."""

import os
import sys
from pathlib import Path
import shutil

# Add src to path
sys.path.insert(0, os.path.abspath("src"))

from semlix.index import create_in
from semlix.fields import Schema, TEXT, ID
from semlix.semantic import (
    HybridSearcher,
    HybridIndexWriter,
    SentenceTransformerProvider,
)
from semlix.semantic.stores import FaissVectorStore

# Sample documents from the docs folder
DOCUMENTS = [
    {
        "id": "quickstart",
        "title": "Quick Start Guide",
        "content": """
semlix is a library of classes and functions for indexing text and then searching the index.
It allows you to develop custom search engines for your content. For example, if you were
creating blogging software, you could use semlix to add a search function to allow users to
search blog entries.

To begin using semlix, you need an index object. The first time you create
an index, you must define the index's schema. The schema lists the fields
in the index. A field is a piece of information for each document in the index,
such as its title or text content.
        """.strip()
    },
    {
        "id": "intro",
        "title": "Introduction to semlix",
        "content": """
semlix is a fast, pure Python search engine library with semantic search capabilities.
Based on Whoosh (the original library), semlix maintains the pure Python philosophy. You should be able to
use semlix anywhere you can use Python, no compiler or Java required.

semlix lets you index free-form or structured text and then quickly find matching
documents based on simple or complex search criteria, including semantic similarity.

By default, semlix uses the Okapi BM25F ranking function, but like most things 
the ranking function can be easily customized.
        """.strip()
    },
    {
        "id": "indexing",
        "title": "How to Index Documents",
        "content": """
To create an index in a directory, use index.create_in.
Once you've created an Index object, you can add documents to the index with an
IndexWriter object. The easiest way to get the IndexWriter is to call Index.writer().

Creating a writer locks the index for writing, so only one thread/process at
a time can have a writer open. The IndexWriter's add_document method accepts 
keyword arguments where the field name is mapped to a value.
        """.strip()
    },
    {
        "id": "searching",
        "title": "How to Search",
        "content": """
Once you've created an index and added documents to it, you can search for those
documents. To get a Searcher object, call searcher() on your Index object.

The Searcher object is the main high-level interface for reading the index. It
has lots of useful methods for getting information about the index.

The most important method on the Searcher object is search, which takes a
Query object and returns a Results object containing matching documents.
        """.strip()
    },
    {
        "id": "schema",
        "title": "Schema and Fields",
        "content": """
The schema lists the fields in the index. A field can be indexed (meaning it can
be searched) and/or stored (meaning the value that gets indexed is returned
with the results).

semlix comes with predefined field types:
- ID: indexes the entire value as a single unit
- TEXT: for body text, supports phrase searching
- KEYWORD: for space- or comma-separated keywords
- NUMERIC: for numbers
- BOOLEAN: for boolean values
- DATETIME: for datetime objects
        """.strip()
    },
    {
        "id": "query",
        "title": "Query Language",
        "content": """
semlix provides a powerful query language for searching. You can use QueryParser
to parse query strings into Query objects. The query language supports:
- Simple terms: "python"
- Phrases: "python tutorial"
- Boolean operators: AND, OR, NOT
- Wildcards: "pyth*"
- Field-specific searches: "title:python"
        """.strip()
    },
]


def main():
    print("=" * 70)
    print("Testing Semantic Search with semlix")
    print("=" * 70)
    print()
    
    # Setup directories
    index_dir = Path("test_semantic_index")
    vector_path = Path("test_vectors.faiss")
    
    # Clean up old test data
    if index_dir.exists():
        print(f"Cleaning up old index at {index_dir}")
        shutil.rmtree(index_dir)
    if vector_path.exists():
        print(f"Cleaning up old vector store at {vector_path}")
        vector_path.unlink()
        vector_path.with_suffix(".meta").unlink()
    
    # Create schema
    print("Creating schema...")
    schema = Schema(
        id=ID(stored=True, unique=True),
        title=TEXT(stored=True),
        content=TEXT(stored=True)
    )
    
    # Create index
    print(f"Creating index at {index_dir}...")
    index_dir.mkdir(exist_ok=True)
    ix = create_in(str(index_dir), schema)
    
    # Create semantic components
    print("Initializing embedding provider (sentence-transformers)...")
    embedder = SentenceTransformerProvider("all-MiniLM-L6-v2")
    print(f"  Model: {embedder.model_name}")
    print(f"  Dimension: {embedder.dimension}")
    
    print("Creating FAISS vector store...")
    vector_store = FaissVectorStore(
        dimension=embedder.dimension,
        index_type="Flat"  # Exact search for small dataset
    )
    print(f"  Vector store dimension: {vector_store.dimension}")
    
    # Index documents
    print("\nIndexing documents...")
    with HybridIndexWriter(
        ix, 
        vector_store, 
        embedder, 
        embedding_field="content",
        id_field="id",
        batch_size=10,
        show_progress=True
    ) as writer:
        for doc in DOCUMENTS:
            print(f"  Adding: {doc['title']}")
            writer.add_document(**doc)
    
    print(f"\nIndexed {len(DOCUMENTS)} documents")
    print(f"Vector store contains {vector_store.count} vectors")
    
    # Save vector store
    print(f"\nSaving vector store to {vector_path}...")
    vector_store.save(vector_path)
    
    # Create hybrid searcher
    print("\nCreating hybrid searcher...")
    searcher = HybridSearcher(
        index=ix,
        vector_store=vector_store,
        embedding_provider=embedder,
        default_field="content",
        id_field="id",
        alpha=0.5,  # 50% lexical, 50% semantic
        fusion_method="rrf"
    )
    print(f"  Alpha: {searcher.alpha} (0=lexical, 1=semantic)")
    print(f"  Fusion method: {searcher.fusion_method.value}")
    
    # Test queries
    print("\n" + "=" * 70)
    print("Testing Search Queries")
    print("=" * 70)
    
    test_queries = [
        "how to create an index",
        "search for documents",
        "field types and schema",
        "query language features",
        "ranking and scoring",
    ]
    
    for query in test_queries:
        print(f"\n{'─' * 70}")
        print(f"Query: '{query}'")
        print(f"{'─' * 70}")
        
        results = searcher.search(query, limit=3, highlight_fields=["content"])
        
        if not results:
            print("  No results found")
            continue
        
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Title: {result.get('title', 'N/A')}")
            print(f"    ID: {result.doc_id}")
            print(f"    Combined Score: {result.score:.4f}")
            if result.lexical_score is not None:
                print(f"    Lexical Score: {result.lexical_score:.4f}")
            else:
                print(f"    Lexical Score: N/A")
            if result.semantic_score is not None:
                print(f"    Semantic Score: {result.semantic_score:.4f}")
            else:
                print(f"    Semantic Score: N/A")
            
            # Show highlight if available
            if result.highlights.get("content"):
                highlight = result.highlights["content"]
                # Truncate if too long
                if len(highlight) > 150:
                    highlight = highlight[:147] + "..."
                print(f"    Highlight: {highlight}")
    
    # Test semantic-only search
    print("\n" + "=" * 70)
    print("Testing Semantic-Only Search")
    print("=" * 70)
    
    semantic_query = "finding matching text documents"
    print(f"\nQuery: '{semantic_query}' (semantic only)")
    print(f"{'─' * 70}")
    
    semantic_results = searcher.search_semantic_only(semantic_query, limit=3)
    
    for i, result in enumerate(semantic_results, 1):
        print(f"\n  Result {i}:")
        print(f"    Title: {result.get('title', 'N/A')}")
        if result.semantic_score is not None:
            print(f"    Semantic Score: {result.semantic_score:.4f}")
        else:
            print(f"    Semantic Score: N/A")
    
    # Test lexical-only search
    print("\n" + "=" * 70)
    print("Testing Lexical-Only Search")
    print("=" * 70)
    
    lexical_query = "index schema"
    print(f"\nQuery: '{lexical_query}' (lexical only)")
    print(f"{'─' * 70}")
    
    lexical_results = searcher.search_lexical_only(lexical_query, limit=3)
    
    for i, result in enumerate(lexical_results, 1):
        print(f"\n  Result {i}:")
        print(f"    Title: {result.get('title', 'N/A')}")
        if result.lexical_score is not None:
            print(f"    Lexical Score: {result.lexical_score:.4f}")
        else:
            print(f"    Lexical Score: N/A")
    
    # Test loading vector store
    print("\n" + "=" * 70)
    print("Testing Vector Store Persistence")
    print("=" * 70)
    
    print("\nLoading vector store from disk...")
    loaded_store = FaissVectorStore.load(vector_path)
    print(f"  Loaded {loaded_store.count} vectors")
    print(f"  Dimension: {loaded_store.dimension}")
    
    # Test search with loaded store
    loaded_searcher = HybridSearcher(
        index=ix,
        vector_store=loaded_store,
        embedding_provider=embedder,
        default_field="content",
        id_field="id"
    )
    
    test_query = "pure python search engine"
    print(f"\nTesting search with loaded store: '{test_query}'")
    loaded_results = loaded_searcher.search(test_query, limit=2)
    
    for i, result in enumerate(loaded_results, 1):
        print(f"  {i}. {result.get('title')} (score: {result.score:.4f})")
    
    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
    print(f"\nTest index: {index_dir}")
    print(f"Vector store: {vector_path} (and .meta)")
    print("\nTo clean up test files, run:")
    print(f"  rm -rf {index_dir}")
    print(f"  rm {vector_path} {vector_path.with_suffix('.meta')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

