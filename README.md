Maintained
==============

About semlix
============

semlix is a fast, featureful full-text indexing and searching library
implemented in pure Python. Based on the excellent Whoosh library, semlix
extends it with modern semantic search capabilities while maintaining full
backward compatibility. Programmers can use it to easily add search
functionality to their applications and websites. Every part of how semlix
works can be extended or replaced to meet your needs exactly.

What does "semlix" mean?
-------------------------

The name **semlix** stands for:

* **Sem**antic - Understanding meaning and context beyond keywords
* **Lex**ical - Traditional keyword matching (BM25/TF-IDF)
* **Ind**ex - Fast, efficient indexing and retrieval

semlix combines all three: it indexes your documents, searches them using both
lexical (keyword) and semantic (meaning-based) methods, then intelligently
combines the results for superior search quality.

Some of semlix's features include:

* Pythonic API.
* Pure-Python. No compilation or binary packages needed, no mysterious crashes.
* Fielded indexing and search.
* Fast indexing and retrieval -- faster than any other pure-Python, scoring,
  full-text search solution I know of.
* Pluggable scoring algorithm (including BM25F), text analysis, storage,
  posting format, etc.
* Powerful query language.
* Pure Python spell-checker (as far as I know, the only one).
* **Semantic search** - Hybrid search combining traditional lexical matching (BM25/TF-IDF) 
  with modern vector-based semantic similarity for understanding meaning beyond keywords. 

semlix might be useful in the following circumstances:

* Anywhere a pure-Python solution is desirable to avoid having to build/compile
  native libraries (or force users to build/compile them).
* As a research platform (at least for programmers that find Python easier to
  read and work with than Java ;)
* When an easy-to-use Pythonic interface is more important to you than raw
  speed. 

semlix is based on Whoosh, which was created and is maintained by Matt Chaput.
Whoosh was originally created for use in the online help system of Side Effects
Software's 3D animation software Houdini. Side Effects Software Inc. graciously
agreed to open-source the code. semlix extends Whoosh with semantic search
capabilities while honoring its pure-Python philosophy.

This software is licensed under the terms of the simplified BSD (A.K.A. "two
clause" or "FreeBSD") license. See LICENSE.txt for information.

Installing semlix
=================

Install from PyPI: https://pypi.org/project/semlix/

Basic installation::

    pip install semlix

For semantic search capabilities::

    pip install semlix[semantic]

For full semantic search with all providers and FAISS support::

    pip install semlix[semantic-full]

Or using `uv`::

    uv pip install semlix[semantic-full]

Semantic Search
===============

semlix includes optional semantic search capabilities that combine traditional 
lexical matching with vector-based semantic similarity. This enables queries like 
"how to fix authentication issues" to match documents containing "resolving login 
problems" even without shared keywords.

Key features:

* **Hybrid Search**: Combines BM25/TF-IDF lexical search with semantic vector search
* **Multiple Embedding Providers**: Support for sentence-transformers, OpenAI, Cohere, 
  and HuggingFace Inference API
* **Flexible Vector Stores**: Pure-Python NumPy backend for small datasets, FAISS backend 
  for large-scale deployments
* **Result Fusion**: Multiple fusion algorithms (RRF, Linear, DBSF) for optimal ranking
* **Backward Compatible**: Existing Whoosh code continues to work without modification

Quick example::

    >>> from semlix.index import create_in
    >>> from semlix.fields import Schema, TEXT, ID
    >>> from semlix.semantic import (
    ...     HybridSearcher, HybridIndexWriter,
    ...     SentenceTransformerProvider
    ... )
    >>> from semlix.semantic.stores import NumpyVectorStore
    >>> 
    >>> # Create schema and index
    >>> schema = Schema(id=ID(stored=True, unique=True), content=TEXT(stored=True))
    >>> ix = create_in("my_index", schema)
    >>> 
    >>> # Create semantic components
    >>> embedder = SentenceTransformerProvider("all-MiniLM-L6-v2")
    >>> vector_store = NumpyVectorStore(dimension=embedder.dimension)
    >>> 
    >>> # Index documents
    >>> with HybridIndexWriter(ix, vector_store, embedder) as writer:
    ...     writer.add_document(id="1", content="Python programming basics")
    ...     writer.add_document(id="2", content="How to fix login issues")
    >>> 
    >>> # Search with hybrid search
    >>> searcher = HybridSearcher(ix, vector_store, embedder)
    >>> results = searcher.search("authentication problems")  # Matches "login issues"!

See the [semantic search documentation](docs/source/semantic.rst) for more details.

Project
=======

* **GitHub**: https://github.com/semlix/semlix
* **PyPI**: https://pypi.org/project/semlix/
* **Documentation**: https://semlix.readthedocs.io/en/latest/

