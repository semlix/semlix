======================
Introduction to semlix
======================

About semlix
------------

semlix is based on Whoosh, which was created by `Matt Chaput <mailto:matt@whoosh.ca>`_.
semlix started as a quick and dirty search server for the online documentation of the
`Houdini <http://www.sidefx.com/>`_ 3D animation software package. Side Effects Software
generously allowed Matt to open source the code. semlix extends semlix with modern semantic
search capabilities while maintaining full backward compatibility and honoring semlix's
pure-Python philosophy.

* semlix is fast, but uses only pure Python, so it will run anywhere Python runs,
  without requiring a compiler.

* By default, semlix uses the `Okapi BM25F <http://en.wikipedia.com/wiki/Okapi_BM25>`_ ranking
  function, but like most things the ranking function can be easily customized.

* semlix creates fairly small indexes compared to many other search libraries.

* All indexed text in semlix must be *unicode*.

* semlix lets you store arbitrary Python objects with indexed documents.

* semlix adds semantic search capabilities, enabling hybrid search that combines
  traditional lexical matching with vector-based semantic similarity.


What is semlix?
---------------

semlix is a fast, pure Python search engine library with semantic search capabilities.

The primary design impetus of semlix (inherited from semlix) is that it is pure Python.
You should be able to use semlix anywhere you can use Python, no compiler or Java required.

Like one of its ancestors, Lucene, semlix is not really a search engine, it's a programmer
library for creating a search engine [1]_.

Practically no important behavior of semlix is hard-coded. Indexing
of text, the level of information stored for each term in each field, parsing of search queries,
the types of queries allowed, scoring algorithms, etc. are all customizable, replaceable, and
extensible.


.. [1] It would of course be possible to build a turnkey search engine on top of semlix,
       like Nutch and Solr use Lucene.


What can semlix do for you?
---------------------------

semlix lets you index free-form or structured text and then quickly find matching
documents based on simple or complex search criteria, including semantic similarity
matching beyond keyword search.


Getting help with semlix
------------------------

You can view the project and report issues on the
`semlix GitHub page <https://github.com/semlix/semlix>`_.
