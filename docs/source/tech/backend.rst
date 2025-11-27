==============================
How to implement a new backend
==============================

Index
=====

* Subclass :class:`semlix.index.Index`.

* Indexes must implement the following methods.

  * :meth:`semlix.index.Index.is_empty`

  * :meth:`semlix.index.Index.doc_count`

  * :meth:`semlix.index.Index.reader`

  * :meth:`semlix.index.Index.writer`

* Indexes that require/support locking must implement the following methods.

  * :meth:`semlix.index.Index.lock`

  * :meth:`semlix.index.Index.unlock`

* Indexes that support deletion must implement the following methods.

  * :meth:`semlix.index.Index.delete_document`

  * :meth:`semlix.index.Index.doc_count_all` -- if the backend has delayed
    deletion.

* Indexes that require/support versioning/transactions *may* implement the following methods.

  * :meth:`semlix.index.Index.latest_generation`

  * :meth:`semlix.index.Index.up_to_date`

  * :meth:`semlix.index.Index.last_modified`

* Index *may* implement the following methods (the base class's versions are no-ops).

  * :meth:`semlix.index.Index.optimize`

  * :meth:`semlix.index.Index.close`


IndexWriter
===========

* Subclass :class:`semlix.writing.IndexWriter`.

* IndexWriters must implement the following methods.

  * :meth:`semlix.writing.IndexWriter.add_document`

  * :meth:`semlix.writing.IndexWriter.add_reader`

* Backends that support deletion must implement the following methods.

  * :meth:`semlix.writing.IndexWriter.delete_document`

* IndexWriters that work as transactions must implement the following methods.

  * :meth:`semlix.reading.IndexWriter.commit` -- Save the additions/deletions done with
    this IndexWriter to the main index, and release any resources used by the IndexWriter.

  * :meth:`semlix.reading.IndexWriter.cancel` -- Throw away any additions/deletions done
    with this IndexWriter, and release any resources used by the IndexWriter.


IndexReader
===========

* Subclass :class:`semlix.reading.IndexReader`.

* IndexReaders must implement the following methods.

  * :meth:`semlix.reading.IndexReader.__contains__`

  * :meth:`semlix.reading.IndexReader.__iter__`

  * :meth:`semlix.reading.IndexReader.iter_from`

  * :meth:`semlix.reading.IndexReader.stored_fields`

  * :meth:`semlix.reading.IndexReader.doc_count_all`

  * :meth:`semlix.reading.IndexReader.doc_count`

  * :meth:`semlix.reading.IndexReader.doc_field_length`

  * :meth:`semlix.reading.IndexReader.field_length`

  * :meth:`semlix.reading.IndexReader.max_field_length`

  * :meth:`semlix.reading.IndexReader.postings`

  * :meth:`semlix.reading.IndexReader.has_vector`

  * :meth:`semlix.reading.IndexReader.vector`

  * :meth:`semlix.reading.IndexReader.doc_frequency`

  * :meth:`semlix.reading.IndexReader.frequency`

* Backends that support deleting documents should implement the following
  methods.

  * :meth:`semlix.reading.IndexReader.has_deletions`
  * :meth:`semlix.reading.IndexReader.is_deleted`

* Backends that support versioning should implement the following methods.

  * :meth:`semlix.reading.IndexReader.generation`

* If the IndexReader object does not keep the schema in the ``self.schema``
  attribute, it needs to override the following methods.

  * :meth:`semlix.reading.IndexReader.field`

  * :meth:`semlix.reading.IndexReader.field_names`

  * :meth:`semlix.reading.IndexReader.scorable_names`

  * :meth:`semlix.reading.IndexReader.vector_names`

* IndexReaders *may* implement the following methods.

  * :meth:`semlix.reading.DocReader.close` -- closes any open resources associated with the
    reader.


Matcher
=======

The :meth:`semlix.reading.IndexReader.postings` method returns a
:class:`semlix.matching.Matcher` object. You will probably need to implement
a custom Matcher class for reading from your posting lists.

* Subclass :class:`semlix.matching.Matcher`.

* Implement the following methods at minimum.

  * :meth:`semlix.matching.Matcher.is_active`

  * :meth:`semlix.matching.Matcher.copy`

  * :meth:`semlix.matching.Matcher.id`

  * :meth:`semlix.matching.Matcher.next`

  * :meth:`semlix.matching.Matcher.value`

  * :meth:`semlix.matching.Matcher.value_as`

  * :meth:`semlix.matching.Matcher.score`

* Depending on the implementation, you *may* implement the following methods
  more efficiently.

  * :meth:`semlix.matching.Matcher.skip_to`

  * :meth:`semlix.matching.Matcher.weight`

* If the implementation supports quality, you should implement the following
  methods.

  * :meth:`semlix.matching.Matcher.supports_quality`

  * :meth:`semlix.matching.Matcher.quality`

  * :meth:`semlix.matching.Matcher.block_quality`

  * :meth:`semlix.matching.Matcher.skip_to_quality`
