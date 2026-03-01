from __future__ import annotations

import bisect
import logging
import math
import os
import re
import threading
import unicodedata
from contextlib import contextmanager
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import timezone
from functools import lru_cache
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING
from typing import Literal

import tantivy
from dateutil.relativedelta import relativedelta
from django.conf import settings
from django.utils import timezone as django_timezone
from django.utils.timezone import get_current_timezone
from django.utils.timezone import now
from guardian.shortcuts import get_users_with_perms
from whoosh.qparser.dateparse import DateParserPlugin
from whoosh.qparser.dateparse import English
from whoosh.util.times import timespan

from documents.models import Document
from documents.models import User

if TYPE_CHECKING:
    from django.db.models import QuerySet
    from whoosh.searching import ResultsPage

logger = logging.getLogger("paperless.index")
index_dir = f"{settings.INDEX_DIR}_tantivy"

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")
WORD_RE = re.compile(r"\w+", flags=re.IGNORECASE)

MAX_RESULT_LIMIT = 10001


def extract_bigram_content(text: str) -> str:
    """
    Extract relevant bigrams from a given text.

    This is used for languages that cannot be properly indexed with simple tokenizer,
    such as CJK languages.
    """
    cjk_sequences = CJK_RE.findall(text)
    return " ".join(cjk_sequences)


# In Tantivy, a text analyzer is a pipeline of
# a tokenizer, optionally followed by filters.
bigram_analyzer: tantivy.TextAnalyzer = (
    tantivy.TextAnalyzerBuilder(
        tantivy.Tokenizer.ngram(2, 2),
    )
    .filter(tantivy.Filter.lowercase())
    .build()
)

# Remove words longer than 64 characters, make the fields case AND diacritic insensitive
simple_analyzer: tantivy.TextAnalyzer = (
    tantivy.TextAnalyzerBuilder(
        tantivy.Tokenizer.simple(),
    )
    .filter(tantivy.Filter.remove_long(65))
    .filter(tantivy.Filter.lowercase())
    .filter(tantivy.Filter.ascii_fold())
    .build()
)


@lru_cache(maxsize=1)
def get_schema():
    """
    Prepare the Tantivy index schema.

    index_options: for text fields. "position" is default (frequency and position), "frequency" or "basic".
    stored: whether the field is stored and can be retrieved directly.
    fast: whether the field is stored in a columnar fashion, needed for sorting.
    indexed: needed for integer fields to be searchable.
    """
    sb = tantivy.SchemaBuilder()
    # TODO are all the has_ really needed?
    sb.add_integer_field("id", stored=True, indexed=True)
    sb.add_text_field("title", stored=True, fast=True, index_option="basic")
    sb.add_text_field(
        "autocomplete_word",
        stored=True,
        index_option="basic",
        tokenizer_name="simple_analyzer",
    )  # Alphabetically sorted list
    sb.add_text_field("content", stored=True, tokenizer_name="simple_analyzer")
    sb.add_text_field(
        "bigram_content",
        tokenizer_name="bigram_analyzer",
        index_option="freq",
    )  # used for languages such as CJK
    sb.add_integer_field("asn", stored=True, fast=True)
    sb.add_text_field("correspondent", stored=True, fast=True, tokenizer_name="default")
    sb.add_integer_field("correspondent_id", stored=True)
    sb.add_boolean_field("has_correspondent", stored=True)
    sb.add_text_field("tag", stored=True, tokenizer_name="simple_analyzer")
    sb.add_integer_field("tag_id", stored=True)
    sb.add_boolean_field("has_tag", stored=True)
    sb.add_text_field("type", stored=True, fast=True, tokenizer_name="default")
    sb.add_integer_field("type_id", stored=True)
    sb.add_boolean_field("has_type", stored=True)
    sb.add_date_field(
        "created",
        stored=True,
        fast=True,
        indexed=True,
    )  # Indexed dates must be timezone-naive datetimes
    sb.add_date_field("modified", stored=True, fast=True, indexed=True)
    sb.add_date_field("added", stored=True, fast=True, indexed=True)
    sb.add_text_field("path", stored=True)
    sb.add_integer_field("path_id", stored=True)
    sb.add_boolean_field("has_path", stored=True)
    sb.add_text_field("notes", stored=True, tokenizer_name="simple_analyzer")
    sb.add_integer_field("num_notes", stored=True, fast=True)
    sb.add_text_field("custom_fields", stored=True, tokenizer_name="simple_analyzer")
    sb.add_integer_field("custom_field_count", stored=True)
    sb.add_integer_field("custom_fields_id", stored=True)
    sb.add_boolean_field("has_custom_fields", stored=True)
    sb.add_text_field("owner", stored=True, fast=True)
    sb.add_integer_field("owner_id", stored=True, indexed=True)
    sb.add_boolean_field("has_owner", indexed=True)
    sb.add_integer_field("viewer_id", stored=True, indexed=True)
    sb.add_text_field("checksum", stored=True, index_option="basic")
    sb.add_integer_field("page_count", stored=True, fast=True)
    sb.add_text_field("original_filename", stored=True)
    sb.add_boolean_field("is_shared", stored=True)
    return sb.build()


def recreate_index_dir(path=index_dir):
    if Path(path).exists():
        rmtree(path)
    Path(path).mkdir(parents=True, exist_ok=True)


@contextmanager
def open_index(*, recreate=False, reload=True):
    path = index_dir
    if recreate or not Path(path).exists():
        recreate_index_dir(path)
    try:
        index = tantivy.Index(schema=get_schema(), path=path)
        index.register_tokenizer("bigram_analyzer", bigram_analyzer)
        index.register_tokenizer("simple_analyzer", simple_analyzer)
    except ValueError as e:
        # Schema has changed
        logger.warning(f"Recreating index due to error: {e}")
        recreate_index_dir(path)
        index = tantivy.Index(schema=get_schema(), path=path)
    if reload:
        index.reload()  # Ensure we have the latest commit?
    yield index


class AsyncWriter(threading.Thread):
    LOCK_EXC_MSG = "Failed to acquire Lockfile"

    def __init__(self, index: tantivy.Index, delay=0.25, **writerargs):
        """
        Asynchronous writer for tantivy, inspired from Whoosh's AsyncWriter.

        :param index: the :class:`whoosh.index.Index` to write to.
        :param delay: the delay (in seconds) between attempts to instantiate
            the actual writer.
        :param writerargs: an optional dictionary specifying keyword arguments
            to to be passed to the index's ``writer()`` method.
        """
        threading.Thread.__init__(self)
        self.running = False
        self.index = index
        self.writerargs = writerargs or {}
        self.delay = delay
        self.events = []
        try:
            self.writer = self.index.writer(**self.writerargs)
        except ValueError as e:
            if self.LOCK_EXC_MSG in str(e):
                self.writer = None
            else:
                raise

    def _record(self, method, *args, **kwargs):
        if self.writer:
            getattr(self.writer, method)(*args, **kwargs)
        else:
            self.events.append((method, args, kwargs))

    def run(self):
        self.running = True
        writer = self.writer
        while writer is None:
            try:
                writer = self.index.writer(**self.writerargs)
            except ValueError as e:
                if self.LOCK_EXC_MSG in str(e):
                    import time as stime

                    stime.sleep(self.delay)
                else:
                    raise
        for method, args, kwargs in self.events:
            getattr(writer, method)(*args, **kwargs)
        writer.commit(*self.commitargs, **self.commitkwargs)
        writer.wait_merging_threads()

    def delete_documents_by_query(self, *args, **kwargs):
        self._record("delete_documents_by_query", *args, **kwargs)

    def add_document(self, *args, **kwargs):
        self._record("add_document", *args, **kwargs)

    def commit_and_wait_merging_threads(self, *args, **kwargs):
        if self.writer:
            self.writer.commit(*args, **kwargs)
            self.writer.wait_merging_threads()
        else:
            self.commitargs, self.commitkwargs = args, kwargs
            self.start()
        if self.is_alive():
            self.join()

    def rollback(self, *args, **kwargs):
        if self.writer:
            return self.writer.rollback(*args, **kwargs)
        # If we never acquired the writer, drop buffered events
        self.events.clear()
        # If a background thread is running, we can't reliably abort tantivy's writer
        # but dropping events is the best effort here.


@contextmanager
def open_index_writer(*, reload=True, **kwargs):
    with open_index(reload=reload) as index:
        writer = AsyncWriter(index, num_threads=os.cpu_count() or 0)
        try:
            yield writer
        except Exception as e:
            logger.exception(str(e))
            writer.rollback()
        finally:
            writer.commit_and_wait_merging_threads()


@contextmanager
def open_index_searcher():
    with open_index() as index:
        index.config_reader(reload_policy="commit")
        yield index.searcher()


def tokenize_for_autocomplete(text: str):
    """Return all distinct autocomplete words from a given text."""
    return {m.group(0).lower() for m in WORD_RE.finditer(text)}


def datetime_to_tantivy(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if not isinstance(dt, datetime):
        raise TypeError(f"Expected datetime, got {type(dt)}")
    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def update_document(
    writer: tantivy.IndexWriter,
    doc: Document,
    effective_content: str | None = None,
    viewer_ids: list[int] | None = None,
) -> None:
    if effective_content is None:
        effective_content = doc.content
    tag_list = list(doc.tags.all())
    tags = ",".join([t.name for t in tag_list])
    tags_ids = [t.id for t in tag_list]
    notes = ",".join([str(n.note) for n in doc.notes.all()])
    custom_field_list = list(doc.custom_fields.all())
    custom_fields = ",".join([str(c) for c in custom_field_list])
    custom_fields_ids = [f.field.id for f in custom_field_list]
    asn: int | None = doc.archive_serial_number
    if asn is not None and (
        asn < Document.ARCHIVE_SERIAL_NUMBER_MIN
        or asn > Document.ARCHIVE_SERIAL_NUMBER_MAX
    ):
        logger.error(
            f"Not indexing Archive Serial Number {asn} of document {doc.pk}. "
            f"ASN is out of range "
            f"[{Document.ARCHIVE_SERIAL_NUMBER_MIN:,}, "
            f"{Document.ARCHIVE_SERIAL_NUMBER_MAX:,}.",
        )
        asn = 0
    if viewer_ids is None:
        # Fallback for single-document indexing (not bulk reindex)
        users_with_perms = get_users_with_perms(
            doc,
            only_with_perms_in=["view_document"],
        )
        viewer_ids = [int(u.id) for u in users_with_perms]

    writer.delete_documents_by_query(
        tantivy.Query.term_query(get_schema(), "id", doc.pk),
    )
    indexed_doc = tantivy.Document(
        id=doc.pk,
        title=doc.title or "",
        content=effective_content,
        bigram_content=extract_bigram_content(effective_content),
        correspondent=doc.correspondent.name if doc.correspondent else "",
        correspondent_id=doc.correspondent.id if doc.correspondent else 0,
        has_correspondent=doc.correspondent is not None,
        has_tag=len(tags) > 0,
        type=doc.document_type.name if doc.document_type else "",
        type_id=doc.document_type.id if doc.document_type else 0,
        has_type=doc.document_type is not None,
        created=datetime_to_tantivy(datetime.combine(doc.created, time.min)),
        added=datetime_to_tantivy(doc.added),
        asn=asn or 0,
        modified=datetime_to_tantivy(doc.modified),
        path=doc.storage_path.name if doc.storage_path else "",
        path_id=doc.storage_path.id if doc.storage_path else 0,
        has_path=doc.storage_path is not None,
        notes=notes or "",
        num_notes=len(notes),
        custom_fields=custom_fields or "",
        custom_field_count=len(custom_field_list),
        has_custom_fields=len(custom_fields) > 0,
        owner=doc.owner.username if doc.owner else "",
        owner_id=int(doc.owner.id if doc.owner and doc.owner.id else 0),
        has_owner=bool(doc.owner and doc.owner.id is not None),
        checksum=doc.checksum or "",
        page_count=doc.page_count or 0,
        original_filename=doc.original_filename,
        is_shared=len(viewer_ids) > 0,
    )
    for tag_id in tags_ids:
        indexed_doc.add_integer("tag_id", tag_id)
    for custom_fields_id in custom_fields_ids:
        indexed_doc.add_integer("custom_fields_id", custom_fields_id)
    for viewer_id in viewer_ids:
        indexed_doc.add_integer("viewer_id", viewer_id)

    autocomplete_words = tokenize_for_autocomplete(effective_content)

    # Add also title and note content for autocomplete
    autocomplete_words.update(tokenize_for_autocomplete(doc.title))
    autocomplete_words.update(tokenize_for_autocomplete(notes))

    # Make sure to sort the autocomplete word lists.
    # We assume it's sorted for autocomplete search function.
    for word in sorted(autocomplete_words):
        indexed_doc.add_text("autocomplete_word", word)
    writer.add_document(indexed_doc)
    logger.debug(f"Index updated for document {doc.pk}.")


def remove_document(writer: tantivy.IndexWriter, doc: Document) -> None:
    remove_document_by_id(writer, doc.pk)


def remove_document_by_id(writer: tantivy.IndexWriter, doc_id) -> None:
    writer.delete_documents_by_query(
        tantivy.Query.term_query(get_schema(), "id", doc_id),
    )


def add_or_update_document(document: Document) -> None:
    with open_index_writer() as writer:
        update_document(writer, document)


def add_or_update_documents(documents: list[Document], batchsize=0) -> None:
    if batchsize <= 0:
        with open_index_writer() as writer:
            for document in documents:
                update_document(writer, document)
    else:
        for i in range(0, len(documents), batchsize):
            batch = documents[i : i + batchsize]
            # do stuff with batch
            with open_index_writer() as writer:
                for document in batch:
                    update_document(writer, document)


def remove_document_from_index(document: Document) -> None:
    with open_index_writer() as writer:
        remove_document(writer, document)


class MappedDocIdSet:
    def __init__(self, filter_queryset: QuerySet):
        self.document_ids = set(filter_queryset.values_list("id", flat=True))

    def filter(self, results):
        # results : list of tantivy.Document
        return [r for r in results if r.get("id") in self.document_ids]


class ResultsPage:
    """Tantivy result page"""

    results: list
    pagelen: int
    pagenum: int
    total: int

    def __init__(self, results, pagelen, pagenum, total):
        self.results = results
        self.pagelen = pagelen
        self.pagenum = pagenum
        self.total = total


class Hit:
    def __init__(
        self,
        id,
        score,
        rank=None,
        content=None,
        highlights=None,
        note_highlights=None,
    ):
        self.id: int = id
        self.score: float = score
        self.rank: int = rank
        self.content: str = content
        self._highlights: str = highlights
        self._note_highlights: str = note_highlights

    def __getitem__(self, key):
        if key == "id":
            return self.id
        raise KeyError(key)

    def highlights(self, field, *args, **kwargs):
        text = self._note_highlights if field == "notes" else self._highlights
        result = text.replace("<b>", '<span class="match">').replace(
            "</b>",
            "</span>",
        )
        return result

    def __repr__(self):
        return f"Hit(id={self.id}, score={self.score}, rank={self.rank})"


class TantivyResultsPage:
    """
    Taken from Whoosh ResultsPage object, for use with Tantivy.

    This contains all results, but with a pagination system."""

    def __init__(self, results: list[Hit], pagenum, pagelen):
        self.results = results
        self.total = len(results)

        if pagenum < 1:
            raise ValueError("pagenum must be >= 1")

        self.pagecount = math.ceil(self.total / pagelen)
        self.pagenum = min(self.pagecount, pagenum)
        offset = (self.pagenum - 1) * pagelen
        if (offset + pagelen) > self.total:
            pagelen = self.total - offset
        self.offset = offset
        self.pagelen = pagelen

    def __getitem__(self, n):
        offset = self.offset
        if isinstance(n, slice):
            start, stop, step = n.indices(self.pagelen)
            return self.results.__getitem__(slice(start + offset, stop + offset, step))
        else:
            return self.results.__getitem__(n + offset)

    def __iter__(self):
        return iter(self.results[self.offset : self.offset + self.pagelen])

    def __len__(self):
        return self.total

    def docnum(self, n):
        """Returns the document number of the hit at the nth position on this
        page.
        """
        return self.results.docnum(n + self.offset)

    @property
    def doc_ids(self):
        """Return the DB ids of the documents in the result page"""
        return [result.id for result in self.results]


class SimplePage:
    """A pre-sliced page of results. No internal re-pagination."""

    def __init__(self, results: list[Hit], total: int):
        self.results = results
        self.total = total

    def __getitem__(self, n):
        return self.results[n]

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return self.total

    @property
    def doc_ids(self):
        return [hit.id for hit in self.results]


class DelayedQuery:
    def _get_query(self):
        raise NotImplementedError  # pragma: no cover

    def _get_query_sortedby(self) -> tuple[None, Literal[False]] | tuple[str, bool]:
        if "ordering" not in self.query_params:
            return None, False

        field: str = self.query_params["ordering"]

        sort_fields_map: dict[str, str] = {
            "created": "created",
            "modified": "modified",
            "added": "added",
            "title": "title",
            "correspondent__name": "correspondent",
            "document_type__name": "type",
            "archive_serial_number": "asn",
            "num_notes": "num_notes",
            "owner": "owner",
            "page_count": "page_count",
        }

        if field.startswith("-"):
            field = field[1:]
            reverse = True
        else:
            reverse = False

        if field not in sort_fields_map:
            return None, False
        else:
            return sort_fields_map[field], reverse

    def _manual_sort_requested(self):
        # See https://github.com/paperless-ngx/paperless-ngx/pull/11383
        # Tantivy implementation was done before this PR. Needs to implement.
        return False

    def __init__(
        self,
        searcher: tantivy.Searcher,
        query_params,
        page_size,
        filter_queryset: QuerySet,
        user=None,
    ) -> None:
        self.searcher = searcher
        self.query_params = query_params
        self.page_size = page_size
        self.saved_results = dict()
        self.first_score = None
        self.filter_queryset = filter_queryset
        self.suggested_correction = None
        self.user = user
        self._count: int | None = None
        self._combined_query = None
        self._sort_field = None
        self._sort_order = None

    def _build_combined_query(self):
        """Build the Tantivy query with permissions baked in. Called once."""
        if self._combined_query is not None:
            return

        q, mask, suggested_correction = self._get_query()
        self.suggested_correction = suggested_correction

        # Combine search query with permissions query
        schema = get_schema()
        perm_q = get_permissions_query(self.user, schema)
        self._combined_query = tantivy.Query.boolean_query(
            [
                (tantivy.Occur.Must, q),
                (tantivy.Occur.Must, perm_q),
            ],
        )

        sortedby, reverse = self._get_query_sortedby()
        self._sort_field = sortedby
        self._sort_order = tantivy.Order.Desc if reverse else tantivy.Order.Asc

        # Handle "more like this" special case
        if isinstance(self, DelayedMoreLikeThisQuery) and sortedby not in [
            "score",
            None,
        ]:
            search_result = self.searcher.search(
                self._combined_query,
                limit=MAX_RESULT_LIMIT,
            ).hits
            more_like_this_ids = [
                self.searcher.doc(doc_addr)["id"][0] for _, doc_addr in search_result
            ]
            self._combined_query = tantivy.Query.boolean_query(
                [
                    (
                        tantivy.Occur.Must,
                        tantivy.Query.term_set_query(
                            schema,
                            "id",
                            more_like_this_ids,
                        ),
                    ),
                    (tantivy.Occur.Must, perm_q),
                ],
            )

    def __len__(self) -> int:
        if self._manual_sort_requested():
            manual_hits = self._manual_hits()
            return len(manual_hits)

        if self._count is None:
            self._build_combined_query()
            # In tantivy, no need to fetch all results to count them, one is enough
            result = self.searcher.search(
                self._combined_query,
                limit=1,
            )
            self._count = result.count
        return self._count

    def _get_all_ids(self) -> list[int]:
        """Get all matching document IDs (used for "select all" in front-end). Lightweight, without snippets."""
        self._build_combined_query()
        result = self.searcher.search(
            self._combined_query,
            limit=MAX_RESULT_LIMIT,
            order_by_field=self._sort_field,
            order=self._sort_order,
        )
        ids = []
        for _, doc_addr in result.hits:
            doc = self.searcher.doc(doc_addr)
            ids.append(doc["id"][0])
        return ids

    def __getitem__(self, item: slice):
        if item.start in self.saved_results:
            return self.saved_results[item.start]

        if self._manual_sort_requested():
            manual_hits = self._manual_hits()
            start = 0 if item.start is None else item.start
            stop = item.stop
            hits = manual_hits[start:stop] if stop is not None else manual_hits[start:]
            page = ManualResultsPage(hits)
            self.saved_results[start] = page
            return page

        self._build_combined_query()
        q = self._combined_query

        start = item.start or 0
        page_size = (item.stop - start) if item.stop else self.page_size

        result = self.searcher.search(
            q,
            limit=page_size,
            offset=start,
            order_by_field=self._sort_field,
            order=self._sort_order,
        )

        if self._count is None:
            self._count = result.count

        # Generate snippets only for this page
        content_snippet_generator = tantivy.SnippetGenerator.create(
            self.searcher,
            q,
            get_schema(),
            "content",
        )
        content_snippet_generator.set_max_num_chars(550)
        note_snippet_generator = tantivy.SnippetGenerator.create(
            self.searcher,
            q,
            get_schema(),
            "notes",
        )
        note_snippet_generator.set_max_num_chars(100)

        hits = []
        for rank, (score, doc_addr) in enumerate(result.hits, start=start + 1):
            doc = self.searcher.doc(doc_addr)
            content_snippet = content_snippet_generator.snippet_from_doc(doc)
            note_snippet = note_snippet_generator.snippet_from_doc(doc)
            hits.append(
                Hit(
                    doc["id"][0],
                    score,
                    rank=rank,
                    content=content_snippet.fragment(),
                    highlights=content_snippet.to_html(),
                    note_highlights=note_snippet.to_html(),
                ),
            )

        page = SimplePage(results=hits, total=result.count)
        self.saved_results[start] = page
        return page


class ManualResultsPage(list):
    def __init__(self, hits):
        super().__init__(hits)
        self.results = ManualResults(hits)


class ManualResults:
    def __init__(self, hits):
        self._docnums = [hit.docnum for hit in hits]

    def docs(self):
        return self._docnums


class LocalDateParser(English):
    def reverse_timezone_offset(self, d):
        return (d.replace(tzinfo=django_timezone.get_current_timezone())).astimezone(
            timezone.utc,
        )

    def date_from(self, *args, **kwargs):
        d = super().date_from(*args, **kwargs)
        if isinstance(d, timespan):
            d.start = self.reverse_timezone_offset(d.start)
            d.end = self.reverse_timezone_offset(d.end)
        elif isinstance(d, datetime):
            d = self.reverse_timezone_offset(d)
        return d


date_parser_plugin = DateParserPlugin(
    basedate=django_timezone.now(),
    dateparser=LocalDateParser(),
)


def parse_natural_date(expr: str):
    """
    Parse une expression de date naturelle en utilisant le parser anglais de Whoosh.
    Retourne (start, end).
    """

    result = date_parser_plugin.dateparser.date_from(expr)

    if not result:
        return (None, None)

    if isinstance(result, timespan):
        return result.start, result.end
    else:
        raise RuntimeError(f"Unexpected result type: {type(result)}")


DATE_FIELDS = ["created", "added", "modified"]
DATE_QUERY_RE = re.compile(
    r"""
    (?P<field>\w+)                    # field ex: created
    \s*:\s*                           # separator ":"
    (?P<op>>=|<=|>|<)?\s*             # optional operator
    (                                 #
        (?P<bracket_expr>             #   if [expr]
            \[[^\[\]]*\]
        )
        |                             #   or
        (?P<quoted_expr>              #   expr' or "expr"
            ['"][^'"]*['"]
        )
        |                             #   or
        (?P<bare_expr>                #   simple expression
            [^\],'\"]+?
        )
    )
    (?=(?:\s+|,)\w+\s*:|$)            # stop at next field, comma or end
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def replace_date_expr(match, date_fields=DATE_FIELDS):
    field = match.group("field").strip()
    operator = match.group("op") or ""
    expr = (
        match.group("bracket_expr")
        or match.group("quoted_expr")
        or match.group("bare_expr")
    )
    expr = expr.strip("[]'\" ").strip()

    # ignore non-date fields
    if field not in date_fields:
        return match.group(0)

    # parse date expression
    try:
        start, end = parse_natural_date(expr)
    except Exception as e:
        print(f"⚠️ parse_natural_date failed for {expr}: {e}")
        return match.group(0)

    # si parse_natural_date a renvoyé None
    if not start and not end:
        return match.group(0)

    # build Tantivy range
    if operator in (">", ">="):
        return f"{field}:[{start.isoformat()} TO *]"
    elif operator in ("<", "<="):
        return f"{field}:[* TO {end.isoformat()}]"
    elif "to" in expr.lower():
        return f"{field}:[{start.isoformat()} TO {end.isoformat()}]"
    else:
        return f"{field}:[{start.isoformat()} TO {end.isoformat()}]"


# def preprocess_query_dates(query: str) -> str:
#     # Convert quotes into brackets first
#     query = re.sub(
#         r"(\w+)\s*:\s*(['\"])(.+?)\2",
#         lambda m: f"{m.group(1)}:[{m.group(3)}]",
#         query,
#     )

#     # Replace date expressions
#     return DATE_QUERY_RE.sub(replace_date_expr, query)


def preprocess_query_dates(query: str, date_fields=DATE_FIELDS) -> str:
    """
    Only convert quoted expressions to [expr] for *date* fields,
    then replace date expressions by parsed Tantivy ranges.
    """

    # Build an alternation of the date field names, e.g. "created|added|modified"
    # Use re.IGNORECASE so fields are matched case-insensitively.
    field_alternation = "|".join(map(re.escape, date_fields))
    quoted_for_date_re = re.compile(
        rf"(?i)\b(?P<field>{field_alternation})\s*:\s*(['\"])(?P<body>.+?)\2",
    )

    # Replace only quoted date fields: created:"today" -> created:[today]
    query = quoted_for_date_re.sub(
        lambda m: f"{m.group('field')}:[{m.group('body')}]",
        query,
    )

    # Now run the main DATE_QUERY_RE substitution which will call replace_date_expr
    # For re.sub with a function that needs extra args, use a lambda capturing date_fields.
    return DATE_QUERY_RE.sub(
        lambda m: replace_date_expr(m, date_fields=date_fields),
        query,
    )


NQL_TOKENS = ["AND", "OR", "NOT", "TO", "+", "-", "(", ")", '"', "[", "]"]


def rewrite_default_and_keywords(query_string: str) -> str:
    """
    Separate keywords by AND when natural query language isn't detected.
    """
    if not any(
        [w in query_string for w in NQL_TOKENS],
    ):
        # No natural language detected, so separate by AND
        query_string = re.sub(r"\s+", " AND ", query_string.strip())
    return query_string


# Known keywords in your search syntax
KEYWORDS = [
    "content",
    "bigram_content",
    "title",
    "correspondent",
    "tag",
    "type",
    "notes",
    "custom_fields",
    "added",
    "created",
    "modified",
]


def normalize_query(query: str) -> str:
    """The front-end can send date filters after a comma.
    This fixes this by replacing the comma by a "AND" condition with the rest of the query."""
    # Split by commas
    parts = [p.strip() for p in query.split(",") if p.strip()]

    normalized_parts = []
    free_text_parts = []

    for part in parts:
        # Check if it starts with a known keyword
        if any(part.startswith(f"{kw}:") for kw in KEYWORDS):
            # If we already collected free text, wrap it in parentheses
            if free_text_parts:
                normalized_parts.append(f"({' '.join(free_text_parts)})")
                free_text_parts = []
            normalized_parts.append(part)
        else:
            free_text_parts.append(part)

    # Add remaining free text if any
    if free_text_parts:
        normalized_parts.append(f"({' '.join(free_text_parts)})")

    # Join everything with AND
    return " AND ".join(normalized_parts)


FIELD_EXPR_RE = re.compile(
    r"""
    (?P<field>\w+)
    \s*:\s*
    (?P<value>
        \[[^\[\]]*\]      # plage entre crochets
        |
        [^,\s]+           # ou valeur simple
    )
    (?:,|(?=[\s)]|$))     # séparée par virgule, escape, parenthèse fermante ou fin
    """,
    re.VERBOSE,
)


def extract_query_parts(query: str):
    """Extract structured filters (field:value) and remaining free text."""
    filters = []
    consumed_spans = []

    for match in FIELD_EXPR_RE.finditer(query):
        field = match.group("field")
        value = match.group("value")
        filters.append((field, value))
        consumed_spans.append(match.span())

    # Remove all matched segments from the original string
    free_text = query
    for start, end in reversed(consumed_spans):
        free_text = free_text[:start] + " " + free_text[end:]

    # Normalize free text
    text_terms = [
        t
        for t in re.findall(r"\w+(?:['’]\w+)?", free_text)  # Noqa RUF001
        if t and t not in NQL_TOKENS
    ]

    return {"filters": filters, "text_terms": text_terms}


class DelayedFullTextQuery(DelayedQuery):
    def _get_query(self) -> tuple:
        q_str = self.query_params["query"]
        print(f"raw query: {q_str}")
        q_str = rewrite_natural_date_keywords(q_str)
        print(f"with natural date keywords: {q_str}")
        q_str = normalize_query(q_str)
        print(f"normalized query: {q_str}")

        q_str = rewrite_default_and_keywords(q_str)
        print(f"with default and keywords: {q_str}")
        q_str = preprocess_query_dates(q_str)
        print(f"with dates: {q_str}")
        text_terms = extract_query_parts(q_str)["text_terms"]
        print(f"text terms: {text_terms}")
        if settings.ADVANCED_FUZZY_SEARCH_TRESHOLD and any(
            [len(t) >= settings.ADVANCED_FUZZY_SEARCH_TRESHOLD for t in text_terms],
        ):
            fuzzy_search = True
        else:
            fuzzy_search = False
        # TODO: date parsing plugin like whoosh
        with open_index() as index:
            queries = list()
            q = index.parse_query_lenient(
                q_str,
                [
                    "content",
                    "bigram_content",
                    "title",
                    "correspondent",
                    "tag",
                    "type",
                    "notes",
                    "custom_fields",
                    "added",
                    "created",
                    "modified",
                ],
                field_boosts={"title": 5.0, "content": 0.5},
                conjunction_by_default=True,
            )[0]
            queries.append(q)
            if fuzzy_search:
                # An exact match should outweigh a fuzzy one
                fuzzy_q = index.parse_query_lenient(
                    # fuzzy field: prefix (prefix must match) bool, distance int, transpose_cost_one (2 letters inverted cost 1 instead of 2): bool
                    q_str,
                    [
                        "content",
                        "bigram_content",
                        "title",
                        "correspondent",
                        "tag",
                        "type",
                        "notes",
                        "custom_fields",
                        "added",
                        "created",
                        "modified",
                    ],
                    field_boosts={"title": 3.0, "content": 0.5},
                    fuzzy_fields={
                        "content": (True, 1, True),
                        "title": (True, 1, True),
                        "correspondent": (True, 1, True),
                        "tag": (True, 1, True),
                        "type": (True, 1, True),
                        "notes": (True, 1, True),
                        "custom_fields": (True, 1, True),
                    },
                    conjunction_by_default=True,
                )[0]
                queries.append(tantivy.Query.boost_query(fuzzy_q, 0.1))
            q = tantivy.Query.boolean_query(
                [(tantivy.Occur.Should, q) for q in queries],
            )
            words = autocomplete(
                index,
                q_str,
                limit=1,
                user=self.user,
                fuzzy_search=True,
            )
            suggested_correction = words[0] if words and words[0] != q_str else None

        return q, None, suggested_correction


class DelayedMoreLikeThisQuery(DelayedQuery):
    def _get_query(self) -> tuple:
        more_like_doc_id = int(self.query_params["more_like_id"])

        # Fetch the current doc's address
        current_doc_q = tantivy.Query.term_query(get_schema(), "id", more_like_doc_id)

        doc_address: tantivy.DocAddress = self.searcher.search(
            current_doc_q,
            limit=1,
        ).hits[0][1]

        # Exclude the current doc for "more like this" search results
        more_like_this_q = tantivy.Query.more_like_this_query(doc_address)
        q = tantivy.Query.boolean_query(
            [
                (tantivy.Occur.Must, more_like_this_q),
                (tantivy.Occur.MustNot, current_doc_q),
            ],
        )

        mask = None  # TODO?

        return q, mask, None


def _normalize(text: str) -> str:
    # Decompose characters (NFKD) and remove diacritics
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    ).lower()


def prefix_view(sorted_words: list[str], prefix: str):
    """Return an iterator over words starting with prefix from a given list of sorted words."""

    prefix = _normalize(prefix)
    start_idx = bisect.bisect_left(sorted_words, prefix)

    # iterate forward lazily, stop when prefix no longer matches
    def generator():
        for i in range(start_idx, len(sorted_words)):
            word = sorted_words[i]
            if not _normalize(word).startswith(prefix):
                break
            yield word

    return generator()


def autocomplete(
    index,
    term: str,
    limit: int = 10,
    user: User | None = None,
    *,
    fuzzy_search=False,
) -> list[str]:
    """
    Returns a list of autocomplete suggestions for the given term.
    Caveat: documents are searched by Tantivy TF-IDF score,
    but for each document found, all prefixes in this document are added to the list.
    So only the first prefix is guaranteed to have the highest score.
    """
    with open_index() as index:
        result: list[str] = list()
        searcher = index.searcher()
        schema = index.schema
        perm_q = get_permissions_query(user, schema)
        normalized_term = _normalize(term)

        # Find the exact match first
        exact_subquery = tantivy.Query.term_query(
            schema,
            "autocomplete_word",
            normalized_term,
        )
        exact_q = tantivy.Query.boolean_query(
            [
                (tantivy.Occur.Must, exact_subquery),
                (tantivy.Occur.Must, perm_q),
            ],
        )

        hits = searcher.search(exact_q, limit=1).hits
        if hits:
            result.append(term)

        # Find prefixed terms until limit is reached
        try:
            if fuzzy_search:
                prefix_subquery = tantivy.Query.fuzzy_term_query(
                    schema,
                    "autocomplete_word",
                    normalized_term,
                    1,
                    transposition_cost_one=True,
                    prefix=True,
                )
            else:
                prefix_subquery = tantivy.Query.regex_query(
                    schema,
                    "autocomplete_word",
                    f"{normalized_term}.*",
                )
        except ValueError as e:
            # Autocomplete doesn't support special terms, e.g. parentheses, +, - etc.
            logger.debug(f"{e}")
            return []
        seen = set()
        remaining_limit = limit - len(result)
        while len(result) < limit:
            seen_q = []
            for word in seen:
                seen_q.append(
                    (
                        tantivy.Occur.MustNot,
                        tantivy.Query.term_query(schema, "autocomplete_word", word),
                    ),
                )
            prefix_q = tantivy.Query.boolean_query(
                [
                    (tantivy.Occur.MustNot, exact_subquery),
                    (tantivy.Occur.Must, prefix_subquery),
                    (tantivy.Occur.Must, perm_q),
                    *seen_q,
                ],
            )

            hits = searcher.search(prefix_q, limit=remaining_limit * 5).hits
            if not hits:
                return result
            found_new_word = False
            for _, addr in hits:
                doc = searcher.doc(addr)
                all_words = doc["autocomplete_word"]
                for word in prefix_view(all_words, normalized_term):
                    if word in seen:
                        continue
                    seen.add(word)
                    found_new_word = True
                    result.append(word)
                    if len(result) >= limit:
                        return result
            if not found_new_word:
                return result
        return result


def get_permissions_query(user: User | None, schema) -> tantivy.Query:
    # No filter if super user
    if user and user.is_superuser:
        return tantivy.Query.all_query()

    # Always return documents with no owner
    queries = [tantivy.Query.term_query(schema, "has_owner", "false")]

    if user:
        user_id = str(user.id)
        queries.extend(
            [
                tantivy.Query.term_query(schema, "owner_id", int(user_id)),
                tantivy.Query.term_query(schema, "viewer_id", int(user_id)),
            ],
        )

    return tantivy.Query.boolean_query([(tantivy.Occur.Should, q) for q in queries])


def rewrite_natural_date_keywords(query_string: str) -> str:
    """
    Rewrites natural date keywords (e.g. added:today or added:"yesterday") to UTC range syntax for Whoosh.
    This resolves timezone issues with date parsing in Whoosh as well as adding support for more
    natural date keywords.
    """

    tz = get_current_timezone()
    local_now = now().astimezone(tz)
    today = local_now.date()

    # all supported Keywords
    pattern = r"(\b(?:added|created|modified))\s*:\s*[\"']?(today|yesterday|this month|previous month|previous week|previous quarter|this year|previous year)[\"']?"

    def repl(m):
        field = m.group(1)
        keyword = m.group(2).lower()

        match keyword:
            case "today":
                start = datetime.combine(today, time.min, tzinfo=tz)
                end = datetime.combine(today, time.max, tzinfo=tz)

            case "yesterday":
                yesterday = today - timedelta(days=1)
                start = datetime.combine(yesterday, time.min, tzinfo=tz)
                end = datetime.combine(yesterday, time.max, tzinfo=tz)

            case "this month":
                start = datetime(local_now.year, local_now.month, 1, 0, 0, 0, tzinfo=tz)
                end = start + relativedelta(months=1) - timedelta(seconds=1)

            case "previous month":
                this_month_start = datetime(
                    local_now.year,
                    local_now.month,
                    1,
                    0,
                    0,
                    0,
                    tzinfo=tz,
                )
                start = this_month_start - relativedelta(months=1)
                end = this_month_start - timedelta(seconds=1)

            case "this year":
                start = datetime(local_now.year, 1, 1, 0, 0, 0, tzinfo=tz)
                end = datetime(local_now.year, 12, 31, 23, 59, 59, tzinfo=tz)

            case "previous week":
                days_since_monday = local_now.weekday()
                this_week_start = datetime.combine(
                    today - timedelta(days=days_since_monday),
                    time.min,
                    tzinfo=tz,
                )
                start = this_week_start - timedelta(days=7)
                end = this_week_start - timedelta(seconds=1)

            case "previous quarter":
                current_quarter = (local_now.month - 1) // 3 + 1
                this_quarter_start_month = (current_quarter - 1) * 3 + 1
                this_quarter_start = datetime(
                    local_now.year,
                    this_quarter_start_month,
                    1,
                    0,
                    0,
                    0,
                    tzinfo=tz,
                )
                start = this_quarter_start - relativedelta(months=3)
                end = this_quarter_start - timedelta(seconds=1)

            case "previous year":
                start = datetime(local_now.year - 1, 1, 1, 0, 0, 0, tzinfo=tz)
                end = datetime(local_now.year - 1, 12, 31, 23, 59, 59, tzinfo=tz)

        # Convert to UTC and format as RFC 3339 for Tantivy date fields
        start_str = start.astimezone(timezone.utc).isoformat()
        end_str = end.astimezone(timezone.utc).isoformat()
        return f"{field}:[{start_str} TO {end_str}]"

    return re.sub(pattern, repl, query_string, flags=re.IGNORECASE)
