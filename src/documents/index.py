from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING
from typing import Literal

from dateutil.relativedelta import relativedelta
import tantivy
from django.conf import settings
from django.utils import timezone as django_timezone
from django.utils.timezone import get_current_timezone
from django.utils.timezone import now
from guardian.shortcuts import get_users_with_perms
from whoosh import classify
from whoosh import highlight
from whoosh import query
from whoosh.highlight import HtmlFormatter
from whoosh.index import FileIndex
from whoosh.qparser.dateparse import English
from whoosh.util.times import timespan

from documents.models import CustomFieldInstance
from documents.models import Document
from documents.models import Note
from documents.models import User

if TYPE_CHECKING:
    from django.db.models import QuerySet
    from whoosh.searching import ResultsPage

logger = logging.getLogger("paperless.index")


def get_schema():
    sb = tantivy.SchemaBuilder()
    # TODO are all the has_ really needed?
    sb.add_unsigned_field("id", stored=True)
    sb.add_text_field("title", stored=True)
    sb.add_text_field("autocomplete_word", stored=True)
    sb.add_text_field("content", stored=False)  # pas nécessairement stored
    sb.add_unsigned_field("asn", stored=True)
    sb.add_text_field("correspondent", stored=True)
    sb.add_unsigned_field("correspondent_id", stored=True)
    sb.add_boolean_field("has_correspondent", stored=True)
    sb.add_text_field("tag", stored=True)
    sb.add_text_field("tag_id", stored=True)
    sb.add_boolean_field("has_tag", stored=True)
    sb.add_text_field("type", stored=True)
    sb.add_unsigned_field("type_id", stored=True)
    sb.add_boolean_field("has_type", stored=True)
    sb.add_date_field("created", stored=True)  # UNIX timestamp or ISO string
    sb.add_date_field("modified", stored=True)
    sb.add_date_field("added", stored=True)
    sb.add_text_field("path", stored=True)
    sb.add_unsigned_field("path_id", stored=True)
    sb.add_boolean_field("has_path", stored=True)
    sb.add_text_field("notes", stored=True)
    sb.add_unsigned_field("num_notes", stored=True)
    sb.add_text_field("custom_fields", stored=True)
    sb.add_unsigned_field("custom_field_count", stored=True)
    sb.add_text_field("custom_fields_id", stored=True)
    sb.add_boolean_field("has_custom_fields", stored=True)
    sb.add_text_field("owner", stored=True)
    sb.add_unsigned_field("owner_id", stored=True)
    sb.add_boolean_field("has_owner", stored=True)
    sb.add_unsigned_field("viewer_id", stored=True)  # tokenization raw preferred
    sb.add_text_field("checksum", stored=True)
    sb.add_unsigned_field("page_count", stored=True)
    sb.add_text_field("original_filename", stored=True)
    sb.add_boolean_field("is_shared", stored=True)  # or integer 0/1

    return sb.build()


def _recreate_index_dir(path):
    if Path(path).exists():
        rmtree(path)
    Path(path).mkdir(parents=True, exist_ok=True)


@contextmanager
def open_index(*, recreate=False, reload=True):
    path = str(settings.INDEX_DIR)
    if recreate or not Path(path).exists():
        _recreate_index_dir(path)
    try:
        index = tantivy.Index(schema=get_schema(), path=path)
    except ValueError as e:
        # Schema has changed
        logger.warning(f"Recreating index due to error: {e}")
        _recreate_index_dir(path)
        index = tantivy.Index(schema=get_schema(), path=path)
    if reload:
        index.reload()  # Ensure we have the latest commit?
    yield index


@contextmanager
def open_index_writer(**kwargs):
    with open_index(reload=False) as index:
        writer = index.writer()
        try:
            yield writer
        except Exception as e:
            logger.exception(str(e))
            writer.rollback()
        finally:
            writer.commit()
            writer.wait_merging_threads()


@contextmanager
def open_index_searcher():
    with open_index() as index:
        index.config_reader(reload_policy="commit")
        yield index.searcher()


def tokenize_for_autocomplete(text: str) -> list[str]:
    # Lowercase, split on non-word characters
    # Matches Tantivy’s default behavior for text fields
    words = re.findall(r"\b\w+\b", text.lower())
    return words


def update_document(writer: tantivy.IndexWriter, doc: Document) -> None:
    tags = ",".join([t.name for t in doc.tags.all()])
    tags_ids = ",".join([str(t.id) for t in doc.tags.all()])
    notes = ",".join([str(c.note) for c in Note.objects.filter(document=doc)])
    custom_fields = ",".join(
        [str(c) for c in CustomFieldInstance.objects.filter(document=doc)],
    )
    custom_fields_ids = ",".join(
        [str(f.field.id) for f in CustomFieldInstance.objects.filter(document=doc)],
    )
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
    users_with_perms = get_users_with_perms(
        doc,
        only_with_perms_in=["view_document"],
    )
    viewer_ids: str = ",".join([str(u.id) for u in users_with_perms])
    writer.delete_documents("id", doc.pk)
    writer.add_document(
        tantivy.Document(
            id=doc.pk,
            title=doc.title,
            content=doc.content,
            correspondent=doc.correspondent.name if doc.correspondent else "",
            correspondent_id=doc.correspondent.id if doc.correspondent else 0,
            has_correspondent=doc.correspondent is not None,
            tag=tags if tags else "",
            tag_id=tags_ids if tags_ids else "",
            has_tag=len(tags) > 0,
            type=doc.document_type.name if doc.document_type else "",
            type_id=doc.document_type.id if doc.document_type else 0,
            has_type=doc.document_type is not None,
            created=datetime.combine(doc.created, time.min).isoformat(),
            added=doc.added.isoformat(),
            asn=asn or 0,
            modified=doc.modified.isoformat(),
            path=doc.storage_path.name if doc.storage_path else "",
            path_id=doc.storage_path.id if doc.storage_path else 0,
            has_path=doc.storage_path is not None,
            notes=notes or "",
            num_notes=len(notes),
            custom_fields=custom_fields or "",
            custom_field_count=len(doc.custom_fields.all()),
            has_custom_fields=len(custom_fields) > 0,
            custom_fields_id=custom_fields_ids if custom_fields_ids else "",
            owner=doc.owner.username if doc.owner else "",
            owner_id=doc.owner.id if doc.owner else 0,
            has_owner=doc.owner is not None,
            viewer_id=viewer_ids if viewer_ids else "",
            checksum=doc.checksum or "",
            page_count=doc.page_count or 0,
            original_filename=doc.original_filename,
            is_shared=len(viewer_ids) > 0,
        ),
    )
    added_words = set()
    for word in tokenize_for_autocomplete(doc.content):
        if word not in added_words:
            added_words.add(word)
            writer.add_document(
                tantivy.Document(
                    id=doc.pk,
                    autocomplete_word=word,
                    owner_id=doc.owner.id if doc.owner else 0,
                    viewer_id=viewer_ids if viewer_ids else "",
                ),
            )
    logger.debug(f"Index updated for document {doc.pk}.")


def remove_document(writer: tantivy.IndexWriter, doc: Document) -> None:
    remove_document_by_id(writer, doc.pk)


def remove_document_by_id(writer: tantivy.IndexWriter, doc_id) -> None:
    writer.delete_documents("id", doc_id)


def add_or_update_document(document: Document) -> None:
    with open_index_writer() as writer:
        update_document(writer, document)


def add_or_update_documents(documents: list[Document], batchsize=100) -> None:
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
    def __init__(self, id, score, rank=None):
        self.id = id
        self.score = score
        self.rank = rank

    def __getitem__(self, key):
        if key == "id":
            return self.id
        raise KeyError(key)

    def highlights(self, *args, **kwargs):
        return None

    def __repr__(self):
        return f"Hit(id={self.id}, score={self.score}, rank={self.rank})"


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

    def __init__(
        self,
        searcher: tantivy.Searcher,
        query_params,
        page_size,
        filter_queryset: QuerySet,
    ) -> None:
        self.searcher = searcher
        self.query_params = query_params
        self.page_size = page_size
        self.saved_results = dict()
        self.first_score = None
        self.filter_queryset = filter_queryset
        self.suggested_correction = None
        self._manual_hits_cache: list | None = None

    def __len__(self) -> int:
        if self._manual_sort_requested():
            manual_hits = self._manual_hits()
            return len(manual_hits)

        page = self[0:1]
        return len(page)

    def _manual_sort_requested(self):
        ordering = self.query_params.get("ordering", "")
        return ordering.lstrip("-").startswith("custom_field_")

    def _manual_hits(self):
        if self._manual_hits_cache is None:
            q, mask, suggested_correction = self._get_query()
            self.suggested_correction = suggested_correction

            results = self.searcher.search(
                q,
                mask=mask,
                filter=MappedDocIdSet(self.filter_queryset, self.searcher.ixreader),
                limit=None,
            )
            results.fragmenter = highlight.ContextFragmenter(surround=50)
            results.formatter = HtmlFormatter(tagname="span", between=" ... ")

            if not self.first_score and len(results) > 0:
                self.first_score = results[0].score

            if self.first_score:
                results.top_n = [
                    (
                        (hit[0] / self.first_score) if self.first_score else None,
                        hit[1],
                    )
                    for hit in results.top_n
                ]

            hits_by_id = {hit["id"]: hit for hit in results}
            matching_ids = list(hits_by_id.keys())

            ordered_ids = list(
                self.filter_queryset.filter(id__in=matching_ids).values_list(
                    "id",
                    flat=True,
                ),
            )
            ordered_ids = list(dict.fromkeys(ordered_ids))

            self._manual_hits_cache = [
                hits_by_id[_id] for _id in ordered_ids if _id in hits_by_id
            ]
        return self._manual_hits_cache

    def __getitem__(self, item):
        print("DelayedQuery: __getitem__ begin")
        import time

        t0 = time.time()
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

        q, mask, suggested_correction = self._get_query()
        self.suggested_correction = suggested_correction
        sortedby, reverse = self._get_query_sortedby()
        pagenum = math.floor(item.start / self.page_size) + 1
        print(f"pagenum: {pagenum}")
        pagelen = self.page_size
        offset = (pagenum - 1) * pagelen
        print(f"offset: {offset}")
        t1 = time.time()
        search_result = self.searcher.search(
            q,
            limit=pagelen,
            offset=(pagenum - 1) * pagelen,
        ).hits
        print(f"query and tantivy search took {time.time() - t1:.3f} seconds")
        results = list()
        print(2)
        # if self.filter_queryset:
        if self.filter_queryset is not None:
            print(3)
            # print(self.filter_queryset)
            # print(self.filter_queryset.query)
            t1 = time.time()
            allowed_ids = set(self.filter_queryset.values_list("id", flat=True))
            print(f"DB query took {time.time() - t1:.3f} seconds")
            print(3.1)
            print(search_result)
            hit_scores = defaultdict(float)  # doc_id -> max score
            for score, doc_addr in search_result:
                doc = self.searcher.doc(doc_addr)
                # print(q.explain(self.searcher, doc_addr).to_json())
                doc_id = doc["id"][0]
                # print(score)
                # print(doc_addr)
                # print(doc)
                # print(doc["id"])
                print(3.2)
                if doc_id in allowed_ids:
                    print(3.3)
                    # results.append({"id": doc["id"][0], "score": score})
                    hit_scores[doc_id] = max(hit_scores[doc_id], score)
                    # results.append(Hit(doc["id"][0], score))
                    print(3.4)
            results = [Hit(doc_id, score) for doc_id, score in hit_scores.items()]
            # results = [r for r in search_result if r.get("id") in allowed_ids]
            for idx, hit in enumerate(results, start=1):
                hit.rank = idx
            print(results)
            print(4)
            print(f"__getitem__ took {time.time() - t0:.3f} seconds")
        else:
            # TODO
            raise NotImplementedError
            results = [r for r in search_result]
        print("DelayedQuery: __getitem__ end")

        page = {
            # "results": Document.objects.filter(id__in=results),
            "results": results,
            # "total": search_result.total_hits,
            # "page_number": pagenum,
            # "page_size": self.page_size,
            # TODO remove mock data
            "pagenum": 1,
            "pagelen": 50,
            "total": 2,
        }
        return results
        # return search_result
        # search_result.hits
        # return search_result.hits

        # searcher.search(query, limit=offset+pagelen)[offset:offset+pagelen]

        page: ResultsPage = self.searcher.search_page(
            q,
            mask=mask,
            filter=MappedDocIdSet(self.filter_queryset, self.searcher),
            pagenum=math.floor(item.start / self.page_size) + 1,
            pagelen=self.page_size,
            sortedby=sortedby,
            reverse=reverse,
        )
        page.results.fragmenter = highlight.ContextFragmenter(surround=50)
        page.results.formatter = HtmlFormatter(tagname="span", between=" ... ")

        if not self.first_score and len(page.results) > 0 and sortedby is None:
            self.first_score = page.results[0].score

        page.results.top_n = [
            (
                (hit[0] / self.first_score) if self.first_score else None,
                hit[1],
            )
            for hit in page.results.top_n
        ]

        self.saved_results[item.start] = page

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


class DelayedFullTextQuery(DelayedQuery):
    def _get_query(self) -> tuple:
        q_str = self.query_params["query"]
        q_str = rewrite_natural_date_keywords(q_str)
        if len(q_str) <= 3 or "NOT " in q_str:
            fuzzy_search = False
        else:
            fuzzy_search = True
        # qp = MultifieldParser(
        #     [
        #         "content",
        #         "title",
        #         "correspondent",
        #         "tag",
        #         "type",
        #         "notes",
        #         "custom_fields",
        #     ],
        #     # self.searcher.ixreader.schema,
        #     get_schema()
        # )
        # qp.add_plugin(
        #     DateParserPlugin(
        #         basedate=django_timezone.now(),
        #         dateparser=LocalDateParser(),
        #     ),
        # )

        # TODO: date parsing plugin like whoosh
        with open_index() as index:
            # q = tantivy.Query.regex_query(schema=get_schema(), field_name="content", regex_pattern=f"{term}.*")
            # q = tantivy.Query.fuzzy_term_query
            q = index.parse_query(
                # fuzzy field: prefix (prefix must match) bool, distance int, transpose_cost_one (2 letters inverted cost 1 instead of 2): bool
                q_str,
                [
                    "content",
                    "title",
                    "correspondent",
                    "tag",
                    "type",
                    "notes",
                    "custom_fields",
                ],
                field_boosts={"title": 5.0, "content": 0.5},
                # "tag": 2.0, "correspondent": 2.0, "type": 2.0, "custom_fields": 2.0
            )
            if fuzzy_search:
                # An exact match should outweigh a fuzzy one
                fuzzy_q = index.parse_query(
                    # fuzzy field: prefix (prefix must match) bool, distance int, transpose_cost_one (2 letters inverted cost 1 instead of 2): bool
                    q_str,
                    [
                        "content",
                        "title",
                        "correspondent",
                        "tag",
                        "type",
                        "notes",
                        "custom_fields",
                    ],
                    field_boosts={"title": 5.0, "content": 0.5},
                    fuzzy_fields={
                        "content": (True, 1, True),
                        "title": (True, 1, True),
                        "correspondent": (True, 1, True),
                        "tag": (True, 1, True),
                        "type": (True, 1, True),
                        "notes": (True, 1, True),
                        "custom_fields": (True, 1, True),
                    },
                )
                # q = tantivy.Query.boolean_query(should=[q, fuzzy_q])
                q = tantivy.Query.boolean_query(
                    [
                        (tantivy.Occur.Should, q),
                        (tantivy.Occur.Should, fuzzy_q),
                    ],
                )
        suggested_correction = None
        # TODO: corrections not available in tantivy?
        # try:
        #     corrected = self.searcher.correct_query(q, q_str)
        #     if corrected.string != q_str:
        #         suggested_correction = corrected.string
        # except Exception as e:
        #     logger.info(
        #         "Error while correcting query %s: %s",
        #         f"{q_str!r}",
        #         e,
        #     )

        # return q, None, suggested_correction
        return q, None, suggested_correction


class DelayedMoreLikeThisQuery(DelayedQuery):
    def _get_query(self) -> tuple:
        more_like_doc_id = int(self.query_params["more_like_id"])
        content = Document.objects.get(id=more_like_doc_id).content

        docnum = self.searcher.document_number(id=more_like_doc_id)
        # TODO: not supported in tantivy?
        kts = self.searcher.key_terms_from_text(
            "content",
            content,
            numterms=20,
            model=classify.Bo1Model,
            normalize=False,
        )
        q = query.Or(
            [query.Term("content", word, boost=weight) for word, weight in kts],
        )
        mask: set = {docnum}

        return q, mask, None


# def get_permissions_queries(user: User | None = None) -> list:
#     queries = []

#     if user is None:
#         # If no user, include only docs with has_owner = False
#         field = schema.get_field("has_owner")
#         queries.append(TermQuery(Term(field, "false")))
#     elif getattr(user, "is_superuser", False):
#         # Superuser sees all docs → no filter
#         return []
#     else:
#         # Regular user: can see docs they own or that are shared with them
#         owner_field = schema.get_field("owner_id")
#         viewer_field = schema.get_field("viewer_id")

#         owner_query = TermQuery(Term(owner_field, str(user.id)))
#         viewer_query = TermQuery(Term(viewer_field, str(user.id)))

#         # Combine with OR
#         queries.append(BooleanQuery.or_([owner_query, viewer_query]))

#     return queries


def autocomplete(
    ix: FileIndex,
    term: str,
    limit: int = 10,
    user: User | None = None,
) -> list:
    """
    Mimics whoosh.reading.IndexReader.most_distinctive_terms with permissions
    and without scoring
    """
    terms = []
    with open_index() as index:
        searcher = index.searcher()
        # TODO: filter by user permissions

        exact_q = tantivy.Query.term_query(
            schema=index.schema,
            field_name="autocomplete_word",
            field_value=term,
        )
        prefix_q = tantivy.Query.regex_query(
            schema=index.schema,
            field_name="autocomplete_word",
            regex_pattern=f"{term}.*",
        )
        q = tantivy.Query.boolean_query(
            [
                (tantivy.Occur.Should, exact_q),
                (tantivy.Occur.Should, prefix_q),
            ],
        )
        # searcher = index.searcher()
        search_result = searcher.search(q, limit=limit).hits
        exact_matches = []
        prefix_matches = []
        seen = set()
        # print(search_result)
        for _, doc_addr in search_result:
            doc = searcher.doc(doc_addr)
            # print(q.explain(self.searcher, doc_addr).to_json())
            word = doc["autocomplete_word"][0]
            if word in seen:
                continue
            seen.add(word)
            if word == term.lower():
                exact_matches.append(word)
            else:
                prefix_matches.append(word)
        terms = exact_matches + prefix_matches
        # print(q.explain(searcher, doc_addr).to_json())
        # q = index.parse_query(
        #     term,
        #     [
        #         "content",
        #         "title",
        #         "correspondent",
        #         "tag",
        #         "type",
        #         "notes",
        #         "custom_fields",
        #     ],

        # )

        # mlt_query = tantivy.Query.more_like_this_query()
        # mlt_query.add_field("content")
        # mlt_query.like_text(user_input)

        # results = searcher.search(mlt_query, limit=10).hits
        # print(hits)

        # prefix_query = tantivy.Query.more_like_this_query()
        # prefix_query = tantivy.PrefixQuery(field, user_input.lower())
        # results = searcher.search(prefix_query, limit=10).hits
    # terms = []
    # TODO : support autocomplete in tantivy
    return terms[:limit]


def get_permissions_criterias(user: User | None = None) -> list:
    user_criterias = [query.Term("has_owner", text=False)]
    if user is not None:
        if user.is_superuser:  # superusers see all docs
            user_criterias = []
        else:
            user_criterias.append(query.Term("owner_id", user.id))
            user_criterias.append(
                query.Term("viewer_id", str(user.id)),
            )
    return user_criterias


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
                end = datetime.combine(today, time.max, tzinfo=tz)

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

        # Convert to UTC and format
        start_str = start.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")
        end_str = end.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{field}:[{start_str} TO {end_str}]"

    return re.sub(pattern, repl, query_string, flags=re.IGNORECASE)
