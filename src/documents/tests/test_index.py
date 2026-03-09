from datetime import datetime
from unittest import mock

from django.contrib.auth.models import User
from django.test import SimpleTestCase
from django.test import TestCase
from django.test import override_settings
from django.utils.timezone import get_current_timezone
from django.utils.timezone import timezone

from documents import index
from documents.index import normalize_query
from documents.index import preprocess_query
from documents.models import Document
from documents.tests.utils import DirectoriesMixin


class TestAutoComplete(DirectoriesMixin, TestCase):
    def test_auto_complete(self):
        doc1 = Document.objects.create(
            title="doc1",
            checksum="A",
            content="test test2 test3",
        )
        doc2 = Document.objects.create(title="doc2", checksum="B", content="test test2")
        doc3 = Document.objects.create(title="doc3", checksum="C", content="test2")

        index.add_or_update_document(doc1)
        index.add_or_update_document(doc2)
        index.add_or_update_document(doc3)

        with index.open_index() as ix:
            self.assertListEqual(
                index.autocomplete(ix, "tes"),
                ["test2", "test", "test3"],
            )
            self.assertListEqual(
                index.autocomplete(ix, "tes", limit=3),
                ["test2", "test", "test3"],
            )
            self.assertListEqual(index.autocomplete(ix, "tes", limit=1), ["test2"])
            self.assertListEqual(index.autocomplete(ix, "tes", limit=0), [])

    def test_archive_serial_number_ranging(self):
        """
        GIVEN:
            - Document with an archive serial number above schema allowed size
        WHEN:
            - Document is provided to the index
        THEN:
            - Error is logged
            - Document ASN is reset to 0 for the index
        """
        doc1 = Document.objects.create(
            title="doc1",
            checksum="A",
            content="test test2 test3",
            # yes, this is allowed, unless full_clean is run
            # DRF does call the validators, this test won't
            archive_serial_number=Document.ARCHIVE_SERIAL_NUMBER_MAX + 1,
        )
        with self.assertLogs("paperless.index", level="ERROR") as cm:
            with mock.patch(
                "documents.index.AsyncWriter.add_document",
            ) as mocked_add_doc:
                index.add_or_update_document(doc1)

                mocked_add_doc.assert_called_once()
                indexed_doc = mocked_add_doc.call_args[0][0]

                self.assertEqual(indexed_doc["asn"], [0])

                error_str = cm.output[0]
                expected_str = "ERROR:paperless.index:Not indexing Archive Serial Number 4294967296 of document 1"
                self.assertIn(expected_str, error_str)

    def test_archive_serial_number_is_none(self):
        """
        GIVEN:
            - Document with no archive serial number
        WHEN:
            - Document is provided to the index
        THEN:
            - ASN is set to 0 in the index
        """
        doc1 = Document.objects.create(
            title="doc1",
            checksum="A",
            content="test test2 test3",
        )
        with mock.patch(
            "documents.index.AsyncWriter.add_document",
        ) as mocked_add_doc:
            index.add_or_update_document(doc1)

            mocked_add_doc.assert_called_once()
            indexed_doc = mocked_add_doc.call_args[0][0]

            self.assertEqual(indexed_doc["asn"], [0])

    @override_settings(TIME_ZONE="Pacific/Auckland")
    def test_added_today_respects_local_timezone_boundary(self):
        tz = get_current_timezone()
        fixed_now = datetime(2025, 7, 20, 15, 0, 0, tzinfo=tz)

        # Fake a time near the local boundary (1 AM NZT = 13:00 UTC on previous UTC day)
        local_dt = datetime(2025, 7, 20, 1, 0, 0).replace(tzinfo=tz)
        utc_dt = local_dt.astimezone(timezone.utc)

        doc = Document.objects.create(
            title="Time zone",
            content="Testing added:today",
            checksum="edgecase123",
            added=utc_dt,
        )

        with index.open_index_writer() as writer:
            index.update_document(writer, doc)

        superuser = User.objects.create_superuser(username="testuser")
        self.client.force_login(superuser)

        with mock.patch("documents.index.now", return_value=fixed_now):
            response = self.client.get("/api/documents/?query=added:today")
            results = response.json()["results"]
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["id"], doc.id)

            response = self.client.get("/api/documents/?query=added:yesterday")
            results = response.json()["results"]
            self.assertEqual(len(results), 0)

    def test_update_doc_replaces_previous_one(self):
        """
        GIVEN:
            - Any document
        WHEN:
            - Document is added to the index
        THEN:
            - Any document with the same ID is removed from the index
        """
        doc1 = Document.objects.create(
            id=123,
            title="doc1",
            checksum="A",
            content="test test2 test3",
        )
        with mock.patch(
            "documents.index.remove_document_by_id",
        ) as mocked_remove_doc:
            index.add_or_update_document(doc1)

            mocked_remove_doc.assert_called_once()
            doc_id = mocked_remove_doc.call_args[0][1]
            self.assertEqual(doc_id, 123)


@override_settings(TIME_ZONE="UTC")
class TestRewriteNaturalDateKeywords(SimpleTestCase):
    """
    Unit tests for rewrite_natural_date_keywords function.
    """

    def _rewrite_with_now(self, query: str, now_dt: datetime) -> str:
        with mock.patch("documents.index.now", return_value=now_dt):
            return index.rewrite_natural_date_keywords(query)

    def _assert_rewrite_contains(
        self,
        query: str,
        now_dt: datetime,
        *expected_fragments: str,
    ) -> str:
        result = self._rewrite_with_now(query, now_dt)
        for fragment in expected_fragments:
            self.assertIn(fragment, result)
        return result

    def test_range_keywords(self):
        """
        Test various different range keywords
        """
        cases = [
            (
                "added:today",
                datetime(2025, 7, 20, 15, 30, 45, tzinfo=timezone.utc),
                (
                    "added:[2025-07-20T00:00:00+00:00 TO 2025-07-20T23:59:59.999999+00:00]",
                ),
            ),
            (
                "added:yesterday",
                datetime(2025, 7, 20, 15, 30, 45, tzinfo=timezone.utc),
                (
                    "added:[2025-07-19T00:00:00+00:00 TO 2025-07-19T23:59:59.999999+00:00]",
                ),
            ),
            (
                "added:this month",
                datetime(2025, 7, 15, 12, 0, 0, tzinfo=timezone.utc),
                (
                    "added:[2025-07-01T00:00:00+00:00 TO 2025-07-31T23:59:59.999999+00:00]",
                ),
            ),
            (
                "added:previous month",
                datetime(2025, 7, 15, 12, 0, 0, tzinfo=timezone.utc),
                (
                    "added:[2025-06-01T00:00:00+00:00 TO 2025-06-30T23:59:59.999999+00:00]",
                ),
            ),
            (
                "added:this year",
                datetime(2025, 7, 15, 12, 0, 0, tzinfo=timezone.utc),
                (
                    "added:[2025-01-01T00:00:00+00:00 TO 2025-12-31T23:59:59.999999+00:00]",
                ),
            ),
            (
                "added:previous year",
                datetime(2025, 7, 15, 12, 0, 0, tzinfo=timezone.utc),
                (
                    "added:[2024-01-01T00:00:00+00:00 TO 2024-12-31T23:59:59.999999+00:00]",
                ),
            ),
            # Previous quarter from July 15, 2025 is April-June.
            (
                "added:previous quarter",
                datetime(2025, 7, 15, 12, 0, 0, tzinfo=timezone.utc),
                (
                    "added:[2025-04-01T00:00:00+00:00 TO 2025-06-30T23:59:59.999999+00:00]",
                ),
            ),
            # July 20, 2025 is a Sunday (weekday 6) so previous week is July 7-13.
            (
                "added:previous week",
                datetime(2025, 7, 20, 12, 0, 0, tzinfo=timezone.utc),
                (
                    "added:[2025-07-07T00:00:00+00:00 TO 2025-07-13T23:59:59.999999+00:00]",
                ),
            ),
        ]

        for query, now_dt, fragments in cases:
            with self.subTest(query=query):
                self._assert_rewrite_contains(query, now_dt, *fragments)

    def test_additional_fields(self):
        fixed_now = datetime(2025, 7, 20, 15, 30, 45, tzinfo=timezone.utc)
        # created
        self._assert_rewrite_contains("created:today", fixed_now, "created:[2025-07-20")
        # modified
        self._assert_rewrite_contains(
            "modified:today",
            fixed_now,
            "modified:[2025-07-20",
        )

    def test_basic_syntax_variants(self):
        """
        Test that quoting, casing, and multi-clause queries are parsed.
        """
        fixed_now = datetime(2025, 7, 20, 15, 30, 45, tzinfo=timezone.utc)

        # quoted keywords
        result1 = self._rewrite_with_now('added:"today"', fixed_now)
        result2 = self._rewrite_with_now("added:'today'", fixed_now)
        self.assertIn("added:[2025-07-20", result1)
        self.assertIn("added:[2025-07-20", result2)

        # case insensitivity
        for query in ("added:TODAY", "added:Today", "added:ToDaY"):
            with self.subTest(case_variant=query):
                self._assert_rewrite_contains(query, fixed_now, "added:[2025-07-20")

        # multiple clauses
        result = self._rewrite_with_now("added:today created:yesterday", fixed_now)
        self.assertIn("added:[2025-07-20", result)
        self.assertIn("created:[2025-07-19", result)

    def test_no_match(self):
        """
        Test that queries without keywords are unchanged.
        """
        query = "title:test content:example"
        result = index.rewrite_natural_date_keywords(query)
        self.assertEqual(query, result)

    @override_settings(TIME_ZONE="Pacific/Auckland")
    def test_timezone_awareness(self):
        """
        Test timezone conversion.
        """
        # July 20, 2025 1:00 AM NZST = July 19, 2025 13:00 UTC
        fixed_now = datetime(2025, 7, 20, 1, 0, 0, tzinfo=get_current_timezone())
        result = self._rewrite_with_now("added:today", fixed_now)
        # Should convert to UTC properly
        # NZST is UTC+12, so "today" July 20 local = July 19 12:00 UTC to July 20 11:59:59 UTC
        self.assertIn(
            "added:[2025-07-19T12:00:00+00:00 TO 2025-07-20T11:59:59.999999+00:00]",
            result,
        )


class TestIndexResilience(DirectoriesMixin, SimpleTestCase):
    def test_transient_error_retries_then_succeeds(self):
        """
        GIVEN:
            - Index directory exists
        WHEN:
            - open_index is called
            - tantivy.Index raises OSError once, then succeeds
        THEN:
            - Index is opened successfully on retry
            - Index is not recreated
        """
        expected_index = mock.MagicMock()

        with (
            self.assertLogs("paperless.index", level="WARNING") as cm,
            mock.patch(
                "documents.index.tantivy.Index",
                side_effect=[OSError("busy"), expected_index],
            ) as mock_index,
            mock.patch("documents.index.recreate_index_dir") as mock_recreate,
        ):
            with index.open_index() as ix:
                self.assertIs(ix, expected_index)

        self.assertEqual(mock_index.call_count, 2)
        # recreate_index_dir is only called at the start if path doesn't exist
        # but not as error recovery since retry succeeded
        mock_recreate.assert_not_called()
        self.assertIn("Error opening index (attempt 1/3)", cm.output[0])

    def test_transient_errors_exhaust_retries_and_recreate(self):
        """
        GIVEN:
            - Index directory exists
        WHEN:
            - open_index is called
            - tantivy.Index always raises OSError
        THEN:
            - Index is recreated after retries are exhausted
        """
        recreated_index = mock.MagicMock()

        with (
            self.assertLogs("paperless.index", level="ERROR") as cm,
            mock.patch(
                "documents.index.tantivy.Index",
                side_effect=[
                    OSError("busy"),
                    OSError("busy"),
                    OSError("busy"),
                    recreated_index,
                ],
            ) as mock_index,
            mock.patch("documents.index.recreate_index_dir") as mock_recreate,
        ):
            with index.open_index() as ix:
                self.assertIs(ix, recreated_index)

        # 3 failed attempts + 1 final call after recreate
        self.assertEqual(mock_index.call_count, 4)
        mock_recreate.assert_called_once()
        self.assertIn("Failed to open index after 3 attempts", cm.output[-1])

    def test_schema_error_recreates_immediately(self):
        """
        GIVEN:
            - Index directory exists
        WHEN:
            - open_index is called
            - tantivy.Index raises ValueError (schema mismatch)
        THEN:
            - Index is recreated immediately without retries
        """
        recreated_index = mock.MagicMock()

        with (
            self.assertLogs("paperless.index", level="WARNING") as cm,
            mock.patch(
                "documents.index.tantivy.Index",
                side_effect=[ValueError("schema changed"), recreated_index],
            ) as mock_index,
            mock.patch("documents.index.recreate_index_dir") as mock_recreate,
        ):
            with index.open_index() as ix:
                self.assertIs(ix, recreated_index)

        # 1 failed attempt + 1 call after recreate = 2
        self.assertEqual(mock_index.call_count, 2)
        mock_recreate.assert_called_once()
        self.assertIn("Recreating index due to schema error", cm.output[0])


class TestNormalizeQuery(SimpleTestCase):
    """normalize_query: comma-separated parts → AND joins."""

    def test_single_term_no_comma(self):
        self.assertEqual(normalize_query("invoice"), "(invoice)")

    def test_comma_separated_free_text(self):
        self.assertEqual(normalize_query("bank, statement"), "(bank statement)")

    def test_comma_with_field_filter(self):
        self.assertEqual(
            normalize_query("bank statement, added:today"),
            "(bank statement) AND added:today",
        )

    def test_multiple_field_filters(self):
        self.assertEqual(
            normalize_query("added:today, created:yesterday"),
            "added:today AND created:yesterday",
        )

    def test_mixed_free_text_and_filters(self):
        self.assertEqual(
            normalize_query("invoice, tag:important, bank"),
            "(invoice) AND tag:important AND (bank)",
        )

    def test_empty_string(self):
        self.assertEqual(normalize_query(""), "")

    def test_whitespace_parts_ignored(self):
        self.assertEqual(normalize_query("test, , ,hello"), "(test hello)")


class TestTantivyQueryParsing(DirectoriesMixin, SimpleTestCase):
    """
    Integration tests: verify that Tantivy's parse_query_lenient correctly
    interprets boolean operators, phrases, ranges, field queries, and boosting.

    Opens a real (empty) Tantivy index and inspects generated query structure.
    """

    default_fields = [
        "content",
        "title",
        "correspondent",
        "tag",
        "type",
        "notes",
        "custom_fields",
    ]

    def _parse(self, query_str, *, conjunction_by_default=True):
        with index.open_index() as ix:
            return ix.parse_query_lenient(
                query_str,
                self.default_fields,
                conjunction_by_default=conjunction_by_default,
            )

    def _parse_ok(self, query_str, **kwargs):
        """Parse and assert no errors. Returns query string repr."""
        query, errors = self._parse(query_str, **kwargs)
        self.assertEqual(
            errors,
            [],
            f"Unexpected parse errors for '{query_str}': {errors}",
        )
        return str(query)

    def _parse_lenient(self, query_str, **kwargs):
        """Parse and return query string repr, ignoring errors."""
        query, _errors = self._parse(query_str, **kwargs)
        return str(query)

    # --- Structural checks (not covered by integration tests) ---

    def test_conjunction_by_default(self):
        and_result = self._parse_ok("bank AND statement")
        default_result = self._parse_ok("bank statement")
        self.assertEqual(and_result, default_result)

    def test_or_produces_different_query_than_and(self):
        and_result = self._parse_ok("bank AND statement")
        or_result = self._parse_ok("bank OR statement")
        self.assertNotEqual(and_result, or_result)

    def test_phrase_different_from_and(self):
        phrase = self._parse_ok('"bank statement"')
        and_query = self._parse_ok("bank AND statement")
        self.assertNotEqual(phrase, and_query)

    def test_field_query_produces_term_query(self):
        """A field-specific query should target a single field, not expand to all defaults."""
        result = self._parse_ok("title:invoice")
        self.assertIn("termquery", result.lower())

    def test_field_query_different_from_free_text(self):
        """title:invoice should differ from just 'invoice' (which expands to all fields)."""
        field_result = self._parse_ok("title:invoice")
        free_result = self._parse_ok("invoice")
        self.assertNotEqual(field_result, free_result)

    # --- Range and boost parsing (no indexed docs in integration tests) ---

    def test_inclusive_range(self):
        result = self._parse_lenient("title:[a TO z]")
        self.assertIn("rangequery", result.lower())

    def test_date_range(self):
        result = self._parse_lenient(
            "added:[2025-01-01T00:00:00+00:00 TO 2025-12-31T23:59:59+00:00]",
        )
        self.assertIn("2025-01-01", result)
        self.assertIn("2025-12-31", result)
        self.assertIn("date", result.lower())

    def test_boost_term(self):
        result = self._parse_ok("bank^2.0")
        self.assertIn("bank", result.lower())
        self.assertIn("2", result)

    def test_boost_phrase(self):
        result = self._parse_ok('"bank statement"^3.0')
        self.assertIn("bank", result.lower())

    # --- Edge cases (malformed input) ---

    def test_empty_query(self):
        result = self._parse_lenient("")
        self.assertIsNotNone(result)

    def test_special_chars_only(self):
        result = self._parse_lenient("!!!")
        self.assertIsNotNone(result)

    def test_unbalanced_quotes_lenient(self):
        result = self._parse_lenient('"bank statement')
        self.assertIn("bank", result.lower())

    def test_unknown_field_lenient(self):
        result = self._parse_lenient("nonexistent_field:test")
        self.assertIsNotNone(result)


class TestPreprocessQuery(SimpleTestCase):
    """
    Unit tests for preprocess_query: the single entry point that chains
    all rewrite steps before Tantivy parsing.
    """

    def test_plain_words_unchanged(self):
        query = preprocess_query("bank statement")
        self.assertIn("bank", query)
        self.assertIn("statement", query)

    def test_comma_joins_with_and(self):
        query = preprocess_query("invoice, tag:important")
        self.assertIn("AND", query)
        self.assertIn("invoice", query)
        self.assertIn("tag:important", query)

    def test_explicit_or_preserved(self):
        query = preprocess_query("bank OR statement")
        self.assertIn("OR", query)

    def test_explicit_not_preserved(self):
        query = preprocess_query("bank NOT statement")
        self.assertIn("NOT", query)

    def test_phrases_preserved(self):
        query = preprocess_query('"bank statement"')
        self.assertIn('"bank statement"', query)

    def test_mixed_comma_and_boolean(self):
        query = preprocess_query("bank OR credit, tag:finance")
        self.assertIn("bank", query)
        self.assertIn("credit", query)
        self.assertIn("tag:finance", query)
        self.assertIn("AND", query)


@override_settings(TIME_ZONE="UTC")
class TestPreprocessQueryDates(SimpleTestCase):
    """
    Tests that preprocess_query correctly rewrites date expressions
    (both natural keywords and Whoosh-style expressions) into Tantivy ISO ranges.
    """

    def _preprocess_with_now(self, query: str, now_dt: datetime) -> str:
        with mock.patch("documents.index.now", return_value=now_dt):
            return preprocess_query(query)

    def test_today_keyword(self):
        result = self._preprocess_with_now(
            "added:today",
            datetime(2025, 7, 20, 15, 0, 0, tzinfo=timezone.utc),
        )
        self.assertIn("added:[2025-07-20T00:00:00+00:00 TO 2025-07-20T23:59:59", result)

    def test_yesterday_keyword(self):
        result = self._preprocess_with_now(
            "added:yesterday",
            datetime(2025, 7, 20, 15, 0, 0, tzinfo=timezone.utc),
        )
        self.assertIn("added:[2025-07-19T00:00:00+00:00 TO 2025-07-19T23:59:59", result)

    def test_date_keyword_with_free_text(self):
        result = self._preprocess_with_now(
            "invoice, added:today",
            datetime(2025, 7, 20, 15, 0, 0, tzinfo=timezone.utc),
        )
        self.assertIn("invoice", result)
        self.assertIn("added:[2025-07-20", result)
        self.assertIn("AND", result)


class TestTantivySearchIntegration(DirectoriesMixin, TestCase):
    """
    Integration tests that index real documents and verify that Tantivy
    query syntax produces correct search results.

    Covers: AND, OR, NOT, phrases, phrase slop (~N), field queries,
    boosting, parentheses grouping.
    """

    default_fields = [
        "content",
        "title",
        "correspondent",
        "tag",
        "type",
        "notes",
        "custom_fields",
    ]

    def setUp(self):
        super().setUp()
        self.d1 = Document.objects.create(
            title="invoice from bank",
            content="the quick brown fox jumps over the lazy dog",
            checksum="int1",
        )
        self.d2 = Document.objects.create(
            title="bank statement august",
            content="monthly statement for credit card payment",
            checksum="int2",
        )
        self.d3 = Document.objects.create(
            title="tax return",
            content="annual tax return filed with government agency",
            checksum="int3",
        )
        self.d4 = Document.objects.create(
            title="receipt from shop",
            content="bought groceries and paid with credit card at the shop",
            checksum="int4",
        )
        for doc in [self.d1, self.d2, self.d3, self.d4]:
            index.add_or_update_document(doc)

    def _search(self, query_str, *, conjunction_by_default=True):
        """Search the index and return matching document PKs."""
        q_str = preprocess_query(query_str)
        with index.open_index() as ix:
            query, _errors = ix.parse_query_lenient(
                q_str,
                self.default_fields,
                conjunction_by_default=conjunction_by_default,
            )
            searcher = ix.searcher()
            results = searcher.search(query, limit=100)
            return {
                searcher.doc(doc_addr)["id"][0] for _score, doc_addr in results.hits
            }

    # --- Single term ---

    def test_single_term(self):
        ids = self._search("bank")
        self.assertIn(self.d1.pk, ids)
        self.assertIn(self.d2.pk, ids)
        self.assertNotIn(self.d3.pk, ids)

    # --- AND (implicit via conjunction_by_default) ---

    def test_implicit_and(self):
        """'bank statement' with conjunction_by_default → both terms required."""
        ids = self._search("bank statement")
        self.assertIn(self.d2.pk, ids)  # has both in title
        # d1 has 'bank' in title but not 'statement' in content or title
        self.assertNotIn(self.d3.pk, ids)

    def test_explicit_and(self):
        ids = self._search("credit AND card")
        self.assertIn(self.d2.pk, ids)
        self.assertIn(self.d4.pk, ids)
        self.assertNotIn(self.d1.pk, ids)

    # --- OR ---

    def test_explicit_or(self):
        ids = self._search("tax OR bank")
        self.assertIn(self.d1.pk, ids)  # bank
        self.assertIn(self.d2.pk, ids)  # bank
        self.assertIn(self.d3.pk, ids)  # tax
        self.assertNotIn(self.d4.pk, ids)

    # --- NOT / exclusion ---

    def test_minus_exclusion(self):
        ids = self._search("credit -shop")
        self.assertIn(self.d2.pk, ids)  # credit, no shop
        self.assertNotIn(self.d4.pk, ids)  # credit + shop

    # --- Phrases ---

    def test_phrase_match(self):
        """Exact phrase 'brown fox' should match d1 only."""
        ids = self._search('"brown fox"')
        self.assertIn(self.d1.pk, ids)
        self.assertEqual(len(ids), 1)

    def test_phrase_no_match(self):
        """'fox brown' (reversed) should not match as a phrase."""
        ids = self._search('"fox brown"')
        self.assertNotIn(self.d1.pk, ids)

    # --- Phrase slop (word distance) ---

    def test_phrase_slop(self):
        """'quick fox'~2 should match d1 ('quick brown fox' — distance 1)."""
        ids = self._search('"quick fox"~2')
        self.assertIn(self.d1.pk, ids)

    def test_phrase_slop_too_small(self):
        """'quick lazy'~1 should NOT match d1 (distance is 4)."""
        ids = self._search('"quick lazy"~1')
        self.assertNotIn(self.d1.pk, ids)

    # --- Field-specific queries ---

    def test_field_title(self):
        ids = self._search("title:invoice")
        self.assertIn(self.d1.pk, ids)
        self.assertNotIn(self.d2.pk, ids)

    def test_field_content(self):
        ids = self._search("content:groceries")
        self.assertIn(self.d4.pk, ids)
        self.assertEqual(len(ids), 1)

    # --- Grouping with parentheses ---

    def test_parentheses_grouping(self):
        """(tax OR bank) -statement → tax return + invoice from bank."""
        ids = self._search("(tax OR bank) -statement")
        self.assertIn(self.d1.pk, ids)  # bank, no statement
        self.assertIn(self.d3.pk, ids)  # tax, no statement
        self.assertNotIn(self.d2.pk, ids)  # bank + statement

    # --- Boosting ---

    def test_boost_affects_ranking(self):
        """'bank^10 OR tax' — bank docs should rank higher than tax docs."""
        with index.open_index() as ix:
            # Use parse_query_lenient directly to avoid preprocess_query wrapping
            query, _ = ix.parse_query_lenient(
                "bank^10 OR tax",
                self.default_fields,
                conjunction_by_default=False,
            )
            searcher = ix.searcher()
            results = searcher.search(query, limit=100)
            ranked_ids = [searcher.doc(addr)["id"][0] for _score, addr in results.hits]
        # bank docs (d1, d2) should appear before tax doc (d3)
        bank_positions = [ranked_ids.index(pk) for pk in [self.d1.pk, self.d2.pk]]
        tax_position = ranked_ids.index(self.d3.pk)
        self.assertTrue(all(bp < tax_position for bp in bank_positions))

    # --- Plus (required) ---

    def test_plus_required(self):
        ids = self._search("+credit +shop")
        self.assertIn(self.d4.pk, ids)
        self.assertNotIn(self.d2.pk, ids)  # credit but no shop

    # --- Complex combinations ---

    def test_complex_query(self):
        """title:bank AND (credit OR groceries) → d2 (bank+credit), not d4 (no bank in title)."""
        ids = self._search("title:bank AND (credit OR groceries)")
        self.assertIn(self.d2.pk, ids)
        self.assertNotIn(self.d4.pk, ids)  # no 'bank' in title
