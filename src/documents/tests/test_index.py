from datetime import datetime
from unittest import mock

from django.contrib.auth.models import User
from django.test import SimpleTestCase
from django.test import TestCase
from django.test import override_settings
from django.utils.timezone import get_current_timezone
from django.utils.timezone import timezone

from documents import index
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

        ix = index.open_index()

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
