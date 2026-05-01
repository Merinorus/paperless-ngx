import json
import logging
import os
import tempfile
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile
from zipfile import is_zipfile

import ijson
from django.conf import settings
from django.contrib.auth.models import Permission
from django.contrib.auth.models import User
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import FieldDoesNotExist
from django.core.management import call_command
from django.core.management.base import CommandError
from django.core.management.color import no_style
from django.core.serializers.base import DeserializationError
from django.core.serializers.python import Deserializer as PythonDeserializer
from django.db import DEFAULT_DB_ALIAS
from django.db import IntegrityError
from django.db import connections
from django.db import router
from django.db import transaction
from django.db.models.signals import m2m_changed
from django.db.models.signals import post_save
from filelock import FileLock

from documents.file_handling import create_source_path_directory
from documents.management.commands.base import PaperlessCommand
from documents.management.commands.mixins import CryptMixin
from documents.models import Correspondent
from documents.models import CustomField
from documents.models import CustomFieldInstance
from documents.models import Document
from documents.models import DocumentType
from documents.models import Note
from documents.models import ShareLinkBundle
from documents.models import Tag
from documents.settings import EXPORTER_ARCHIVE_NAME
from documents.settings import EXPORTER_CRYPTO_SETTINGS_NAME
from documents.settings import EXPORTER_FILE_NAME
from documents.settings import EXPORTER_SHARE_LINK_BUNDLE_NAME
from documents.settings import EXPORTER_THUMBNAIL_NAME
from documents.signals.handlers import check_paths_and_prune_custom_fields
from documents.signals.handlers import update_filename_and_move_files
from documents.signals.handlers import update_llm_suggestions_cache
from documents.utils import copy_file_with_basic_stats
from paperless import version

if settings.AUDIT_LOG_ENABLED:
    from auditlog.registry import auditlog


def iter_manifest_records(path: Path) -> Generator[dict, None, None]:
    """Yield records one at a time from a manifest JSON array via ijson."""
    try:
        with path.open("rb") as f:
            yield from ijson.items(f, "item")
    except ijson.JSONError as e:
        raise CommandError(f"Failed to parse manifest file {path}: {e}") from e


def _iter_batches(iterable, size: int) -> Generator[list, None, None]:
    """Yield successive batches of *size* items from an iterable."""
    from itertools import islice

    it = iter(iterable)
    while batch := list(islice(it, size)):
        yield batch


@contextmanager
def disable_signal(sig, receiver, sender, *, weak: bool | None = None) -> Generator:
    try:
        sig.disconnect(receiver=receiver, sender=sender)
        yield
    finally:
        kwargs = {"weak": weak} if weak is not None else {}
        sig.connect(receiver=receiver, sender=sender, **kwargs)


class Command(CryptMixin, PaperlessCommand):
    help = (
        "Using a manifest.json file, load the data from there, and import the "
        "documents it refers to."
    )

    supports_progress_bar = True
    supports_multiprocessing = False

    def add_arguments(self, parser) -> None:
        super().add_arguments(parser)
        parser.add_argument("source")

        parser.add_argument(
            "--data-only",
            default=False,
            action="store_true",
            help="If set, only the database will be exported, not files",
        )

        parser.add_argument(
            "--passphrase",
            help="If provided, is used to sensitive fields in the export",
        )

    def pre_check(self) -> None:
        """
        Runs some initial checks against the state of the install and source, including:
        - Does the target exist?
        - Can we access the target?
        - Does the target have a manifest file?
        - Are there existing files in the document folders?
        - Are there existing users or documents in the database?
        """

        def pre_check_maybe_not_empty() -> None:
            # Skip this check if operating only on the database
            # We can expect data to exist in that case
            if not self.data_only:
                for document_dir in [settings.ORIGINALS_DIR, settings.ARCHIVE_DIR]:
                    if document_dir.exists() and document_dir.is_dir():
                        for entry in document_dir.glob("**/*"):
                            if entry.is_dir():
                                continue
                            self.stdout.write(
                                self.style.WARNING(
                                    f"Found file {entry.relative_to(document_dir)}, this might indicate a non-empty installation",
                                ),
                            )
                            break
            # But existing users or other data still matters in a data only
            if (
                User.objects.exclude(username__in=["consumer", "AnonymousUser"]).count()
                != 0
            ):
                self.stdout.write(
                    self.style.WARNING(
                        "Found existing user(s), this might indicate a non-empty installation",
                    ),
                )
            if Document.global_objects.count() != 0:
                self.stdout.write(
                    self.style.WARNING(
                        "Found existing documents(s), this might indicate a non-empty installation",
                    ),
                )

        def pre_check_manifest_exists() -> None:
            if not (self.source / "manifest.json").exists():
                raise CommandError(
                    "That directory doesn't appear to contain a manifest.json file.",
                )

        if not self.source.exists():
            raise CommandError("That path doesn't exist")

        if not os.access(self.source, os.R_OK):
            raise CommandError("That path doesn't appear to be readable")

        pre_check_maybe_not_empty()
        pre_check_manifest_exists()

    def load_manifest_files(self) -> None:
        """
        Loads manifest data from the various JSON files for parsing and loading the database
        """
        main_manifest_path: Path = self.source / "manifest.json"
        self.manifest_paths.append(main_manifest_path)

        for file in Path(self.source).glob("**/*-manifest.json"):
            self.manifest_paths.append(file)

    def load_metadata(self) -> None:
        """
        Loads either just the version information or the version information and extra data

        Must account for the old style of export as well, with just version.json
        """
        version_path: Path = self.source / "version.json"
        metadata_path: Path = self.source / "metadata.json"
        if not version_path.exists() and not metadata_path.exists():
            self.stdout.write(
                self.style.NOTICE("No version.json or metadata.json file located"),
            )
            return

        if metadata_path.exists():
            with metadata_path.open() as infile:
                data = json.load(infile)
                self.version = data["version"]
                if not self.passphrase and EXPORTER_CRYPTO_SETTINGS_NAME in data:
                    raise CommandError(
                        "No passphrase was given, but this export contains encrypted fields",
                    )
                elif EXPORTER_CRYPTO_SETTINGS_NAME in data:
                    self.load_crypt_params(data)
        elif version_path.exists():
            with version_path.open() as infile:
                self.version = json.load(infile)["version"]

        if self.version and self.version != version.__full_version_str__:
            self.stdout.write(
                self.style.WARNING(
                    "Version mismatch: "
                    f"Currently {version.__full_version_str__},"
                    f" importing {self.version}."
                    " Continuing, but import may fail.",
                ),
            )

    def load_data_to_database(self) -> None:
        """
        As the name implies, loads data from the JSON file(s) into the database.

        Streams each manifest via ijson and feeds records in fixed-size batches
        to Django's PythonDeserializer. Within a batch, rows are grouped per
        model and inserted via ``bulk_create`` (one INSERT per model), and M2M
        relations are inserted directly into their through-tables in bulk.
        FK checks are disabled during the load and re-checked at the end,
        allowing arbitrary order of rows in the manifest.

        Behaves identically across SQLite, MariaDB and PostgreSQL: backend-
        specific concerns (FK toggle, sequence reset) go through Django's DB
        operations layer which abstracts them.
        """
        _BATCH_SIZE = 100
        using = DEFAULT_DB_ALIAS
        connection = connections[using]

        try:
            with transaction.atomic(using=using):
                # delete these since pk can change, re-created from import
                ContentType.objects.all().delete()
                Permission.objects.all().delete()

                models_loaded: set = set()
                objs_with_deferred_fields: list = []
                object_count = 0

                with connection.constraint_checks_disabled():
                    for manifest_path in self.manifest_paths:
                        record_iter = iter_manifest_records(manifest_path)
                        for batch in _iter_batches(record_iter, _BATCH_SIZE):
                            objs_by_model: dict = defaultdict(list)
                            m2m_pending: list = []
                            for deserialized in PythonDeserializer(
                                batch,
                                using=using,
                                ignorenonexistent=True,
                                handle_forward_references=True,
                            ):
                                object_count += 1
                                obj = deserialized.object
                                model_cls = type(obj)
                                objs_by_model[model_cls].append(obj)
                                if router.allow_migrate_model(using, model_cls):
                                    models_loaded.add(model_cls)
                                if deserialized.m2m_data:
                                    m2m_pending.append(
                                        (obj, deserialized.m2m_data),
                                    )
                                if deserialized.deferred_fields:
                                    objs_with_deferred_fields.append(
                                        deserialized,
                                    )

                            # One UPSERT per distinct model in the batch.
                            # Mirrors loaddata's Model.save() semantics: rows
                            # already present (e.g. singletons created by
                            # post_migrate like ApplicationConfiguration) are
                            # updated rather than rejected on PK conflict.
                            for model_cls, objs in objs_by_model.items():
                                pk_name = model_cls._meta.pk.name
                                # Skip the PK (target of the conflict) and
                                # GeneratedField columns: PostgreSQL forbids
                                # writing to generated columns and Django's
                                # save() filters them automatically, but
                                # bulk_create with explicit update_fields
                                # respects our list verbatim.
                                update_fields = [
                                    f.name
                                    for f in model_cls._meta.concrete_fields
                                    if not f.primary_key
                                    and not getattr(f, "generated", False)
                                ]
                                manager = model_cls._default_manager.db_manager(
                                    using,
                                )
                                if update_fields:
                                    manager.bulk_create(
                                        objs,
                                        batch_size=_BATCH_SIZE,
                                        update_conflicts=True,
                                        unique_fields=[pk_name],
                                        update_fields=update_fields,
                                    )
                                else:
                                    # Pure-PK models: nothing to update on
                                    # conflict, just skip duplicates.
                                    manager.bulk_create(
                                        objs,
                                        batch_size=_BATCH_SIZE,
                                        ignore_conflicts=True,
                                    )

                            # M2M: build through-table rows ourselves and
                            # bulk_create them. set() dedups per-pair like
                            # RelatedManager.set() would. ignore_conflicts
                            # protects against pre-existing relations.
                            through_rows: dict = defaultdict(list)
                            for obj, m2m_data in m2m_pending:
                                for accessor, target_pks in m2m_data.items():
                                    field = obj._meta.get_field(accessor)
                                    through = field.remote_field.through
                                    src_attr = f"{field.m2m_field_name()}_id"
                                    tgt_attr = f"{field.m2m_reverse_field_name()}_id"
                                    for target_pk in set(target_pks):
                                        through_rows[through].append(
                                            through(
                                                **{
                                                    src_attr: obj.pk,
                                                    tgt_attr: target_pk,
                                                },
                                            ),
                                        )
                            for through_cls, rows in through_rows.items():
                                through_cls._default_manager.db_manager(
                                    using,
                                ).bulk_create(
                                    rows,
                                    batch_size=_BATCH_SIZE,
                                    ignore_conflicts=True,
                                )

                    for obj in objs_with_deferred_fields:
                        obj.save_deferred_fields(using=using)

                # FKs were disabled during the load; re-check them now that
                # all rows are in. No-op on backends that don't implement it.
                table_names = [m._meta.db_table for m in models_loaded]
                connection.check_constraints(table_names=table_names)

                # Recalibrate auto-increment sequences (PostgreSQL only;
                # returns [] on MariaDB and SQLite, so this is a no-op there).
                if object_count > 0 and models_loaded:
                    sequence_sql = connection.ops.sequence_reset_sql(
                        no_style(),
                        list(models_loaded),
                    )
                    if sequence_sql:
                        with connection.cursor() as cursor:
                            for line in sequence_sql:
                                cursor.execute(line)
        except (FieldDoesNotExist, DeserializationError, IntegrityError) as e:
            self.stdout.write(self.style.ERROR("Database import failed"))
            if (
                self.version is not None
                and self.version != version.__full_version_str__
            ):  # pragma: no cover
                self.stdout.write(
                    self.style.ERROR(
                        "Version mismatch: "
                        f"Currently {version.__full_version_str__},"
                        f" importing {self.version}",
                    ),
                )
                raise e
            else:
                self.stdout.write(
                    self.style.ERROR("No version information present"),
                )
                raise e

    def handle(self, *args, **options) -> None:
        logging.getLogger().handlers[0].level = logging.ERROR

        self.source = Path(options["source"]).resolve()
        self.data_only: bool = options["data_only"]
        self.passphrase: str | None = options.get("passphrase")
        self.version: str | None = None
        self.salt: str | None = None
        self.manifest_paths = []

        # Create a temporary directory for extracting a zip file into it, even if supplied source is no zip file to keep code cleaner.
        with tempfile.TemporaryDirectory() as tmp_dir:
            if is_zipfile(self.source):
                with ZipFile(self.source) as zf:
                    zf.extractall(tmp_dir)
                self.source = Path(tmp_dir)
            self._run_import()

    def _run_import(self) -> None:
        self.pre_check()
        self.load_metadata()
        self.load_manifest_files()
        self.check_manifest_validity()
        self.decrypt_secret_fields()

        # see /src/documents/signals/handlers.py
        with (
            disable_signal(
                post_save,
                receiver=update_filename_and_move_files,
                sender=Document,
                weak=False,
            ),
            disable_signal(
                post_save,
                receiver=update_llm_suggestions_cache,
                sender=Document,
                weak=False,
            ),
            disable_signal(
                m2m_changed,
                receiver=update_filename_and_move_files,
                sender=Document.tags.through,
                weak=False,
            ),
            disable_signal(
                post_save,
                receiver=update_filename_and_move_files,
                sender=CustomFieldInstance,
                weak=False,
            ),
            disable_signal(
                post_save,
                receiver=check_paths_and_prune_custom_fields,
                sender=CustomField,
            ),
        ):
            if settings.AUDIT_LOG_ENABLED:
                auditlog.unregister(Document)
                auditlog.unregister(Correspondent)
                auditlog.unregister(Tag)
                auditlog.unregister(DocumentType)
                auditlog.unregister(Note)
                auditlog.unregister(CustomField)
                auditlog.unregister(CustomFieldInstance)

            # Fill up the database with whatever is in the manifest
            self.load_data_to_database()
            if not self.data_only:
                self._import_files_from_manifest()
            else:
                self.stdout.write(self.style.NOTICE("Data only import completed"))

            for tmp in getattr(self, "_decrypted_tmp_paths", []):
                tmp.unlink(missing_ok=True)

        self.stdout.write("Updating search index...")
        call_command(
            "document_index",
            "reindex",
            no_progress_bar=self.no_progress_bar,
        )

    def check_manifest_validity(self) -> None:
        """
        Attempts to verify the manifest is valid.  Namely checking the files
        referred to exist and the files can be read from
        """

        def check_document_validity(document_record: dict) -> None:
            if EXPORTER_FILE_NAME not in document_record:
                raise CommandError(
                    "The manifest file contains a record which does not "
                    "refer to an actual document file.",
                )

            doc_file = document_record[EXPORTER_FILE_NAME]
            doc_path: Path = self.source / doc_file
            if not doc_path.exists():
                raise CommandError(
                    f'The manifest file refers to "{doc_file}" which does not '
                    "appear to be in the source directory.",
                )
            try:
                with doc_path.open(mode="rb"):
                    pass
            except Exception as e:
                raise CommandError(
                    f"Failed to read from original file {doc_path}",
                ) from e

            if EXPORTER_ARCHIVE_NAME in document_record:
                archive_file = document_record[EXPORTER_ARCHIVE_NAME]
                doc_archive_path: Path = self.source / archive_file
                if not doc_archive_path.exists():
                    raise CommandError(
                        f"The manifest file refers to {archive_file} which "
                        f"does not appear to be in the source directory.",
                    )
                try:
                    with doc_archive_path.open(mode="rb"):
                        pass
                except Exception as e:
                    raise CommandError(
                        f"Failed to read from archive file {doc_archive_path}",
                    ) from e

        def check_share_link_bundle_validity(bundle_record: dict) -> None:
            if EXPORTER_SHARE_LINK_BUNDLE_NAME not in bundle_record:
                return

            bundle_file = bundle_record[EXPORTER_SHARE_LINK_BUNDLE_NAME]
            bundle_path: Path = self.source / bundle_file
            if not bundle_path.exists():
                raise CommandError(
                    f'The manifest file refers to "{bundle_file}" which does not '
                    "appear to be in the source directory.",
                )
            try:
                with bundle_path.open(mode="rb"):
                    pass
            except Exception as e:
                raise CommandError(
                    f"Failed to read from share link bundle file {bundle_path}",
                ) from e

        self.stdout.write("Checking the manifest")
        self._document_count = 0
        self._bundle_count = 0
        for manifest_path in self.manifest_paths:
            for record in iter_manifest_records(manifest_path):
                if record["model"] == "documents.document":
                    self._document_count += 1
                    if not self.data_only:
                        check_document_validity(record)
                elif record["model"] == "documents.sharelinkbundle":
                    if record.get(EXPORTER_SHARE_LINK_BUNDLE_NAME):
                        self._bundle_count += 1
                    if not self.data_only:
                        check_share_link_bundle_validity(record)

    def _iter_manifest_by_model(self, model: str) -> Generator[dict, None, None]:
        """Stream records of a given model from all manifest files."""
        for manifest_path in self.manifest_paths:
            for record in iter_manifest_records(manifest_path):
                if record["model"] == model:
                    yield record

    def _import_files_from_manifest(self) -> None:
        settings.ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
        settings.THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)
        settings.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        settings.SHARE_LINK_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

        self.stdout.write("Copy files into paperless...")

        self._import_document_files()
        self._import_share_link_bundle_files()

    def _import_document_files(self) -> None:
        """Stream document records from manifest and copy files in batches."""
        _BATCH_SIZE = 100

        record_iter = (
            {
                "pk": r["pk"],
                EXPORTER_FILE_NAME: r[EXPORTER_FILE_NAME],
                EXPORTER_THUMBNAIL_NAME: r.get(EXPORTER_THUMBNAIL_NAME),
                EXPORTER_ARCHIVE_NAME: r.get(EXPORTER_ARCHIVE_NAME),
            }
            for r in self._iter_manifest_by_model("documents.document")
        )

        processed = 0
        for record_batch in self.track(
            _iter_batches(record_iter, _BATCH_SIZE),
            description="Copying files...",
            total=(self._document_count + _BATCH_SIZE - 1) // _BATCH_SIZE,
        ):
            # Batch-fetch all Document instances for this chunk
            pks = [r["pk"] for r in record_batch]
            doc_map = {d.pk: d for d in Document.global_objects.filter(pk__in=pks)}

            for record in record_batch:
                document = doc_map[record["pk"]]

                doc_file = record[EXPORTER_FILE_NAME]
                document_path = self.source / doc_file

                thumbnail_path = None
                if record[EXPORTER_THUMBNAIL_NAME]:
                    thumbnail_path = (
                        self.source / record[EXPORTER_THUMBNAIL_NAME]
                    ).resolve()

                archive_path = None
                if record[EXPORTER_ARCHIVE_NAME]:
                    archive_path = self.source / record[EXPORTER_ARCHIVE_NAME]

                with FileLock(settings.MEDIA_LOCK):
                    if Path(document.source_path).is_file():
                        raise FileExistsError(document.source_path)

                    create_source_path_directory(document.source_path)
                    copy_file_with_basic_stats(document_path, document.source_path)

                    if thumbnail_path:
                        copy_file_with_basic_stats(
                            thumbnail_path,
                            document.thumbnail_path,
                        )

                    if archive_path:
                        create_source_path_directory(document.archive_path)
                        copy_file_with_basic_stats(
                            archive_path,
                            document.archive_path,
                        )

                document.save()

            processed += len(record_batch)

    def _import_share_link_bundle_files(self) -> None:
        """Stream share link bundle records from manifest and copy files."""
        _BATCH_SIZE = 100

        record_iter = (
            {
                "pk": r["pk"],
                EXPORTER_SHARE_LINK_BUNDLE_NAME: r.get(
                    EXPORTER_SHARE_LINK_BUNDLE_NAME,
                ),
            }
            for r in self._iter_manifest_by_model("documents.sharelinkbundle")
            if r.get(EXPORTER_SHARE_LINK_BUNDLE_NAME)
        )

        for record_batch in self.track(
            _iter_batches(record_iter, _BATCH_SIZE),
            description="Copying share link bundles...",
            total=(self._bundle_count + _BATCH_SIZE - 1) // _BATCH_SIZE,
        ):
            pks = [r["pk"] for r in record_batch]
            bundle_map = {b.pk: b for b in ShareLinkBundle.objects.filter(pk__in=pks)}

            for record in record_batch:
                bundle = bundle_map[record["pk"]]
                bundle_file = record[EXPORTER_SHARE_LINK_BUNDLE_NAME]
                bundle_source_path = (self.source / bundle_file).resolve()
                bundle_target_path = bundle.absolute_file_path
                if bundle_target_path is None:
                    raise CommandError(
                        f"Share link bundle {bundle.pk} does not have a valid file path.",
                    )

                with FileLock(settings.MEDIA_LOCK):
                    bundle_target_path.parent.mkdir(parents=True, exist_ok=True)
                    copy_file_with_basic_stats(
                        bundle_source_path,
                        bundle_target_path,
                    )

    def _decrypt_record_if_needed(self, record: dict) -> dict:
        fields = self.CRYPT_FIELDS_BY_MODEL.get(record.get("model", ""))
        if fields:
            for field in fields:
                if record["fields"].get(field):
                    record["fields"][field] = self.decrypt_string(
                        value=record["fields"][field],
                    )
        return record

    def decrypt_secret_fields(self) -> None:
        """
        The converse decryption of some fields out of the export before importing to database.
        Streams records from each manifest path and writes decrypted content to a temp file.
        """
        if not self.passphrase:
            return
        # Salt has been loaded from metadata.json at this point, so it cannot be None
        self.setup_crypto(passphrase=self.passphrase, salt=self.salt)
        self._decrypted_tmp_paths: list[Path] = []
        new_paths: list[Path] = []
        for manifest_path in self.manifest_paths:
            tmp = manifest_path.with_name(manifest_path.stem + ".decrypted.json")
            with tmp.open("w", encoding="utf-8") as out:
                out.write("[\n")
                first = True
                for record in iter_manifest_records(manifest_path):
                    if not first:
                        out.write(",\n")
                    json.dump(
                        self._decrypt_record_if_needed(record),
                        out,
                        indent=2,
                        ensure_ascii=False,
                    )
                    first = False
                out.write("\n]\n")
            self._decrypted_tmp_paths.append(tmp)
            new_paths.append(tmp)
        self.manifest_paths = new_paths
