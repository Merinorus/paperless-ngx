from __future__ import annotations

import logging
import re
from fnmatch import fnmatch
from typing import TYPE_CHECKING
from typing import NamedTuple

from django.contrib.auth import get_user_model

from documents.classifier import load_classifier
from documents.data_models import ConsumableDocument
from documents.data_models import DocumentSource
from documents.models import Correspondent
from documents.models import Document
from documents.models import DocumentType
from documents.models import MatchingModel
from documents.models import StoragePath
from documents.models import Tag
from documents.models import Workflow
from documents.models import WorkflowTrigger
from documents.permissions import get_objects_for_user_owner_aware

if TYPE_CHECKING:
    from collections.abc import Iterable

    from documents.classifier import DocumentClassifier


logger = logging.getLogger("paperless.matching")


def log_reason(
    matching_model: MatchingModel | WorkflowTrigger,
    document: Document | SerializedDocument,
    reason: str,
):
    class_name = type(matching_model).__name__
    name = (
        matching_model.name if hasattr(matching_model, "name") else str(matching_model)
    )
    logger.debug(
        f"{class_name} {name} matched on document {document} because {reason}",
    )


class SerializedUser(NamedTuple):
    id: int

    @classmethod
    def from_model(cls, model: User | None) -> SerializedUser | None:
        if model is None:
            return None
        return cls(id=model.id)


class SerializedDocument(NamedTuple):
    pk: int
    content: str
    owner: SerializedUser | None
    str_repr: str

    def __str__(self):
        return self.str_repr

    @classmethod
    def from_model(cls, model: Document):
        owner = SerializedUser.from_model(model.owner)
        return cls(pk=model.pk, content=model.content, owner=owner, str_repr=str(model))

    @property
    def id(self):
        return self.pk


class SerializedMatching(NamedTuple):
    is_insensitive: bool
    match: str
    matching_algorithm: int
    pk: int

    @classmethod
    def from_model(cls, model: MatchingModel):
        return cls(
            is_insensitive=model.is_insensitive,
            match=model.match,
            matching_algorithm=model.matching_algorithm,
            pk=model.pk,
        )


User = get_user_model()


def match_correspondents(
    document: Document | SerializedDocument,
    classifier: DocumentClassifier | None = None,
    user_id: int | None = None,
    correspondents: Iterable[MatchingModel | SerializedMatching] | None = None,
):
    if not classifier:
        classifier = load_classifier()

    pred_id = classifier.predict_correspondent(document.content) if classifier else None

    if not correspondents:
        if not user_id and document.owner:
            user_id = document.owner.id
        user = User.objects.get(id=user_id) if user_id else None
        correspondents = list(
            get_objects_for_user_owner_aware(
                user,
                "documents.view_correspondent",
                Correspondent,
            )
            if user
            else Correspondent.objects.all(),
        )

    matching_ids: list[int] = [
        o.pk
        for o in correspondents
        if matches(o, document)
        or (o.pk == pred_id and o.matching_algorithm == MatchingModel.MATCH_AUTO)
    ]
    return matching_ids


def match_document_types(
    document: Document | SerializedDocument,
    classifier: DocumentClassifier | None = None,
    user_id: int | None = None,
    document_types: Iterable[MatchingModel | SerializedMatching] | None = None,
):
    if not classifier:
        classifier = load_classifier()
    pred_id = classifier.predict_document_type(document.content) if classifier else None

    if not document_types:
        if not user_id and document.owner:
            user_id = document.owner.id
        user = User.objects.get(id=user_id) if user_id else None
        document_types = list(
            get_objects_for_user_owner_aware(
                user,
                "documents.view_documenttype",
                DocumentType,
            )
            if user
            else DocumentType.objects.all(),
        )

    matching_ids: list[int] = [
        o.pk
        for o in document_types
        if matches(o, document)
        or (o.pk == pred_id and o.matching_algorithm == MatchingModel.MATCH_AUTO)
    ]
    return matching_ids


def match_tags(
    document: Document | SerializedDocument,
    classifier: DocumentClassifier | None = None,
    user_id: int | None = None,
    tags: Iterable[MatchingModel | SerializedMatching] | None = None,
):
    if not classifier:
        classifier = load_classifier()
    predicted_tag_ids = classifier.predict_tags(document.content) if classifier else []

    if not tags:
        if not user_id and document.owner:
            user_id = document.owner.id
        user = User.objects.get(id=user_id) if user_id else None
        tags = list(
            get_objects_for_user_owner_aware(user, "documents.view_tag", Tag)
            if user
            else Tag.objects.all(),
        )

    matching_ids: list[int] = [
        o.pk
        for o in tags
        if matches(o, document)
        or (
            o.pk in predicted_tag_ids
            and o.matching_algorithm == MatchingModel.MATCH_AUTO
        )
    ]
    return matching_ids


def match_storage_paths(
    document: Document | SerializedDocument,
    classifier: DocumentClassifier | None = None,
    user_id: int | None = None,
    storage_paths: Iterable[MatchingModel | SerializedMatching] | None = None,
):
    if not classifier:
        classifier = load_classifier()
    pred_id = classifier.predict_storage_path(document.content) if classifier else None

    if not storage_paths:
        if not user_id and document.owner:
            user_id = document.owner.id
        user = User.objects.get(id=user_id) if user_id else None
        storage_paths = list(
            get_objects_for_user_owner_aware(
                user,
                "documents.view_storagepath",
                StoragePath,
            )
            if user
            else StoragePath.objects.all(),
        )

    matching_ids: list[int] = [
        o.pk
        for o in storage_paths
        if matches(o, document)
        or (o.pk == pred_id and o.matching_algorithm == MatchingModel.MATCH_AUTO)
    ]
    return matching_ids


def matches(
    matching_model: MatchingModel | SerializedMatching,
    document: Document | SerializedDocument,
):
    search_kwargs = {}

    document_content = document.content

    # Check that match is not empty
    if not matching_model.match.strip():
        return False

    if matching_model.is_insensitive:
        search_kwargs = {"flags": re.IGNORECASE}

    if matching_model.matching_algorithm == MatchingModel.MATCH_NONE:
        return False

    elif matching_model.matching_algorithm == MatchingModel.MATCH_ALL:
        for word in _split_match(matching_model):
            search_result = re.search(rf"\b{word}\b", document_content, **search_kwargs)
            if not search_result:
                return False
        log_reason(
            matching_model,
            document,
            f"it contains all of these words: {matching_model.match}",
        )
        return True

    elif matching_model.matching_algorithm == MatchingModel.MATCH_ANY:
        for word in _split_match(matching_model):
            if re.search(rf"\b{word}\b", document_content, **search_kwargs):
                log_reason(matching_model, document, f"it contains this word: {word}")
                return True
        return False

    elif matching_model.matching_algorithm == MatchingModel.MATCH_LITERAL:
        result = bool(
            re.search(
                rf"\b{re.escape(matching_model.match)}\b",
                document_content,
                **search_kwargs,
            ),
        )
        if result:
            log_reason(
                matching_model,
                document,
                f'it contains this string: "{matching_model.match}"',
            )
        return result

    elif matching_model.matching_algorithm == MatchingModel.MATCH_REGEX:
        try:
            match = re.search(
                re.compile(matching_model.match, **search_kwargs),
                document_content,
            )
        except re.error:
            logger.error(
                f"Error while processing regular expression {matching_model.match}",
            )
            return False
        if match:
            log_reason(
                matching_model,
                document,
                f"the string {match.group()} matches the regular expression "
                f"{matching_model.match}",
            )
        return bool(match)

    elif matching_model.matching_algorithm == MatchingModel.MATCH_FUZZY:
        from rapidfuzz import fuzz

        match = re.sub(r"[^\w\s]", "", matching_model.match)
        text = re.sub(r"[^\w\s]", "", document_content)
        if matching_model.is_insensitive:
            match = match.lower()
            text = text.lower()
        if fuzz.partial_ratio(match, text, score_cutoff=90):
            # TODO: make this better
            log_reason(
                matching_model,
                document,
                f"parts of the document content somehow match the string "
                f"{matching_model.match}",
            )
            return True
        else:
            return False

    elif matching_model.matching_algorithm == MatchingModel.MATCH_AUTO:
        # this is done elsewhere.
        return False

    else:
        raise NotImplementedError("Unsupported matching algorithm")


def _split_match(matching_model):
    """
    Splits the match to individual keywords, getting rid of unnecessary
    spaces and grouping quoted words together.

    Example:
      '  some random  words "with   quotes  " and   spaces'
        ==>
      ["some", "random", "words", "with+quotes", "and", "spaces"]
    """
    findterms = re.compile(r'"([^"]+)"|(\S+)').findall
    normspace = re.compile(r"\s+").sub
    return [
        # normspace(" ", (t[0] or t[1]).strip()).replace(" ", r"\s+")
        re.escape(normspace(" ", (t[0] or t[1]).strip())).replace(r"\ ", r"\s+")
        for t in findterms(matching_model.match)
    ]


def consumable_document_matches_workflow(
    document: ConsumableDocument,
    trigger: WorkflowTrigger,
) -> tuple[bool, str]:
    """
    Returns True if the ConsumableDocument matches all filters from the workflow trigger,
    False otherwise. Includes a reason if doesn't match
    """

    trigger_matched = True
    reason = ""

    # Document source vs trigger source
    if len(trigger.sources) > 0 and document.source not in [
        int(x) for x in list(trigger.sources)
    ]:
        reason = (
            f"Document source {document.source.name} not in"
            f" {[DocumentSource(int(x)).name for x in trigger.sources]}",
        )
        trigger_matched = False

    # Document mail rule vs trigger mail rule
    if (
        trigger.filter_mailrule is not None
        and document.mailrule_id != trigger.filter_mailrule.pk
    ):
        reason = (
            f"Document mail rule {document.mailrule_id}"
            f" != {trigger.filter_mailrule.pk}",
        )
        trigger_matched = False

    # Document filename vs trigger filename
    if (
        trigger.filter_filename is not None
        and len(trigger.filter_filename) > 0
        and not fnmatch(
            document.original_file.name.lower(),
            trigger.filter_filename.lower(),
        )
    ):
        reason = (
            f"Document filename {document.original_file.name} does not match"
            f" {trigger.filter_filename.lower()}",
        )
        trigger_matched = False

    # Document path vs trigger path
    if (
        trigger.filter_path is not None
        and len(trigger.filter_path) > 0
        and not fnmatch(
            document.original_file,
            trigger.filter_path,
        )
    ):
        reason = (
            f"Document path {document.original_file}"
            f" does not match {trigger.filter_path}",
        )
        trigger_matched = False

    return (trigger_matched, reason)


def existing_document_matches_workflow(
    document: Document,
    trigger: WorkflowTrigger,
) -> tuple[bool, str]:
    """
    Returns True if the Document matches all filters from the workflow trigger,
    False otherwise. Includes a reason if doesn't match
    """

    trigger_matched = True
    reason = ""

    if trigger.matching_algorithm > MatchingModel.MATCH_NONE and not matches(
        trigger,
        document,
    ):
        reason = (
            f"Document content matching settings for algorithm '{trigger.matching_algorithm}' did not match",
        )
        trigger_matched = False

    # Document tags vs trigger has_tags
    if (
        trigger.filter_has_tags.all().count() > 0
        and document.tags.filter(
            id__in=trigger.filter_has_tags.all().values_list("id"),
        ).count()
        == 0
    ):
        reason = (
            f"Document tags {document.tags.all()} do not include"
            f" {trigger.filter_has_tags.all()}",
        )
        trigger_matched = False

    # Document correspondent vs trigger has_correspondent
    if (
        trigger.filter_has_correspondent is not None
        and document.correspondent != trigger.filter_has_correspondent
    ):
        reason = (
            f"Document correspondent {document.correspondent} does not match {trigger.filter_has_correspondent}",
        )
        trigger_matched = False

    # Document document_type vs trigger has_document_type
    if (
        trigger.filter_has_document_type is not None
        and document.document_type != trigger.filter_has_document_type
    ):
        reason = (
            f"Document doc type {document.document_type} does not match {trigger.filter_has_document_type}",
        )
        trigger_matched = False

    # Document original_filename vs trigger filename
    if (
        trigger.filter_filename is not None
        and len(trigger.filter_filename) > 0
        and document.original_filename is not None
        and not fnmatch(
            document.original_filename.lower(),
            trigger.filter_filename.lower(),
        )
    ):
        reason = (
            f"Document filename {document.original_filename} does not match"
            f" {trigger.filter_filename.lower()}",
        )
        trigger_matched = False

    return (trigger_matched, reason)


def document_matches_workflow(
    document: ConsumableDocument | Document,
    workflow: Workflow,
    trigger_type: WorkflowTrigger.WorkflowTriggerType,
) -> bool:
    """
    Returns True if the ConsumableDocument or Document matches all filters and
    settings from the workflow trigger, False otherwise
    """

    trigger_matched = True
    if workflow.triggers.filter(type=trigger_type).count() == 0:
        trigger_matched = False
        logger.info(f"Document did not match {workflow}")
        logger.debug(f"No matching triggers with type {trigger_type} found")
    else:
        for trigger in workflow.triggers.filter(type=trigger_type):
            if trigger_type == WorkflowTrigger.WorkflowTriggerType.CONSUMPTION:
                trigger_matched, reason = consumable_document_matches_workflow(
                    document,
                    trigger,
                )
            elif (
                trigger_type == WorkflowTrigger.WorkflowTriggerType.DOCUMENT_ADDED
                or trigger_type == WorkflowTrigger.WorkflowTriggerType.DOCUMENT_UPDATED
                or trigger_type == WorkflowTrigger.WorkflowTriggerType.SCHEDULED
            ):
                trigger_matched, reason = existing_document_matches_workflow(
                    document,
                    trigger,
                )
            else:
                # New trigger types need to be explicitly checked above
                raise Exception(f"Trigger type {trigger_type} not yet supported")

            if trigger_matched:
                logger.info(f"Document matched {trigger} from {workflow}")
                # matched, bail early
                return True
            else:
                logger.info(f"Document did not match {workflow}")
                logger.debug(reason)

    return trigger_matched
