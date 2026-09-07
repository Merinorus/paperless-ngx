import json

import pytest
from cachalot.settings import cachalot_settings
from django.contrib.auth.models import Group
from django.contrib.auth.models import Permission
from django.contrib.auth.models import User
from django.core.cache import caches
from django.test import override_settings
from guardian.shortcuts import assign_perm
from guardian.shortcuts import remove_perm

from documents.models import Correspondent
from documents.models import CustomField
from documents.models import CustomFieldInstance
from documents.models import Document
from documents.models import DocumentType
from documents.models import StoragePath
from documents.models import Tag
from documents.tests.factories import DocumentFactory
from documents.views import DocumentViewSet

pytestmark = pytest.mark.django_db(transaction=True)


@pytest.fixture(params=[False, True], ids=["uncached", "cached"])
def selection_cache(request, settings):
    cache_settings = {
        **settings.CACHES,
        "selection-tests": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "selection-tests",
        },
    }
    with override_settings(
        CACHES=cache_settings,
        CACHALOT_CACHE="selection-tests",
        CACHALOT_ENABLED=request.param,
    ):
        cachalot_settings.reload()
        caches["selection-tests"].clear()
        try:
            yield request.param
        finally:
            caches["selection-tests"].clear()
    cachalot_settings.reload()


@pytest.fixture
def metadata(selection_cache):
    # Insert in reverse alphabetical order to check the response ordering too.
    result = {}
    for key, model in (
        ("selected_correspondents", Correspondent),
        ("selected_tags", Tag),
        ("selected_document_types", DocumentType),
        ("selected_storage_paths", StoragePath),
        ("selected_custom_fields", CustomField),
    ):
        kwargs = (
            {"data_type": CustomField.FieldDataType.STRING}
            if model is CustomField
            else {}
        )
        unused = model.objects.create(name="z-unused", **kwargs)
        used = model.objects.create(name="a-used", **kwargs)
        result[key] = (used, unused)
    return result


def expected_counts(metadata, count):
    result = {
        key: [
            {"id": used.pk, "document_count": count},
            {"id": unused.pk, "document_count": 0},
        ]
        for key, (used, unused) in metadata.items()
    }
    # Custom fields are ordered by creation date, not by name.
    result["selected_custom_fields"].reverse()
    return result


def add_document(metadata, **kwargs):
    document = DocumentFactory(
        correspondent=metadata["selected_correspondents"][0],
        document_type=metadata["selected_document_types"][0],
        storage_path=metadata["selected_storage_paths"][0],
        **kwargs,
    )
    document.tags.add(metadata["selected_tags"][0])
    CustomFieldInstance.objects.create(
        document=document,
        field=metadata["selected_custom_fields"][0],
        value_text="value",
    )
    return document


def get_selection(client, **params):
    response = client.get(
        "/api/documents/",
        {"include_selection_data": "true", **params},
    )
    assert response.status_code == 200
    return response.data["selection_data"]


def test_selection_contract_and_filters(admin_client, metadata):
    first = add_document(metadata, title="matching first")
    second = add_document(metadata, title="matching second")
    add_document(metadata, title="excluded")
    # Matching two tags must not count a document twice.
    first.tags.add(metadata["selected_tags"][1])
    expected = expected_counts(metadata, 2)
    expected["selected_tags"][1]["document_count"] = 1
    params = {
        "title__icontains": "matching",
        "tags__id__in": ",".join(str(tag.pk) for tag in metadata["selected_tags"]),
        "page_size": 1,
    }
    for page, ordering in ((1, "title"), (2, "title"), (1, "-created")):
        assert (
            get_selection(admin_client, **params, page=page, ordering=ordering)
            == expected
        )
    assert get_selection(admin_client, id__in=f"{first.pk},{second.pk}") == expected
    assert get_selection(admin_client, id__in="0") == expected_counts(metadata, 0)
    assert "selection_data" not in admin_client.get("/api/documents/").data


def test_selection_cache_reuses_counts_across_ordering(
    metadata,
    selection_cache,
    django_assert_num_queries,
):
    add_document(metadata)
    view = DocumentViewSet()
    expected = expected_counts(metadata, 1)
    assert (
        view._get_selection_data_for_queryset(Document.objects.order_by("created"))
        == expected
    )
    if selection_cache:
        with django_assert_num_queries(0):
            assert (
                view._get_selection_data_for_queryset(
                    Document.objects.order_by("-title"),
                )
                == expected
            )


def test_selection_updates_after_document_changes(admin_client, metadata):
    document = add_document(metadata)
    assert get_selection(admin_client) == expected_counts(metadata, 1)
    add_document(metadata, root_document=document)
    # Versions are not additional documents in the default list.
    assert get_selection(admin_client) == expected_counts(metadata, 1)
    document.delete()
    assert get_selection(admin_client) == expected_counts(metadata, 0)
    document.restore(strict=False)
    assert get_selection(admin_client) == expected_counts(metadata, 1)
    document.correspondent = metadata["selected_correspondents"][1]
    document.document_type = metadata["selected_document_types"][1]
    document.storage_path = metadata["selected_storage_paths"][1]
    document.save()
    expected = expected_counts(metadata, 1)
    for key in (
        "selected_correspondents",
        "selected_document_types",
        "selected_storage_paths",
    ):
        expected[key][0]["document_count"] = 0
        expected[key][1]["document_count"] = 1
    assert get_selection(admin_client) == expected


def test_selection_updates_after_relation_changes(admin_client, metadata):
    document = add_document(metadata)
    tag = metadata["selected_tags"][0]
    params = {"tags__id__in": str(tag.pk)}
    assert get_selection(admin_client, **params) == expected_counts(metadata, 1)
    document.tags.remove(tag)
    assert get_selection(admin_client, **params) == expected_counts(metadata, 0)
    document.tags.add(tag)
    assert get_selection(admin_client, **params) == expected_counts(metadata, 1)
    instance = document.custom_fields.get()
    instance.delete()
    expected = expected_counts(metadata, 1)
    expected["selected_custom_fields"][1]["document_count"] = 0
    assert get_selection(admin_client, **params) == expected
    instance.restore(strict=False)
    assert get_selection(admin_client, **params) == expected_counts(metadata, 1)
    added = Correspondent.objects.create(name="b-new")
    expected = expected_counts(metadata, 1)
    expected["selected_correspondents"].insert(1, {"id": added.pk, "document_count": 0})
    assert get_selection(admin_client, **params) == expected


def test_selection_permissions_and_cache_invalidation(rest_api_client, metadata):
    user = User.objects.create_user(username="reader")
    other = User.objects.create_user(username="other")
    permission = Permission.objects.get(
        codename="view_document",
        content_type__app_label="documents",
    )
    user.user_permissions.add(permission)
    other.user_permissions.add(permission)
    document = add_document(metadata, owner=other)
    rest_api_client.force_authenticate(user=user)
    assert get_selection(rest_api_client) == expected_counts(metadata, 0)
    assign_perm("view_document", user, document)
    assert get_selection(rest_api_client) == expected_counts(metadata, 1)

    remove_perm("view_document", user, document)
    assert get_selection(rest_api_client) == expected_counts(metadata, 0)
    group = Group.objects.create(name="readers")
    assign_perm("view_document", group, document)
    user.groups.add(group)
    assert get_selection(rest_api_client) == expected_counts(metadata, 1)
    user.groups.remove(group)
    assert get_selection(rest_api_client) == expected_counts(metadata, 0)
    rest_api_client.force_authenticate(user=other)
    assert get_selection(rest_api_client) == expected_counts(metadata, 1)
    # Changing ownership invalidates the cached permission-filtered counts.
    document.owner = user
    document.save()
    assert get_selection(rest_api_client) == expected_counts(metadata, 0)
    rest_api_client.force_authenticate(user=user)
    assert get_selection(rest_api_client) == expected_counts(metadata, 1)


def test_selection_custom_field_filter_invalidation(admin_client, metadata):
    document = add_document(metadata)
    field = metadata["selected_custom_fields"][0]
    params = {"custom_field_query": json.dumps([field.pk, "exact", "value"])}
    assert get_selection(admin_client, **params) == expected_counts(metadata, 1)
    instance = document.custom_fields.get()
    instance.value_text = "changed"
    instance.save()
    assert get_selection(admin_client, **params) == expected_counts(metadata, 0)
