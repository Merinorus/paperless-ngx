from django.db import migrations
from django.db import models


class Migration(migrations.Migration):
    dependencies = [
        ("paperless", "0015_applicationconfiguration_remote_ocr_mode"),
    ]

    operations = [
        migrations.AlterField(
            model_name="applicationconfiguration",
            name="remote_ocr_mode",
            field=models.CharField(
                blank=True,
                choices=[
                    ("always", "All supported documents"),
                    ("auto", "Only for documents with no embedded text"),
                    ("workflow_only", "Only when a workflow enables it"),
                ],
                max_length=32,
                null=True,
                verbose_name="Sets which documents are sent to the remote OCR engine",
            ),
        ),
    ]
