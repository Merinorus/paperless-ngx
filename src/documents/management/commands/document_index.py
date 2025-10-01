from django.core.management import BaseCommand
from django.db import transaction

from documents.index import recreate_index_dir
from documents.management.commands.mixins import ProgressBarMixin
from documents.tasks import index_optimize
from documents.tasks import index_reindex


class Command(ProgressBarMixin, BaseCommand):
    help = "Manages the document index."

    def add_arguments(self, parser):
        parser.add_argument("command", choices=["reindex", "optimize"])
        self.add_argument_progress_bar_mixin(parser)

    def handle(self, *args, **options):
        self.handle_progress_bar_mixin(**options)
        with transaction.atomic():
            if options["command"] == "reindex":
                recreate_index_dir()
                index_reindex(progress_bar_disable=self.no_progress_bar)
            elif options["command"] == "optimize":
                index_optimize()
