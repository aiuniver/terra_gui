from django.apps import AppConfig
from django.conf import settings

from .scheduler import scheduler


class PeriodicTaskConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.periodic_task'

    def ready(self):
        if settings.TERRA_AI_SYNC_LOGS:
            scheduler.start()
