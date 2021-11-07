from django.apps import AppConfig

from .scheduler import scheduler


class PeriodicTaskConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.periodic_task"

    def ready(self):
        scheduler.start()
