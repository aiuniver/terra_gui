#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""

import os
import sys
import subprocess

from pathlib import Path
from apscheduler.executors.pool import ProcessPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler

from django.conf import settings


def _sync_logs():
    scheduler = BackgroundScheduler()
    executors = {
        "default": {"type": "threadpool", "max_workers": 1},
        "processpool": ProcessPoolExecutor(max_workers=1),
    }
    scheduler.configure(executors=executors)

    @scheduler.scheduled_job("interval", minutes=0.1)
    def scheduler_rsync():
        date = settings.TERRA_AI_DATE_START
        destination = Path(
            settings.TERRA_AI_BASE_DIR,
            "logs",
            "gui",
            str(settings.USER_PORT),
            f"{date.year}{date.month}{date.day}{date.hour}{date.minute}{date.second}.log",
        )
        cmd = f'rsync -P -avz -e "ssh -i {Path(settings.BASE_DIR, "rsa.key")} -o StrictHostKeyChecking=no" {Path(settings.BASE_DIR, "logs.txt")} yu-maksimov@81.90.181.251:{destination}'
        subprocess.call(cmd, shell=True)

    scheduler.start()


def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    try:
        from django.core.management import execute_from_command_line

        if settings.TERRA_AI_SYNC_LOGS:
            _sync_logs()

    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
