#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""

import os
import sys
import subprocess

from pathlib import Path
from apscheduler.executors.pool import ProcessPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler

from django.conf import settings


def main():
    """Run administrative tasks."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    try:
        from django.core.management import execute_from_command_line

        scheduler = BackgroundScheduler()
        executors = {
            "default": {"type": "threadpool", "max_workers": 1},
            "processpool": ProcessPoolExecutor(max_workers=1),
        }
        scheduler.configure(executors=executors)
        
        @scheduler.scheduled_job("interval", minutes=5)
        def scheduler_rsync():
            # subprocess.call(
            #     f'rsync -P -avz -e "ssh -i {Path(settings.BASE_DIR, "rsa.key")} -o StrictHostKeyChecking=no" \
            #         {Path(settings.BASE_DIR, "logs.txt")} yu-maksimov@81.90.181.251:/',
            #     shell=True
            # )
            pass

        scheduler.start()

    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
