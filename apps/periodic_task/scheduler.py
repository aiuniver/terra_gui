import subprocess

from pathlib import Path
from apscheduler.executors.pool import ProcessPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler

from django.conf import settings


scheduler = BackgroundScheduler()
executors = {
    "default": {"type": "threadpool", "max_workers": 1},
    "processpool": ProcessPoolExecutor(max_workers=1),
}
scheduler.configure(executors=executors)


if settings.TERRA_AI_SYNC_LOGS:

    @scheduler.scheduled_job("interval", minutes=1)
    def scheduler_rsync():
        date = settings.TERRA_AI_DATE_START
        source_log = Path(settings.BASE_DIR, "logs.txt")
        source_error = Path(settings.BASE_DIR, "errors.txt")
        destination_log = Path(
            settings.TERRA_AI_BASE_DIR,
            "logs",
            "gui",
            "%d-%04d%02d%02d%02d%02d%02d.log"
            % (
                int(settings.USER_PORT),
                date.year,
                date.month,
                date.day,
                date.hour,
                date.minute,
                date.second,
            ),
            )
        data = []
        if source_log.exists():
            source_log.chmod(0o777)
            data.append((source_log, destination_log))
        if source_error.exists():
            source_error.chmod(0o777)
            destination_error = destination_log.with_suffix('.error')
            data.append((source_error, destination_error))
        for source, destination in data:
            cmd = f'rsync -avzqP -e "ssh -i {Path(settings.BASE_DIR, "rsa.key")} -o StrictHostKeyChecking=no" {source} '\
                  f'terra_log@81.90.181.251:{destination}'
            subprocess.call(cmd, shell=True)
        source_error.unlink(missing_ok=True)
