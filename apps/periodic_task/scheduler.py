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

@scheduler.scheduled_job("interval", minutes=1)
def scheduler_rsync():
    date = settings.TERRA_AI_DATE_START
    destination = Path(
        settings.TERRA_AI_BASE_DIR,
        "logs",
        "gui",
        f"{settings.USER_PORT}-{date.year}{date.month}{date.day}{date.hour}{date.minute}{date.second}.log",
    )
    cmd = f'rsync -avzqP -e "ssh -i {Path(settings.BASE_DIR, "rsa.key")} -o StrictHostKeyChecking=no" {Path(settings.BASE_DIR, "logs.txt")} terra_log@81.90.181.251:{destination}'
    subprocess.call(cmd, shell=True)
    