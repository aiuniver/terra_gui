import os
import re
import time
import requests

from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
from django.conf import settings as django_settings

from .. import progress, settings
from ..data.deploy.stages import StageUploadData, StageCompleteData, StageResponseData
from ..exceptions.deploy import RequestAPIException

from ..progress import utils as progress_utils

DEPLOY_PREPARE_TITLE = "Подготовка данных"
DEPLOY_UPLOAD_TITLE = "Загрузка архива"


def __run_rsync(progress_name: str, file: str, destination: str):
    cmd = f'rsync -P -avz -e "ssh -i {Path(django_settings.BASE_DIR, "rsa.key")} -o StrictHostKeyChecking=no" {file} terra@188.124.47.137:{destination}'
    proc = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    while True:
        output = proc.stdout.readline().decode("utf-8")
        if re.search(r"error", output):
            progress.pool(progress_name, error=output)
            break
        if output.startswith("total size"):
            progress.pool(progress_name, percent=100)
            break
        time.sleep(1)


@progress.threading
def upload(source: Path, data: dict):
    # Сброс прогресс-бара
    progress_name = "deploy_upload"
    progress.pool.reset(progress_name)

    try:
        # Подготовка данных (архивация исходников)
        zip_destination = progress_utils.pack(
            progress_name, DEPLOY_PREPARE_TITLE, source
        )
        destination = Path(f"{zip_destination.absolute()}.zip")
        os.rename(zip_destination, destination)
        data.update({"file": {"path": destination.absolute()}})
        upload_data = StageUploadData(**data)
        upload_response = requests.post(
            settings.DEPLOY_URL,
            json=upload_data.native(),
            headers={"Content-Type": "application/json"},
        )
        if upload_response.ok:
            upload_response = upload_response.json()
            if upload_response.get("success"):
                progress.pool(progress_name, message=DEPLOY_UPLOAD_TITLE, percent=0)
                # import shutil
                #
                # print(upload_data.file.path)
                # shutil.copyfile(
                #     upload_data.file.path, "/home/bl146u/Virtualenv/test/file.zip"
                # )
                __run_rsync(
                    progress_name,
                    upload_data.file.path,
                    upload_response.get("destination"),
                )
                complete_data = StageCompleteData(
                    stage=2,
                    deploy=upload_response.get("deploy"),
                    login=upload_data.user.login,
                    project=upload_data.project.slug,
                )
                complete_response = requests.post(
                    settings.DEPLOY_URL,
                    json=complete_data.native(),
                    headers={"Content-Type": "application/json"},
                )
                os.remove(destination)
                if complete_response.ok:
                    progress.pool(
                        progress_name,
                        data=StageResponseData(**complete_response.json()),
                        finished=True,
                    )
                else:
                    raise RequestAPIException()
            else:
                os.remove(destination)
                raise RequestAPIException()
        else:
            os.remove(destination)
            raise RequestAPIException()
    except Exception as error:
        progress.pool(progress_name, finished=True, error=error)
