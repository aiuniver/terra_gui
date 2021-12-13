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
        print(1)
        zip_destination = progress_utils.pack(
            progress_name, DEPLOY_PREPARE_TITLE, source
        )
        print(2)
        destination = Path(f"{zip_destination.absolute()}.zip")
        print(3)
        os.rename(zip_destination, destination)
        print(4)
        data.update({"file": {"path": destination.absolute()}})
        print(5)
        upload_data = StageUploadData(**data)
        print(6)
        upload_response = requests.post(
            settings.DEPLOY_URL,
            json=upload_data.native(),
            headers={"Content-Type": "application/json"},
        )
        print(7)
        if upload_response.ok:
            print(8)
            upload_response = upload_response.json()
            print(9)
            if upload_response.get("success"):
                print(10)
                progress.pool(progress_name, message=DEPLOY_UPLOAD_TITLE, percent=0)
                print(11)
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
                print(12)
                complete_data = StageCompleteData(
                    stage=2,
                    deploy=upload_response.get("deploy"),
                    login=upload_data.user.login,
                    project=upload_data.project.slug,
                )
                print(13)
                complete_response = requests.post(
                    settings.DEPLOY_URL,
                    json=complete_data.native(),
                    headers={"Content-Type": "application/json"},
                )
                print(14)
                os.remove(destination)
                print(15)
                if complete_response.ok:
                    print(16)
                    progress.pool(
                        progress_name,
                        data=StageResponseData(**complete_response.json()),
                        finished=True,
                    )
                else:
                    print(17)
                    raise RequestAPIException()
            else:
                print(18)
                os.remove(destination)
                raise RequestAPIException()
        else:
            print(19)
            os.remove(destination)
            raise RequestAPIException()
    except Exception as error:
        print(20)
        progress.pool(progress_name, finished=True, error=error)
