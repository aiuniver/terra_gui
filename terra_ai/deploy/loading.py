import os
import re
import time
import requests

from pathlib import Path
from subprocess import Popen, PIPE, STDOUT

from terra_ai import progress
from terra_ai.progress import utils as progress_utils
from terra_ai.data.deploy.stages import (
    StageUploadData,
    StageCompleteData,
    StageResponseData,
)
from terra_ai.exceptions.deploy import RequestAPIException


DEPLOY_PREPARE_TITLE = "Подготовка данных"
DEPLOY_UPLOAD_TITLE = "Загрузка архива"


def __run_rsync(progress_name: str, data: StageUploadData, destination: str):
    rsa_path = Path(f'./{data.server.get("domain_name")}.rsa.key')
    try:
        os.remove(rsa_path)
    except Exception:
        pass
    with open(rsa_path, "w") as rsa_path_ref:
        rsa_path_ref.write(f'{data.server.get("private_ssh_key")}\n')
        rsa_path.chmod(0o600)
    cmd = f'rsync -P -avz -e "ssh -i {rsa_path} -o StrictHostKeyChecking=no" {data.file.path} {data.server.get("user")}@{data.server.get("domain_name")}:{destination}'
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
        deploy_url = (
            f'https://{upload_data.server.get("domain_name")}/autodeployterra_upload/'
        )
        upload_data_dict = upload_data.native()
        upload_data_dict.pop("server")
        upload_response = requests.post(
            deploy_url,
            json=upload_data_dict,
            headers={"Content-Type": "application/json"},
        )
        if upload_response.ok:
            upload_response = upload_response.json()
            if upload_response.get("success"):
                progress.pool(progress_name, message=DEPLOY_UPLOAD_TITLE, percent=0)
                __run_rsync(
                    progress_name, upload_data, upload_response.get("destination")
                )
                complete_data = StageCompleteData(
                    stage=2,
                    deploy=upload_response.get("deploy"),
                    login=upload_data.user.login,
                    project=upload_data.project.slug,
                )
                complete_response = requests.post(
                    deploy_url,
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
