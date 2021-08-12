import requests

from pathlib import Path

from .. import progress, settings
from ..data.deploy.stages import StageUploadData, StageCompleteData, StageResponseData
from ..exceptions.deploy import RequestAPIException

from ..progress import utils as progress_utils

DEPLOY_PREPARE_TITLE = "Подготовка данных"
DEPLOY_UPLOAD_TITLE = "Загрузка архива"


@progress.threading
def upload(source: Path, data: dict):
    # Сброс прогресс-бара
    progress_name = progress.PoolName.deploy_upload

    try:
        # Подготовка данных (архивация исходников)
        zip_destination = progress_utils.pack(
            progress_name, DEPLOY_PREPARE_TITLE, source
        )
        data.update({"file": {"path": zip_destination.name}})
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
                print("Run rsync to upload to:", upload_response.get("destination"))
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
                if complete_response.ok:
                    progress.pool(
                        progress_name,
                        data=StageResponseData(**complete_response.json()),
                        finished=True,
                    )
                else:
                    raise RequestAPIException()
            else:
                raise RequestAPIException()
        else:
            raise RequestAPIException()
    except Exception as error:
        progress.pool(progress_name, error=str(error))
