import os
import sys
import uuid
import json
import shutil
import base64
import tempfile
import requests

from pathlib import Path
from pydantic import ValidationError
from pydantic.networks import HttpUrl
from pydantic.errors import PathNotExistsError

from .. import progress, settings
from ..data.datasets.creation import SourceData
from ..data.datasets.dataset import (
    DatasetLoadData,
    DatasetsGroupsList,
    CustomDatasetConfigData,
    DatasetInfo,
)
from ..data.datasets.extra import DatasetGroupChoice
from ..data.presets.datasets import DatasetsGroups
from ..exceptions.datasets import (
    DatasetSourceLoadUndefinedMethodException,
    DatasetChoiceUndefinedMethodException,
    UnknownDatasetException,
)
from ..progress import utils as progress_utils

DOWNLOAD_SOURCE_TITLE = "Загрузка исходников датасета"
DATASET_SOURCE_UNPACK_TITLE = "Распаковка исходников датасета"
DATASET_CHOICE_TITLE = "Загрузка датасета %s.%s"
DATASET_CHOICE_UNPACK_TITLE = "Распаковка датасета %s.%s"
DATASET_CHOICE_TERRA_URL = "https://storage.googleapis.com/terra_ai/DataSets/Numpy/"


os.makedirs(settings.DATASETS_SOURCE_DIR, exist_ok=True)
os.makedirs(settings.DATASETS_LOADED_DIR, exist_ok=True)
for item in DatasetGroupChoice:
    os.makedirs(Path(settings.DATASETS_LOADED_DIR, item.name), exist_ok=True)


@progress.threading
def __load_from_url(folder: Path, url: HttpUrl):
    # Получение папки датасета
    folder_name = base64.b64encode(url.encode("UTF-8")).decode("UTF-8")
    dataset_path = Path(folder, folder_name)

    # Сброс прогресс-бара
    progress_name = "dataset_source_load"

    # Что делаем, если папка датасета уже существует
    if dataset_path.exists():
        shutil.rmtree(dataset_path, ignore_errors=True)
    #     progress.pool(
    #         progress_name,
    #         percent=100,
    #         data=dataset_path.absolute(),
    #         finished=True,
    #     )
    #     return

    # Запускаем загрузку
    try:
        zipfile_path = progress_utils.download(
            progress_name, DOWNLOAD_SOURCE_TITLE, url
        )
        zip_destination = progress_utils.unpack(
            progress_name, DATASET_SOURCE_UNPACK_TITLE, zipfile_path
        )
        shutil.move(zip_destination, dataset_path)
        os.remove(zipfile_path.absolute())
        progress.pool(progress_name, data=dataset_path.absolute(), finished=True)
    except (Exception, requests.exceptions.ConnectionError) as error:
        progress.pool(progress_name, error=str(error))


@progress.threading
def __load_from_googledrive(folder: Path, zipfile_path: Path):
    # Получение папки датасета
    folder_name = zipfile_path.name[: zipfile_path.name.rfind(".")]
    dataset_path = Path(folder, folder_name)

    # Имя прогресс-бара
    progress_name = "dataset_source_load"

    # Что делаем, если папка датасета уже существует
    if dataset_path.exists():
        shutil.rmtree(dataset_path, ignore_errors=True)
    #     progress.pool(
    #         progress_name,
    #         percent=100,
    #         data=dataset_path.absolute(),
    #         finished=True,
    #     )
    #     return

    # Запускаем загрузку
    try:
        zip_destination = progress_utils.unpack(
            progress_name, DATASET_SOURCE_UNPACK_TITLE, zipfile_path
        )
        shutil.move(zip_destination, dataset_path)
        progress.pool(progress_name, data=dataset_path.absolute(), finished=True)
    except Exception as error:
        progress.pool(progress_name, error=str(error))


def source(strict_object: SourceData):
    __method_name = f"__load_from_{strict_object.mode.lower()}"
    __method = getattr(sys.modules.get(__name__), __method_name, None)
    if __method:
        mode_folder = Path(settings.DATASETS_SOURCE_DIR, strict_object.mode.lower())
        os.makedirs(mode_folder, exist_ok=True)
        __method(mode_folder, strict_object.value)
    else:
        raise DatasetSourceLoadUndefinedMethodException(strict_object.mode.value)


@progress.threading
def _choice_from_keras(
    progress_name: str,
    destination: Path,
    name: str,
    reset_model: bool,
    **kwargs,
):
    dataset = (
        DatasetsGroupsList(DatasetsGroups)
        .get(DatasetGroupChoice.keras)
        .datasets.get(name)
    )
    if dataset:
        shutil.rmtree(destination, ignore_errors=True)
        os.makedirs(destination, exist_ok=True)
        with open(Path(destination, settings.DATASET_CONFIG), "w") as config_ref:
            json.dump(dataset.native(), config_ref)
        progress.pool(
            progress_name,
            percent=100,
            data={
                "info": DatasetInfo(group=DatasetGroupChoice.keras, alias=name),
                "reset_model": reset_model,
            },
            finished=True,
        )
    else:
        progress.pool(
            progress_name,
            finished=True,
            error=str(UnknownDatasetException(DatasetGroupChoice.keras.value, name)),
        )


@progress.threading
def _choice_from_terra(
    progress_name: str,
    destination: Path,
    name: str,
    reset_model: bool,
    **kwargs,
):
    try:
        zipfile_path = progress_utils.download(
            progress_name,
            DATASET_CHOICE_TITLE % (DatasetGroupChoice.terra.value, name),
            f"{DATASET_CHOICE_TERRA_URL}{name}.zip",
        )
        zip_destination = progress_utils.unpack(
            progress_name,
            DATASET_CHOICE_UNPACK_TITLE % (DatasetGroupChoice.terra.value, name),
            zipfile_path,
        )
        data = CustomDatasetConfigData(path=Path(zip_destination))
        zip_dirpath = Path(tempfile.gettempdir(), str(uuid.uuid4()))
        shutil.copytree(data.path, zip_dirpath)
        os.chmod(zip_dirpath, 0o755)
        zip_filepath = Path(zip_dirpath, "dataset.zip")
        unpacked = progress_utils.unpack(
            progress_name,
            DATASET_CHOICE_UNPACK_TITLE % (DatasetGroupChoice.terra.value, name),
            zip_filepath,
        )
        os.remove(zip_filepath)
        for item in os.listdir(unpacked):
            shutil.move(str(Path(unpacked, item).absolute()), zip_dirpath)
        shutil.rmtree(destination, ignore_errors=True)
        os.rename(zip_dirpath, destination)
        shutil.rmtree(unpacked)
        if Path(destination, settings.DATASET_CONFIG).is_file():
            progress.pool(
                progress_name,
                percent=100,
                data={
                    "info": DatasetInfo(group=DatasetGroupChoice.terra, alias=name),
                    "reset_model": reset_model,
                },
                finished=True,
            )
        else:
            progress.pool(
                progress_name,
                finished=True,
                error=str(
                    UnknownDatasetException(DatasetGroupChoice.terra.value, name)
                ),
            )
    except ValidationError as error:
        for item in error.args[0]:
            if isinstance(item.exc, PathNotExistsError):
                progress.pool(
                    progress_name,
                    finished=True,
                    error=str(
                        UnknownDatasetException(DatasetGroupChoice.terra.value, name)
                    ),
                )
                return
        progress.pool(progress_name, finished=True, error=str(error))
    except (Exception, requests.exceptions.ConnectionError) as error:
        progress.pool(progress_name, finished=True, error=str(error))
    except Exception as error:
        progress.pool(progress_name, finished=True, error=str(error))


@progress.threading
def _choice_from_custom(
    progress_name: str,
    destination: Path,
    name: str,
    source: Path,
    reset_model: bool,
    **kwargs,
):
    try:
        data = CustomDatasetConfigData(
            path=Path(source, f"{name}.{settings.DATASET_EXT}")
        )
        zip_dirpath = Path(tempfile.gettempdir(), str(uuid.uuid4()))
        shutil.copytree(data.path, zip_dirpath)
        os.chmod(zip_dirpath, 0o755)
        zip_filepath = Path(zip_dirpath, "dataset.zip")
        unpacked = progress_utils.unpack(
            progress_name,
            DATASET_CHOICE_UNPACK_TITLE % (DatasetGroupChoice.custom.value, name),
            zip_filepath,
        )
        os.remove(zip_filepath)
        for item in os.listdir(unpacked):
            shutil.move(str(Path(unpacked, item).absolute()), zip_dirpath)
        shutil.rmtree(destination, ignore_errors=True)
        os.rename(zip_dirpath, destination)
        shutil.rmtree(unpacked)
        if Path(destination, settings.DATASET_CONFIG).is_file():
            progress.pool(
                progress_name,
                percent=100,
                data={
                    "info": DatasetInfo(group=DatasetGroupChoice.custom, alias=name),
                    "reset_model": reset_model,
                },
                finished=True,
            )
        else:
            progress.pool(
                progress_name,
                finished=True,
                error=str(
                    UnknownDatasetException(DatasetGroupChoice.custom.value, name)
                ),
            )
    except ValidationError as error:
        for item in error.args[0]:
            if isinstance(item[0].exc, PathNotExistsError):
                progress.pool(
                    progress_name,
                    finished=True,
                    error=str(
                        UnknownDatasetException(DatasetGroupChoice.custom.value, name)
                    ),
                )
                return
        progress.pool(progress_name, finished=True, error=str(error))
    except Exception as error:
        progress.pool(progress_name, finished=True, error=str(error))


def choice(
    progress_name: str, dataset_choice: DatasetLoadData, reset_model: bool = False
):
    _method = getattr(
        sys.modules.get(__name__), f"_choice_from_{dataset_choice.group.name}", None
    )
    if _method:
        progress.pool.reset(
            progress_name,
            message=DATASET_CHOICE_TITLE
            % (dataset_choice.group.value, dataset_choice.alias),
            finished=False,
        )
        _destination = Path(
            settings.DATASETS_LOADED_DIR,
            dataset_choice.group.name,
            dataset_choice.alias,
        )
        if Path(_destination, settings.DATASET_CONFIG).is_file():
            progress.pool(
                progress_name,
                percent=100,
                data={
                    "info": DatasetInfo(
                        group=dataset_choice.group.name, alias=dataset_choice.alias
                    ),
                    "reset_model": reset_model,
                },
                finished=True,
            )
        else:
            _method(
                progress_name=progress_name,
                destination=_destination,
                name=dataset_choice.alias,
                source=dataset_choice.path,
                reset_model=reset_model,
            )
    else:
        raise DatasetChoiceUndefinedMethodException(dataset_choice.group.value)
