import os
import sys
import shutil
import base64
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
    DatasetData,
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
DATASET_CHOICE_TERRA_URL = "https://storage.googleapis.com/terra_ai/DataSets/terra_v2/"
DATASETS_SOURCE_DIR = Path(settings.TMP_DIR, "datasets", "sources")

os.makedirs(DATASETS_SOURCE_DIR, exist_ok=True)


@progress.threading
def __load_from_url(folder: Path, url: HttpUrl):
    # Получение папки датасета
    folder_name = base64.b64encode(url.encode("UTF-8")).decode("UTF-8")
    dataset_path = Path(folder, folder_name)

    # Сброс прогресс-бара
    progress_name = "dataset_source_load"

    # Если папка датасета уже существует, просто выходим и говорим прогресс-бару,
    # что загрузка завершена и возвращаем путь в прогресс-бар
    if dataset_path.exists():
        progress.pool(
            progress_name,
            percent=100,
            data=dataset_path.absolute(),
            finished=True,
        )
        return

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

    # Если папка датасета уже существует, просто выходим и говорим прогресс-бару,
    # что загрузка завершена и возвращаем путь в прогресс-бар
    if dataset_path.exists():
        progress.pool(
            progress_name,
            percent=100,
            data=dataset_path.absolute(),
            finished=True,
        )
        return

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
        mode_folder = Path(DATASETS_SOURCE_DIR, strict_object.mode.lower())
        os.makedirs(mode_folder, exist_ok=True)
        __method(mode_folder, strict_object.value)
    else:
        raise DatasetSourceLoadUndefinedMethodException(strict_object.mode.value)


@progress.threading
def __choice_from_keras(name: str, destination: Path, **kwargs):
    # Имя прогресс-бара
    progress_name = "dataset_choice"
    progress.pool.reset(
        progress_name,
        message=DATASET_CHOICE_TITLE % (DatasetGroupChoice.keras.value, name),
        finished=False,
    )

    # Выбор датасета
    dataset = (
        DatasetsGroupsList(DatasetsGroups)
        .get(DatasetGroupChoice.keras)
        .datasets.get(name)
    )
    if dataset:
        shutil.rmtree(destination, ignore_errors=True)
        os.makedirs(destination, exist_ok=True)
        progress.pool(progress_name, percent=100, data=dataset, finished=True)
    else:
        progress.pool(
            progress_name,
            error=str(UnknownDatasetException(DatasetGroupChoice.keras.value, name)),
        )


@progress.threading
def __choice_from_custom(name: str, destination: Path, source: Path, **kwargs):
    # Имя прогресс-бара
    progress_name = "dataset_choice"
    progress.pool.reset(
        progress_name,
        message=DATASET_CHOICE_TITLE % (DatasetGroupChoice.custom.value, name),
        finished=False,
    )

    # Выбор датасета
    try:
        data = CustomDatasetConfigData(
            path=Path(source, f"{name}.{settings.DATASET_EXT}")
        )
        shutil.rmtree(destination)
        shutil.copytree(data.path, destination)
        dataset = DatasetData(**data.config)
        if dataset:
            progress.pool(progress_name, percent=100, data=dataset, finished=True)
        else:
            progress.pool(
                progress_name,
                error=str(
                    UnknownDatasetException(DatasetGroupChoice.custom.value, name)
                ),
            )
    except ValidationError as error:
        for item in error.args[0]:
            if isinstance(item.exc, PathNotExistsError):
                progress.pool(
                    progress_name,
                    error=str(
                        UnknownDatasetException(DatasetGroupChoice.custom.value, name)
                    ),
                )
                return
        progress.pool(progress_name, error=str(error))
    except Exception as error:
        progress.pool(progress_name, error=str(error))


@progress.threading
def __choice_from_terra(name: str, destination: Path, **kwargs):
    # Имя прогресс-бара
    progress_name = "dataset_choice"
    progress.pool.reset(
        progress_name,
        message=DATASET_CHOICE_TITLE % (DatasetGroupChoice.terra.value, name),
        finished=False,
    )

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
        shutil.rmtree(destination, ignore_errors=True)
        os.makedirs(destination, exist_ok=True)
        for file in os.listdir(zip_destination):
            shutil.move(str(Path(zip_destination, file).absolute()), destination)
        os.remove(zipfile_path.absolute())
    except (Exception, requests.exceptions.ConnectionError) as error:
        progress.pool(progress_name, error=str(error))
        return

    # Выбор датасета
    try:
        data = CustomDatasetConfigData(path=destination)
        dataset = DatasetData(**data.config)
        if dataset:
            progress.pool(progress_name, percent=100, data=dataset, finished=True)
        else:
            progress.pool(
                progress_name,
                error=str(
                    UnknownDatasetException(DatasetGroupChoice.terra.value, name)
                ),
            )
    except ValidationError as error:
        for item in error.args[0]:
            if isinstance(item.exc, PathNotExistsError):
                progress.pool(
                    progress_name,
                    error=str(
                        UnknownDatasetException(DatasetGroupChoice.terra.value, name)
                    ),
                )
                return
        progress.pool(progress_name, error=str(error))
    except Exception as error:
        progress.pool(progress_name, error=str(error))


def choice(dataset_choice: DatasetLoadData, destination: Path):
    __method_name = f"__choice_from_{dataset_choice.group.lower()}"
    __method = getattr(sys.modules.get(__name__), __method_name, None)
    if __method:
        __method(
            name=dataset_choice.alias,
            destination=destination,
            source=dataset_choice.path,
        )
    else:
        raise DatasetChoiceUndefinedMethodException(dataset_choice.group.value)
