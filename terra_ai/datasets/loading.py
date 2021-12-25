import os
import sys
import json
import shutil
import base64

from pathlib import Path
from typing import List, Callable
from pydantic.networks import HttpUrl

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
    DatasetNotFoundInGroupException,
)
from ..utils import get_tempdir
from ..progress import utils as progress_utils

DOWNLOAD_SOURCE_TITLE = "Загрузка исходников датасета"
DATASET_SOURCE_UNPACK_TITLE = "Распаковка исходников датасета"
DATASET_CHOICE_TITLE = "Загрузка датасета `%s.%s`"
DATASET_CHOICE_UNPACK_TITLE = "Распаковка датасета `%s.%s`"
DATASET_CHOICE_TERRA_URL = f"{settings.GOOGLE_STORAGE_URL}DataSets/Numpy/"


os.makedirs(settings.DATASETS_SOURCE_DIR, exist_ok=True)
os.makedirs(settings.DATASETS_LOADED_DIR, exist_ok=True)
for item in DatasetGroupChoice:
    os.makedirs(Path(settings.DATASETS_LOADED_DIR, item.name), exist_ok=True)


def __load_from_url(progress_name: str, folder: Path, url: HttpUrl):
    folder_name = base64.b64encode(url.encode("UTF-8")).decode("UTF-8")
    dataset_path = Path(folder, folder_name)

    if dataset_path.exists():
        shutil.rmtree(dataset_path, ignore_errors=True)

    try:
        zipfile_path = progress_utils.download(
            progress_name, DOWNLOAD_SOURCE_TITLE, url
        )
        zip_destination = progress_utils.unpack(
            progress_name, DATASET_SOURCE_UNPACK_TITLE, zipfile_path
        )
        shutil.move(zip_destination, dataset_path)
        os.remove(zipfile_path.absolute())
        progress.pool(progress_name, finished=True, data=dataset_path.absolute())
    except Exception as error:
        progress.pool(progress_name, finished=True, error=error)


def __load_from_googledrive(progress_name: str, folder: Path, zipfile_path: Path):
    folder_name = zipfile_path.name[: zipfile_path.name.rfind(".")]
    dataset_path = Path(folder, folder_name)

    if dataset_path.exists():
        shutil.rmtree(dataset_path, ignore_errors=True)

    try:
        zip_destination = progress_utils.unpack(
            progress_name, DATASET_SOURCE_UNPACK_TITLE, zipfile_path
        )
        shutil.move(zip_destination, dataset_path)
        progress.pool(progress_name, finished=True, data=dataset_path.absolute())
    except Exception as error:
        progress.pool(progress_name, finished=True, error=error)


@progress.threading
def source(strict_object: SourceData):
    progress_name = "dataset_source_load"
    progress.pool.reset(progress_name, message=DOWNLOAD_SOURCE_TITLE)
    try:
        __method_name = f"__load_from_{strict_object.mode.lower()}"
        __method = getattr(sys.modules.get(__name__), __method_name, None)
        if __method:
            mode_folder = Path(settings.DATASETS_SOURCE_DIR, strict_object.mode.lower())
            os.makedirs(mode_folder, exist_ok=True)
            __method(progress_name, mode_folder, strict_object.value)
        else:
            progress.pool(
                progress_name,
                finished=True,
                error=DatasetSourceLoadUndefinedMethodException(
                    strict_object.mode.value
                ),
            )
    except Exception as error:
        progress.pool(progress_name, finished=True, error=error)


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
            error=DatasetNotFoundInGroupException(name, DatasetGroupChoice.keras.value),
        )


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
        os.remove(zipfile_path)
        data = CustomDatasetConfigData(path=zip_destination)
        zip_filepath = Path(zip_destination, "dataset.zip")
        progress_utils.unpack(
            progress_name,
            DATASET_CHOICE_UNPACK_TITLE % (DatasetGroupChoice.terra.value, name),
            zip_filepath,
            zip_destination,
        )
        os.remove(zip_filepath)
        shutil.rmtree(destination, ignore_errors=True)
        os.rename(zip_destination, destination)
        config_path = Path(destination, settings.DATASET_CONFIG)
        if config_path.is_file():
            with open(config_path) as config_ref:
                config_data = json.load(config_ref)
                config_data["group"] = DatasetGroupChoice.terra.name
            with open(config_path, "w") as config_ref:
                json.dump(config_data, config_ref)
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
            shutil.rmtree(destination, ignore_errors=True)
            progress.pool(
                progress_name,
                finished=True,
                error=DatasetNotFoundInGroupException(
                    name, DatasetGroupChoice.terra.value
                ),
            )
    except Exception as error:
        progress.pool(progress_name, finished=True, error=error)


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
        zip_destination = get_tempdir(False)
        shutil.copytree(data.path, zip_destination)
        zip_filepath = Path(zip_destination, "dataset.zip")
        progress_utils.unpack(
            progress_name,
            DATASET_CHOICE_UNPACK_TITLE % (DatasetGroupChoice.custom.value, name),
            zip_filepath,
            zip_destination,
        )
        os.remove(zip_filepath)
        shutil.rmtree(destination, ignore_errors=True)
        os.rename(zip_destination, destination)
        config_path = Path(destination, settings.DATASET_CONFIG)
        if config_path.is_file():
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
            shutil.rmtree(destination, ignore_errors=True)
            progress.pool(
                progress_name,
                finished=True,
                error=DatasetNotFoundInGroupException(
                    name, DatasetGroupChoice.custom.value
                ),
            )
    except Exception as error:
        progress.pool(progress_name, finished=True, error=error)


def _run_choice_method(
    method: Callable,
    progress_name: str,
    dataset_choice: DatasetLoadData,
    reset_model: bool = False,
    set_finished: bool = True,
):
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
            finished=set_finished,
        )
    else:
        method(
            progress_name=progress_name,
            destination=_destination,
            name=dataset_choice.alias,
            source=dataset_choice.path,
            reset_model=reset_model,
        )


def choice_no_thread(
    progress_name: str, dataset_choice: DatasetLoadData, reset_model: bool = False
):
    _method = getattr(
        sys.modules.get(__name__), f"_choice_from_{dataset_choice.group.name}", None
    )
    progress.pool.reset(
        progress_name,
        message=DATASET_CHOICE_TITLE
        % (dataset_choice.group.value, dataset_choice.alias),
        finished=False,
    )
    if _method:
        _run_choice_method(
            method=_method,
            progress_name=progress_name,
            dataset_choice=dataset_choice,
            reset_model=reset_model,
            set_finished=True,
        )
    else:
        progress.pool(
            progress_name,
            finished=True,
            error=DatasetChoiceUndefinedMethodException(dataset_choice.group.value),
        )


@progress.threading
def choice(
    progress_name: str, dataset_choice: DatasetLoadData, reset_model: bool = False
):
    choice_no_thread(progress_name, dataset_choice, reset_model)


def multiload_no_thread(
    progress_name: str, datasets_load_data: List[DatasetLoadData], **kwargs
):
    progress.pool.reset(progress_name, finished=False)
    for dataset_choice in datasets_load_data:
        _method = getattr(
            sys.modules.get(__name__), f"_choice_from_{dataset_choice.group.name}", None
        )
        if _method:
            progress.pool(
                progress_name,
                message=DATASET_CHOICE_TITLE
                % (dataset_choice.group.value, dataset_choice.alias),
                percent=0,
            )
            _run_choice_method(
                method=_method,
                progress_name=progress_name,
                dataset_choice=dataset_choice,
                reset_model=False,
                set_finished=False,
            )
        else:
            progress.pool(
                progress_name,
                finished=True,
                error=DatasetChoiceUndefinedMethodException(dataset_choice.group.value),
            )
            return
    progress.pool(
        progress_name,
        finished=True,
        percent=0,
        message="",
        data={"datasets": datasets_load_data, "kwargs": kwargs},
    )


@progress.threading
def multiload(progress_name: str, datasets_load_data: List[DatasetLoadData], **kwargs):
    multiload_no_thread(progress_name, datasets_load_data, **kwargs)
