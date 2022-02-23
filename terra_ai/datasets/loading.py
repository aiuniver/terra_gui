import os
import sys
import json
import shutil
import base64

from pathlib import Path
from typing import List, Callable
from pydantic.networks import HttpUrl

from terra_ai import progress, settings
from terra_ai.data.datasets.creation import SourceData
from terra_ai.data.datasets.dataset import (
    DatasetData,
    DatasetLoadData,
    DatasetCommonGroupList,
    DatasetCommonPathsData,
)
from terra_ai.data.datasets.extra import DatasetGroupChoice
from terra_ai.exceptions.datasets import (
    DatasetSourceLoadUndefinedMethodException,
    DatasetChoiceUndefinedMethodException,
)
from terra_ai.data.training.extra import ArchitectureChoice
from terra_ai.utils import get_tempdir
from terra_ai.progress import utils as progress_utils

DOWNLOAD_SOURCE_TITLE = "Загрузка исходников датасета"
DATASET_SOURCE_UNPACK_TITLE = "Распаковка исходников датасета"
DATASET_CHOICE_TITLE = "Загрузка датасета `%s.%s.%s`"
DATASET_CHOICE_UNPACK_TITLE = "Распаковка датасета `%s.%s.%s`"
DATASET_CHOICE_TERRA_URL = f"{settings.GOOGLE_STORAGE_URL}DataSets/Numpy_new/"


os.makedirs(settings.DATASETS_SOURCE_DIR, exist_ok=True)
os.makedirs(settings.DATASETS_LOADED_DIR, exist_ok=True)
for item in list(DatasetGroupChoice):
    os.makedirs(Path(settings.DATASETS_LOADED_DIR, item.name), exist_ok=True)


def __load_from_url(progress_name: str, folder: Path, url: HttpUrl, extra: dict):
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
        extra.get("source").update({"path": dataset_path.absolute()})
        progress.pool(
            progress_name,
            finished=True,
            data={"path": dataset_path.absolute(), "extra": extra},
        )
    except Exception as error:
        progress.pool(progress_name, finished=True, error=error)


def __load_from_googledrive(
    progress_name: str,
    folder: Path,
    folder_name: str,
    extra: dict,
):
    zipfile_path = Path(settings.TERRA_PATH.sources, f"{folder_name}.zip")
    dataset_path = Path(folder, folder_name)

    if dataset_path.exists():
        shutil.rmtree(dataset_path, ignore_errors=True)

    try:
        zip_destination = progress_utils.unpack(
            progress_name, DATASET_SOURCE_UNPACK_TITLE, zipfile_path
        )
        shutil.move(zip_destination, dataset_path)
        extra.get("source").update({"path": dataset_path.absolute()})
        progress.pool(
            progress_name,
            finished=True,
            data={"path": dataset_path.absolute(), "extra": extra},
        )
    except Exception as error:
        progress.pool(progress_name, finished=True, error=error)


@progress.threading
def source(strict_object: SourceData, extra: dict):
    extra.update({"source": strict_object.dict()})
    progress_name = "dataset_source_load"
    progress.pool.reset(progress_name, message=DOWNLOAD_SOURCE_TITLE)
    try:
        __method_name = f"__load_from_{strict_object.mode.lower()}"
        __method = getattr(sys.modules.get(__name__), __method_name, None)
        if __method:
            mode_folder = Path(settings.DATASETS_SOURCE_DIR, strict_object.mode.lower())
            os.makedirs(mode_folder, exist_ok=True)
            __method(progress_name, mode_folder, strict_object.value, extra)
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
    version: str,
    reset_model: bool,
    **kwargs,
):
    try:
        dataset_config = (
            DatasetCommonGroupList().get(DatasetGroupChoice.keras).datasets.get(name)
        )
        if not dataset_config:
            raise Exception("Dataset not found")
        version_config = None
        for item in dataset_config.versions:
            if item.get("alias") == version:
                version_config = {**item}
                break
        if not version_config:
            raise Exception("Dataset not found")
        shutil.rmtree(str(destination), ignore_errors=True)
        os.makedirs(str(destination), exist_ok=True)
        dataset_config = dataset_config.native()
        dataset_config.update(
            {
                "version": version_config,
                "path": destination,
                "group": DatasetGroupChoice.keras,
            }
        )
        dataset = DatasetData(**dataset_config)
        # dataset_config_path = Path(destination, settings.DATASET_CONFIG)
        # with open(dataset_config_path, "w") as dataset_config_path_ref:
        #     json.dump(dataset.native(), dataset_config_path_ref)
        progress.pool(
            progress_name,
            percent=100,
            data={"dataset": dataset, "reset_model": reset_model},
            finished=True,
        )
    except Exception as error:
        shutil.rmtree(str(destination), ignore_errors=True)
        progress.pool(progress_name, finished=True, error=error)


def _choice_from_terra(
    progress_name: str,
    destination: Path,
    name: str,
    version: str,
    reset_model: bool,
    **kwargs,
):
    source = None
    zip_destination = get_tempdir(False)
    try:
        zipfile_path = progress_utils.download(
            progress_name,
            DATASET_CHOICE_TITLE % (DatasetGroupChoice.terra.value, name, version),
            f"{DATASET_CHOICE_TERRA_URL}{name}.zip",
        )
        source = progress_utils.unpack(
            progress_name,
            DATASET_CHOICE_UNPACK_TITLE
            % (DatasetGroupChoice.terra.value, name, version),
            zipfile_path,
        )
        os.remove(str(zipfile_path))
        dataset_path = DatasetCommonPathsData(basepath=source)
        shutil.copytree(
            str(
                Path(dataset_path.versions, f"{version}.{settings.DATASET_VERSION_EXT}")
            ),
            str(zip_destination),
        )
        zip_filepath = Path(zip_destination, "version.zip")
        progress_utils.unpack(
            progress_name,
            DATASET_CHOICE_UNPACK_TITLE
            % (DatasetGroupChoice.terra.value, name, version),
            zip_filepath,
            zip_destination,
        )
        os.remove(str(zip_filepath))
        shutil.rmtree(str(destination), ignore_errors=True)
        os.makedirs(str(destination), exist_ok=True)
        os.rename(str(zip_destination), str(destination))
        with open(Path(dataset_path.basepath, settings.DATASET_CONFIG)) as config_ref:
            dataset_config = json.load(config_ref)
        version_path = Path(destination, settings.DATASET_VERSION_CONFIG)
        with open(version_path) as version_ref:
            version_config = json.load(version_ref)
        os.remove(str(version_path))
        shutil.rmtree(str(source), ignore_errors=True)
        dataset_config.update(
            {
                "version": version_config,
                "path": destination,
                "group": DatasetGroupChoice.terra,
            }
        )
        dataset = DatasetData(**dataset_config)
        dataset_config_path = Path(destination, settings.DATASET_CONFIG)
        with open(dataset_config_path, "w") as dataset_config_path_ref:
            json.dump(dataset.native(), dataset_config_path_ref)
        progress.pool(
            progress_name,
            percent=100,
            data={"dataset": dataset, "reset_model": reset_model},
            finished=True,
        )
    except Exception as error:
        if source:
            shutil.rmtree(str(source), ignore_errors=True)
        shutil.rmtree(str(zip_destination), ignore_errors=True)
        shutil.rmtree(str(destination), ignore_errors=True)
        progress.pool(progress_name, finished=True, error=error)


def _choice_from_custom(
    progress_name: str,
    destination: Path,
    name: str,
    version: str,
    source: Path,
    reset_model: bool,
    **kwargs,
):
    try:
        zip_destination = get_tempdir(False)
        dataset_path = DatasetCommonPathsData(
            basepath=Path(source, f"{name}.{settings.DATASET_EXT}")
        )
        shutil.copytree(
            str(
                Path(dataset_path.versions, f"{version}.{settings.DATASET_VERSION_EXT}")
            ),
            str(zip_destination),
        )
        zip_filepath = Path(zip_destination, "version.zip")
        progress_utils.unpack(
            progress_name,
            DATASET_CHOICE_UNPACK_TITLE
            % (DatasetGroupChoice.custom.value, name, version),
            zip_filepath,
            zip_destination,
        )
        os.remove(str(zip_filepath))
        shutil.rmtree(str(destination), ignore_errors=True)
        os.makedirs(str(destination), exist_ok=True)
        os.rename(str(zip_destination), str(destination))
        with open(Path(dataset_path.basepath, settings.DATASET_CONFIG)) as config_ref:
            dataset_config = json.load(config_ref)
        version_path = Path(destination, settings.DATASET_VERSION_CONFIG)
        with open(version_path) as version_ref:
            version_config = json.load(version_ref)
        os.remove(str(version_path))
        dataset_config.update(
            {
                "version": version_config,
                "path": destination,
                "group": DatasetGroupChoice.custom,
            }
        )
        dataset = DatasetData(**dataset_config)
        dataset_config_path = Path(destination, settings.DATASET_CONFIG)
        with open(dataset_config_path, "w") as dataset_config_path_ref:
            json.dump(dataset.native(), dataset_config_path_ref)
        progress.pool(
            progress_name,
            percent=100,
            data={"dataset": dataset, "reset_model": reset_model},
            finished=True,
        )
    except Exception as error:
        shutil.rmtree(str(destination), ignore_errors=True)
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
        dataset_choice.version,
    )
    config_path = Path(_destination, settings.DATASET_CONFIG)
    if config_path.is_file():
        with open(config_path) as config_path_ref:
            config = json.load(config_path_ref)
        dataset = DatasetData(**config)
        progress.pool(
            progress_name,
            percent=100,
            data={"dataset": dataset, "reset_model": reset_model},
            finished=set_finished,
        )
    else:
        method(
            progress_name=progress_name,
            destination=_destination,
            name=dataset_choice.alias,
            version=dataset_choice.version,
            source=dataset_choice.path,
            reset_model=reset_model,
        )


def choice_no_thread(
    progress_name: str, dataset_choice: DatasetLoadData, reset_model: bool = False
):
    _method = getattr(
        sys.modules.get(__name__), f"_choice_from_{dataset_choice.group.name}", None
    )
    message = DATASET_CHOICE_TITLE % (
        dataset_choice.group.value,
        dataset_choice.alias,
        dataset_choice.version,
    )
    progress.pool.reset(progress_name, message=message, finished=False)
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
                % (
                    dataset_choice.group.value,
                    dataset_choice.alias,
                    dataset_choice.version,
                ),
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
