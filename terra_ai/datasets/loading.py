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
    UnknownKerasDatasetException,
    UnknownCustomDatasetException,
)
from ..progress import utils as progress_utils

DOWNLOAD_SOURCE_TITLE = "Загрузка исходников датасета"
DATASET_SOURCE_UNPACK_TITLE = "Распаковка исходников датасета"
DATASET_CHOICE_TITLE = "Загрузка датасета %s.%s"
DATASETS_SOURCE_DIR = Path(settings.TMP_DIR, "datasets_sources")
DATASETS_CHOICE_DIR = Path(settings.TMP_DIR, "datasets_choices")

os.makedirs(DATASETS_SOURCE_DIR, exist_ok=True)
os.makedirs(DATASETS_CHOICE_DIR, exist_ok=True)


# class Dataloader:
#     def __init__(self):
#         self.save_path = mkdtemp()
#         self._file_folder = None
#
#     @property
#     def file_folder(self) -> str:
#         return self._file_folder
#
#     @file_folder.setter
#     def file_folder(self, name):
#         self._file_folder = Path(os.path.join(self.save_path, name))
#
#     def unzip(self, zip_name: str):
#
#         file_path = Path(os.path.join(self.file_folder, "tmp", zip_name))
#         temp_folder = os.path.join(self.file_folder, "tmp")
#         os.makedirs(temp_folder, exist_ok=True)
#         shutil.unpack_archive(file_path, self.file_folder)
#         shutil.rmtree(temp_folder, ignore_errors=True)
#
#     def download(self, link: str, file_name: str):
#         os.makedirs(self.file_folder, exist_ok=True)
#         os.makedirs(os.path.join(self.file_folder, "tmp"), exist_ok=True)
#
#         resp = requests.get(link, stream=True)
#         total = int(resp.headers.get("content-length", 0))
#         idx = 0
#         with open(
#             os.path.join(self.file_folder, "tmp", file_name), "wb"
#         ) as out_file, tqdm(
#             desc=f"Загрузка архива {file_name}",
#             total=total,
#             unit="iB",
#             unit_scale=True,
#             unit_divisor=1024,
#         ) as progress_bar:
#             for data in resp.iter_content(chunk_size=1024):
#                 size = out_file.write(data)
#                 progress_bar.update(size)
#                 idx += size
#                 if idx % 143360 == 0 or idx == progress_bar.total:
#                     progress_bar_status = (
#                         progress_bar.desc,
#                         str(round(idx / progress_bar.total, 2)),
#                         f"{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.",
#                     )
#                     # if idx == progress_bar.total:
#                     #     out_exchange.print_progress_bar(progress_bar_status, stop_flag=True)
#                     # else:
#                     #     out_exchange.print_progress_bar(progress_bar_status)
#
#     def load_data(self, strict_object: SourceData):
#         __method_name = f"load_from_{strict_object.mode.lower()}"
#         __method = getattr(self, __method_name, None)
#         return __method(strict_object.value)
#
#     def load_from_terra(self, name: str):
#         file_name = DatasetArchives[name]
#         self.file_folder = name
#         link = "https://storage.googleapis.com/terra_ai/DataSets/Numpy/" + file_name
#         self.download(link, file_name)
#         if "zip" in file_name:
#             self.unzip(file_name)
#         return {"message": f"Файлы скачаны в директорию {self.file_folder}"}


@progress.threading
def __load_from_url(folder: Path, url: HttpUrl):
    # Получение папки датасета
    folder_name = base64.b64encode(url.encode("UTF-8")).decode("UTF-8")
    dataset_path = Path(folder, folder_name)

    # Сброс прогресс-бара
    progress_name = progress.PoolName.dataset_source_load

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
    progress_name = progress.PoolName.dataset_source_load

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
def __choice_from_keras(name: str, **kwargs):
    # Имя прогресс-бара
    progress_name = progress.PoolName.dataset_choice
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
        progress.pool(progress_name, percent=100, data=dataset, finished=True)
    else:
        progress.pool(progress_name, error=str(UnknownKerasDatasetException(name)))


@progress.threading
def __choice_from_custom(name: str, source: Path, **kwargs):
    # Имя прогресс-бара
    progress_name = progress.PoolName.dataset_choice
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
        dataset = DatasetData(**data.config)
        if dataset:
            progress.pool(progress_name, percent=100, data=dataset, finished=True)
        else:
            progress.pool(progress_name, error=str(UnknownCustomDatasetException(name)))
    except ValidationError as error:
        for item in error.args[0]:
            if isinstance(item.exc, PathNotExistsError):
                progress.pool(
                    progress_name, error=str(UnknownCustomDatasetException(name))
                )
                return
        progress.pool(progress_name, error=str(error))


@progress.threading
def __choice_from_terra(name: str, **kwargs):
    # Имя прогресс-бара
    progress_name = progress.PoolName.dataset_choice
    progress.pool.reset(
        progress_name,
        message=DATASET_CHOICE_TITLE % (DatasetGroupChoice.terra.value, name),
        finished=False,
    )

    # Выбор датасета
    print(name)


def choice(dataset_choice: DatasetLoadData):
    __method_name = f"__choice_from_{dataset_choice.group.lower()}"
    __method = getattr(sys.modules.get(__name__), __method_name, None)
    if __method:
        choice_folder = Path(DATASETS_CHOICE_DIR, dataset_choice.group.lower())
        os.makedirs(choice_folder, exist_ok=True)
        __method(
            name=dataset_choice.alias, folder=choice_folder, source=dataset_choice.path
        )
    else:
        raise DatasetChoiceUndefinedMethodException(dataset_choice.group.value)
