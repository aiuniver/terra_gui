import os
import sys
import shutil
import zipfile
import requests

from tqdm import tqdm
from pathlib import Path
from tempfile import mkdtemp, gettempdir

from .data import DatasetArchives

from .. import progress
from ..data.datasets.creation import SourceData
from ..exceptions.datasets import DatasetSourceLoadUndefinedMethodException


DATASETS_SOURCE_DIR = Path(gettempdir(), "terraai", "datasets")

os.makedirs(DATASETS_SOURCE_DIR, exist_ok=True)


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
#
#     def load_from_url(self, link: str):
#         file_name = link.split("/")[-1]
#         self.file_folder = (
#             file_name[: file_name.rfind(".")] if "." in file_name else file_name
#         )
#         self.download(link, file_name)
#         if "zip" in file_name or "zip" in link:
#             self.unzip(file_name)
#         return {"message": f"Файлы скачаны в директорию {self.file_folder}"}


@progress.threading
def __load_from_googledrive(folder: Path, zipfile_path: Path):
    # Сброс прогресс-бара
    progress_name = progress.PoolName.dataset_source_load
    progress.pool.reset(progress_name, message="Загрузка датасета")

    # Получение папки датасета
    folder_name = zipfile_path.name[: zipfile_path.name.rfind(".")]
    dataset_path = Path(folder, folder_name)

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

    # Создаем временную папку, в которую будем распаковывать архив исходников,
    # и запускаем распаковку
    tmp_destination = mkdtemp()
    try:
        with zipfile.ZipFile(zipfile_path) as zipfile_ref:
            __tqdm = tqdm(zipfile_ref.infolist())
            for member in __tqdm:
                zipfile_ref.extract(member, tmp_destination)
                progress.pool(progress_name, percent=__tqdm.n / __tqdm.total * 100)
            shutil.move(tmp_destination, dataset_path)
            progress.pool(
                progress_name,
                percent=__tqdm.n / __tqdm.total * 100,
                data=dataset_path.absolute(),
                finished=True,
            )
    except Exception as error:
        shutil.rmtree(tmp_destination, ignore_errors=True)
        progress.pool(progress_name, error=str(error))


def load(strict_object: SourceData):
    __method_name = f"__load_from_{strict_object.mode.lower()}"
    __method = getattr(sys.modules.get(__name__), __method_name, None)
    if __method:
        mode_folder = Path(DATASETS_SOURCE_DIR, strict_object.mode.lower())
        os.makedirs(mode_folder, exist_ok=True)
        __method(mode_folder, strict_object.value)
    else:
        raise DatasetSourceLoadUndefinedMethodException(strict_object.mode.value)
