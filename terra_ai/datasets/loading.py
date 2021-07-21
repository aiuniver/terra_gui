import os
import pathlib
import shutil
from typing import Any

import requests

from tempfile import mkdtemp
from tqdm.notebook import tqdm

from .data import DatasetArchives
# from terra_ai import out_exchange
from ..data.datasets.creation import SourceData


class Dataloader:

    def __init__(self):
        self.save_path = mkdtemp()
        self.trds_path = '/content/drive/MyDrive/TerraAI/datasets'
        self._file_folder = None

    def _get_zipfiles(self) -> list:

        return os.listdir(os.path.join(self.trds_path, 'sources'))

    @property
    def file_folder(self) -> str:
        return self._file_folder

    @file_folder.setter
    def file_folder(self, name):
        self._file_folder = pathlib.Path(os.path.join(self.save_path, name))

    def unzip(self, zip_name: str):

        file_path = pathlib.Path(os.path.join(self.file_folder, 'tmp', zip_name))
        temp_folder = os.path.join(self.file_folder, 'tmp')
        os.makedirs(temp_folder, exist_ok=True)
        shutil.unpack_archive(file_path, self.file_folder)
        shutil.rmtree(temp_folder, ignore_errors=True)

    def download(self, link: str, file_name: str):
        os.makedirs(self.file_folder, exist_ok=True)
        os.makedirs(os.path.join(self.file_folder, 'tmp'), exist_ok=True)

        resp = requests.get(link, stream=True)
        total = int(resp.headers.get('content-length', 0))
        idx = 0
        with open(os.path.join(self.file_folder, 'tmp', file_name), 'wb') as out_file, tqdm(
                desc=f"Загрузка архива {file_name}", total=total, unit='iB', unit_scale=True,
                unit_divisor=1024) as progress_bar:
            for data in resp.iter_content(chunk_size=1024):
                size = out_file.write(data)
                progress_bar.update(size)
                idx += size
                if idx % 143360 == 0 or idx == progress_bar.total:
                    progress_bar_status = (progress_bar.desc, str(round(idx / progress_bar.total, 2)),
                                           f'{str(round(progress_bar.last_print_t - progress_bar.start_t, 2))} сек.')
                    # if idx == progress_bar.total:
                    #     out_exchange.print_progress_bar(progress_bar_status, stop_flag=True)
                    # else:
                    #     out_exchange.print_progress_bar(progress_bar_status)

    def load_data(self, strict_object: SourceData):
        __method_name = f'load_from_{strict_object.mode.lower()}'
        __method = getattr(self, __method_name, None)
        return __method(strict_object.value)

    def load_from_terra(self, name: str):
        file_name = DatasetArchives[name]
        self.file_folder = name
        link = 'https://storage.googleapis.com/terra_ai/DataSets/Numpy/' + file_name
        self.download(link, file_name)
        if 'zip' in file_name:
            self.unzip(file_name)
        return {"message": f"Файлы скачаны в директорию {self.file_folder}"}

    def load_from_url(self, link: str):
        file_name = link.split('/')[-1]
        self.file_folder = file_name[:file_name.rfind('.')] if '.' in file_name else file_name
        self.download(link, file_name)
        if 'zip' in file_name or 'zip' in link:
            self.unzip(file_name)
        return {"message": f"Файлы скачаны в директорию {self.file_folder}"}

    def load_from_googledrive(self, filepath: str):
        zip_name = str(filepath).split('/')[-1]
        name = zip_name[:zip_name.rfind('.')]
        self.file_folder = name
        shutil.unpack_archive(filepath, self.file_folder)
        return {"message": f"Файлы скачаны в директорию {self.file_folder}"}
