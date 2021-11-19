import os
import shutil
import zipfile
import requests

from stat import S_IWRITE
from pathlib import Path
from tempfile import mkdtemp, NamedTemporaryFile
from pydantic.networks import HttpUrl

from ..utils import context_cwd
from . import pool


NOT_ZIP_FILE_URL = "Неверная ссылка на zip-файл «%s»"
URL_DOWNLOAD_DIVISOR = 1024


def download(progress_name: str, title: str, url: HttpUrl) -> Path:
    pool(progress_name, message=title, finished=False)
    file_destination = NamedTemporaryFile(delete=False)
    try:
        response = requests.get(url, stream=True)
        if requests.status_codes.codes.get("ok") != response.status_code:
            raise Exception(NOT_ZIP_FILE_URL % url)
        length = int(response.headers.get("Content-Length", 0))
        size = 0
        with open(file_destination.name, "wb") as file_destination_ref:
            for data in response.iter_content(chunk_size=URL_DOWNLOAD_DIVISOR):
                size += file_destination_ref.write(data)
                pool(progress_name, percent=size / length * 100)
    except requests.exceptions.ConnectionError as error:
        os.remove(file_destination.name)
        raise requests.exceptions.ConnectionError(error)
    return Path(file_destination.name)


def pack(progress_name: str, title: str, source: Path, delete=True) -> Path:
    pool.reset(progress_name, message=title, finished=False)
    zip_destination = NamedTemporaryFile(suffix=".zip", delete=delete)
    try:
        with context_cwd(source), zipfile.ZipFile(
            zip_destination.name, "w"
        ) as zipfile_ref:
            quantity = sum(list(map(lambda item: len(item[2]), os.walk("./"))))
            __num = 0
            for path, dirs, files in os.walk("./"):
                for file in files:
                    zipfile_ref.write(Path(path, file))
                    __num += 1
                    pool(progress_name, percent=__num / quantity * 100)
    except Exception as error:
        raise Exception(error)
    os.chmod(zip_destination.name, S_IWRITE)
    return zip_destination


def unpack(progress_name: str, title: str, zipfile_path: Path) -> Path:
    zip_destination: Path = mkdtemp()
    pool.reset(progress_name, message=title, finished=False)
    try:
        with zipfile.ZipFile(zipfile_path) as zipfile_ref:
            files_list = zipfile_ref.infolist()
            for _index, _member in enumerate(files_list):
                zipfile_ref.extract(_member, zip_destination)
                pool(progress_name, percent=(_index + 1) / len(files_list) * 100)
            pool(progress_name, percent=100)
    except Exception as error:
        shutil.rmtree(zip_destination, ignore_errors=True)
        raise Exception(error)
    return zip_destination
