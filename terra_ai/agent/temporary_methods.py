import sys
import zipfile

from tqdm.asyncio import tqdm
from time import sleep
from pathlib import Path
from pydantic.networks import HttpUrl

from ..data.datasets.creation import SourceData
from ..threading import threading


@threading
def __dataset_load_google_drive(path: Path, progress: callable = None):
    with zipfile.ZipFile(path) as file_ref:
        __tqdm = tqdm(file_ref.infolist())
        for member in __tqdm:
            try:
                file_ref.extract(member, ".")
            except zipfile.error:
                pass
            progress(__tqdm.n / __tqdm.total * 100)
        progress(__tqdm.n / __tqdm.total * 100)


@threading
def __dataset_load_url(url: HttpUrl, progress: callable = None):
    sleep(3)
    print("URL:", url)
    sleep(3)
    print("URL:", "stop")


def dataset_load(source: SourceData, progress: callable = None):
    method_name = f"__dataset_load_{source.mode}"
    method = getattr(sys.modules.get(__name__), method_name)
    method(source.value, progress)
