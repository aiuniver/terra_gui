import sys
import zipfile

from tqdm import tqdm
from time import sleep
from pathlib import Path
from pydantic.networks import HttpUrl

from ..data.datasets.creation import SourceData
from ..threading import threading


@threading
def __dataset_load_google_drive(path: Path):
    print("Google drive:", "start")
    with zipfile.ZipFile(path) as file_ref:
        for member in tqdm(file_ref.infolist(), desc="Extracting "):
            try:
                file_ref.extract(member, ".")
            except zipfile.error as error:
                print("ERROR:", error)
    print("Google drive:", "stop")


@threading
def __dataset_load_url(url: HttpUrl):
    sleep(3)
    print("URL:", url)
    sleep(3)
    print("URL:", "stop")


def dataset_load(source: SourceData):
    method_name = f"__dataset_load_{source.mode}"
    method = getattr(sys.modules.get(__name__), method_name)
    method(source.value)
