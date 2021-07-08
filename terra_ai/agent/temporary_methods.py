import sys
import zipfile

from tqdm import tqdm
from pathlib import Path
from pydantic.networks import HttpUrl

from ..data.datasets.creation import SourceData
from .. import progress


@progress.threading
def __dataset_source_load_google_drive(path: Path):
    name = progress.PoolName.dataset_source_load
    progress.pool.reset(name, message="Загрузка исходников")
    try:
        with zipfile.ZipFile(path) as file_ref:
            __tqdm = tqdm(file_ref.infolist())
            for member in __tqdm:
                file_ref.extract(member, ".")
                progress.pool(name, percent=__tqdm.n / __tqdm.total * 100)
            progress.pool(
                name,
                percent=__tqdm.n / __tqdm.total * 100,
                data={"my": "dict"},
                finished=True,
            )
    except Exception as error:
        progress.pool(name, error=str(error))


@progress.threading
def __dataset_source_load_url(url: HttpUrl):
    print("URL:", url)
    print("URL:", "stop")


def dataset_source_load(source: SourceData):
    method_name = f"__dataset_source_load_{source.mode}"
    method = getattr(sys.modules.get(__name__), method_name)
    method(source.value)
