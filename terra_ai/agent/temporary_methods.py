import sys
import zipfile
import tempfile

from tqdm import tqdm
from pathlib import Path
from pydantic.networks import HttpUrl

from ..data.datasets.creation import SourceData
from ..data.modeling.model import ModelLoadData
from .. import progress


@progress.threading
def __dataset_source_load_google_drive(path: Path):
    name = progress.PoolName.dataset_source_load
    progress.pool.reset(name, message="Загрузка исходников")
    destination = tempfile.mkdtemp(prefix="dataset-source-")
    try:
        with zipfile.ZipFile(path) as file_ref:
            __tqdm = tqdm(file_ref.infolist())
            for member in __tqdm:
                file_ref.extract(member, destination)
                progress.pool(name, percent=__tqdm.n / __tqdm.total * 100)
            progress.pool(
                name,
                percent=__tqdm.n / __tqdm.total * 100,
                data={"path": destination},
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


@progress.threading
def model_load(model: ModelLoadData):
    name = progress.PoolName.model_load
    progress.pool.reset(name, message="Загрузка модели")
    destination = tempfile.mkdtemp(prefix="model-")
    try:
        with zipfile.ZipFile(model.value) as file_ref:
            __tqdm = tqdm(file_ref.infolist())
            for member in __tqdm:
                file_ref.extract(member, destination)
                progress.pool(name, percent=__tqdm.n / __tqdm.total * 100)
            progress.pool(
                name,
                percent=__tqdm.n / __tqdm.total * 100,
                data={"path": destination},
                finished=True,
            )
    except Exception as error:
        progress.pool(name, error=str(error))
