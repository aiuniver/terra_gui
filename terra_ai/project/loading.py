import json
import os
import shutil

from pathlib import Path

from terra_ai import progress, settings
from terra_ai.utils import get_tempfile
from terra_ai.data.datasets.dataset import DatasetLoadData
from terra_ai.datasets import loading as datasets_loading
from terra_ai.progress import utils as progress_utils


PROJECT_LOAD_TITLE = "Загрузка проекта %s"
PROJECT_LOAD_NAME = "project_load"


@progress.threading
def load(dataset_path: Path, source: Path, destination: Path):
    progress.pool.reset(PROJECT_LOAD_NAME, finished=False)
    try:
        zip_filepath = get_tempfile(False)
        shutil.copy(source, zip_filepath)
        unpacked = progress_utils.unpack(
            PROJECT_LOAD_NAME, PROJECT_LOAD_TITLE % source.name, zip_filepath
        )
        os.remove(zip_filepath)
        shutil.rmtree(destination, ignore_errors=True)
        shutil.move(unpacked, destination)

        with open(Path(destination, settings.DATASET_CONFIG)) as config_ref:
            config = json.load(config_ref)
            dataset = config.get("dataset")
            if dataset:
                dataset_info = {
                    "alias": dataset.get("alias"),
                    "group": dataset.get("group"),
                }
            else:
                dataset_info = config.get("dataset_info")
            if dataset_info:
                datasets_loading.choice(
                    PROJECT_LOAD_NAME,
                    DatasetLoadData(path=dataset_path, **dataset_info),
                )
            else:
                progress.pool.reset(PROJECT_LOAD_NAME, finished=True)

    except Exception as error:
        progress.pool(PROJECT_LOAD_NAME, finished=True, error=error)
