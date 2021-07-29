import os
import shutil

from .. import progress
from ..data.modeling.model import ModelLoadData
from ..progress import utils as progress_utils

MODEL_UNPACK_TITLE = "Распаковка модели"


@progress.threading
def model(data: ModelLoadData):
    # Имя прогресс-бара
    progress_name = progress.PoolName.model_load

    # Запускаем загрузку
    try:
        zip_destination = progress_utils.unpack(
            progress_name, MODEL_UNPACK_TITLE, data.value
        )
        shutil.rmtree(data.destination)
        os.rename(zip_destination, data.destination)
        progress.pool(progress_name, data=data.destination.absolute(), finished=True)
    except Exception as error:
        progress.pool(progress_name, error=str(error))
