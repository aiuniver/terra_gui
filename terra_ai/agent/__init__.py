import os
import json
import tensorflow

from pathlib import Path
from transliterate import slugify

from ..data.datasets.dataset import CustomDataset, DatasetsGroupsList
from ..data.datasets.creation import SourceData
from ..data.datasets.creation import FilePathSourcesList

from ..data.modeling.model import ModelsGroupsList, ModelLoadData

from ..data.presets.datasets import DatasetsGroups
from ..data.presets.models import ModelsGroups
from ..data.extra import HardwareAcceleratorData, HardwareAcceleratorChoice

from .. import ASSETS_PATH
from .. import progress
from . import exceptions
from . import temporary_methods
from ..datasets import loader


class Exchange:
    def __call__(self, method: str, *args, **kwargs) -> dict:
        # Получаем метод для вызова
        __method_name = f"_call_{method}"
        __method = getattr(self, __method_name, None)

        # Проверяем существует ли метод
        if __method is None:
            raise exceptions.CallMethodNotFoundException(self.__class__, __method_name)

        # Проверяем является ли методом вызываемым
        if not callable(__method):
            raise exceptions.MethodNotCallableException(self.__class__, __method_name)

        # Вызываем метод
        return __method(**kwargs)

    @property
    def is_colab(self) -> bool:
        return "COLAB_GPU" in os.environ.keys()

    def _call_hardware_accelerator(self) -> dict:
        device_name = tensorflow.test.gpu_device_name()
        if device_name != "/device:GPU:0":
            if self.is_colab:
                try:
                    tensorflow.distribute.cluster_resolver.TPUClusterResolver()
                    __type = HardwareAcceleratorChoice.TPU
                except ValueError:
                    __type = HardwareAcceleratorChoice.CPU
            else:
                __type = HardwareAcceleratorChoice.CPU
        else:
            __type = HardwareAcceleratorChoice.GPU
        hardware = HardwareAcceleratorData(type=__type)
        return hardware.native()

    def _call_datasets_info(self, path: str) -> dict:
        """
        Получение данных для страницы датасетов: датасеты и теги
        """
        info = DatasetsGroupsList(DatasetsGroups)
        for dirname in os.listdir(path):
            try:
                dataset = CustomDataset(path=Path(path, dirname))
                alias = slugify(dataset.config.get("name"), language_code="ru")
                info.get("custom").datasets.append(
                    {
                        "alias": alias,
                        "name": dataset.config.get("name"),
                        "date": dataset.config.get("date"),
                        "size": {"value": dataset.config.get("size")},
                        "tags": dataset.tags.dict(),
                    }
                )
            except Exception:
                pass
        return info.native()

    def _call_dataset_source_load(self, mode: str, value: str):
        """
        Загрузка исходников датасета
        """
        source = SourceData(mode=mode, value=value)
        data = loader.load_data(source)
        return data

    def _call_dataset_source_load_progress(self) -> dict:
        """
        Прогресс загрузки исходников датасета
        """
        return progress.pool(progress.PoolName.dataset_source_load).native()

    def _call_datasets_sources(self, path: str) -> list:
        """
        Получение списка исходников датасетов
        """
        files = FilePathSourcesList()
        for filename in os.listdir(path):
            filepath = Path(path, filename)
            try:
                files.append({"value": filepath})
            except Exception:
                pass
        files.sort(key=lambda item: item.label)
        return files.native()

    def _call_models(self, path: str) -> list:
        """
        Получение списка моделей
        """
        models = ModelsGroupsList(ModelsGroups)
        models_path = Path(ASSETS_PATH, "models")
        for filename in os.listdir(models_path):
            try:
                models.get("preset").models.append(
                    {"value": Path(models_path, filename)}
                )
            except Exception:
                pass
        models.get("preset").models.sort(key=lambda item: item.label)
        for filename in os.listdir(path):
            try:
                models.get("custom").models.append({"value": Path(path, filename)})
            except Exception:
                pass
        models.get("custom").models.sort(key=lambda item: item.label)
        return models.native()

    def _call_model_load(self, value: str):
        """
        Загрузка модели
        """
        model = ModelLoadData(value=value)
        temporary_methods.model_load(model)

    def _call_model_load_progress(self) -> dict:
        """
        Прогресс загрузки модели
        """
        return progress.pool(progress.PoolName.model_load).native()


agent_exchange = Exchange()
