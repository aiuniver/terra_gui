import os
import tensorflow

from pathlib import Path

from ..data.datasets.dataset import (
    DatasetLoadData,
    CustomDatasetConfigData,
    DatasetsGroupsList,
    DatasetData,
)
from ..data.datasets.creation import SourceData
from ..data.datasets.creation import FilePathSourcesList
from ..data.datasets.extra import DatasetGroupChoice

from ..data.modeling.model import ModelsGroupsList, ModelLoadData

from ..data.presets.datasets import DatasetsGroups
from ..data.presets.models import ModelsGroups
from ..data.extra import HardwareAcceleratorData, HardwareAcceleratorChoice

from ..datasets import loading as datasets_loading

from .. import ASSETS_PATH, DATASET_EXT
from .. import progress
from ..progress import ProgressData
from . import exceptions


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

    def _call_hardware_accelerator(self) -> HardwareAcceleratorData:
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
        return HardwareAcceleratorData(type=__type)

    def _call_dataset_choice(self, path: str, group: str, alias: str) -> DatasetData:
        """
        Выбор датасета
        """
        dataset_choice = DatasetLoadData(path=path, group=group, alias=alias)
        if dataset_choice.group == DatasetGroupChoice.keras:
            dataset = (
                DatasetsGroupsList(DatasetsGroups)
                .get(DatasetGroupChoice.keras)
                .datasets.get(dataset_choice.alias)
            )
            if not dataset:
                raise exceptions.UnknownKerasDatasetException(dataset_choice.alias)
            return dataset
        elif dataset_choice.group == DatasetGroupChoice.custom:
            data = CustomDatasetConfigData(
                path=Path(path, f"{dataset_choice.alias}.{DATASET_EXT}")
            )
            return DatasetData(**data.config)
        else:
            raise exceptions.DatasetGroupUndefinedMethodException(
                dataset_choice.group.value
            )

    def _call_datasets_info(self, path: str) -> DatasetsGroupsList:
        """
        Получение данных для страницы датасетов: датасеты и теги
        """
        info = DatasetsGroupsList(DatasetsGroups)
        for dirname in os.listdir(path):
            try:
                dataset_config = CustomDatasetConfigData(path=Path(path, dirname))
                info.get("custom").datasets.append(DatasetData(**dataset_config.config))
            except Exception:
                pass
        return info

    def _call_dataset_source_load(self, mode: str, value: str) -> dict:
        """
        Загрузка исходников датасета
        """
        source = SourceData(mode=mode, value=value)
        return datasets_loading.source(source)

    def _call_dataset_source_create(self, **kwargs) -> dict:
        """
        Создание датасета из исходников
        """
        print(kwargs)
        return {}

    def _call_dataset_source_load_progress(self) -> ProgressData:
        """
        Прогресс загрузки исходников датасета
        """
        return progress.pool(progress.PoolName.dataset_source_load)

    def _call_datasets_sources(self, path: str) -> FilePathSourcesList:
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
        return files

    def _call_models(self, path: str) -> ModelsGroupsList:
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
        return models

    def _call_model_load(self, value: str) -> dict:
        """
        Загрузка модели
        """
        model = ModelLoadData(value=value)
        # temporary_methods.model_load(model)

    def _call_model_load_progress(self) -> ProgressData:
        """
        Прогресс загрузки модели
        """
        return progress.pool(progress.PoolName.model_load)


agent_exchange = Exchange()
