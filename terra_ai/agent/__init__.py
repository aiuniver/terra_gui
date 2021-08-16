import os
import json
import tensorflow

from typing import Any
from pathlib import Path
from transliterate import slugify

from ..data.datasets.dataset import (
    DatasetLoadData,
    CustomDatasetConfigData,
    DatasetsGroupsList,
    DatasetData,
)
from ..data.datasets.creation import SourceData, CreationData
from ..data.datasets.creation import FilePathSourcesList
from ..data.datasets.extra import DatasetGroupChoice

from ..data.modeling.model import ModelsGroupsList, ModelLoadData, ModelDetailsData
from ..data.modeling.extra import ModelGroupChoice

from ..data.presets.datasets import DatasetsGroups
from ..data.presets.models import ModelsGroups
from ..data.extra import (
    HardwareAcceleratorData,
    HardwareAcceleratorChoice,
    FileManagerItem,
)

from ..datasets import loading as datasets_loading
from ..deploy import loading as deploy_loading

from .. import settings, progress
from . import exceptions


class Exchange:
    def __call__(self, method: str, *args, **kwargs) -> Any:
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
        datasets_loading.choice(DatasetLoadData(path=path, group=group, alias=alias))

    def _call_dataset_choice_progress(self) -> progress.ProgressData:
        """
        Прогресс выбора датасета
        """
        return progress.pool(progress.PoolName.dataset_choice)

    def _call_datasets_info(self, path: str) -> DatasetsGroupsList:
        """
        Получение данных для страницы датасетов: датасеты и теги
        """
        info = DatasetsGroupsList(DatasetsGroups)
        for dirname in os.listdir(path):
            try:
                dataset_config = CustomDatasetConfigData(path=Path(path, dirname))
                info.get(DatasetGroupChoice.custom.name).datasets.append(
                    DatasetData(**dataset_config.config)
                )
            except Exception:
                pass
        return info

    def _call_dataset_source_load(self, mode: str, value: str):
        """
        Загрузка исходников датасета
        """
        datasets_loading.source(SourceData(mode=mode, value=value))

    def _call_dataset_source_load_progress(self) -> progress.ProgressData:
        """
        Прогресс загрузки исходников датасета
        """
        progress_data = progress.pool(progress.PoolName.dataset_source_load)
        if progress_data.finished and progress_data.data:
            __path = progress_data.data.absolute()
            progress_data.data = {
                "file_manager": (FileManagerItem(path=__path).native().get("children")),
                "source_path": __path,
            }
        else:
            progress.data = []
        return progress_data

    def _call_dataset_create(self, **kwargs) -> dict:
        """
        Создание датасета из исходников
        """
        kwargs.update(
            {
                "alias": slugify(kwargs.get("name")),
            }
        )
        creation = CreationData(**kwargs)
        print(creation)
        return {}

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
        models_path = Path(settings.ASSETS_PATH, "models")
        for filename in os.listdir(models_path):
            try:
                models.get(ModelGroupChoice.preset.name).models.append(
                    {"value": Path(models_path, filename)}
                )
            except Exception:
                pass
        models.get(ModelGroupChoice.preset.name).models.sort(
            key=lambda item: item.label
        )
        for filename in os.listdir(path):
            try:
                models.get(ModelGroupChoice.custom.name).models.append(
                    {"value": Path(path, filename)}
                )
            except Exception:
                pass
        models.get(ModelGroupChoice.custom.name).models.sort(
            key=lambda item: item.label
        )
        return models

    def _call_model_get(self, value: str) -> ModelDetailsData:
        """
        Получение модели
        """
        data = ModelLoadData(value=value)
        with open(data.value.absolute(), "r") as config_ref:
            config = json.load(config_ref)
            return ModelDetailsData(**config)

    def _call_model_update(self, model: dict, **kwargs) -> ModelDetailsData:
        """
        Обновление модели
        """
        if len(kwargs.keys()):
            model.update(kwargs)
        return ModelDetailsData(**model)

    def _call_model_layer_save(self, model: dict, **kwargs) -> ModelDetailsData:
        """
        Обновление слоя модели
        """
        model = ModelDetailsData(**model)
        if len(kwargs.keys()):
            model.layers.append(kwargs)
        return model

    def _call_deploy_upload(self, source: Path, **kwargs):
        """
        Деплой: загрузка
        """
        deploy_loading.upload(source, kwargs)

    def _call_deploy_upload_progress(self) -> progress.ProgressData:
        """
        Деплой: прогресс загрузки
        """
        return progress.pool(progress.PoolName.deploy_upload)


agent_exchange = Exchange()
