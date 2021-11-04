import os
import json
import shutil
import pynvml
import tensorflow

from pathlib import Path
from typing import Any, NoReturn

from . import exceptions as agent_exceptions
from . import utils as agent_utils
from .. import settings, progress
from ..exceptions import tensor_flow as tf_exceptions
from ..data.datasets.creation import FilePathSourcesList
from ..data.datasets.creation import SourceData, CreationData
from ..data.datasets.dataset import (
    DatasetLoadData,
    CustomDatasetConfigData,
    DatasetsGroupsList,
    DatasetData,
)
from ..data.datasets.extra import DatasetGroupChoice
from ..data.extra import (
    HardwareAcceleratorData,
    HardwareAcceleratorChoice,
    FileManagerItem,
)
from ..data.modeling.extra import ModelGroupChoice
from ..data.modeling.model import ModelsGroupsList, ModelLoadData, ModelDetailsData
from ..data.presets.datasets import DatasetsGroups
from ..data.presets.models import ModelsGroups
from ..data.projects.project import ProjectsInfoData, ProjectsList
from ..data.training.train import TrainingDetailsData, InteractiveData
from ..data.training.extra import StateStatusChoice
from ..datasets import loading as datasets_loading
from ..datasets import utils as datasets_utils
from ..datasets.creating import CreateDataset
from ..deploy import loading as deploy_loading
from ..modeling.validator import ModelValidator
from ..progress import utils as progress_utils
from ..training import training_obj
from ..training.guinn import interactive


class Exchange:
    def __call__(self, method: str, *args, **kwargs) -> Any:
        # Получаем метод для вызова
        __method_name = f"_call_{method}"
        __method = getattr(self, __method_name, None)

        # Проверяем, существует ли метод
        if __method is None:
            raise agent_exceptions.CallMethodNotFoundException(
                self.__class__, __method_name
            )

        # Проверяем, является ли метод вызываемым
        if not callable(__method):
            raise agent_exceptions.MethodNotCallableException(
                __method_name, self.__class__
            )

        # Вызываем метод
        try:
            return __method(**kwargs)
        except tensorflow.errors.OpError as error:
            err_msg = str(
                getattr(
                    tf_exceptions, error.__class__.__name__, tf_exceptions.UnknownError
                )(error.message)
            )
            raise getattr(
                agent_utils.ExceptionClasses,
                method,
                agent_utils.ExceptionClasses.unknown,
            ).value(err_msg)
        except agent_exceptions.ExchangeBaseException as error:
            raise error
        except Exception as error:
            raise getattr(
                agent_utils.ExceptionClasses,
                method,
                agent_utils.ExceptionClasses.unknown,
            ).value(str(error))

    @property
    def is_colab(self) -> bool:
        return "COLAB_GPU" in os.environ.keys()

    def _call_hardware_accelerator(self) -> HardwareAcceleratorData:
        try:
            pynvml.nvmlInit()
            _is_gpu = True
        except Exception:
            _is_gpu = False

        if not _is_gpu:
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

    def _call_projects_info(self, path: Path) -> ProjectsInfoData:
        """
        Получение списка проектов
        """
        projects = ProjectsList()
        for filename in os.listdir(path):
            if filename.endswith("project"):
                projects.append({"value": Path(path, filename)})
        projects.sort(key=lambda item: item.label)
        return ProjectsInfoData(projects=projects.native())

    def _call_project_save(
        self, source: Path, target: Path, name: str, overwrite: bool
    ):
        """
        Сохранение проекта
        """
        project_path = Path(target, f"{name}.{settings.PROJECT_EXT}")
        if not overwrite and project_path.is_file():
            raise agent_exceptions.ProjectAlreadyExistsException(name)
        zip_destination = progress_utils.pack(
            "project_save", "Сохранение проекта", source, delete=False
        )
        shutil.move(zip_destination.name, project_path)

    def _call_project_load(self, source: Path, target: Path):
        """
        Загрузка проекта
        """
        shutil.rmtree(target, ignore_errors=True)
        destination = progress_utils.unpack("project_load", "Загрузка проекта", source)
        shutil.move(destination, target)

    def _call_dataset_choice(
        self,
        custom_path: Path,
        destination: Path,
        group: str,
        alias: str,
        reset_model: bool = False,
    ) -> NoReturn:
        """
        Выбор датасета
        """
        datasets_loading.choice(
            DatasetLoadData(path=custom_path, group=group, alias=alias),
            destination=destination,
            reset_model=reset_model,
        )

    def _call_dataset_choice_progress(self) -> progress.ProgressData:
        """
        Прогресс выбора датасета
        """
        return progress.pool("dataset_choice")

    def _call_dataset_delete(self, path: str, group: str, alias: str):
        """
        Удаление датасета
        """
        if group == DatasetGroupChoice.custom:
            shutil.rmtree(
                Path(path, f"{alias}.{settings.DATASET_EXT}"), ignore_errors=True
            )
        else:
            raise agent_exceptions.DatasetCanNotBeDeletedException(alias, group)

    def _call_datasets_info(self, path: Path) -> DatasetsGroupsList:
        """
        Получение данных для страницы датасетов: датасеты и теги
        """
        info = DatasetsGroupsList(DatasetsGroups)
        for dirname in os.listdir(str(path.absolute())):
            if dirname.endswith(settings.DATASET_EXT):
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
        progress_data = progress.pool("dataset_source_load")
        if progress_data.finished and progress_data.data:
            __path = progress_data.data.absolute()
            file_manager = FileManagerItem(path=__path).native().get("children")
            progress_data.data = {
                "file_manager": file_manager,
                "source_path": __path,
            }
        else:
            progress.data = []
        return progress_data

    def _call_dataset_source_segmentation_classes_auto_search(
        self, path: Path, num_classes: int, mask_range: int
    ) -> dict:
        """
        Автопоиск классов для сегментации при создании датасета
        """
        return datasets_utils.get_classes_autosearch(
            path, num_classes, mask_range
        ).native()

    def _call_dataset_source_segmentation_classes_annotation(self, path: Path) -> dict:
        """
        Получение классов для сегментации при создании датасета с использованием файла аннотации
        """
        return datasets_utils.get_classes_annotation(path).native()

    def _call_dataset_create(self, creation_data: CreationData):
        """
        Создание датасета из исходников
        """
        CreateDataset(creation_data)

    def _call_dataset_create_progress(self) -> progress.ProgressData:
        """
        Прогресс создание датасета из исходников
        """
        return progress.pool("create_dataset")

    def _call_datasets_sources(self, path: str) -> FilePathSourcesList:
        """
        Получение списка исходников датасетов
        """
        files = FilePathSourcesList()
        for filename in os.listdir(path):
            if filename.endswith(".zip"):
                filepath = Path(path, filename)
                files.append({"value": filepath})
        files.sort(key=lambda item: item.label)
        return files

    def _call_models(self, path: str) -> ModelsGroupsList:
        """
        Получение списка моделей
        """
        models = ModelsGroupsList(ModelsGroups)
        preset_models_path = Path(settings.ASSETS_PATH, "models")
        custom_models_path = path
        couples = (("preset", preset_models_path), ("custom", custom_models_path))
        for models_group, models_path in couples:
            for filename in os.listdir(models_path):
                if filename.endswith(".model"):
                    models.get(
                        getattr(ModelGroupChoice, models_group).name
                    ).models.append({"value": Path(models_path, filename)})
            models.get(getattr(ModelGroupChoice, models_group).name).models.sort(
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

    def _call_model_update(self, model: dict) -> ModelDetailsData:
        """
        Обновление модели
        """
        return ModelDetailsData(**model)

    def _call_model_validate(self, model: ModelDetailsData) -> tuple:
        """
        Валидация модели
        """
        return ModelValidator(model).get_validated()

    def _call_model_create(self, model: dict, path: Path, overwrite: bool):
        """
        Создание модели
        """
        model_path = Path(path, f'{model.get("name")}.{settings.MODEL_EXT}')
        if not overwrite and model_path.is_file():
            raise agent_exceptions.ModelAlreadyExistsException(model.get("name"))
        with open(model_path, "w") as model_ref:
            json.dump(model, model_ref)

    def _call_model_delete(self, path: Path):
        """
        Удаление модели
        """
        os.remove(path)

    def _call_training_start(
        self,
        dataset: DatasetData,
        model: ModelDetailsData,
        training: TrainingDetailsData,
    ):
        """
        Старт обучения
        """
        if (
            training.state.status == StateStatusChoice.stopped
            or training.state.status == StateStatusChoice.trained
        ):
            training.state.set(StateStatusChoice.addtrain)
        else:
            training.state.set(StateStatusChoice.addtrain.training)
        training_obj.terra_fit(dataset=dataset, gui_model=model, training=training)

    def _call_training_stop(self, training: TrainingDetailsData):
        """
        Остановить обучение
        """
        training.state.set(StateStatusChoice.stopped)

    def _call_training_clear(self, training: TrainingDetailsData):
        """
        Очистить обучение
        """
        training.state.set(StateStatusChoice.no_train)

    def _call_training_interactive(self, config: InteractiveData) -> dict:
        """
        Обновление интерактивных параметров обучения
        """
        return interactive.get_train_results(config=config)

    def _call_training_progress(self) -> progress.ProgressData:
        """
        Прогресс обучения
        """
        return progress.pool("training")

    def _call_training_save(self):
        """
        Сохранение обучения
        """
        pass

    def _call_deploy_cascades_create(self, training_path: str, model_name: str):
        pass

    def _call_deploy_upload(self, source: Path, **kwargs):
        """
        Деплой: загрузка
        """
        deploy_loading.upload(source, kwargs)

    def _call_deploy_upload_progress(self) -> progress.ProgressData:
        """
        Деплой: прогресс загрузки
        """
        return progress.pool("deploy_upload")


agent_exchange = Exchange()
