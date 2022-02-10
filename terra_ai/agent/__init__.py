import os
import json
import shutil
import pynvml
import tensorflow

from pathlib import Path
from typing import Any, Dict, List

from terra_ai import exceptions as terra_ai_exceptions
from terra_ai import progress as terra_ai_progress
from terra_ai.settings import DATASET_EXT, ASSETS_PATH, MODEL_EXT, PROJECT_PATH
from terra_ai.agent.exceptions import (
    CallMethodNotFoundException,
    MethodNotCallableException,
    ModelAlreadyExistsException,
)
from terra_ai.data.extra import HardwareAcceleratorData, HardwareAcceleratorChoice
from terra_ai.data.projects.project import ProjectsInfoData, ProjectsList
from terra_ai.data.datasets.dataset import (
    DatasetVersionList,
    DatasetCommonGroupList,
    DatasetData,
    DatasetLoadData,
)
from terra_ai.data.datasets.creation import (
    CreationData,
    FilePathSourcesList,
    SourceData,
)
from terra_ai.data.datasets.extra import DatasetGroupChoice
from terra_ai.data.modeling.model import (
    ModelsGroupsList,
    ModelDetailsData,
    ModelLoadData,
)
from terra_ai.data.modeling.extra import ModelGroupChoice, LayerTypeChoice
from terra_ai.data.training.train import TrainingDetailsData
from terra_ai.data.training.extra import StateStatusChoice, ArchitectureChoice
from terra_ai.data.cascades.cascade import (
    CascadeDetailsData,
    CascadesList,
    CascadeLoadData,
)
from terra_ai.data.cascades.extra import BlockGroupChoice
from terra_ai.data.deploy.tasks import DeployPageData
from terra_ai.data.presets.datasets import DatasetCommonGroup
from terra_ai.data.presets.models import ModelsGroups
from terra_ai.project.loading import load as project_load
from terra_ai.datasets import utils as datasets_utils
from terra_ai.datasets.creating import CreateDataset
from terra_ai.datasets.loading import (
    source as dataset_source,
    choice as dataset_choice,
    multiload as dataset_multiload,
    multiload_no_thread as dataset_multiload_no_thread,
)
from terra_ai.modeling.validator import ModelValidator
from terra_ai.training import training_obj
from terra_ai.training.training import interactive
from terra_ai.cascades.cascade_validator import CascadeValidator
from terra_ai.cascades.cascade_runner import CascadeRunner
from terra_ai.deploy.loading import upload as deploy_upload


class Exchange:
    def __call__(self, method: str, *args, **kwargs) -> Any:
        # Получаем метод для вызова
        __method_name = f"_call_{method}"
        __method = getattr(self, __method_name, None)

        # Проверяем, существует ли метод
        if __method is None:
            raise CallMethodNotFoundException(self.__class__, __method_name)

        # Проверяем, является ли метод вызываемым
        if not callable(__method):
            raise MethodNotCallableException(__method_name, self.__class__)

        # Вызываем метод
        return __method(**kwargs)

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

    def _call_project_load(self, dataset_path: Path, source: Path, target: Path):
        """
        Загрузка проекта
        """
        project_load(Path(dataset_path), Path(source), Path(target))

    def _call_datasets_info(self) -> DatasetCommonGroupList:
        """
        Получение данных для страницы датасетов: датасеты и теги
        """
        return DatasetCommonGroupList()

    def _call_datasets_versions(self, group: str, alias: str) -> DatasetVersionList:
        """
        Получение списка версий датасетов
        """
        return DatasetVersionList(group=group, alias=alias)

    def _call_dataset_choice(
        self, group: str, alias: str, version: str, reset_model: bool = False
    ):
        """
        Выбор датасета
        """
        dataset_choice(
            "dataset_choice",
            DatasetLoadData(group=group, alias=alias, version=version),
            reset_model=reset_model,
        )

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

    def _call_dataset_source_load(self, mode: str, value: str):
        """
        Загрузка исходников датасета
        """
        dataset_source(SourceData(mode=mode, value=value))

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

    def _call_dataset_delete(self, path: str, group: str, alias: str):
        """
        Удаление датасета
        """
        if group == DatasetGroupChoice.custom:
            shutil.rmtree(Path(path, f"{alias}.{DATASET_EXT}"), ignore_errors=True)
        else:
            raise terra_ai_exceptions.datasets.DatasetCanNotBeDeletedException(
                alias, group
            )

    def _call_models(self, path: str) -> ModelsGroupsList:
        """
        Получение списка моделей
        """
        models = ModelsGroupsList(ModelsGroups)
        preset_models_path = Path(ASSETS_PATH, "models")
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

    def _call_model_validate(
        self, model: ModelDetailsData, dataset_data: DatasetData
    ) -> tuple:
        """
        Валидация модели
        """
        return ModelValidator(model, dataset_data).get_validated()

    def _call_model_create(self, model: dict, path: Path, overwrite: bool):
        """
        Создание модели
        """
        model_path = Path(path, f'{model.get("name")}.{MODEL_EXT}')
        if not overwrite and model_path.is_file():
            raise ModelAlreadyExistsException(model.get("name"))
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
        for layer in model.layers:
            if layer.type == LayerTypeChoice.PretrainedYOLO:
                layer.parameters.weight_load()

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

    def _call_training_interactive(self, training: TrainingDetailsData):
        """
        Обновление интерактивных параметров обучения
        """
        if training.state.status not in (StateStatusChoice.no_train,):
            interactive.get_train_results()

    def _call_training_kill(self, training: TrainingDetailsData):
        """
        Удаление незавершенного обучения
        """
        training.state.set("kill")

    def _call_training_save(self):
        """
        Сохранение обучения
        """
        pass

    def _call_cascade_get(self, value: str) -> CascadeDetailsData:
        """
        Получение каскада
        """
        data = CascadeLoadData(value=value)
        with open(Path(data.value).absolute(), "r") as config_ref:
            config = json.load(config_ref)
            return CascadeDetailsData(**config)

    def _call_cascades_info(self, path: str) -> CascadesList:
        """
        Получение списка каскадов
        """
        return CascadesList(path)

    def _call_cascade_update(self, cascade: dict) -> CascadeDetailsData:
        """
        Обновление каскада
        """
        return CascadeDetailsData(**cascade)

    def _call_cascade_validate(self, path: Path, cascade: CascadeDetailsData):
        """
        Валидация каскада
        """
        return CascadeValidator().get_validate(cascade_data=cascade, training_path=path)

    @terra_ai_progress.threading
    def _call_cascade_start(
        self,
        training_path: Path,
        datasets_path: Path,
        sources: Dict[int, Dict[str, str]],
        cascade: CascadeDetailsData,
        example_count: int = None,
    ):
        """
        Запуск каскада
        """
        progress_name = "cascade_start"
        datasets = list(
            map(
                lambda item: DatasetLoadData(path=datasets_path, **dict(item)),
                sources.values(),
            )
        )
        for block in cascade.blocks:
            if block.group == BlockGroupChoice.Model:
                _path = Path(
                    training_path, block.parameters.main.path, "model", "dataset.json"
                )
                if not _path.is_file():
                    _path = Path(
                        training_path,
                        block.parameters.main.path,
                        "model",
                        "dataset",
                        "config.json",
                    )
                with open(_path) as config_ref:
                    data = json.load(config_ref)
                    datasets.append(
                        DatasetLoadData(
                            path=datasets_path,
                            alias=data.get("alias"),
                            group=data.get("group"),
                        )
                    )
        dataset_multiload_no_thread(progress_name, datasets, sources=sources)
        terra_ai_progress.pool(progress_name, finished=False)
        progress_data = terra_ai_progress.pool(progress_name).data
        sources_data = progress_data.get("kwargs", {}).get("sources", {})
        dataset_sources = progress_data.get("datasets", [])
        sources = {}
        for block in cascade.blocks:
            if block.group != BlockGroupChoice.InputData:
                continue
            source_data = sources_data.get(str(block.id))
            if not source_data:
                continue
            datasets_source = list(
                filter(
                    lambda item: item.alias == source_data.get("alias")
                    and item.group == source_data.get("group"),
                    dataset_sources,
                )
            )
            if not len(datasets_source):
                continue
            sources.update(
                {
                    # block.id: DatasetInfo(
                    #     alias=datasets_source[0].alias,
                    #     group=datasets_source[0].group,
                    # ).dataset.sources
                }
            )
        self(
            "cascade_execute",
            sources=sources,
            cascade=cascade,
            training_path=PROJECT_PATH.training,
            example_count=example_count,
        )

    def _call_cascade_execute(
        self,
        sources: Dict[int, List[str]],
        cascade: CascadeDetailsData,
        training_path,
        example_count: int = None,
    ):
        """
        Исполнение каскада
        """
        for block in cascade.blocks:
            if block.group == BlockGroupChoice.Service:
                block.parameters.model_load()
        CascadeRunner().start_cascade(
            sources=sources,
            cascade_data=cascade,
            training_path=training_path,
            example_count=example_count,
        )

    def _call_deploy_get(self, datasets: List[DatasetLoadData], page: DeployPageData):
        """
        Получение данных для отображения пресетов на странице деплоя
        """
        dataset_multiload("deploy_get", datasets, page=page)

    def _call_deploy_cascades_create(self, training_path: str, model_name: str):
        pass

    def _call_deploy_upload(self, source: Path, **kwargs):
        """
        Деплой: загрузка
        """
        deploy_upload(source, kwargs)


agent_exchange = Exchange()
