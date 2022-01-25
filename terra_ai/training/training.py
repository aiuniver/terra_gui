import gc
import json
import os
import threading
from pathlib import Path

from typing import Optional, Union

from tensorflow.keras.models import Model
from tensorflow import keras

from terra_ai import progress
from terra_ai.callbacks.utils import YOLO_ARCHITECTURE, get_dataset_length, GAN_ARCHITECTURE
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.datasets.extra import LayerOutputTypeChoice, LayerInputTypeChoice
from terra_ai.data.modeling.model import ModelDetailsData
from terra_ai.data.training.extra import ArchitectureChoice, StateStatusChoice
from terra_ai.data.training.train import TrainingDetailsData
from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.exceptions.base import TerraBaseException
from terra_ai.exceptions.deploy import MethodNotImplementedException
from terra_ai.exceptions.training import TooBigBatchSize, DatasetPrepareMissing, ModelSettingMissing, \
    NoYoloParamsException, TrainingException
from terra_ai.logging import logger
from terra_ai.modeling.validator import ModelValidator
from terra_ai.training.terra_models import BaseTerraModel, YoloTerraModel, GANTerraModel, ConditionalGANTerraModel, \
    TextToImageGANTerraModel, ImageToImageGANTerraModel
from terra_ai.callbacks.base_callback import FitCallback

from terra_ai.callbacks import interactive
import terra_ai.exceptions.callbacks as exception

__version__ = 0.02

# noinspection PyTypeChecker,PyBroadException
from terra_ai.utils import check_error


class GUINN:
    name = "GUINN"

    def __init__(self) -> None:
        self.callback = None
        self.params: TrainingDetailsData
        self.nn_name: str = ''
        self.dataset: Optional[PrepareDataset] = None
        self.deploy_type = None
        self.model: Optional[Model] = None
        self.json_model = None
        self.model_path: Path = Path("./")
        self.deploy_path: Path = Path("./")
        self.dataset_path: Path = Path("./")
        self.loss: dict = {}
        self.metrics: dict = {}
        self.yolo_pred = None
        self.batch_size = 128
        self.val_batch_size = 1
        self.epochs = 5
        self.sum_epoch = 0
        self.stop_training = False
        self.retrain_flag = False
        self.shuffle: bool = True
        self.model_is_trained: bool = False
        self.progress_name = "training"

    def _set_training_params(self, dataset: DatasetData, params: TrainingDetailsData) -> None:
        # logger.debug(f"{GUINN.name}, {GUINN._set_training_params.__name__}")
        method_name = '_set_training_params'
        logger.info("Установка параметров обучения...", extra={"type": "info"})
        try:
            self.params = params
            self.dataset = self._prepare_dataset(
                dataset=dataset, model_path=params.model_path, state=params.state.status)
            self.train_length, self.val_length = get_dataset_length(self.dataset)

            if not self.dataset.data.architecture or self.dataset.data.architecture == ArchitectureChoice.Basic:
                self.deploy_type = self._set_deploy_type(self.dataset)
            else:
                self.deploy_type = self.dataset.data.architecture

            self.nn_name = "trained_model"

            train_size = len(self.dataset.dataframe.get("train")) if self.dataset.data.use_generator \
                else len(self.dataset.dataset.get('train'))
            if params.base.batch > train_size:
                if params.state.status == "addtrain":
                    params.state.set("stopped")
                else:
                    params.state.set("no_train")
                raise TooBigBatchSize(params.base.batch, train_size)

            self.batch_size = params.base.batch

            interactive.set_attributes(dataset=self.dataset, params=params)
            logger.info("Установка параметров обучения завершена", extra={"type": "success"})
        except Exception as error:
            raise check_error(error, self.__class__.__name__, method_name)

    def _set_callbacks(self, dataset: PrepareDataset, train_details: TrainingDetailsData) -> None:
        # logger.debug(f"{GUINN.name}, {GUINN._set_callbacks.__name__}")
        self.callback = FitCallback(dataset=dataset, training_details=train_details, model_name=self.nn_name,
                                    deploy_type=self.deploy_type.name)
        logger.info("Добавление колбэков выполнено", extra={"type": "success"})

    @staticmethod
    def _set_deploy_type(dataset: PrepareDataset) -> str:
        # logger.debug(f"{GUINN.name}, {GUINN._set_deploy_type.__name__}")
        method_name = '_set_deploy_type'
        try:
            data = dataset.data
            inp_tasks = []
            out_tasks = []
            for key, val in data.inputs.items():
                if val.task == LayerInputTypeChoice.Dataframe:
                    tmp = []
                    for value in data.columns[key].values():
                        tmp.append(value.get("task"))
                    unique_vals = list(set(tmp))
                    if len(unique_vals) == 1 and unique_vals[0] in LayerInputTypeChoice.__dict__.keys() and \
                            unique_vals[0] in [LayerInputTypeChoice.Image, LayerInputTypeChoice.Text,
                                               LayerInputTypeChoice.Audio, LayerInputTypeChoice.Video]:
                        inp_tasks.append(unique_vals[0])
                    else:
                        inp_tasks.append(val.task)
                else:
                    inp_tasks.append(val.task)
            for key, val in data.outputs.items():
                if val.task == LayerOutputTypeChoice.Dataframe:
                    tmp = []
                    for value in data.columns[key].values():
                        tmp.append(value.task)
                    unique_vals = list(set(tmp))
                    if len(unique_vals) == 1 and unique_vals[0] in LayerOutputTypeChoice.__dict__.keys():
                        out_tasks.append(unique_vals[0])
                    else:
                        out_tasks.append(val.task)
                else:
                    out_tasks.append(val.task)

            inp_task_name = list(set(inp_tasks))[0] if len(set(inp_tasks)) == 1 else LayerInputTypeChoice.Dataframe
            out_task_name = list(set(out_tasks))[0] if len(set(out_tasks)) == 1 else LayerOutputTypeChoice.Dataframe

            if inp_task_name + out_task_name in ArchitectureChoice.__dict__.keys():
                deploy_type = ArchitectureChoice.__dict__[inp_task_name + out_task_name]
            elif out_task_name in ArchitectureChoice.__dict__.keys():
                deploy_type = ArchitectureChoice.__dict__[out_task_name]
            elif out_task_name == LayerOutputTypeChoice.ObjectDetection:
                deploy_type = ArchitectureChoice.__dict__[dataset.instructions.get(2).parameters.model.title() +
                                                          dataset.instructions.get(2).parameters.yolo.title()]
            else:
                raise MethodNotImplementedException(
                    __method=inp_task_name + out_task_name, __class="ArchitectureChoice")
            return deploy_type
        except Exception as error:
            raise check_error(error, GUINN().name, method_name)

    def _prepare_dataset(self, dataset: DatasetData, model_path: Path, state: str) -> PrepareDataset:
        # logger.debug(f"{GUINN.name}, {GUINN._prepare_dataset.__name__}")
        method_name = '_prepare_dataset'
        try:
            logger.info("Загрузка датасета...", extra={"type": "info"})
            prepared_dataset = PrepareDataset(data=dataset, datasets_path=dataset.path)
            prepared_dataset.prepare_dataset()
            if state != "addtrain":
                prepared_dataset.deploy_export(os.path.join(model_path))
            logger.info("Загрузка датасета завершена", extra={"type": "success"})
            return prepared_dataset
        except Exception as error:
            raise DatasetPrepareMissing(
                self.__class__.__name__, method_name
            ).with_traceback(error.__traceback__)

    def _set_model(self, model: ModelDetailsData, train_details: TrainingDetailsData,
                   dataset: PrepareDataset) -> Union[BaseTerraModel, YoloTerraModel]:
        # logger.debug(f"{GUINN.name}, {GUINN._set_model.__name__}")
        method_name = 'set model'
        try:
            logger.info("Загрузка модели...", extra={"type": "info"})
            base_model = None
            if train_details.state.status == "training":
                validator = ModelValidator(model, dataset.data.architecture)
                base_model = validator.get_keras_model()

            if dataset.data.architecture == ArchitectureChoice.GAN:
                train_model = GANTerraModel(
                    model=base_model, model_name=self.nn_name, model_path=train_details.model_path)
            elif dataset.data.architecture == ArchitectureChoice.CGAN:
                train_model = ConditionalGANTerraModel(
                    model=base_model, model_name=self.nn_name, model_path=train_details.model_path,
                    options=dataset)
            elif dataset.data.architecture == ArchitectureChoice.TextToImageGAN:
                train_model = TextToImageGANTerraModel(
                    model=base_model, model_name=self.nn_name, model_path=train_details.model_path,
                    options=dataset)
            elif dataset.data.architecture == ArchitectureChoice.ImageToImageGAN:
                train_model = ImageToImageGANTerraModel(
                    model=base_model, model_name=self.nn_name, model_path=train_details.model_path,
                    options=dataset)
            elif dataset.data.architecture in YOLO_ARCHITECTURE:
                options = self.get_yolo_init_parameters(dataset=dataset)
                train_model = YoloTerraModel(
                    model=base_model, model_name=self.nn_name, model_path=train_details.model_path, **options)
            else:
                train_model = BaseTerraModel(
                    model=base_model, model_name=self.nn_name, model_path=train_details.model_path)

            logger.info("Загрузка модели завершена", extra={"type": "success"})
            return train_model
        except Exception as error:
            raise ModelSettingMissing(
                self.__class__.__name__, method_name
            ).with_traceback(error.__traceback__)

    def get_yolo_init_parameters(self, dataset: PrepareDataset):
        # logger.debug(f"{GUINN.name}, {GUINN.get_yolo_init_parameters.__name__}")
        method_name = 'get_yolo_init_parameters'
        try:
            version = dataset.instructions.get(list(dataset.data.outputs.keys())[0]).get(
                '2_object_detection').get('yolo')
            classes = dataset.data.outputs.get(list(dataset.data.outputs.keys())[0]).classes_names
            options = {"training": True, "classes": classes, "version": version}
            return options
        except Exception as error:
            raise NoYoloParamsException(
                self.__class__.__name__, method_name
            ).with_traceback(error.__traceback__)

    def _kill_last_training(self, state):
        # logger.debug(f"{GUINN.name}, {GUINN._kill_last_training.__name__}")
        method_name = '_kill_last_training'
        try:
            for one_thread in threading.enumerate():
                if one_thread.getName() == "current_train":
                    current_status = state.state.status
                    state.state.set("stopped")
                    progress.pool(self.progress_name,
                                  message="Найдено незавершенное обучение. Идет очистка. Подождите.", finished=False)
                    one_thread.join()
                    state.state.set(current_status)
        except Exception as error:
            raise check_error(error, self.__class__.__name__, method_name)

    def terra_fit(self, dataset: DatasetData, gui_model: ModelDetailsData, training: TrainingDetailsData) -> None:
        # logger.debug(f"{GUINN.name}, {GUINN.terra_fit.__name__}")
        method_name = 'terra_fit'
        logger.info(f"start {method_name}")
        try:
            # check and kill last training if it detect
            self._kill_last_training(state=training)
            progress.pool.reset(self.progress_name, finished=False)

            # save base training params for deploy
            if training.state.status != "addtrain":
                with open(os.path.join(training.model_path, "config.train"), "w", encoding="utf-8") as train_config:
                    json.dump(training.base.native(), train_config)

            self.nn_cleaner(retrain=True if training.state.status == "training" else False)
            self._set_training_params(dataset=dataset, params=training)
            self.model_fit(params=training, dataset=self.dataset, model=gui_model)
        except Exception as error:
            logger.error(error)
            raise check_error(error, self.__class__.__name__, method_name)

    def nn_cleaner(self, retrain: bool = False) -> None:
        # logger.debug(f"{GUINN.name}, {GUINN.nn_cleaner.__name__}")
        keras.backend.clear_session()
        self.dataset = None
        self.deploy_type = None
        self.model = None
        if retrain:
            self.sum_epoch = 0
            self.loss = {}
            self.metrics = {}
            self.callback = None
            interactive.clear_history()
        gc.collect()

    @progress.threading
    def model_fit(self, params: TrainingDetailsData, model: ModelDetailsData, dataset: PrepareDataset) -> None:
        # logger.debug(f"{GUINN.name}, {GUINN.model_fit.__name__}")
        method_name = 'model_fit'
        try:
            logger.info(f"Старт обучения модели...", extra={"front_level": "info"})
            self._set_callbacks(dataset=dataset, train_details=params)
            threading.enumerate()[-1].setName("current_train")
            progress.pool(self.progress_name, finished=False, message="Компиляция модели ...")
            compiled_model = self._set_model(model=model, train_details=params, dataset=dataset)
            compiled_model.set_callback(self.callback)
            progress.pool(self.progress_name, finished=False, message="\n Компиляция модели выполнена")
            if params.state.status == "training":
                compiled_model.save()
            progress.pool(self.progress_name, finished=False, message="\n Начало обучения ...")
            compiled_model.fit(params=params, dataset=dataset)
        except Exception as error:
            if self.callback and self.callback.last_epoch <= 1 and params.state.status == StateStatusChoice.training:
                params.state.set(StateStatusChoice.no_train)
            else:
                params.state.set(StateStatusChoice.stopped)
            progress.pool(self.progress_name, data=params, finished=True, error=error)
            raise check_error(error, self.__class__.__name__, method_name)
