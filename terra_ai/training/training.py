import gc
import json
import os
import threading
from pathlib import Path

from typing import Optional, Union

from tensorflow.keras.models import Model
from tensorflow import keras


from terra_ai import progress
from terra_ai.callbacks.utils import print_error, YOLO_ARCHITECTURE, get_dataset_length
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.datasets.extra import LayerOutputTypeChoice, LayerInputTypeChoice
from terra_ai.data.modeling.model import ModelDetailsData
from terra_ai.data.training.extra import ArchitectureChoice
from terra_ai.data.training.train import TrainingDetailsData
from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.exceptions.deploy import MethodNotImplementedException
from terra_ai.modeling.validator import ModelValidator
from terra_ai.exceptions import training as exceptions
from terra_ai.training.terra_models import BaseTerraModel, YoloTerraModel
from terra_ai.callbacks.base_callback import FitCallback

from terra_ai.callbacks import interactive

__version__ = 0.02


# noinspection PyTypeChecker,PyBroadException
class GUINN:

    def __init__(self) -> None:
        self.name = "GUINN"
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
        method_name = '_set_training_params'
        try:
            print(method_name)
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
                raise exceptions.TooBigBatchSize(params.base.batch, train_size)

            self.batch_size = params.base.batch

            interactive.set_attributes(dataset=self.dataset, params=params)
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def _set_callbacks(self, dataset: PrepareDataset, train_details: TrainingDetailsData) -> None:
        method_name = '_set_callbacks'
        try:
            print(method_name)
            progress.pool(self.progress_name, finished=False, message="Добавление колбэков...")

            self.callback = FitCallback(dataset=dataset, training_details=train_details, model_name=self.nn_name,
                                        deploy_type=self.deploy_type.name)
            progress.pool(self.progress_name, finished=False, message="Добавление колбэков выполнено")
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    @staticmethod
    def _set_deploy_type(dataset: PrepareDataset) -> str:
        method_name = '_set_deploy_type'
        try:
            print(method_name)
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
                raise MethodNotImplementedException(__method=inp_task_name + out_task_name,
                                                    __class="ArchitectureChoice")
            return deploy_type
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    @staticmethod
    def _prepare_dataset(dataset: DatasetData, model_path: Path, state: str) -> PrepareDataset:
        method_name = '_prepare_dataset'
        try:
            print(method_name)
            prepared_dataset = PrepareDataset(data=dataset, datasets_path=dataset.path)
            prepared_dataset.prepare_dataset()
            if state != "addtrain":
                prepared_dataset.deploy_export(os.path.join(model_path, "dataset"))
            return prepared_dataset
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def _set_model(self, model: ModelDetailsData, train_details: TrainingDetailsData,
                   dataset: PrepareDataset) -> Union[BaseTerraModel, YoloTerraModel]:
        method_name = '_set_model'
        try:
            print(method_name)
            if train_details.state.status == "training":
                validator = ModelValidator(model)
                base_model = validator.get_keras_model()
                train_model = BaseTerraModel(model=base_model,
                                             model_name=self.nn_name,
                                             model_path=train_details.model_path)
                if dataset.data.architecture in YOLO_ARCHITECTURE:
                    options = self.get_yolo_init_parameters(dataset=dataset)
                    train_model = YoloTerraModel(model=base_model,
                                                 model_name=self.nn_name,
                                                 model_path=train_details.model_path,
                                                 **options).yolo_model
            else:
                train_model = BaseTerraModel(model=None,
                                             model_name=self.nn_name,
                                             model_path=train_details.model_path)
                train_model.load()
                if dataset.data.architecture in YOLO_ARCHITECTURE:
                    options = self.get_yolo_init_parameters(dataset=dataset)
                    train_model = YoloTerraModel(model=train_model.base_model,
                                                 model_name=self.nn_name,
                                                 model_path=train_details.model_path,
                                                 **options)
                weight = None
                for i in os.listdir(train_details.model_path):
                    if i[-3:] == '.h5' and 'best' not in i:
                        weight = i
                if weight:
                    train_model.load_weights()
            return train_model
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    @staticmethod
    def get_yolo_init_parameters(dataset: PrepareDataset):
        version = dataset.instructions.get(list(dataset.data.outputs.keys())[0]).get(
            '2_object_detection').get('yolo')
        classes = dataset.data.outputs.get(list(dataset.data.outputs.keys())[0]).classes_names
        options = {"training": True, "classes": classes, "version": version}
        return options

    @staticmethod
    def _save_params_for_deploy(params: TrainingDetailsData):
        method_name = '_save_params_for_deploy'
        try:
            print(method_name)
            with open(os.path.join(params.model_path, "config.train"), "w", encoding="utf-8") as train_config:
                json.dump(params.base.native(), train_config)
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def _kill_last_training(self, state):
        method_name = '_kill_last_training'
        try:
            print(method_name)
            for one_thread in threading.enumerate():
                if one_thread.getName() == "current_train":
                    current_status = state.state.status
                    state.state.set("stopped")
                    progress.pool(self.progress_name,
                                  message="Найдено незавершенное обучение. Идет очистка. Подождите.")
                    one_thread.join()
                    state.state.set(current_status)
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def terra_fit(self, dataset: DatasetData, gui_model: ModelDetailsData, training: TrainingDetailsData) -> None:
        method_name = 'model_fit'
        try:
            print(method_name)
            self._kill_last_training(state=training)
            progress.pool.reset(self.progress_name)

            if training.state.status != "addtrain":
                self._save_params_for_deploy(params=training)
            self.nn_cleaner(retrain=True if training.state.status == "training" else False)
            self._set_training_params(dataset=dataset, params=training)
            self.model_fit(params=training, dataset=self.dataset, model=gui_model)

        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def nn_cleaner(self, retrain: bool = False) -> None:
        method_name = 'nn_cleaner'
        try:
            print(method_name)
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
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    @progress.threading
    def model_fit(self, params: TrainingDetailsData, model: ModelDetailsData, dataset: PrepareDataset) -> None:
        method_name = 'model_fit'
        try:
            self._set_callbacks(dataset=dataset, train_details=params)
            threading.enumerate()[-1].setName("current_train")
            progress.pool(self.progress_name, finished=False, message="Компиляция модели ...")
            compiled_model = self._set_model(model=model, train_details=params, dataset=dataset)
            compiled_model.set_callback(self.callback)
            progress.pool(self.progress_name, finished=False, message="\n Компиляция модели выполнена")
            progress.pool(self.progress_name, finished=False, message="\n Начало обучения ...")
            if params.state.status == "training":
                compiled_model.save()
            compiled_model.fit(params=params, dataset=dataset)

        except Exception as e:
            print_error(GUINN().name, method_name, e)
