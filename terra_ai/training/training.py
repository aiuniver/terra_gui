import gc
import importlib
import json
import math
import copy
import os
import threading
from pathlib import Path

import psutil
import time
import pynvml as N

from typing import Optional

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.python.keras.utils.np_utils import to_categorical

from config import settings
from terra_ai import progress
from terra_ai.callbacks.classification_callbacks import BaseClassificationCallback
from terra_ai.callbacks.interactive_callback import InteractiveCallback
# from terra_ai.training.customcallback import InteractiveCallback
from terra_ai.callbacks.utils import print_error, loss_metric_config, BASIC_ARCHITECTURE, CLASS_ARCHITECTURE, \
    YOLO_ARCHITECTURE, round_loss_metric, class_metric_list, CLASSIFICATION_ARCHITECTURE, get_dataset_length
from terra_ai.customLayers import terra_custom_layers
from terra_ai.data.datasets.dataset import DatasetData, DatasetOutputsData
from terra_ai.data.datasets.extra import LayerOutputTypeChoice, LayerInputTypeChoice, LayerEncodingChoice
from terra_ai.data.deploy.extra import DeployTypeChoice
from terra_ai.data.deploy.tasks import DeployData
from terra_ai.data.modeling.model import ModelDetailsData
from terra_ai.data.presets.training import Metric
from terra_ai.data.training.extra import CheckpointTypeChoice, ArchitectureChoice
from terra_ai.data.training.train import TrainingDetailsData

from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.deploy.create_deploy_package import CascadeCreator
from terra_ai.exceptions.deploy import MethodNotImplementedException
from terra_ai.modeling.validator import ModelValidator
from terra_ai.exceptions import training as exceptions
from terra_ai.training.terra_models import BaseTerraModel, YoloTerraModel
from terra_ai.training.training_history import History
from terra_ai.training.yolo_utils import create_yolo, compute_loss, get_mAP
from terra_ai.utils import decamelize
from terra_ai.callbacks import interactive

__version__ = 0.02


# noinspection PyTypeChecker,PyBroadException
class GUINN:

    def __init__(self) -> None:
        self.name = "GUINN"
        self.callbacks = None
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

            if params.state.status == "addtrain":
                if params.logs.get("addtrain_epochs")[-1] >= self.sum_epoch:
                    self.sum_epoch += params.base.epochs
                if params.logs.get("addtrain_epochs")[-1] < self.sum_epoch:
                    self.epochs = self.sum_epoch - params.logs.get("addtrain_epochs")[-1]
                else:
                    self.epochs = params.base.epochs
            else:
                self.epochs = params.base.epochs

            self.batch_size = params.base.batch

            interactive.set_attributes(dataset=self.dataset, params=params)
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def _set_callbacks(self, dataset: PrepareDataset, train_details: TrainingDetailsData) -> None:
        method_name = '_set_callbacks'
        try:
            print(method_name)
            progress.pool(self.progress_name, finished=False, message="Добавление колбэков...")
            retrain_epochs = self.sum_epoch if train_details.state.status == "addtrain" else self.epochs

            self.callback = FitCallback(dataset=dataset, retrain_epochs=retrain_epochs,
                                        training_details=train_details, model_name=self.nn_name,
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
                   dataset: PrepareDataset) -> Optional[BaseTerraModel, YoloTerraModel]:
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
                    if i[-3:] == '.h5' and 'last' in i:
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

    def terra_fit(self, dataset: DatasetData, gui_model: ModelDetailsData, training: TrainingDetailsData) -> dict:
        method_name = 'model_fit'
        try:
            """
               This method created for using wth externally compiled models

               Args:
                   dataset: DatasetData
                   gui_model: Keras model for fit - ModelDetailsData
                   training: TrainingDetailsData

               Return:
                   dict
               """
            print(method_name)
            # self._kill_last_training(state=training)
            progress.pool.reset(self.progress_name)

            if training.state.status != "addtrain":
                self._save_params_for_deploy(params=training)

            self.nn_cleaner(retrain=True if training.state.status == "training" else False)

            self._set_training_params(dataset=dataset, params=training)

            self.model_fit(params=training, dataset=self.dataset, model=gui_model)
            return {"dataset": self.dataset}
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
                self.callbacks = []
                interactive.clear_history()
            gc.collect()
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def get_nn(self):
        self.nn_cleaner(retrain=True)
        return self

    @progress.threading
    def model_fit(self, params: TrainingDetailsData, model: ModelDetailsData, dataset: PrepareDataset) -> None:
        method_name = 'base_model_fit'
        try:
            yolo_arch = True if dataset.data.architecture in YOLO_ARCHITECTURE else False
            self._set_callbacks(dataset=dataset, train_details=params)
            # callback = FitCallback(dataset, params)
            threading.enumerate()[-1].setName("current_train")
            progress.pool(self.progress_name, finished=False, message="Компиляция модели ...")
            compiled_model = self._set_model(model=model, train_details=params, dataset=dataset)
            compiled_model.set_callback(self.callback)
            if params.state.status == "training":
                compiled_model.save()
            if yolo_arch:
                model = compiled_model.yolo_model
            else:
                model = compiled_model.base_model
            compiled_model.fit(params=params, dataset=dataset, model=model)

            progress.pool(self.progress_name, finished=False, message="\n Компиляция модели выполнена")
            progress.pool(self.progress_name, finished=False, message="\n Начало обучения ...")
            if (params.state.status == "stopped" and self.callbacks[0].last_epoch < params.base.epochs) or \
                    (params.state.status == "trained" and self.callbacks[0].last_epoch - 1 == params.base.epochs):
                self.sum_epoch = params.base.epochs
        except Exception as e:
            print_error(GUINN().name, method_name, e)


# noinspection PyBroadException
class MemoryUsage:
    def __init__(self, debug=False):
        self.debug = debug
        try:
            N.nvmlInit()
            self.gpu = settings.USE_GPU
        except:
            self.gpu = False

    def get_usage(self):
        usage_dict = {}
        if self.gpu:
            gpu_name = N.nvmlDeviceGetName(N.nvmlDeviceGetHandleByIndex(0))
            gpu_utilization = N.nvmlDeviceGetUtilizationRates(N.nvmlDeviceGetHandleByIndex(0))
            gpu_memory = N.nvmlDeviceGetMemoryInfo(N.nvmlDeviceGetHandleByIndex(0))
            usage_dict["GPU"] = {
                'gpu_name': gpu_name,
                'gpu_utilization': f'{gpu_utilization.gpu: .2f}',
                'gpu_memory_used': f'{gpu_memory.used / 1024 ** 3: .2f}GB',
                'gpu_memory_total': f'{gpu_memory.total / 1024 ** 3: .2f}GB'
            }
            if self.debug:
                print(f'GPU usage: {gpu_utilization.gpu: .2f} ({gpu_memory.used / 1024 ** 3: .2f}GB / '
                      f'{gpu_memory.total / 1024 ** 3: .2f}GB)')
        else:
            cpu_usage = psutil.cpu_percent(percpu=True)
            usage_dict["CPU"] = {
                'cpu_utilization': f'{sum(cpu_usage) / len(cpu_usage): .2f}',
            }
            if self.debug:
                print(f'Average CPU usage: {sum(cpu_usage) / len(cpu_usage): .2f}')
                print(f'Max CPU usage: {max(cpu_usage): .2f}')
        usage_dict["RAM"] = {
            'ram_utilization': f'{psutil.virtual_memory().percent: .2f}',
            'ram_memory_used': f'{psutil.virtual_memory().used / 1024 ** 3: .2f}GB',
            'ram_memory_total': f'{psutil.virtual_memory().total / 1024 ** 3: .2f}GB'
        }
        usage_dict["Disk"] = {
            'disk_utilization': f'{psutil.disk_usage("/").percent: .2f}',
            'disk_memory_used': f'{psutil.disk_usage("/").used / 1024 ** 3: .2f}GB',
            'disk_memory_total': f'{psutil.disk_usage("/").total / 1024 ** 3: .2f}GB'
        }
        if self.debug:
            print(f'RAM usage: {psutil.virtual_memory().percent: .2f} '
                  f'({psutil.virtual_memory().used / 1024 ** 3: .2f}GB / '
                  f'{psutil.virtual_memory().total / 1024 ** 3: .2f}GB)')
            print(f'Disk usage: {psutil.disk_usage("/").percent: .2f} '
                  f'({psutil.disk_usage("/").used / 1024 ** 3: .2f}GB / '
                  f'{psutil.disk_usage("/").total / 1024 ** 3: .2f}GB)')
        return usage_dict


class FitCallback:
    """CustomCallback for all task type"""

    def __init__(self, dataset: PrepareDataset, training_details: TrainingDetailsData, retrain_epochs: int = None,
                 model_name: str = "model", deploy_type: str = ""):
        super().__init__()
        print('\n FitCallback')
        self.name = "FitCallback"
        self.current_logs = {}
        self.usage_info = MemoryUsage(debug=False)
        self.training_detail = training_details
        self.training_detail.logs = None
        self.dataset = dataset
        self.dataset_path = dataset.data.path
        self.deploy_type = getattr(DeployTypeChoice, deploy_type)
        self.is_yolo = True if dataset.data.architecture in YOLO_ARCHITECTURE else False
        self.batch_size = training_details.base.batch
        self.epochs = training_details.base.epochs
        self.retrain_epochs = retrain_epochs
        self.still_epochs = training_details.base.epochs
        self.nn_name = model_name
        self.deploy_path = training_details.deploy_path
        self.model_path = training_details.model_path
        self.stop_training = False

        self.history = History(dataset=dataset, training_details=training_details)

        self.batch = 0
        self.num_batches = 0
        self.last_epoch = 0
        self._start_time = time.time()
        self._time_batch_step = time.time()
        self._time_first_step = time.time()
        self._sum_time = 0
        self._sum_epoch_time = 0
        self.progress_name = "training"
        self.result = {
            'info': None,
            "train_usage": {
                "hard_usage": self.usage_info.get_usage(),
                "timings": {
                    "estimated_time": 0,
                    "elapsed_time": 0,
                    "still_time": 0,
                    "avg_epoch_time": 0,
                    "elapsed_epoch_time": 0,
                    "still_epoch_time": 0,
                    "epoch": {
                        "current": 0,
                        "total": 0
                    },
                    "batch": {
                        "current": 0,
                        "total": 0
                    },
                }
            },
            'train_data': None,
            'states': {}
        }
        # аттрибуты для чекпоинта
        self.checkpoint_config = training_details.base.architecture.parameters.checkpoint
        self.checkpoint_mode = self._get_checkpoint_mode()  # min max
        self.num_outputs = len(self.dataset.data.outputs.keys())
        self.metric_checkpoint = self.checkpoint_config.metric_name  # "val_mAP50" if self.is_yolo else "loss"

        self.samples_train = []
        self.samples_val = []
        self.samples_target_train = []
        self.samples_target_val = []

    def _get_checkpoint_mode(self):
        method_name = '_get_checkpoint_mode'
        try:
            print(method_name, self.checkpoint_config)
            if self.checkpoint_config.type == CheckpointTypeChoice.Loss:
                return 'min'
            elif self.checkpoint_config.type == CheckpointTypeChoice.Metrics:
                metric_name = self.checkpoint_config.metric_name
                return loss_metric_config.get("metric").get(metric_name).get("mode")
            else:
                print('\nClass FitCallback method _get_checkpoint_mode: No checkpoint types are found\n')
                return None
        except Exception as e:
            print_error('FitCallback', method_name, e)

    @staticmethod
    def _logs_predict_extract(logs, prefix):
        pred_on_batch = []
        for key in logs.keys():
            if key.startswith(prefix):
                pred_on_batch.append(logs[key])
        return pred_on_batch

    def _best_epoch_monitoring(self):
        method_name = '_best_epoch_monitoring'
        try:
            print(method_name)
            output = "output" if self.is_yolo else f"{self.checkpoint_config.layer}"
            checkpoint_type = self.checkpoint_config.type.name.lower()
            metric = self.checkpoint_config.metric_name.name
            indicator = self.checkpoint_config.indicator.name.lower()
            checkpoint_list = self.history.get_checkpoint_data(output, checkpoint_type, metric, indicator)
            if self.checkpoint_mode == 'min' and checkpoint_list[-1] == min(checkpoint_list):
                return True
            elif self.checkpoint_mode == "max" and checkpoint_list[-1] == max(checkpoint_list):
                return True
            else:
                return False
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _set_result_data(self, param: dict) -> None:
        method_name = '_set_result_data'
        try:
            for key in param.keys():
                if key in self.result.keys():
                    self.result[key] = param[key]
                elif key == "timings":
                    self.result["train_usage"]["timings"]["estimated_time"] = param[key][1] + param[key][2]
                    self.result["train_usage"]["timings"]["elapsed_time"] = param[key][1]
                    self.result["train_usage"]["timings"]["still_time"] = param[key][2]
                    self.result["train_usage"]["timings"]["avg_epoch_time"] = int(
                        self._sum_epoch_time / self.last_epoch)
                    self.result["train_usage"]["timings"]["elapsed_epoch_time"] = param[key][3]
                    self.result["train_usage"]["timings"]["still_epoch_time"] = param[key][4]
                    self.result["train_usage"]["timings"]["epoch"] = param[key][5]
                    self.result["train_usage"]["timings"]["batch"] = param[key][6]
            self.result["train_usage"]["hard_usage"] = self.usage_info.get_usage()
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _get_result_data(self):
        return self.result

    def _get_train_status(self) -> str:
        return self.training_detail.state.status

    def _get_predict(self, current_model=None):
        method_name = '_get_predict'
        try:
            print(method_name)
            # current_model = deploy_model if deploy_model else self.model
            if self.is_yolo:
                current_predict = [np.concatenate(elem, axis=0) for elem in zip(*self.samples_val)]
                current_target = [np.concatenate(elem, axis=0) for elem in zip(*self.samples_target_val)]
            else:
                # TODO: настроить вывод массивов их обучения, выводить словарь
                #  {'train_true': train_true, 'train_pred': train_pred, 'val_true': val_true, 'val_pred': val_pred}
                if self.dataset.data.use_generator:
                    current_predict = current_model.predict(
                        self.dataset.dataset.get('val').batch(1), batch_size=1)
                else:
                    current_predict = current_model.predict(self.dataset.X.get('val'), batch_size=self.batch_size)
                # current_predict = None
                current_target = None
            return current_predict, current_target
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _deploy_predict(self, presets_predict):
        method_name = '_deploy_predict'
        try:
            print(method_name)
            result = CreateArray().postprocess_results(
                array=presets_predict, options=self.dataset, save_path=str(self.training_detail.model_path),
                dataset_path=str(self.dataset.data.path))
            deploy_presets = []
            if result:
                deploy_presets = list(result.values())[0]
            return deploy_presets
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _create_form_data_for_dataframe_deploy(self):
        method_name = '_create_form_data_for_dataframe_deploy'
        try:
            print(method_name)
            form_data = []
            with open(os.path.join(self.dataset.data.path, "config.json"), "r", encoding="utf-8") as dataset_conf:
                dataset_info = json.load(dataset_conf).get("columns", {})
            for inputs, input_data in dataset_info.items():
                if int(inputs) not in list(self.dataset.data.outputs.keys()):
                    for column, column_data in input_data.items():
                        label = column
                        available = column_data.get("classes_names") if column_data.get("classes_names") else None
                        widget = "select" if available else "input"
                        input_type = "text"
                        if widget == "select":
                            table_column_data = {
                                "label": label,
                                "widget": widget,
                                "available": available
                            }
                        else:
                            table_column_data = {
                                "label": label,
                                "widget": widget,
                                "type": input_type
                            }
                        form_data.append(table_column_data)
            with open(os.path.join(self.training_detail.deploy_path, "form.json"), "w", encoding="utf-8") as form_file:
                json.dump(form_data, form_file, ensure_ascii=False)
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _create_cascade(self, **data):
        method_name = '_create_cascade'
        try:
            print(method_name)
            if self.dataset.data.alias not in ["imdb", "boston_housing", "reuters"]:
                if "Dataframe" in self.deploy_type:
                    self._create_form_data_for_dataframe_deploy()
                if self.is_yolo:
                    func_name = "object_detection"
                else:
                    func_name = decamelize(self.deploy_type)
                config = CascadeCreator()
                config.create_config(
                    deploy_path=self.training_detail.deploy_path,
                    model_path=self.training_detail.model_path,
                    func_name=func_name
                )
                config.copy_package(
                    deploy_path=self.training_detail.deploy_path,
                    model_path=self.training_detail.model_path
                )
                config.copy_script(
                    deploy_path=self.training_detail.deploy_path,
                    function_name=func_name
                )
                if self.deploy_type == ArchitectureChoice.TextSegmentation:
                    with open(os.path.join(self.training_detail.deploy_path, "format.txt"),
                              "w", encoding="utf-8") as format_file:
                        format_file.write(str(data.get("tags_map", "")))
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _prepare_deploy(self, model):
        method_name = '_prepare_deploy'
        try:
            print(method_name, self.training_detail.deploy_path)
            weight = None
            cascade_data = {"deploy_path": self.training_detail.deploy_path}
            for i in os.listdir(self.training_detail.model_path):
                if i[-3:] == '.h5' and 'best' in i:
                    weight = i
            if weight:
                model.load_weights(os.path.join(self.training_detail.model_path, weight))
            deploy_predict, y_true = self._get_predict(current_model=model)
            deploy_presets_data = self._deploy_predict(deploy_predict)
            out_deploy_presets_data = {"data": deploy_presets_data}
            if self.deploy_type == ArchitectureChoice.TextSegmentation:
                cascade_data.update({"tags_map": deploy_presets_data.get("color_map")})
                out_deploy_presets_data = {
                    "data": deploy_presets_data.get("data", {}),
                    "color_map": deploy_presets_data.get("color_map")
                }
            elif "Dataframe" in self.deploy_type:
                columns = []
                predict_column = ""
                for inp, input_columns in self.dataset.data.columns.items():
                    for column_name in input_columns.keys():
                        columns.append(column_name[len(str(inp)) + 1:])
                        if input_columns[column_name].__class__ == DatasetOutputsData:
                            predict_column = column_name[len(str(inp)) + 1:]
                if self.deploy_type == ArchitectureChoice.DataframeRegression:
                    tmp_data = list(zip(deploy_presets_data.get("preset"), deploy_presets_data.get("label")))
                    tmp_deploy = [{"preset": elem[0], "label": elem[1]} for elem in tmp_data]
                    out_deploy_presets_data = {"data": tmp_deploy}
                out_deploy_presets_data["columns"] = columns
                out_deploy_presets_data[
                    "predict_column"] = predict_column if predict_column else "Предсказанные значения"

            out_deploy_data = dict([
                ("path", Path(self.training_detail.deploy_path)),
                ("path_model", Path(self.training_detail.model_path)),
                ("type", self.deploy_type),
                ("data", out_deploy_presets_data)
            ])
            print(self.deploy_type, type(self.deploy_type))
            self.training_detail.deploy = DeployData(**out_deploy_data)
            self._create_cascade(**cascade_data)
        except Exception as e:
            print_error('FitCallback', method_name, e)

    @staticmethod
    def _estimate_step(current, start, now):
        method_name = '_estimate_step'
        try:
            # print(method_name)
            if current:
                _time_per_unit = (now - start) / current
            else:
                _time_per_unit = (now - start)
            return _time_per_unit
        except Exception as e:
            print_error('FitCallback', method_name, e)

    @staticmethod
    def eta_format(eta):
        method_name = 'eta_format'
        try:
            print(method_name)
            if eta > 3600:
                eta_format = '%d ч %02d мин %02d сек' % (eta // 3600,
                                                         (eta % 3600) // 60, eta % 60)
            elif eta > 60:
                eta_format = '%d мин %02d сек' % (eta // 60, eta % 60)
            else:
                eta_format = '%d сек' % eta
            return ' %s' % eta_format
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def update_progress(self, target, current, start_time, finalize=False, stop_current=0, stop_flag=False):
        method_name = 'update_progress'
        try:
            # print(method_name)
            """
            Updates the progress bar.
            """
            if finalize:
                _now_time = time.time()
                eta = _now_time - start_time
            else:
                _now_time = time.time()
                if stop_flag:
                    time_per_unit = self._estimate_step(stop_current, start_time, _now_time)
                else:
                    time_per_unit = self._estimate_step(current, start_time, _now_time)
                eta = time_per_unit * (target - current)
            return int(eta)
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def on_train_begin(self):
        method_name = 'on_train_begin'
        try:
            print(method_name, self.dataset.dataframe.keys())
            status = self._get_train_status()
            self._start_time = time.time()
            if status != "addtrain":
                self.batch = 1
            if not self.dataset.data.use_generator:
                if len(list(self.dataset.X['train'].values())[0]) % self.batch_size:
                    self.num_batches = len(list(self.dataset.X['train'].values())[0]) // self.batch_size + 1
                else:
                    self.num_batches = len(list(self.dataset.X['train'].values())[0]) // self.batch_size
            else:
                if len(self.dataset.dataframe['train']) % self.batch_size:
                    self.num_batches = len(self.dataset.dataframe['train']) // self.batch_size + 1
                else:
                    self.num_batches = len(self.dataset.dataframe['train']) // self.batch_size
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def on_epoch_begin(self):
        print('on_epoch_begin')
        self.last_epoch += 1
        self._time_first_step = time.time()

    def on_train_batch_end(self, batch, arrays=None, train_data_idxs=None):
        method_name = 'on_train_batch_end'
        try:
            if self._get_train_status() == "stopped":
                print('_get_train_status() == "stopped"')
                self.stop_training = True
                progress.pool(
                    self.progress_name,
                    percent=self.last_epoch / (
                        self.retrain_epochs if self._get_train_status() == "addtrain" else self.epochs) * 100,
                    message="Обучение остановлено пользователем, ожидайте остановку...",
                    finished=False,
                )
            else:
                # bt = time.time()
                msg_batch = {"current": batch, "total": self.num_batches}
                msg_epoch = {"current": self.last_epoch,
                             "total": self.retrain_epochs if self._get_train_status() == "addtrain"
                             else self.epochs}
                still_epoch_time = self.update_progress(self.num_batches, batch, self._time_first_step)
                elapsed_epoch_time = time.time() - self._time_first_step
                elapsed_time = time.time() - self._start_time
                estimated_time = self.update_progress(
                    self.num_batches * self.still_epochs, self.batch, self._start_time, finalize=True)
                # print('-- 1', method_name, round(time.time()-bt, 3))
                still_time = self.update_progress(
                    self.num_batches * self.still_epochs, self.batch, self._start_time)
                self.batch = batch
                if interactive.urgent_predict:
                    # print('\n interactive.urgent_predict\n')
                    # if self.is_yolo:
                    # self.samples_train.append(self._logs_predict_extract(logs, prefix='pred'))
                    # self.samples_target_train.append(self._logs_predict_extract(logs, prefix='target'))

                    # y_pred, y_true = self._get_predict()
                    train_batch_data = interactive.update_state(arrays=arrays, train_idx=train_data_idxs)
                else:
                    train_batch_data = interactive.update_state(arrays=None, train_idx=None)
                    # print("train_batch_data", train_batch_data)
                # print('-- 2', method_name, round(time.time() - bt, 3))
                if train_batch_data:
                    result_data = {
                        'timings': [estimated_time, elapsed_time, still_time,
                                    elapsed_epoch_time, still_epoch_time, msg_epoch, msg_batch],
                        'train_data': train_batch_data
                    }
                else:
                    result_data = {'timings': [estimated_time, elapsed_time, still_time,
                                               elapsed_epoch_time, still_epoch_time, msg_epoch, msg_batch]}

                # print('-- 3', method_name, round(time.time()-bt, 3))
                # print('batch', self.batch, result_data)
                self._set_result_data(result_data)
                self.training_detail.result = self._get_result_data()
                progress.pool(
                    self.progress_name,
                    percent=self.last_epoch / (
                        self.retrain_epochs if self._get_train_status() == "addtrain" else self.epochs
                    ) * 100,
                    message=f"Обучение. Эпоха {self.last_epoch} из "
                            f"{self.retrain_epochs if self._get_train_status() in ['addtrain', 'stopped'] else self.epochs}",
                    finished=False,
                )
                # print('-- 4', method_name, round(time.time() - bt, 3))
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def on_epoch_end(self, epoch, arrays=None, logs=None, train_data_idxs=None):
        method_name = 'on_epoch_end'
        try:
            print(method_name, epoch)
            # self.last_epoch = epoch
            # total_epochs = self.retrain_epochs if self._get_train_status() in ['addtrain', 'stopped'] else self.epochs
            if self.is_yolo:
                self.history.current_logs = logs
                # if self.last_epoch < total_epochs and not self.stop_training:
                #     self.samples_train = []
                #     self.samples_val = []
                #     self.samples_target_train = []
                #     self.samples_target_val = []
            else:
                self.history.current_basic_logs(epoch=epoch, arrays=arrays, train_idx=train_data_idxs)
            # print('\nFitCallback _update_log_history: start')
            # t = time.time()
            self.history.update_log_history()
            # print('\nFitCallback _update_log_history', round(time.time() - t, 3))
            if epoch == 1:
                interactive.log_history = self.history.get_history()
            current_epoch_time = time.time() - self._time_first_step
            self._sum_epoch_time += current_epoch_time
            # print('\nFitCallback interactive.update_state: start')
            # t = time.time()
            train_epoch_data = interactive.update_state(
                fit_logs=self.history.get_history(),
                arrays=arrays,
                current_epoch_time=current_epoch_time,
                on_epoch_end_flag=True,
                train_idx=train_data_idxs
            )
            # print('\nFitCallback interactive.update_state', round(time.time() - t, 3))
            # print(method_name, 'train_epoch_data', train_epoch_data)
            self._set_result_data({'train_data': train_epoch_data})
            progress.pool(
                self.progress_name,
                percent=self.last_epoch / (
                    self.retrain_epochs
                    if self._get_train_status() == "addtrain" or self._get_train_status() == "stopped" else self.epochs
                ) * 100,
                message=f"Обучение. Эпоха {self.last_epoch} из "
                        f"{self.retrain_epochs if self._get_train_status() in ['addtrain', 'stopped'] else self.epochs}",
                finished=False,
            )
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def save_best_weights(self):
        method_name = 'save_best_weights'
        try:
            print(method_name)
            if self.last_epoch > 1:
                if self._best_epoch_monitoring():
                    if not os.path.exists(self.model_path):
                        os.mkdir(self.model_path)
                    if not os.path.exists(os.path.join(self.training_detail.model_path, "deploy_presets")):
                        os.mkdir(os.path.join(self.training_detail.model_path, "deploy_presets"))
                    file_path_best: str = os.path.join(
                        self.training_detail.model_path, f"best_weights_{self.metric_checkpoint}.h5"
                    )
                    return file_path_best
        except Exception as e:
            print_error('FitCallback', method_name, e)
            print('\nself.model.save_weights failed', e)

    def on_train_end(self, model):
        method_name = 'on_train_end'
        try:
            # print(method_name)
            interactive.addtrain_epochs.append(self.last_epoch)
            self.history.save_logs()

            if self.last_epoch > 1:
                file_path_last: str = os.path.join(
                    self.training_detail.model_path, f"last_weights_{self.metric_checkpoint}.h5"
                )
                model.save_weights(file_path_last)
            if not os.path.exists(os.path.join(self.training_detail.model_path, "deploy_presets")):
                os.mkdir(os.path.join(self.training_detail.model_path, "deploy_presets"))
            # self._prepare_deploy(model)

            time_end = self.update_progress(
                self.num_batches * self.epochs, self.batch, self._start_time, finalize=True)
            self._sum_time += time_end
            total_epochs = self.retrain_epochs \
                if self._get_train_status() in ['addtrain', 'trained'] else self.epochs
            if self.stop_training:
                progress.pool(
                    self.progress_name,
                    message=f"Обучение остановлено. Эпоха {self.last_epoch} из {total_epochs}. Модель сохранена.",
                    data=self._get_result_data(),
                    finished=True,
                )
            else:
                percent = self.last_epoch / (
                    self.retrain_epochs
                    if self._get_train_status() == "addtrain" or self._get_train_status() == "stopped"
                    else self.epochs
                ) * 100
                # print('percent', percent, self.progress_name)

                self.training_detail.state.set("trained")
                self.training_detail.result = self._get_result_data()
                progress.pool(
                    self.progress_name,
                    percent=percent,
                    message=f"Обучение завершено. Эпоха {self.last_epoch} из {total_epochs}",
                    finished=True,
                )
        except Exception as e:
            print_error('FitCallback', method_name, e)
