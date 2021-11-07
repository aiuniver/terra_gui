import gc
import importlib
import json
import re
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
from tensorflow.keras.models import load_model

from terra_ai import progress
from terra_ai.callbacks.interactive_callback import InteractiveCallback
from terra_ai.callbacks.utils import loss_metric_config
from terra_ai.data.datasets.dataset import DatasetData, DatasetOutputsData
from terra_ai.data.datasets.extra import LayerOutputTypeChoice, LayerInputTypeChoice
from terra_ai.data.deploy.tasks import DeployData
from terra_ai.data.modeling.model import ModelDetailsData, ModelData
from terra_ai.data.training.extra import CheckpointIndicatorChoice, CheckpointTypeChoice, MetricChoice, ArchitectureChoice
from terra_ai.data.training.train import TrainingDetailsData, StateData

from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.deploy.create_deploy_package import CascadeCreator
from terra_ai.exceptions.deploy import MethodNotImplementedException
from terra_ai.modeling.validator import ModelValidator
# from terra_ai.training.customcallback import InteractiveCallback
from terra_ai.training.customlosses import DiceCoef, UnscaledMAE, BalancedRecall, BalancedDiceCoef, \
    BalancedPrecision, BalancedFScore, FScore
from terra_ai.training.yolo_utils import create_yolo, CustomModelYolo, compute_loss, get_mAP
from terra_ai.exceptions import training as exceptions, terra_exception

__version__ = 0.02

from terra_ai.utils import camelize, decamelize

interactive = InteractiveCallback()


class GUINN:
    """
    GUINN: class, for train model
    """

    def __init__(self) -> None:
        """
        GUINN init method
        """
        self.callbacks = []
        self.chp_monitor = 'loss'

        """
        For model settings
        """
        self.nn_name: str = ''
        self.dataset = None
        self.deploy_type = None
        self.model: Optional[Model] = None
        self.optimizer = None
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
        """
        Logs
        """
        self.history: dict = {}
        self.progress_name = "training"

    @staticmethod
    def _check_metrics(metrics: list, options: DatasetOutputsData, num_classes: int = 2, ) -> list:
        output = []
        for metric in metrics:
            if metric == MetricChoice.MeanIoU.value:
                output.append(getattr(importlib.import_module("tensorflow.keras.metrics"), metric)(num_classes))
            elif metric == MetricChoice.DiceCoef:
                output.append(DiceCoef())
            # elif metric == MetricChoice.RecallPercent:
            #     output.append(RecallPercent())
            elif metric == MetricChoice.BalancedPrecision:
                output.append(BalancedPrecision())
            elif metric == MetricChoice.BalancedFScore:
                output.append(BalancedFScore())
            elif metric == MetricChoice.FScore:
                output.append(FScore())
            elif metric == MetricChoice.BalancedRecall:
                output.append(BalancedRecall())
            elif metric == MetricChoice.BalancedDiceCoef:
                output.append(BalancedDiceCoef(encoding=options.encoding.value))
            elif metric == MetricChoice.UnscaledMAE:
                output.append(UnscaledMAE())
            elif metric == MetricChoice.mAP50 or metric == MetricChoice.mAP95:
                pass
            else:
                output.append(getattr(importlib.import_module("tensorflow.keras.metrics"), metric)())
        return output

    def _set_training_params(self, dataset: DatasetData, params: TrainingDetailsData) -> None:
        self.dataset = self._prepare_dataset(dataset=dataset, model_path=params.model_path,
                                             state=params.state.status)
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
            if self.callbacks[0].last_epoch - 1 >= self.sum_epoch:
                self.sum_epoch += params.base.epochs
            if (self.callbacks[0].last_epoch - 1) < self.sum_epoch:
                self.epochs = self.sum_epoch - self.callbacks[0].last_epoch + 1
            else:
                self.epochs = params.base.epochs
        else:
            self.epochs = params.base.epochs
        self.batch_size = params.base.batch
        self.set_optimizer(params)

        for output_layer in params.base.architecture.outputs_dict:
            self.metrics.update({
                str(output_layer["id"]):
                    self._check_metrics(
                        metrics=output_layer.get("metrics", []),
                        num_classes=output_layer.get("classes_quantity"),
                        options=self.dataset.data.outputs.get(output_layer["id"])
                    )
            })
            self.loss.update({str(output_layer["id"]): output_layer["loss"]})

        interactive.set_attributes(dataset=self.dataset, metrics=self.metrics, losses=self.loss,
                                   dataset_path=dataset.path, params=params)

    def _set_callbacks(self, dataset: PrepareDataset, dataset_data: DatasetData,
                       train_details: TrainingDetailsData, initial_model=None) -> None:
        progress.pool(self.progress_name, finished=False, data={'status': 'Добавление колбэков...'})
        retrain_epochs = self.sum_epoch if train_details.state.status == "addtrain" else self.epochs

        callback = FitCallback(dataset=dataset, dataset_data=dataset_data, retrain_epochs=retrain_epochs,
                               training_details=train_details, model_name=self.nn_name,
                               deploy_type=self.deploy_type, initialed_model=initial_model)
        self.callbacks = [callback]
        progress.pool(self.progress_name, finished=False, data={'status': 'Добавление колбэков выполнено'})

    @staticmethod
    def _set_deploy_type(dataset: PrepareDataset) -> str:
        data = dataset.data
        inp_tasks = []
        out_tasks = []
        for key, val in data.inputs.items():
            if val.task == LayerInputTypeChoice.Dataframe:
                tmp = []
                for value in data.columns[key].values():
                    tmp.append(value.get("task"))
                unique_vals = list(set(tmp))
                if len(unique_vals) == 1 and unique_vals[0] in LayerInputTypeChoice.__dict__.keys() and unique_vals[0] \
                        in [LayerInputTypeChoice.Image, LayerInputTypeChoice.Text,
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
            raise MethodNotImplementedException(__method=inp_task_name + out_task_name, __class="ArchitectureChoice")
        return deploy_type

    @staticmethod
    def _prepare_dataset(dataset: DatasetData, model_path: Path, state: str) -> PrepareDataset:
        prepared_dataset = PrepareDataset(data=dataset, datasets_path=dataset.path)
        prepared_dataset.prepare_dataset()
        if state != "addtrain":
            prepared_dataset.deploy_export(os.path.join(model_path, "dataset"))
        return prepared_dataset

    def _set_model(self, model: ModelDetailsData, train_details: TrainingDetailsData) -> ModelData:
        if train_details.state.status == "training":
            validator = ModelValidator(model)
            train_model = validator.get_keras_model()
        else:
            model_file = f"{self.nn_name}.trm"
            train_model = load_model(os.path.join(train_details.model_path, f"{model_file}"),
                                     compile=False)
            weight = None
            for i in os.listdir(train_details.model_path):
                if i[-3:] == '.h5' and 'last' in i:
                    weight = i
            if weight:
                train_model.load_weights(os.path.join(train_details.model_path, weight))
        return train_model

    @staticmethod
    def _save_params_for_deploy(params: TrainingDetailsData):
        with open(os.path.join(params.model_path, "config.train"), "w", encoding="utf-8") as train_config:
            json.dump(params.base.native(), train_config)

    def set_optimizer(self, params: TrainingDetailsData) -> None:
        """
        Set optimizer method for using terra w/o gui
        """
        optimizer_object = getattr(keras.optimizers, params.base.optimizer.type.value)
        self.optimizer = optimizer_object(**params.base.optimizer.parameters_dict)

    def save_model(self, model_path: Path) -> None:
        """
        Saving last model on each epoch end

        Returns:
            None
        """
        model = f"{self.nn_name}.trm"
        file_path_model: str = os.path.join(model_path, f"{model}")
        self.model.save(file_path_model)

    def _kill_last_training(self, state):
        for one_thread in threading.enumerate():
            if one_thread.getName() == "current_train":
                current_status = state.state.status
                state.state.set("stopped")
                progress.pool(self.progress_name, message="Найдено незавершенное обучение. Идет очистка. Подождите.")
                one_thread.join()
                state.state.set(current_status)

    @staticmethod
    def _get_val_batch_size(batch_size, len_val):
        def issimple(a):
            lst = []
            for i in range(2, a):
                if a % i == 0:
                    if issimple(i) == []:
                        lst.append(i)
            return lst

        min_step = 0
        for i in range(3):
            r = issimple(len_val - i)
            if len(r):
                try:
                    min_step = min(r)
                    if len_val // min_step > batch_size:
                        for k in r:
                            if len_val // k <= batch_size:
                                min_step = k
                                break
                            else:
                                min_step = len_val
                    break
                except ValueError:
                    pass
            else:
                min_step = len_val
        val_batch_size = len_val // min_step
        return val_batch_size

    def terra_fit(self,
                  dataset: DatasetData,
                  gui_model: ModelDetailsData,
                  training: TrainingDetailsData
                  ) -> dict:
        """
        This method created for using wth externally compiled models

        Args:
            dataset: DatasetData
            gui_model: Keras model for fit - ModelDetailsData
            training: TrainingDetailsData

        Return:
            dict
        """
        self._kill_last_training(state=training)
        progress.pool.reset(self.progress_name)

        if training.state.status != "addtrain":
            self._save_params_for_deploy(params=training)

        self.nn_cleaner(retrain=True if training.state.status == "training" else False)
        self._set_training_params(dataset=dataset, params=training)

        self.model = self._set_model(model=gui_model, train_details=training)
        if training.state.status == "training":
            self.save_model(training.model_path)

        self.base_model_fit(training_details=training, dataset=self.dataset, dataset_data=dataset, verbose=0)
        return {"dataset": self.dataset, "metrics": self.metrics, "losses": self.loss}

    def nn_cleaner(self, retrain: bool = False) -> None:
        keras.backend.clear_session()
        self.dataset = None
        self.deploy_type = None
        self.model = None
        if retrain:
            self.sum_epoch = 0
            self.chp_monitor = ""
            self.optimizer = None
            self.loss = {}
            self.metrics = {}
            self.callbacks = []
            self.history = {}
            interactive.clear_history()
        gc.collect()

    def get_nn(self):
        self.nn_cleaner(retrain=True)

        return self

    @progress.threading
    def base_model_fit(self, training_details: TrainingDetailsData, dataset: PrepareDataset,
                       dataset_data: DatasetData, verbose=0) -> None:

        yolo_arch = True if self.deploy_type in [ArchitectureChoice.YoloV3, ArchitectureChoice.YoloV4] else False
        model_yolo = None

        threading.enumerate()[-1].setName("current_train")
        progress.pool(self.progress_name, finished=False, data={'status': 'Компиляция модели ...'})
        if yolo_arch:
            warmup_epoch = training_details.base.architecture.parameters.yolo.train_warmup_epochs
            lr_init = training_details.base.architecture.parameters.yolo.train_lr_init
            lr_end = training_details.base.architecture.parameters.yolo.train_lr_end
            iou_thresh = training_details.base.architecture.parameters.yolo.yolo_iou_loss_thresh

            yolo = create_yolo(self.model, input_size=416, channels=3, training=True,
                               classes=self.dataset.data.outputs.get(2).classes_names,
                               version=self.dataset.instructions.get(2).get('2_object_detection').get('yolo'))
            model_yolo = CustomModelYolo(yolo, self.dataset, self.dataset.data.outputs.get(2).classes_names,
                                         training_details.base.epochs, training_details.base.batch,
                                         warmup_epoch=warmup_epoch, lr_init=lr_init,
                                         lr_end=lr_end, iou_thresh=iou_thresh)
            model_yolo.compile(optimizer=self.optimizer,
                               loss=compute_loss)
        else:
            self.model.compile(loss=self.loss,
                               optimizer=self.optimizer,
                               metrics=self.metrics
                               )
        progress.pool(self.progress_name, finished=False, data={'status': 'Компиляция модели выполнена'})
        self._set_callbacks(
            dataset=dataset, dataset_data=dataset_data, train_details=training_details,
            initial_model=self.model if yolo_arch else None
        )
        progress.pool(self.progress_name, finished=False, data={'status': 'Начало обучения ...'})
        if self.dataset.data.use_generator:
            print('use generator')
            critical_val_size = len(self.dataset.dataframe.get("val"))
            buffer_size = 100
        else:
            print('dont use generator')
            critical_val_size = len(self.dataset.dataset.get('val'))
            buffer_size = 1000

        if (critical_val_size == self.batch_size) or ((critical_val_size % self.batch_size) == 0):
            self.val_batch_size = self.batch_size
        elif critical_val_size < self.batch_size:
            self.val_batch_size = critical_val_size
        else:
            self.val_batch_size = self._get_val_batch_size(self.batch_size, critical_val_size)

        trained_model = model_yolo if model_yolo else self.model

        try:
            self.history = trained_model.fit(
                self.dataset.dataset.get('train').shuffle(buffer_size).batch(
                    self.batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE).take(-1),
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                validation_data=self.dataset.dataset.get('val').batch(
                    self.val_batch_size,
                    drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE).take(-1),
                validation_batch_size=self.val_batch_size,
                epochs=self.epochs,
                verbose=verbose,
                callbacks=self.callbacks
            )
        except Exception as error:
            training_details.state.set("stopped") if training_details.state.status == "addtrain" \
                else training_details.state.set("no_train")
            progress.pool(self.progress_name, error=terra_exception(error).__str__(), finished=True)

        if (training_details.state.status == "stopped" and
            self.callbacks[0].last_epoch < training_details.base.epochs) or \
                (training_details.state.status == "trained" and
                 self.callbacks[0].last_epoch - 1 == training_details.base.epochs):
            self.sum_epoch = training_details.base.epochs
        with open(os.path.join(training_details.path, "test_train_details"), "wb", encoding="utf-8") as ff:
            json.dump(training_details, ff)
        # training_details.deploy = self.callbacks[0].deploy


class MemoryUsage:
    def __init__(self, debug=False):
        self.debug = debug
        try:
            N.nvmlInit()
            self.gpu = True
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


class FitCallback(keras.callbacks.Callback):
    """CustomCallback for all task type"""

    def __init__(self, dataset: PrepareDataset, dataset_data: DatasetData,
                 training_details: TrainingDetailsData, retrain_epochs: int = None,
                 model_name: str = "model", deploy_type: str = "", initialed_model=None):
        super().__init__()
        print('\n FitCallback')
        self.usage_info = MemoryUsage(debug=False)
        self.training_detail = training_details
        self.dataset = dataset
        self.dataset_data = dataset_data
        self.deploy_type = deploy_type
        self.is_yolo = True if self.deploy_type in [ArchitectureChoice.YoloV3, ArchitectureChoice.YoloV4] else False
        self.batch_size = training_details.base.batch
        self.epochs = training_details.base.epochs
        self.retrain_epochs = retrain_epochs
        self.still_epochs = training_details.base.epochs
        self.nn_name = model_name

        self.batch = 0
        self.num_batches = 0
        self.last_epoch = 1
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
        self.checkpoint_mode = self._get_checkpoint_mode(
            training_details.base.architecture.parameters.checkpoint.native()
        )  # min max
        self.num_outputs = len(self.dataset.data.outputs.keys())
        self.metric_checkpoint = "val_mAP50" if self.is_yolo else "loss"

        self.log_history = self._load_logs()

        # yolo params
        self.yolo_model = initialed_model
        self.samples_train = []
        self.samples_val = []
        self.samples_target_train = []
        self.samples_target_val = []

    @staticmethod
    def _get_checkpoint_mode(checkpoint_data: dict):
        if checkpoint_data.get("type") == CheckpointTypeChoice.Loss:
            return 'min'
        elif checkpoint_data.get("type") == CheckpointTypeChoice.Metrics:
            metric_name = checkpoint_data.get("metric_name")
            return loss_metric_config.get("metric").get(metric_name).get("mode")
        else:
            print('\nClass FitCallback method _get_checkpoint_mode: No checkpoint types are found\n')
            return None

    def _get_metric_name_checkpoint(self, logs: dict):
        """Поиск среди fit_logs нужного параметра"""
        checkpoint_config = self.training_detail.base.architecture.parameters.checkpoint.native()
        for log in logs.keys():
            if checkpoint_config.get("type") == CheckpointTypeChoice.Loss and \
                    checkpoint_config.get("indicator") == CheckpointIndicatorChoice.Val and \
                    'val' in log and 'loss' in log:
                if self.num_outputs == 1:
                    self.metric_checkpoint = log
                    break
                else:
                    if f"{checkpoint_config.get('layer')}" in log:
                        self.metric_checkpoint = log
                        break

            elif checkpoint_config.get("type") == CheckpointTypeChoice.Loss and \
                    checkpoint_config.get("indicator") == CheckpointIndicatorChoice.Train and \
                    'val' not in log and 'loss' in log:
                if self.num_outputs == 1:
                    self.metric_checkpoint = log
                    break
                else:
                    if f"{checkpoint_config.get('layer')}" in log:
                        self.metric_checkpoint = log
                        break

            elif checkpoint_config.get("type") == CheckpointTypeChoice.Metrics and \
                    checkpoint_config.get("indicator") == CheckpointIndicatorChoice.Val and \
                    "val" in log:
                camelize_log = self._clean_and_camelize_log_name(log)
                if self.num_outputs == 1 and camelize_log == checkpoint_config.get("metric_name"):
                    self.metric_checkpoint = log
                    break
                else:
                    if f"{self.checkpoint_config.get('layer')}" in log and \
                            camelize_log == checkpoint_config.get("metric_name"):
                        self.metric_checkpoint = log
                        break

            elif checkpoint_config.get("type") == CheckpointTypeChoice.Metrics and \
                    checkpoint_config.get("indicator") == CheckpointIndicatorChoice.Train and \
                    'val' not in log:
                camelize_log = self._clean_and_camelize_log_name(log)
                if self.num_outputs == 1 and camelize_log == checkpoint_config.get("metric_name"):
                    self.metric_checkpoint = log
                    break
                else:
                    if f"{checkpoint_config.get('layer')}" in log and \
                            camelize_log == checkpoint_config.get('metric_name'):
                        self.metric_checkpoint = log
                        break
            else:
                pass

    @staticmethod
    def _clean_and_camelize_log_name(fit_log_name):
        """Камелизатор исключительно для fit_logs"""
        if re.search(r'_\d+$', fit_log_name):
            end = len(f"_{fit_log_name.split('_')[-1]}")
            fit_log_name = fit_log_name[:-end]
        if "val" in fit_log_name:
            fit_log_name = fit_log_name[4:]
        if re.search(r'^\d+_', fit_log_name):
            start = len(f"{fit_log_name.split('_')[0]}_")
            fit_log_name = fit_log_name[start:]
        return camelize(fit_log_name)

    def _fill_log_history(self, epoch, logs):
        """Заполнение истории логов"""
        if epoch == 1:
            self.log_history = {
                'epoch': [],
                'logs': {}
            }
            for metric in logs:
                self.log_history['logs'][metric] = []
            self._get_metric_name_checkpoint(logs)
        self.log_history['epoch'].append(epoch)
        for metric in logs:
            self.log_history['logs'][metric].append(float(logs.get(metric)))

    def _save_logs(self):
        logs = {
            "fit_log": self.log_history,
            "interactive_log": interactive.log_history,
            "progress_table": interactive.progress_table,
            "addtrain_epochs": interactive.addtrain_epochs
        }
        self.training_detail.logs = logs

        interactive_path = os.path.join(self.training_detail.model_path, "interactive.history")
        if not os.path.exists(interactive_path):
            os.mkdir(interactive_path)
        with open(os.path.join(self.training_detail.model_path, "log.history"), "w", encoding="utf-8") as history:
            json.dump(self.log_history, history)
        with open(os.path.join(interactive_path, "log.int"), "w", encoding="utf-8") as log:
            json.dump(interactive.log_history, log)
        with open(os.path.join(interactive_path, "table.int"), "w", encoding="utf-8") as table:
            json.dump(interactive.progress_table, table)
        with open(os.path.join(interactive_path, "addtraining.int"), "w", encoding="utf-8") as addtraining:
            json.dump({"addtrain_epochs": interactive.addtrain_epochs}, addtraining)

    def _load_logs(self):
        if self.training_detail.state.status == "addtrain":
            if self.training_detail.logs:
                logs = self.training_detail.logs.get("fit_log")
                interactive.log_history = self.training_detail.logs.get("interactive_log")
                interactive.progress_table = self.training_detail.logs.get("progress_table")
                interactive.addtrain_epochs = self.training_detail.logs.get("addtrain_epochs")
            else:
                interactive_path = os.path.join(self.training_detail.model_path, "interactive.history")
                with open(os.path.join(self.training_detail.model_path, "log.history"), "r", encoding="utf-8") as history:
                    logs = json.load(history)
                with open(os.path.join(interactive_path, "log.int"), "r", encoding="utf-8") as int_log:
                    interactive.log_history = json.load(int_log)
                with open(os.path.join(interactive_path, "table.int"), "r", encoding="utf-8") as table_int:
                    interactive.progress_table = json.load(table_int)
                with open(os.path.join(interactive_path, "addtraining.int"), "r", encoding="utf-8") as addtraining_int:
                    interactive.addtrain_epochs = json.load(addtraining_int)["addtrain_epochs"]
            self.last_epoch = max(logs.get('epoch')) + 1
            self.still_epochs = self.retrain_epochs - self.last_epoch + 1
            self._get_metric_name_checkpoint(logs.get('logs'))
            return logs
        else:
            return {
                'epoch': [],
                'logs': {}
            }

    @staticmethod
    def _logs_predict_extract(logs, prefix):
        pred_on_batch = []
        for key in logs.keys():
            if key.startswith(prefix):
                pred_on_batch.append(logs[key])
        return pred_on_batch

    @staticmethod
    def _logs_losses_extract(logs, prefixes: list):
        losses = {}
        for key in logs.keys():
            if key.find(prefixes[0]) != -1 or key.find(prefixes[1]) != -1:
                pass
            else:
                losses.update({key: logs[key]})
        return losses

    def _best_epoch_monitoring(self, logs):
        """Оценка текущей эпохи"""
        try:
            if logs.get(self.metric_checkpoint):
                if self.checkpoint_mode == 'min' and \
                        logs.get(self.metric_checkpoint) < min(self.log_history.get("logs").get(self.metric_checkpoint)):
                    return True
                elif self.checkpoint_mode == 'max' and \
                        logs.get(self.metric_checkpoint) > max(self.log_history.get("logs").get(self.metric_checkpoint)):
                    return True
                else:
                    return False
            else:
                return False
        except Exception as e:
            print('\n_best_epoch_monitoring failed', e)

    def _set_result_data(self, param: dict) -> None:
        for key in param.keys():
            if key in self.result.keys():
                self.result[key] = param[key]
            elif key == "timings":
                self.result["train_usage"]["timings"]["estimated_time"] = param[key][1] + param[key][2]
                self.result["train_usage"]["timings"]["elapsed_time"] = param[key][1]
                self.result["train_usage"]["timings"]["still_time"] = param[key][2]
                self.result["train_usage"]["timings"]["avg_epoch_time"] = int(self._sum_epoch_time / self.last_epoch)
                self.result["train_usage"]["timings"]["elapsed_epoch_time"] = param[key][3]
                self.result["train_usage"]["timings"]["still_epoch_time"] = param[key][4]
                self.result["train_usage"]["timings"]["epoch"] = param[key][5]
                self.result["train_usage"]["timings"]["batch"] = param[key][6]
        self.result["train_usage"]["hard_usage"] = self.usage_info.get_usage()

    def _get_result_data(self):
        self.result["states"] = self.training_detail.state.native()
        return self.result

    def _get_train_status(self) -> str:
        return self.training_detail.state.status

    def _get_predict(self, deploy_model=None):
        current_model = deploy_model if deploy_model else self.model
        if self.is_yolo:
            current_predict = [np.concatenate(elem, axis=0) for elem in zip(*self.samples_val)]
            current_target = [np.concatenate(elem, axis=0) for elem in zip(*self.samples_target_val)]
        else:
            if self.dataset.data.use_generator:
                current_predict = current_model.predict(self.dataset.dataset.get('val').batch(1),
                                                        batch_size=1)
            else:
                current_predict = current_model.predict(self.dataset.X.get('val'), batch_size=self.batch_size)
            current_target = None
        return current_predict, current_target

    def _deploy_predict(self, presets_predict):
        result = CreateArray().postprocess_results(array=presets_predict,
                                                   options=self.dataset,
                                                   save_path=str(self.training_detail.model_path),
                                                   dataset_path=str(self.dataset_path))
        deploy_presets = []
        if result:
            deploy_presets = list(result.values())[0]
        return deploy_presets

    def _create_form_data_for_dataframe_deploy(self):
        form_data = []
        with open(os.path.join(self.dataset_path, "config.json"), "r", encoding="utf-8") as dataset_conf:
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

    def _create_cascade(self, **data):
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

    def _prepare_deploy(self):
        weight = None
        cascade_data = {"deploy_path": self.training_detail.deploy_path}
        for i in os.listdir(self.training_detail.model_path):
            if i[-3:] == '.h5' and 'best' in i:
                weight = i
        if weight:
            if self.yolo_model:
                self.yolo_model.load_weights(os.path.join(self.training_detail.model_path, weight))
            else:
                self.model.load_weights(os.path.join(self.training_detail.model_path, weight))
        deploy_predict, y_true = self._get_predict()
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
            for input, input_columns in self.dataset.data.columns.items():
                for column_name in input_columns.keys():
                    columns.append(column_name[len(str(input)) + 1:])
                    if input_columns[column_name].__class__ == DatasetOutputsData:
                        predict_column = column_name[len(str(input)) + 1:]
            if self.deploy_type == ArchitectureChoice.DataframeRegression:
                tmp_data = list(zip(deploy_presets_data.get("preset"), deploy_presets_data.get("label")))
                tmp_deploy = [{"preset": elem[0], "label": elem[1]} for elem in tmp_data]
                out_deploy_presets_data = {"data": tmp_deploy}
            out_deploy_presets_data["columns"] = columns
            out_deploy_presets_data["predict_column"] = predict_column if predict_column else "Предсказанные значения"

        out_deploy_data = dict([
            ("path", self.training_detail.deploy_path),
            ("type", self.deploy_type),
            ("data", out_deploy_presets_data)
        ])
        self.training_detail.deploy = DeployData(**out_deploy_data)
        self._create_cascade(**cascade_data)

    @staticmethod
    def _estimate_step(current, start, now):
        if current:
            _time_per_unit = (now - start) / current
        else:
            _time_per_unit = (now - start)
        return _time_per_unit

    @staticmethod
    def eta_format(eta):
        if eta > 3600:
            eta_format = '%d ч %02d мин %02d сек' % (eta // 3600,
                                                     (eta % 3600) // 60, eta % 60)
        elif eta > 60:
            eta_format = '%d мин %02d сек' % (eta // 60, eta % 60)
        else:
            eta_format = '%d сек' % eta
        return ' %s' % eta_format

    def update_progress(self, target, current, start_time, finalize=False, stop_current=0, stop_flag=False):
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

    def on_train_begin(self, logs=None):
        status = self._get_train_status()
        self._start_time = time.time()
        if status != "addtrain":
            self.batch = 0
        if not self.dataset.data.use_generator:
            self.num_batches = len(list(self.dataset.X.get('train').values())[0]) // self.batch_size
        else:
            self.num_batches = len(self.dataset.dataframe['train']) // self.batch_size

    def on_epoch_begin(self, epoch, logs=None):
        self._time_first_step = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if self._get_train_status() == "stopped":
            self.model.stop_training = True
            msg = f'ожидайте остановку...'
            # self.batch += 1
            self._set_result_data({'info': f"'Обучение остановлено пользователем, '{msg}"})
        else:
            msg_batch = {"current": batch + 1, "total": self.num_batches}
            msg_epoch = {"current": self.last_epoch,
                         "total": self.retrain_epochs if self._get_train_status() == "addtrain"
                         else self.epochs}
            still_epoch_time = self.update_progress(self.num_batches, batch, self._time_first_step)
            elapsed_epoch_time = time.time() - self._time_first_step
            elapsed_time = time.time() - self._start_time
            estimated_time = self.update_progress(self.num_batches * self.still_epochs,
                                                  self.batch, self._start_time, finalize=True)

            still_time = self.update_progress(self.num_batches * self.still_epochs,
                                              self.batch, self._start_time)
            self.batch += 1
            if interactive.urgent_predict:

                if self.is_yolo:
                    self.samples_train.append(self._logs_predict_extract(logs, prefix='pred'))
                    self.samples_target_train.append(self._logs_predict_extract(logs, prefix='target'))

                y_pred, y_true = self._get_predict()
                train_batch_data = interactive.update_state(y_pred=y_pred, y_true=y_true)
            else:
                train_batch_data = interactive.update_state(y_pred=None)
            if train_batch_data:
                result_data = {
                    'timings': [estimated_time, elapsed_time, still_time,
                                elapsed_epoch_time, still_epoch_time, msg_epoch, msg_batch],
                    'train_data': train_batch_data
                }
            else:
                result_data = {'timings': [estimated_time, elapsed_time, still_time,
                                           elapsed_epoch_time, still_epoch_time, msg_epoch, msg_batch]}
            self._set_result_data(result_data)
            self.training_detail.result = self._get_result_data()
            progress.pool(
                self.progress_name,
                percent=(self.last_epoch - 1) / (
                    self.retrain_epochs if self._get_train_status() == "addtrain" else self.epochs
                ) * 100,
                message=f"Обучение. Эпоха {self.last_epoch} из "
                        f"{self.retrain_epochs if self._get_train_status() in ['addtrain', 'stopped'] else self.epochs}",
                data=self._get_result_data(),
                finished=False,
            )

    def on_test_batch_end(self, batch, logs=None):
        if self.is_yolo:
            self.samples_val.append(self._logs_predict_extract(logs, prefix='pred'))
            self.samples_target_val.append(self._logs_predict_extract(logs, prefix='target'))

    def on_epoch_end(self, epoch, logs=None):
        """
        Returns:
            {}:
        """
        y_pred, y_true = self._get_predict()
        total_epochs = self.retrain_epochs if self._get_train_status() in ['addtrain',
                                                                           'stopped'] else self.epochs
        if self.is_yolo:
            mAP = get_mAP(self.model, self.dataset, score_threshold=0.05, iou_threshold=[0.50],
                          TRAIN_CLASSES=self.dataset.data.outputs.get(2).classes_names, dataset_path=self.dataset_path)
            interactive_logs = self._logs_losses_extract(logs, prefixes=['pred', 'target'])
            # interactive_logs.update({'mAP': mAP})
            interactive_logs.update(mAP)
            # output_path = self.image_path.format(epoch)
            if self.last_epoch < total_epochs and not self.model.stop_training:
                self.samples_train = []
                self.samples_val = []
                self.samples_target_train = []
                self.samples_target_val = []
        else:
            interactive_logs = copy.deepcopy(logs)

        interactive_logs['epoch'] = self.last_epoch
        current_epoch_time = time.time() - self._time_first_step
        self._sum_epoch_time += current_epoch_time
        train_epoch_data = interactive.update_state(
            fit_logs=interactive_logs,
            y_pred=y_pred,
            y_true=y_true,
            current_epoch_time=current_epoch_time,
            on_epoch_end_flag=True
        )
        self._set_result_data({'train_data': train_epoch_data})
        # print('/nprogress.pool', self.last_epoch, self.retrain_epochs, self.epochs)
        progress.pool(
            self.progress_name,
            percent=(self.last_epoch - 1) / (
                self.retrain_epochs if self._get_train_status() == "addtrain" or self._get_train_status() == "stopped"
                else self.epochs
            ) * 100,
            message=f"Обучение. Эпоха {self.last_epoch} из "
                    f"{self.retrain_epochs if self._get_train_status() in ['addtrain', 'stopped'] else self.epochs}",
            data=self._get_result_data(),
            finished=False,
        )

        # сохранение лучших весов
        if self.last_epoch > 1:
            try:
                if self._best_epoch_monitoring(interactive_logs):
                    if not os.path.exists(os.path.join(self.training_detail.model_path, "deploy_presets")):
                        os.mkdir(os.path.join(self.training_detail.model_path, "deploy_presets"))
                    file_path_best: str = os.path.join(
                        self.training_detail.model_path, f"best_weights_{self.metric_checkpoint}.h5"
                    )
                    if self.yolo_model:
                        self.yolo_model.save_weights(file_path_best)
                    else:
                        self.model.save_weights(file_path_best)
                    print(f"\nEpoch {self.last_epoch} - best weights was successfully saved\n")
            except Exception as e:
                print('\nself.model.save_weights failed', e)
        self._fill_log_history(self.last_epoch, interactive_logs)
        self.last_epoch += 1

    def on_train_end(self, logs=None):
        interactive.addtrain_epochs.append(self.last_epoch - 1)
        self._save_logs()

        if (self.last_epoch - 1) > 1:
            file_path_last: str = os.path.join(
                self.training_detail.model_path, f"last_weights_{self.metric_checkpoint}.h5"
            )
            if self.yolo_model:
                self.yolo_model.save_weights(file_path_last)
            else:
                self.model.save_weights(file_path_last)
        if not os.path.exists(os.path.join(self.training_detail.model_path, "deploy_presets")):
            os.mkdir(os.path.join(self.training_detail.model_path, "deploy_presets"))
        self._prepare_deploy()

        time_end = self.update_progress(self.num_batches * self.epochs + 1,
                                        self.batch, self._start_time, finalize=True)
        self._sum_time += time_end
        total_epochs = self.retrain_epochs if self._get_train_status() in ['addtrain',
                                                                           'trained'] else self.epochs
        if self.model.stop_training:
            self._set_result_data({'info': f"Обучение остановлено. Модель сохранена."})
            progress.pool(
                self.progress_name,
                message=f"Обучение остановлено. Эпоха {self.last_epoch - 1} из "
                        f"{total_epochs}",
                data=self._get_result_data(),
                finished=True,
            )
        else:
            if self._get_train_status() == "retrain":
                msg = f'Затрачено времени на обучение: ' \
                      f'{self.eta_format(time_end)} '
            else:
                msg = f'Затрачено времени на обучение: ' \
                      f'{self.eta_format(self._sum_time)} '
            self._set_result_data({'info': f"Обучение закончено. {msg}"})
            percent = (self.last_epoch - 1) / (
                self.retrain_epochs if self._get_train_status() ==
                                       "addtrain" or self._get_train_status() == "stopped"
                else self.epochs
            ) * 100

            # if os.path.exists(self.save_model_path) and interactive.deploy_presets_data:
            #     with open(os.path.join(self.save_model_path, "config.presets"), "w", encoding="utf-8") as presets:
            #         presets.write(str(interactive.deploy_presets_data))
            self.training_detail.state.set("trained")
            self.training_detail.result = self._get_result_data()
            progress.pool(
                self.progress_name,
                percent=percent,
                message=f"Обучение завершено. Эпоха {self.last_epoch - 1} из "
                        f"{total_epochs}",
                data=self._get_result_data(),
                finished=True,
            )
