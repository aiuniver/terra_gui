import gc
import importlib
import json
import os
import re
import shutil
import sys

import copy
import os
import psutil
import time
import pynvml as N

from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.models import load_model

from terra_ai import progress
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.datasets.extra import LayerOutputTypeChoice
from terra_ai.data.modeling.model import ModelDetailsData, ModelData
from terra_ai.data.training.extra import CheckpointIndicatorChoice, CheckpointTypeChoice, MetricChoice, \
    CheckpointModeChoice
from terra_ai.data.training.train import TrainData
from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.modeling.validator import ModelValidator
from terra_ai.training.customcallback import InteractiveCallback
from terra_ai.training.customlosses import DiceCoefficient, yolo_loss
from terra_ai.training.data import custom_losses_dict

from matplotlib import pyplot as plt

__version__ = 0.01

from terra_ai.utils import camelize

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
        self.DTS = None
        self.model = None
        self.training_path: str = ""
        self.optimizer = None
        self.loss: dict = {}
        self.metrics: dict = {}

        self.batch_size = 128
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
    def _check_metrics(metrics: list, num_classes: int = 2) -> list:
        output = []
        for metric in metrics:
            if metric == MetricChoice.MeanIoU.value:
                output.append(getattr(importlib.import_module("tensorflow.keras.metrics"), metric)(num_classes))
            elif metric == MetricChoice.DiceCoef:
                output.append(DiceCoefficient())
            else:
                output.append(getattr(importlib.import_module("tensorflow.keras.metrics"), metric)())
        return output

    def _set_training_params(self, dataset: DatasetData, params: TrainData, model_name: str,
                             training_path: Path, dataset_path: str, initial_config: dict) -> None:
        self.dataset = self._prepare_dataset(dataset, dataset_path)
        self.training_path = training_path
        self.nn_name = model_name
        self.epochs = params.epochs if interactive.get_states().get("status") != "addtrain" \
            else self.epochs - self.callbacks[0].last_epoch + 1
        self.batch_size = params.batch
        self.set_optimizer(params)
        # self.set_chp_monitor(params)
        for output_layer in params.architecture.outputs_dict:
            self.metrics.update({
                str(output_layer["id"]):
                    self._check_metrics(metrics=output_layer.get("metrics", []),
                                        num_classes=output_layer.get("classes_quantity"))
            })
            self.loss.update({str(output_layer["id"]): output_layer["loss"]})
        # if interactive.get_states().get("status") != "addtrain":
        interactive.set_attributes(dataset=self.dataset, metrics=self.metrics, losses=self.loss,
                                   dataset_path=dataset_path, initial_config=initial_config)

    def _set_callbacks(self, dataset: PrepareDataset, batch_size: int, epochs: int,
                       checkpoint: dict, save_model_path: str) -> None:
        progress.pool(self.progress_name, finished=False, data={'status': 'Добавление колбэков...'})
        callback = FitCallback(dataset=dataset, checkpoint_config=checkpoint,
                               batch_size=batch_size, epochs=epochs, save_model_path=save_model_path,
                               model_name=self.nn_name)
        self.callbacks = [callback]
        # checkpoint.update([('filepath', 'test_model.h5')])
        # self.callbacks.append(keras.callbacks.ModelCheckpoint(**checkpoint))
        progress.pool(self.progress_name, finished=False, data={'status': 'Добавление колбэков выполнено'})

    @staticmethod
    def _prepare_dataset(dataset: DatasetData, datasets_path: str) -> PrepareDataset:
        prepared_dataset = PrepareDataset(data=dataset, datasets_path=datasets_path)
        prepared_dataset.prepare_dataset()
        return prepared_dataset

    @staticmethod
    def _set_model(model: ModelDetailsData) -> ModelData:
        validator = ModelValidator(model)
        train_model = validator.get_keras_model()
        return train_model

    @staticmethod
    def _save_params_for_deploy(dataset_path: str, training_path: str, params: TrainData):
        if not os.path.exists(training_path):
            os.mkdir(training_path)
        if os.path.exists(os.path.join(training_path, "dataset")):
            shutil.rmtree(os.path.join(training_path, "dataset"), ignore_errors=True)
        shutil.copytree(dataset_path, os.path.join(training_path, "dataset"),
                        ignore=shutil.ignore_patterns("sources", "arrays"))
        with open(os.path.join(training_path, "config.train"), "w", encoding="utf-8") as train_config:
            json.dump(params.native(), train_config)

    def set_optimizer(self, params: TrainData) -> None:
        """
        Set optimizer method for using terra w/o gui
        """

        optimizer_object = getattr(keras.optimizers, params.optimizer.type.value)
        self.optimizer = optimizer_object(**params.optimizer.parameters_dict)
        # print(params.optimizer.parameters_dict)
        # print(self.optimizer)

    # def set_custom_metrics(self, params=None) -> None:
    #     for i_key in self.metrics.keys():
    #         for idx, metric in enumerate(self.metrics[i_key]):
    #             if metric in custom_losses_dict.keys():
    #                 if metric == "mean_io_u":  # TODO определить или заменить self.output_params (возможно на params)
    #                     self.metrics[i_key][idx] = custom_losses_dict[metric](
    #                         num_classes=self.output_params[i_key]['num_classes'], name=metric)
    #                 else:
    #                     self.metrics[i_key][idx] = custom_losses_dict[metric](name=metric)

    def show_training_params(self) -> None:
        """
        output the parameters of the neural network: batch_size, epochs, shuffle, callbacks, loss, metrics,
        x_train_shape, num_classes
        """

        # print("self.DTS.X", self.DTS.X)
        # print("self.DTS.Y", self.DTS.Y)
        print("\nself.DTS.classes_names", self.DTS.classes_names)
        x_shape = []
        v_shape = []
        t_shape = []
        for i_key in self.DTS.X.keys():
            x_shape.append([i_key, self.DTS.X[i_key]['data'][0].shape])
            v_shape.append([i_key, self.DTS.X[i_key]['data'][1].shape])
            t_shape.append([i_key, self.DTS.X[i_key]['data'][2].shape])

        msg = f'num_classes = {self.DTS.num_classes}, x_Train_shape = {x_shape}, x_Val_shape = {v_shape}, \n' \
              f'x_Test_shape = {t_shape}, epochs = {self.epochs}, \n' \
              f'callbacks = {self.callbacks}, batch_size = {self.batch_size},shuffle = {self.shuffle}, \n' \
              f'loss = {self.loss}, metrics = {self.metrics} \n'

        print(msg)
        pass

    def terra_fit(self,
                  dataset: DatasetData,
                  gui_model: ModelDetailsData,
                  training_path: Path = "",
                  dataset_path: str = "",
                  training_params: TrainData = None,
                  initial_config: dict = None
                  ) -> dict:
        """
        This method created for using wth externally compiled models

        Args:
            dataset: DatasetData
            gui_model: Keras model for fit
            training_params:
            training_path: TrainData
            dataset_path: str

        Return:
            None
        """
        model_path = os.path.join(training_path, gui_model.name)
        self._save_params_for_deploy(dataset_path=dataset_path, training_path=model_path, params=training_params)
        self.nn_cleaner(retrain=True if interactive.get_states().get("status") == "training" else False)
        self._set_training_params(dataset=dataset, dataset_path=dataset_path, model_name=gui_model.name,
                                  params=training_params, training_path=training_path, initial_config=initial_config)
        nn_model = self._set_model(model=gui_model)

        if interactive.get_states().get("status") == "trained":
            try:
                list_files = os.listdir(self.training_path)
                model_name = [x for x in list_files if x.endswith("last.h5")]
                custom_objects = {}
                for output_key in self.metrics.keys():
                    for metric_name in self.metrics[output_key]:
                        if not isinstance(metric_name, str):
                            metric_name = metric_name.name
                        if metric_name == "dice_coef":
                            custom_objects.update({"DiceCoefficient": DiceCoefficient})
                if not custom_objects:
                    custom_objects = None
                self.model = load_model(os.path.join(self.training_path, model_name[0]), compile=False,
                                        custom_objects=custom_objects)

                self.nn_name = f"{self.model.name}"
                progress.pool(self.progress_name, finished=False, data={'status': f'Загружена модель {model_name[0]}'})
            except Exception:
                progress.pool(self.progress_name, finished=False, data={'status': 'Ошибка загрузки модели!!!'})

            if interactive.get_states().get("status") == "stopped" and (self.callbacks[0].last_epoch != self.sum_epoch):
                if interactive.get_states().get("status") == "retrain":
                    self.epochs = self.sum_epoch - self.callbacks[0].last_epoch
                else:
                    self.epochs = self.epochs - self.callbacks[0].last_epoch
            else:
                # self.retrain_flag = True
                # self.callbacks[0].stop_flag = False
                self.sum_epoch += self.epochs
                self.callbacks[0].batch_size = self.batch_size
                # self.callbacks[0].retrain_flag = True
                # self.callbacks[0].retrain_epochs = self.epochs
                self.callbacks[0].epochs = self.epochs + self.callbacks[0].last_epoch

            # self.model.stop_training = False
            # self.stop_training = False
            # self.model_is_trained = False
            if list(self.dataset.data.outputs.values())[0].task == LayerOutputTypeChoice.ObjectDetection:
                self.yolo_model_fit(params=training_params, dataset=self.dataset, verbose=1, retrain=True)
            else:
                self.base_model_fit(params=training_params, dataset=self.dataset, verbose=0, save_model_path=model_path)

        else:
            self.model = nn_model
            if list(self.dataset.data.outputs.values())[0].task == LayerOutputTypeChoice.ObjectDetection:
                self.yolo_model_fit(params=training_params, dataset=self.dataset, verbose=1, retrain=False)
            else:
                self.base_model_fit(params=training_params, dataset=self.dataset, verbose=0, save_model_path=model_path)

            self.sum_epoch += self.epochs
        return {"dataset": self.dataset, "metrics": self.metrics, "losses": self.loss}
        # self.stop_training = self.callbacks[0].stop_training

    def nn_cleaner(self, retrain: bool = False) -> None:
        keras.backend.clear_session()
        self.DTS = None
        self.model = None
        if retrain:
            # self.stop_training = False
            # self.model_is_trained = False
            # self.retrain_flag = False
            self.sum_epoch = 0
            self.chp_monitor = ""
            self.optimizer = None
            self.loss = {}
            self.metrics = {}
            self.callbacks = []
            self.history = {}
        gc.collect()

    def get_nn(self):
        self.nn_cleaner(retrain=True)

        return self

    @progress.threading
    def base_model_fit(self, params: TrainData, dataset: PrepareDataset, save_model_path: str, verbose=0) -> None:
        progress.pool(self.progress_name, finished=False, data={'status': 'Компиляция модели ...'})
        # self.set_custom_metrics()
        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=self.metrics
                           )
        # self.model.load_weight(os.path.join(self.training_path, self.nn_name))
        progress.pool(self.progress_name, finished=False, data={'status': 'Компиляция модели выполнена'})
        # if interactive.get_states().get("status") != "addtrain":
        self._set_callbacks(dataset=dataset, batch_size=params.batch,
                            epochs=params.epochs, save_model_path=save_model_path,
                            checkpoint=params.architecture.parameters.checkpoint.native())
        progress.pool(self.progress_name, finished=False, data={'status': 'Начало обучения ...'})
        self.history = self.model.fit(
            self.dataset.dataset.get('train').batch(self.batch_size, drop_remainder=True).take(-1),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            validation_data=self.dataset.dataset.get('val').batch(self.batch_size, drop_remainder=True).take(-1),
            epochs=self.epochs,
            verbose=verbose,
            callbacks=self.callbacks
        )
        # self.model_is_trained = True

    def yolo_model_fit(self, params: TrainData, dataset: PrepareDataset, verbose=0, retrain=False) -> None:
        # Массив используемых анкоров (в пикселях). Используется по 3 анкора на каждый из 3 уровней сеток
        # данные значения коррелируются с размерностью входного изображения input_shape
        anchors = np.array(
            [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
        num_anchors = len(anchors)  # Сохраняем количество анкоров

        @tf.autograph.experimental.do_not_convert
        def create_model(
                input_shape: Tuple[int, int, int],
                num_anchor: int,
                model: Model,
                num_classes: int,
        ) -> Model:
            """
                Функция создания полной модели
                    Входные параметры:
                      input_shape - размерность входного изображения для модели YOLO
                      num_anchors - общее количество анкоров
                      model - спроектированная модель
                      num_classes - количество классов
            """
            w, h, ch = input_shape  # Получаем ширину и высоту и глубину входного изображения
            # inputs = keras.layers.Input(shape=(w, h, ch))  # Создаем входной слой модели, добавляя размерность для
            # глубины цвета

            # Создаем три входных слоя y_true с размерностями ((None, 13, 13, 3, 6), (None, 26, 26, 3, 6) и (None,
            # 52, 52, 3, 6)) 2 и 3 параметры (13, 13) указывают размерность сетки, на которую условно будет разбито
            # исходное изображение каждый уровень сетки отвечает за обнаружение объектов различных размеров (13 -
            # крупных, 26 - средних, 52 - мелких) 4 параметр - количество анкоров на каждый уровень сетки 5 параметр
            # - 4 параметра описывающие параметры анкора (координаты центра, ширина и высота) + вероятность
            # обнаружения объекта + OHE номер класса
            y_true = [
                keras.layers.Input(shape=(w // 32, h // 32, num_anchor // 3, num_classes + 5), name="input_2"),
                keras.layers.Input(shape=(w // 16, h // 16, num_anchor // 3, num_classes + 5), name="input_3"),
                keras.layers.Input(shape=(w // 8, h // 8, num_anchor // 3, num_classes + 5), name="input_4")
            ]

            yolo_model = model  # create_YOLOv3(inputs, num_anchors // 3)  # Создаем модель YOLOv3
            print('Создана модель YOLO. Количество классов: {}.'.format(
                num_classes))  # Выводим сообщение о создании модели
            print('model_yolo.summary()', yolo_model.summary())
            # Создаем выходной слой Lambda (выходом которого будет значение ошибки модели)
            # На вход слоя подается:
            #   - model_yolo.output (выход модели model_yolo (то есть то, что посчитала сеть))
            #   - y_true (оригинальные данные из обучающей выборки)
            outputs = keras.layers.Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                                          arguments={'num_anchors': num_anchor})([*yolo_model.output, *y_true])

            return Model([yolo_model.input, *y_true], outputs)  # Возвращаем модель

        # Создаем модель
        model_yolo = create_model(input_shape=(416, 416, 3), num_anchor=num_anchors, model=self.model,
                                  num_classes=list(self.dataset.data.num_classes.values())[0])
        print(model_yolo.summary())

        # Компилируем модель
        print(('Компиляция модели', '...'))
        # self.set_custom_metrics()
        model_yolo.compile(optimizer=self.optimizer,
                           loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        print(('Компиляция модели', 'выполнена'))
        print(('Начало обучения', '...'))

        if not retrain:
            self._set_callbacks(dataset=dataset, batch_size=params.batch,
                                epochs=params.epochs, checkpoint=params.architecture.parameters.checkpoint.native())

        print(('Начало обучения', '...'))
        self.history = model_yolo.fit(
            self.dataset.dataset.get('train'),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            validation_data=self.dataset.dataset.get('val'),
            epochs=self.epochs,
            verbose=verbose,
            callbacks=self.callbacks
        )


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
            gpu_utilization = N.nvmlDeviceGetUtilizationRates(N.nvmlDeviceGetHandleByIndex(0))
            gpu_memory = N.nvmlDeviceGetMemoryInfo(N.nvmlDeviceGetHandleByIndex(0))
            usage_dict["GPU"] = {
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

    def __init__(self, dataset: PrepareDataset, checkpoint_config: dict, batch_size: int = None, epochs: int = None,
                 save_model_path: str = "./", model_name: str = "noname"):
        """
        Для примера
        "checkpoint": {
                "layer": 2,
                "type": "Metrics",
                "indicator": "Val",
                "mode": "max",
                "metric_name": "Accuracy",
                "save_best": True,
                "save_weights": False,
            },
        """

        super().__init__()
        self.usage_info = MemoryUsage(debug=False)
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.batch = 0
        self.num_batches = 0
        self.last_epoch = 1
        self._start_time = time.time()
        self._time_batch_step = time.time()
        self._time_first_step = time.time()
        self._sum_time = 0
        self._sum_epoch_time = 0
        # self.stop_training = False
        # self.retrain_flag = False
        # self.stop_flag = False
        self.retrain_epochs = 0
        self.save_model_path = save_model_path
        self.nn_name = model_name
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
        self.checkpoint_config = checkpoint_config
        self.num_outputs = len(self.dataset.data.outputs.keys())
        # аттрибуты для чекпоинта
        self.log_history = self._load_logs()

    def _get_metric_name_checkpoint(self, logs: dict):
        """Поиск среди fit_logs нужного параметра"""
        self.metric_checkpoint = "loss"
        for log in logs.keys():
            if self.checkpoint_config.get("type") == CheckpointTypeChoice.Loss and \
                    self.checkpoint_config.get("indicator") == CheckpointIndicatorChoice.Val and \
                    'val' in log and 'loss' in log:
                if self.num_outputs == 1:
                    self.metric_checkpoint = log
                    break
                else:
                    if f"{self.checkpoint_config.get('layer')}" in log:
                        self.metric_checkpoint = log
                        break

            elif self.checkpoint_config.get("type") == CheckpointTypeChoice.Loss and \
                    self.checkpoint_config.get("indicator") == CheckpointIndicatorChoice.Train and \
                    'val' not in log and 'loss' in log:
                if self.num_outputs == 1:
                    self.metric_checkpoint = log
                    break
                else:
                    if f"{self.checkpoint_config.get('layer')}" in log:
                        self.metric_checkpoint = log
                        break

            elif self.checkpoint_config.get("type") == CheckpointTypeChoice.Metrics and \
                    self.checkpoint_config.get("indicator") == CheckpointIndicatorChoice.Val and \
                    "val" in log:
                camelize_log = self._clean_and_camelize_log_name(log)
                if self.num_outputs == 1 and camelize_log == self.checkpoint_config.get("metric_name"):
                    self.metric_checkpoint = log
                    break
                else:
                    if f"{self.checkpoint_config.get('layer')}" in log and \
                            camelize_log == self.checkpoint_config.get("metric_name"):
                        self.metric_checkpoint = log
                        break

            elif self.checkpoint_config.get("type") == CheckpointTypeChoice.Metrics and \
                    self.checkpoint_config.get("indicator") == CheckpointIndicatorChoice.Train and \
                    'val' not in log:
                camelize_log = self._clean_and_camelize_log_name(log)
                if self.num_outputs == 1 and camelize_log == self.checkpoint_config.get("metric_name"):
                    self.metric_checkpoint = log
                    break
                else:
                    if f"{self.checkpoint_config.get('layer')}" in log and \
                            camelize_log == self.checkpoint_config.get('metric_name'):
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
            print(f"Chosen {self.metric_checkpoint} for monitoring")
        self.log_history['epoch'].append(epoch)
        for metric in logs:
            self.log_history['logs'][metric].append(logs.get(metric))

    def _save_logs(self):
        interactive_path = os.path.join(self.save_model_path, "interactive.history")
        if not os.path.exists(interactive_path):
            os.mkdir(interactive_path)
        with open(os.path.join(self.save_model_path, "log.history"), "w", encoding="utf-8") as history:
            json.dump(self.log_history, history)
        with open(os.path.join(interactive_path, "log.int"), "w", encoding="utf-8") as log:
            json.dump(interactive.log_history, log)
        with open(os.path.join(interactive_path, "table.int"), "w", encoding="utf-8") as table:
            json.dump(interactive.progress_table, table)

    def _load_logs(self):
        interactive_path = os.path.join(self.save_model_path, "interactive.history")
        if interactive.get_states().get("status") == "addtrain":
            with open(os.path.join(self.save_model_path, "log.history"), "r", encoding="utf-8") as history:
                logs = json.load(history)
            with open(os.path.join(interactive_path, "log.int"), "r", encoding="utf-8") as int_log:
                interactive_logs = json.load(int_log)
            with open(os.path.join(interactive_path, "table.int"), "r", encoding="utf-8") as table_int:
                interactive_table = json.load(table_int)
            self.last_epoch = max(logs.get('epoch')) + 1
            self._get_metric_name_checkpoint(logs.get('logs'))
            interactive.log_history = interactive_logs
            interactive.progress_table = interactive_table
            return logs
        else:
            return {
                'epoch': [],
                'logs': {}
            }

    def _best_epoch_monitoring(self, logs):
        """Оценка текущей эпохи"""
        if self.checkpoint_config.get("mode") == CheckpointModeChoice.Min and \
                logs.get(self.metric_checkpoint) < min(self.log_history.get("logs").get(self.metric_checkpoint)):
            return True
        elif self.checkpoint_config.get("mode") == CheckpointModeChoice.Max and \
                logs.get(self.metric_checkpoint) > max(self.log_history.get("logs").get(self.metric_checkpoint)):
            return True
        else:
            return False

    def _set_result_data(self, param: dict) -> None:
        for key in param.keys():
            if key in self.result.keys():
                self.result[key] = param[key]
            elif key == "timings":
                self.result["train_usage"]["timings"]["estimated_time"] = param[key][0]
                self.result["train_usage"]["timings"]["elapsed_time"] = param[key][1]
                self.result["train_usage"]["timings"]["still_time"] = param[key][0] - param[key][1]
                self.result["train_usage"]["timings"]["avg_epoch_time"] = self._sum_epoch_time / self.last_epoch
                self.result["train_usage"]["timings"]["elapsed_epoch_time"] = self._sum_epoch_time / self.last_epoch + param[key][2]
                self.result["train_usage"]["timings"]["still_epoch_time"] = param[key][2]
                self.result["train_usage"]["timings"]["epoch"] = param[key][3]
                self.result["train_usage"]["timings"]["batch"] = param[key][4]

    def _get_result_data(self):
        self.result["states"] = interactive.get_states()
        return self.result

    @staticmethod
    def _get_train_status() -> str:
        return interactive.get_states().get("status")

    def save_lastmodel(self) -> None:
        """
        Saving last model on each epoch end

        Returns:
            None
        """
        model_name = f"{self.nn_name}.trm"
        file_path_model: str = os.path.join(
            self.save_model_path, f"{model_name}"
        )
        self.model.save(file_path_model)
        self._set_result_data({'info': f"Последняя модель сохранена как {file_path_model}"})
        progress.pool(
            self.progress_name,
            percent=(self.last_epoch - 1) / self.epochs * 100,
            message=f"Обучение. Эпоха {self.last_epoch - 1} из {self.epochs}",
            data=self._get_result_data(),
            finished=False,
        )
        pass

    def _estimate_step(self, current, start, now):
        if current:
            _time_per_unit = (now - start) / current
        else:
            _time_per_unit = (now - start)
        return _time_per_unit

    def eta_format(self, eta):
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
            self.num_batches = self.dataset.X['train']['1'].shape[0] // self.batch_size
        else:
            self.num_batches = len(self.dataset.dataframe['train']) // self.batch_size

    def on_epoch_begin(self, epoch, logs=None):
        self._time_first_step = time.time()

    def on_train_batch_end(self, batch, logs=None):
        if self._get_train_status() == "stopped":
            self.model.stop_training = True
            msg = f'ожидайте остановку...'
            self.batch += 1
            self._set_result_data({'info': f"'Обучение остановлено пользователем, '{msg}"})
        else:
            train_batch_data = None
            msg_batch = {"batch": {"current": batch, "total": self.num_batches}}
            msg_epoch = {"epoch": {"current": self.last_epoch, "total": self.epochs}}
            elapsed_epoch_time = self.update_progress(self.num_batches, batch, self._time_first_step)
            elapsed_time = self.update_progress(self.num_batches * self.epochs + 1,
                                                self.batch, self._start_time, finalize=True)
            if self._get_train_status() == "retrain":
                estimated_time = self.update_progress(self.num_batches * self.retrain_epochs + 1,
                                                      self.batch, self._start_time)
            elif self._get_train_status() == "addtrain":
                estimated_time = self.update_progress(self.num_batches * self.epochs + 1, self.batch,
                                                      self._start_time, stop_current=batch, stop_flag=True)
                elapsed_time = self._sum_time + elapsed_time
            else:
                estimated_time = self.update_progress(self.num_batches * self.epochs + 1, self.batch, self._start_time)
            self.batch += 1

            if interactive.urgent_predict:
                if self.dataset.data.use_generator:
                    upred = self.model.predict(self.dataset.dataset.get('val').batch(1))
                else:
                    upred = self.model.predict(self.dataset.X.get('val'))
                # for data_type in ['train', 'val']:
                #     upred[data_type] = self.model.predict(self.dataset.X.get(data_type))

                train_batch_data = interactive.update_state(y_pred=upred)
            if train_batch_data:
                result_data = {
                    'timings': [estimated_time, elapsed_time, elapsed_epoch_time, msg_epoch, msg_batch],
                    'train_data': train_batch_data
                }
            else:
                result_data = {'timings': [estimated_time, elapsed_time, elapsed_epoch_time, msg_epoch, msg_batch]}
            self._set_result_data(result_data)
            progress.pool(
                self.progress_name,
                percent=(self.last_epoch - 1) / self.epochs * 100,
                message=f"Обучение. Эпоха {self.last_epoch} из {self.epochs}",
                data=self._get_result_data(),
                finished=False,
            )

    def on_epoch_end(self, epoch, logs=None):
        """
        Returns:
            {}:
        """
        if self.dataset.data.use_generator:
            scheduled_predict = self.model.predict(self.dataset.dataset.get('val').batch(1))
        else:
            scheduled_predict = self.model.predict(self.dataset.X.get('val'))
        interactive_logs = copy.deepcopy(logs)
        interactive_logs['epoch'] = self.last_epoch
        current_epoch_time = time.time() - self._time_first_step
        self._sum_epoch_time += current_epoch_time
        train_epoch_data = interactive.update_state(
            fit_logs=interactive_logs,
            y_pred=scheduled_predict,
            current_epoch_time=current_epoch_time,
            on_epoch_end_flag=True
        )
        self._set_result_data({'train_data': train_epoch_data})
        progress.pool(
            self.progress_name,
            percent=(self.last_epoch - 1) / self.epochs * 100,
            message=f"Обучение. Эпоха {self.last_epoch} из {self.epochs}",
            data=self._get_result_data(),
            finished=False,
        )

        # сохранение лучшей эпохи
        if self.last_epoch > 1:
            if self._best_epoch_monitoring(logs):
                if not os.path.exists(self.save_model_path):
                    os.mkdir(self.save_model_path)
                file_path_best: str = os.path.join(
                    self.save_model_path, f"best_weights_{self.metric_checkpoint}.h5"
                )
                self.model.save_weights(file_path_best)
                print(f"Epoch {self.last_epoch} - best weights was successfully saved")
        self._fill_log_history(self.last_epoch, logs)
        self.last_epoch += 1

    def on_train_end(self, logs=None):
        self._save_logs()
        self.save_lastmodel()
        time_end = self.update_progress(self.num_batches * self.epochs + 1,
                                        self.batch, self._start_time, finalize=True)
        self._sum_time += time_end
        if self.model.stop_training:
            msg = f'Модель сохранена.'
            self._set_result_data({'info': f"'Обучение остановлено пользователем. '{msg}"})
        else:
            if self._get_train_status() == "retrain":
                msg = f'Затрачено времени на обучение: ' \
                      f'{self.eta_format(time_end)} '
            else:
                msg = f'Затрачено времени на обучение: ' \
                      f'{self.eta_format(self._sum_time)} '
            self._set_result_data({'info': f"Обучение закончено. {msg}"})
            interactive.set_status("trained")
            progress.pool(
                self.progress_name,
                percent=(self.last_epoch - 1) / self.epochs * 100,
                message=f"Обучение завершено. Эпоха {self.last_epoch - 1} из {self.epochs}",
                data=self._get_result_data(),
                finished=True,
            )
            # print(msg)
