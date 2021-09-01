import base64
import copy
import itertools
import json
import os
import tempfile
import colorsys
import random

import psutil
from PIL import Image, ImageDraw, ImageFont  # Модули работы с изображениями

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from keras import Model
from keras.utils.np_utils import to_categorical
from pydub import AudioSegment
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tensorflow import keras
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
import numpy as np
import types
import time
import moviepy.editor as moviepy_editor

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import BatchNormalization, MaxPooling2D, Flatten, Dense
from tensorflow.python.layers.convolutional import Conv2D

from terra_ai.customLayers import InstanceNormalization
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.datasets.extra import LayerInputTypeChoice, LayerOutputTypeChoice
from terra_ai.data.presets.training import Task, TasksGroups, Metric, Loss
from terra_ai.data.training.extra import TaskChoice
from terra_ai.datasets.preparing import PrepareDTS
from terra_ai.guiexchange import Exchange
from .. import progress
# from terra_ai.trds import DTS
import pynvml as N

__version__ = 0.17


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
                'gpu_utilization': f'{gpu_utilization.gpu: .2f}%',
                'gpu_memory_used': f'{gpu_memory.used / 1024 ** 3: .2f}GB',
                'gpu_memory_total': f'{gpu_memory.total / 1024 ** 3: .2f}GB'
            }
            if self.debug:
                print(f'GPU usage: {gpu_utilization.gpu: .2f}% ({gpu_memory.used / 1024 ** 3: .2f}GB / '
                      f'{gpu_memory.total / 1024 ** 3: .2f}GB)')
        else:
            cpu_usage = psutil.cpu_percent(percpu=True)
            usage_dict["CPU"] = {
                'cpu_utilization': f'{sum(cpu_usage) / len(cpu_usage): .2f}%',
            }
            if self.debug:
                print(f'Average CPU usage: {sum(cpu_usage) / len(cpu_usage): .2f}%')
                print(f'Max CPU usage: {max(cpu_usage): .2f}%')
        usage_dict["RAM"] = {
            'ram_utilization': f'{psutil.virtual_memory().percent: .2f}%',
            'ram_memory_used': f'{psutil.virtual_memory().used / 1024 ** 3: .2f}GB',
            'ram_memory_total': f'{psutil.virtual_memory().total / 1024 ** 3: .2f}GB'
        }
        usage_dict["Disk"] = {
            'disk_utilization': f'{psutil.disk_usage("/").percent: .2f}%',
            'disk_memory_used': f'{psutil.disk_usage("/").used / 1024 ** 3: .2f}GB',
            'disk_memory_total': f'{psutil.disk_usage("/").total / 1024 ** 3: .2f}GB'
        }
        if self.debug:
            print(f'RAM usage: {psutil.virtual_memory().percent: .2f}% '
                  f'({psutil.virtual_memory().used / 1024 ** 3: .2f}GB / '
                  f'{psutil.virtual_memory().total / 1024 ** 3: .2f}GB)')
            print(f'Disk usage: {psutil.disk_usage("/").percent: .2f}% '
                  f'({psutil.disk_usage("/").used / 1024 ** 3: .2f}GB / '
                  f'{psutil.disk_usage("/").total / 1024 ** 3: .2f}GB)')
        return usage_dict


def sort_dict(dict_to_sort: dict, mode='by_name'):
    if mode == 'by_name':
        sorted_keys = sorted(dict_to_sort)
        sorted_values = []
        for w in sorted_keys:
            sorted_values.append(dict_to_sort[w])
        return tuple(sorted_keys), tuple(sorted_values)
    elif mode == 'ascending':
        sorted_keys = sorted(dict_to_sort, key=dict_to_sort.get)
        sorted_values = []
        for w in sorted_keys:
            sorted_values.append(dict_to_sort[w])
        return tuple(sorted_keys), tuple(sorted_values)
    elif mode == 'descending':
        sorted_keys = sorted(dict_to_sort, key=dict_to_sort.get, reverse=True)
        sorted_values = []
        for w in sorted_keys:
            sorted_values.append(dict_to_sort[w])
        return tuple(sorted_keys), tuple(sorted_values)
    else:
        return tuple(dict_to_sort.keys()), tuple(dict_to_sort.values())


def class_counter(y_array, classes_names: list, ohe=True):
    """
    class_dict = {
        "class_name": int
    }
    """
    class_dict = {}
    for cl in classes_names:
        class_dict[cl] = 0
    if ohe:
        y_array = np.argmax(y_array, axis=-1)
    for y in y_array:
        class_dict[classes_names[y]] += 1
    return class_dict


progress_mode = {
    "BinaryCrossentropy": "min",
    "CategoricalCrossentropy": "min",
    "CategoricalHinge": "min",
    "CosineSimilarity": "min",
    "Hinge": "min",
    "Huber": "min",
    "KLDivergence": "min",
    "LogCosh": "min",
    "MeanAbsoluteError": "min",
    "MeanAbsolutePercentageError": "min",
    "MeanSquaredError": "min",
    "MeanSquaredLogarithmicError": "min",
    "Poisson": "min",
    "SparseCategoricalCrossentropy": "min",
    "SquaredHinge": "min",
    "AUC": "max",
    "Accuracy": "max",
    "BinaryAccuracy": "max",
    "CategoricalAccuracy": "max",
    "FalseNegatives": "min",
    "FalsePositives": "min",
    "LogCoshError": "min",
    "MeanIoU": "max",
    "Precision": "max",
    "Recall": "max",
    "RootMeanSquaredError": "min",
    "SparseCategoricalAccuracy": 'max',
    "SparseTopKCategoricalAccuracy": "max",
    "TopKCategoricalAccuracy": "max",
    "TrueNegatives": "max",
    "TruePositives": "max"
}


class InteractiveCallback:
    """Callback for interactive requests"""

    def __init__(self, dataset, metrics: dict, losses: dict):
        """
        log_history:    epoch_num -> metrics/loss -> output_idx - > metric/loss -> train ->  total/classes
        """

        # self.model = null_model
        self.losses = losses
        self.metrics = metrics
        self.dataset = dataset
        # self.X = {'train': {}, 'val': {}}
        # self.Y = {'train': {}, 'val': {}}
        # self._prepare_dataset()
        self.y_pred = {}
        self.current_epoch = None

        # overfitting params
        self.log_gap = 10
        # self.progress_mode = None
        self.progress_threashold = 5

        self.log_history = {}
        self._prepare_null_log_history_template()
        self.progress_table = []
        self.dataset_balance = self._prepare_dataset_balance()
        self.class_idx = self._get_class_idx()

        self.show_examples = 10
        self.ex_type_choice = 'seed'
        self.seed_idx = self._get_seed()
        self.example_idx = []

        self.interactive_config = {
            'loss_graphs': [
                {
                    'id': 1,
                    'output_idx': 2,
                    'show_for_model': True,
                    'show_for_classes': False
                },
            ],
            'metric_graphs': [
                {
                    'id': 1,
                    'output_idx': 2,
                    'show_for_model': True,
                    'show_for_classes': False,
                    'show_metric': 'Accuracy'
                }
            ],
            'intermediate_result': {
                'show_results': False,
                'data_for_calculation': 'val',
                'example_choice_type': 'seed',
                'main_output': 2,
                'num_examples': 10,
                'show_statistic': False,
                'autoupdate': False
            },
            'progress_table': [
                {
                    'output_idx': 2,
                    'show_loss': True,
                    'show_metrics': True,
                }
            ],
            'statistic_data': {
                'output_id': [2, 3],
                'autoupdate': False
            },
            'data_balance': {
                'show_train': True,
                'show_val': True,
                'sorted': 'by_name'  # 'descending', 'ascending'
            }
        }
        pass

    def _prepare_null_log_history_template(self):
        """
        self.log_history_example = {
            'output_id': {
                'loss': {
                    'CategoricalCrossentropy': {
                        'train': [],
                        'val': []
                    }
                },
                'metrics': {
                    'Accuracy': {
                        'train': [],
                        'val': []
                    },
                    'CategoricalAccuracy': {
                        'train': [],
                        'val': []
                    }
                },
                'progress_state': {
                    'loss': {
                        'CategoricalCrossentropy': {
                                'mean_log_history': [],
                                'normal_state': [],
                                'underfittng': [],
                                'overfitting': []
                        }
                    },
                    'metrics': {
                        'Accuracy': {
                            'mean_log_history': [],
                            'normal_state': [],
                            'underfittng': [],
                            'overfitting': []
                        },
                        'CategoricalAccuracy': {
                            'mean_log_history': [],
                            'normal_state': [],
                            'underfittng': [],
                            'overfitting': []
                        }
                    }
                },
                'class_loss': {
                    'Мерседес': {
                        'CategoricalCrossentropy': {'val': []}
                    },
                    'Рено': {
                        'CategoricalCrossentropy': {'val': []}
                    },
                    'Феррари': {
                        'CategoricalCrossentropy': {'val': []}
                    }
                },
                'class_metrics': {
                    'Мерседес': {
                        'Accuracy': {'val': []},
                        'CategoricalAccuracy': {'val': []}
                    },
                    'Рено': {
                        'Accuracy': {'val': []},
                        'CategoricalAccuracy': {'val': []}
                    },
                    'Феррари': {
                        'Accuracy': {'val': []},
                        'CategoricalAccuracy': {'val': []}
                    }
                }
            }
        }
        """
        self.log_history['epochs'] = []
        for out in self.dataset.Y.get('train').keys():
            self.log_history[out] = {
                'loss': {},
                'metrics': {},
                'progress_state': {
                    'loss': {},
                    'metrics': {}
                }
            }
            if self.losses.get(out) and isinstance(self.losses.get(out), str):
                self.losses[out] = [self.losses.get(out)]
            if self.metrics.get(out) and isinstance(self.metrics.get(out), str):
                self.metrics[out] = [self.metrics.get(out)]

            for loss in self.losses.get(out):
                self.log_history[out]['loss'][f"{loss}"] = {'train': [], 'val': []}
                self.log_history[out]['progress_state']['loss'][f"{loss}"] = {
                    'mean_log_history': [], 'normal_state': [], 'underfittng': [], 'overfitting': []
                }
            for metric in self.metrics.get(out):
                self.log_history[out]['metrics'][f"{metric}"] = {'train': [], 'val': []}
                self.log_history[out]['progress_state']['metrics'][f"{metric}"] = {
                    'mean_log_history': [], 'normal_state': [], 'underfittng': [], 'overfitting': []
                }
            if self.dataset.data.task_type.get(out) == TaskChoice.Classification:
                self.log_history[out]['class_loss'] = {}
                self.log_history[out]['class_metrics'] = {}
                for class_name in self.dataset.data.classes_names.get(out):
                    self.log_history[out]['class_metrics'][f"{class_name}"] = {}
                    self.log_history[out]['class_loss'][f"{class_name}"] = {}

                    for metric in self.metrics.get(out):
                        self.log_history[out]['class_metrics'][f"{class_name}"][f"{metric}"] = {'val': []}
                    for loss in self.losses.get(out):
                        self.log_history[out]['class_loss'][f"{class_name}"][f"{loss}"] = {'val': []}

    # def _prepare_dataset(self):
    #     for ins in self.dataset.X.get('train').keys():
    #         self.X['train'][f'{ins}'] = self.dataset.X.get('train').get(ins)
    #         if self.dataset.X.get('val').get(ins) is not None:
    #             self.X['val'][f'{ins}'] = self.dataset.X.get('val').get(ins)
    #         elif self.dataset.X.get('test').get(ins) is not None:
    #             self.X['val'][f'{ins}'] = self.dataset.X.get('test').get(ins)
    #         else:
    #             self.X['val'][f'{ins}'] = None
    #     for out in self.dataset.Y.get('train').keys():
    #         self.Y['train'][f'{out}'] = self.dataset.Y.get('train').get(out)
    #         if self.dataset.Y.get('val').get(out) is not None:
    #             self.Y['val'][f'{out}'] = self.dataset.Y.get('val').get(out)
    #         elif self.dataset.Y.get('test').get(out) is not None:
    #             self.Y['val'][f'{out}'] = self.dataset.Y.get('test').get(out)
    #         else:
    #             self.X['val'][f'{out}'] = None

    def _prepare_dataset_balance(self):
        """
        return = {
            "output_name": {
                'data_type': {
                    'class_name': int,
                },
                'test': {...},
                'val': {...}
            }
        }
        """
        dataset_balance = {}
        for out in self.dataset.Y.get('train').keys():
            dataset_balance[out] = {}
            if self.dataset.data.task_type.get(out) == TaskChoice.Classification:
                for datatype in self.dataset.Y.keys():
                    if datatype:
                        dataset_balance[out][datatype] = class_counter(
                            self.dataset.Y.get(datatype).get(out),
                            self.dataset.data.classes_names.get(out),
                            self.dataset.data.encoding.get(out) == 'ohe'
                        )
                    else:
                        dataset_balance[datatype] = {}
        return dataset_balance

    def _get_class_idx(self):
        """
        class_idx_dict -> train -> output_idx -> class_name
        """
        class_idx = {}
        for data_type in self.dataset.Y.keys():
            for out in self.dataset.Y.get(data_type).keys():
                class_idx[data_type] = {out: {}}
                for name in self.dataset.data.classes_names.get(out):
                    class_idx[data_type][out][name] = []
                y_true = np.argmax(self.dataset.Y.get(data_type).get(out), axis=-1) \
                    if self.dataset.data.encoding.get(out) == 'ohe' else self.dataset.Y.get(data_type).get(out)
                for idx in range(len(y_true)):
                    class_idx[data_type][out][self.dataset.data.classes_names.get(out)[y_true[idx]]].append(idx)
        return class_idx

    def _get_seed(self):
        for ins in self.dataset.X.get('train').keys():
            seed_idx = {}
            for data_type in self.dataset.X.keys():
                data_lenth = np.arange(len(self.dataset.X.get(data_type).get(ins)))
                np.random.shuffle(data_lenth)
                seed_idx[data_type] = data_lenth
            break
        return seed_idx

    def update_state(self, current_epoch, y_pred, current_epoch_time, on_epoch_end=True):
        self.current_epoch = current_epoch
        self._reformat_y_pred(y_pred)
        self.log_history['epochs'].append(current_epoch)
        self._fill_loss_log_history()
        self._update_progress_table(current_epoch, current_epoch_time)

    def _reformat_y_pred(self, y_pred):
        # predict = self.model.predict(self.X.get('train'))
        for data_type in self.dataset.Y.keys():
            self.y_pred[data_type] = {}
            for idx, out in enumerate(self.dataset.Y.get(data_type).keys()):
                if len(self.dataset.Y.get(data_type).keys()) == 1:
                    self.y_pred[data_type][out] = y_pred
                else:
                    self.y_pred[data_type][out] = y_pred[idx]

    def _get_loss_calculation(self, loss_name, y_true, y_pred, ohe=True, num_classes=10):
        loss_obj = getattr(tf.keras.losses, loss_name)()
        if loss_name == Loss.SparseCategoricalCrossentropy:
            return loss_obj(np.argmax(y_true, axis=-1) if ohe else y_true, y_pred).numpy()
        else:
            return loss_obj(y_true if ohe else to_categorical(y_true, num_classes), y_pred).numpy()

    def _get_metric_calculation(self, metric_name, y_true, y_pred, ohe=True, num_classes=10):
        if metric_name == Metric.MeanIoU:
            metric_obj = getattr(tf.keras.metrics, metric_name)(num_classes)
        else:
            metric_obj = getattr(tf.keras.metrics, metric_name)()

        if metric_name == Metric.Accuracy:
            metric_obj.update_state(np.argmax(y_true, axis=-1) if ohe else y_true, np.argmax(y_pred, axis=-1))
        elif metric_name == Metric.SparseCategoricalAccuracy or \
                metric_name == Metric.SparseTopKCategoricalAccuracy or \
                metric_name == Metric.SparseCategoricalCrossentropy:
            metric_obj.update_state(np.argmax(y_true, axis=-1) if ohe else y_true, y_pred)
        else:
            metric_obj.update_state(y_true if ohe else to_categorical(y_true, num_classes), y_pred)
        return metric_obj.result().numpy()

    def _get_mean_log(self, logs):
        if len(logs) < self.log_gap:
            return np.mean(logs)
        else:
            return np.mean(logs[-self.log_gap:])

    def _fill_loss_log_history(self):
        for out in self.dataset.Y.get('train').keys():
            if self.dataset.data.encoding.get(out) == 'ohe':
                ohe = True
            else:
                ohe = False
            num_classes = self.dataset.data.num_classes.get(out)

            for loss_name in self.log_history.get(out).get('loss').keys():
                for data_type in ['train', 'val']:
                    # fill losses
                    self.log_history[out]['loss'][loss_name][data_type].append(
                        self._get_loss_calculation(
                            loss_name,
                            self.dataset.Y.get('train').get(out),
                            self.y_pred.get(data_type).get(out),
                            ohe,
                            num_classes
                        )
                    )

                # fill loss progress state
                self.log_history[out]['progress_state']['loss'][loss_name]['mean_log_history'].append(
                    self._get_mean_log(self.log_history[out]['loss'][loss_name]['val'])
                )

                # get progress state data
                loss_underfittng = self._evaluate_underfitting(
                    loss_name,
                    self.log_history[out]['loss'][loss_name]['train'][-1],
                    self.log_history[out]['loss'][loss_name]['val'][-1],
                    self.log_history[out]['progress_state']['loss'][loss_name]['underfitting'],
                    out
                )
                loss_overfittng = self._evaluate_overfitting(
                    loss_name,
                    self.log_history[out]['progress_state']['loss'][loss_name]['mean_log_history'],
                    self.log_history[out]['progress_state']['loss'][loss_name]['overfitting'],
                    out
                )
                if loss_underfittng or loss_overfittng:
                    normal_state = False
                else:
                    normal_state = True

                self.log_history[out]['progress_state']['loss'][loss_name]['underfitting'].append(loss_underfittng)
                self.log_history[out]['progress_state']['loss'][loss_name]['overfitting'].append(loss_overfittng)
                self.log_history[out]['progress_state']['loss'][loss_name]['normal_state'].append(normal_state)

                # get Classification loss logs
                if self.dataset.data.task_type.get(out) == TaskChoice.Classification:
                    # fill class losses
                    for cls in self.log_history.get(out).get('class_loss').keys():
                        self.log_history[out]['class_loss'][cls][loss_name]['val'].append(
                            self._get_loss_calculation(
                                loss_name,
                                self.dataset.Y.get('val').get(out)[self.class_idx.get('val').get(out).get(cls)],
                                self.y_pred.get('val').get(out)[self.class_idx.get('val').get(out).get(cls)],
                                ohe, num_classes)
                        )

            for metric in self.log_history.get(out).get('metrics').keys():
                for data_type in ['train', 'val']:
                    # fill metrics
                    self.log_history[out]['metrics'][metric][data_type].append(self._get_metric_calculation(
                        metric, self.dataset.Y.get('train').get(out), self.y_pred.get(data_type).get(out), ohe,
                        num_classes))

                # fill loss progress state
                self.log_history[out]['progress_state']['metrics'][metric]['mean_log_history'].append(
                    self._get_mean_log(self.log_history[out]['metrics'][metric]['val'])
                )
                loss_underfittng = self._evaluate_underfitting(
                    metric,
                    self.log_history[out]['metrics'][metric]['train'][-1],
                    self.log_history[out]['metrics'][metric]['val'][-1],
                    self.log_history[out]['progress_state']['metrics'][metric]['underfitting'],
                    out
                )
                loss_overfittng = self._evaluate_overfitting(
                    metric,
                    self.log_history[out]['progress_state']['metrics'][metric]['mean_log_history'],
                    self.log_history[out]['progress_state']['metrics'][metric]['overfitting'],
                    out
                )
                if loss_underfittng or loss_overfittng:
                    normal_state = False
                else:
                    normal_state = True
                self.log_history[out]['progress_state']['metrics'][metric]['underfitting'].append(loss_underfittng)
                self.log_history[out]['progress_state']['metrics'][metric]['overfitting'].append(loss_overfittng)
                self.log_history[out]['progress_state']['metrics'][metric]['normal_state'].append(normal_state)

                if self.dataset.data.task_type.get(out) == TaskChoice.Classification:
                    # fill class losses
                    for cls in self.log_history.get(out).get('class_metrics').keys():
                        self.log_history[out]['class_metrics'][cls][metric]['val'].append(
                            self._get_metric_calculation(
                                metric,
                                self.dataset.Y.get('val').get(out)[self.class_idx.get('val').get(out).get(cls)],
                                self.y_pred.get('val').get(out)[self.class_idx.get('val').get(out).get(cls)],
                                ohe, num_classes)
                        )

    def _evaluate_overfitting(self, metric_name, mean_log, overfit_logs, output):
        count = sum(overfit_logs[-self.log_gap:])
        if progress_mode.get(metric_name) == 'min' and \
                mean_log[-1] > min(mean_log) and \
                (mean_log[-1] - min(mean_log)) * 100 / min(mean_log) > 10:
            count += 1
        elif progress_mode.get(metric_name) == 'max' and \
                mean_log[-1] < max(mean_log) and \
                (max(mean_log) - mean_log[-1]) * 100 / mean_log[-1] > 10:
            count += 1
        else:
            count = 0

        if count >= self.progress_threashold:
            return True
        else:
            return False

    def _evaluate_underfitting(self, metric_name, train_log, val_log, underfit_logs):
        count = sum(underfit_logs[-self.log_gap:])
        if progress_mode.get(metric_name) == 'min' and \
                (val_log[-1] - train_log[-1]) / train_log[-1] * 100 > 10:
            count += 1
        elif progress_mode.get(metric_name) == 'max' and \
                (train_log[-1] - val_log[-1]) / train_log[-1] * 100 > 10:
            count += 1
        else:
            count = 0
        if count >= self.log_gap:
            return True
        else:
            return False

    def return_interactive_results(self, config: dict):
        """Return dict with data for current interactive request"""
        self.interactive_config = config
        self.example_idx = self._prepare_example_idx_to_show()
        return {
            'loss_graphs': self._get_loss_graph_data_request(),
            'metric_graphs': self._get_metric_graph_data_request(),
            'intermediate_result': self._get_intermediate_result_request(),
            'progress_table': self.progress_table,
            'statistic_data': self._get_statistic_data_request(),
            'data_balance': self._get_balance_data_request(),
        }

    def _get_loss_graph_data_request(self):
        """
        'loss_graphs': [

            # пример для всей модели
            {
                'id': 1,
                'graph_name': f'Output_{output_idx} - График ошибки обучения - Эпоха №{epoch_num}',
                'x_label': 'Эпоха',
                'y_label': 'Значение',
                'plot_data': [
                    {
                        'label': 'Тренировочная выборка',
                        'epochs': []:
                        'values': []
                    },
                    {
                        'label': 'Проверочная выборка',
                        'epochs': []:
                        'values': []
                    },
                ],
                "progress_state": "normal",
            },

            # Пример для классов
            {
                'graph_name': f'Output_{output_idx} - График ошибки обучения по классам - Эпоха №{epoch_num}',
                'x_label': 'Эпоха',
                'y_label': 'Значение',
                'plot_data': [
                    {
                        'class_label': f'Класс {class_name}',
                        'epochs': [],
                        'values': []
                    },
                    {
                        'class_label': f'Класс {class_name}',
                        'epochs': [],
                        'values': []
                    },
                    {
                        'class_label': f'Класс {class_name}',
                        'epochs': [],
                        'values': []
                    },
                    etc...
                ],
            }
        ]
        """
        if self.interactive_config.get('loss_graphs'):
            data_return = []
            for loss_graph_config in self.interactive_config.get('loss_graphs'):
                if loss_graph_config.get('show_for_model'):
                    data_return.append(
                        {
                            "graph_name": f"'Output_{loss_graph_config.get('output_idx')} - "
                                          f"График ошибки обучения - Эпоха №{self.log_history.get('epochs')[-1]}'",
                            "x_label": "Эпоха",
                            "y_label": "Значение",
                            "plot_data": [
                                {
                                    "label": "Тренировочная выборка",
                                    "epochs": self.log_history.get("epochs"),
                                    "values": self.log_history.get(
                                        f"{loss_graph_config.get('output_idx')}").get('loss').get(
                                        self.losses.get(loss_graph_config.get("output_idx"))).get("train")
                                },
                                {
                                    "label": "Проверочная выборка",
                                    "epochs": self.log_history.get("epochs"),
                                    "values": self.log_history.get(
                                        f"{loss_graph_config.get('output_idx')}").get('loss').get(
                                        self.losses.get(loss_graph_config.get("output_idx"))).get("val")
                                }
                            ],
                            "progress_state": {
                                "normal": self.log_history.get(f"{loss_graph_config.get('output_idx')}").get(
                                    "progress_state").get("loss").get(
                                    self.losses.get(loss_graph_config.get("output_idx"))).get('normal_state')[-1],
                                "underfitting": self.log_history.get(f"{loss_graph_config.get('output_idx')}").get(
                                    "progress_state").get("loss").get(
                                    self.losses.get(loss_graph_config.get("output_idx"))).get('underfittng')[-1],
                                "overfitting": self.log_history.get(f"{loss_graph_config.get('output_idx')}").get(
                                    "progress_state").get("loss").get(
                                    self.losses.get(loss_graph_config.get("output_idx"))).get('overfitting')[-1]
                            }
                        }
                    )
                elif loss_graph_config.get('show_for_classes'):
                    data_return.append(
                        {
                            "graph_name": f"Output_{loss_graph_config.get('output_idx')} - "
                                          f"График ошибки обучения по классам - "
                                          f"Эпоха №{self.log_history.get('epochs')[-1]}",
                            "x_label": "Эпоха",
                            "y_label": "Значение",
                            "plot_data": [
                                {
                                    'class_label': f'Класс {class_name}',
                                    'epochs': self.log_history.get("epochs"),
                                    'values': self.log_history.get(
                                        f"{loss_graph_config.get('output_idx')}").get('class_loss').get(class_name).get(
                                        self.losses.get(loss_graph_config.get("output_idx"))).get("val")
                                } for class_name in self.dataset.data.classes_names.get(
                                    f"{loss_graph_config.get('output_idx')}")
                            ]
                        }
                    )
                else:
                    data_return.append(
                        {
                            "graph_name": "",
                            "x_label": "",
                            "y_label": "",
                            "plot_data": []
                        }
                    )
        else:
            return []

    def _get_metric_graph_data_request(self):
        """
        'metric_graphs': [

            # пример для всей модели
            {
                'graph_name': f'Output_{output_idx} - График метрики {metric_name} - Эпоха №{epoch_num}',
                'x_label': 'Эпоха',
                'y_label': 'Значение',
                'plot_data': [
                    {
                        'label': 'Тренировочная выборка',
                        'epochs': []:
                        'values': []
                    },
                    {
                        'label': 'Проверочная выборка',
                        'epochs': []:
                        'values': []
                    },
                ],
                "progress_state": "normal",
            },

            # Пример для классов
            {
                'graph_name': f'Output_{output_idx} - График метрики {metric_name} по классам - Эпоха №{epoch_num}',
                'x_label': 'Эпоха',
                'y_label': 'Значение',
                'plot_data': [
                    {
                        'class_label': f'Класс {class_name}',
                        'epochs': [],
                        'values': []
                    },
                    etc...
                ],
            }
        ]
        """
        if self.interactive_config.get('metric_graphs'):
            data_return = []
            for metric_graph_config in self.interactive_config.get('metric_graphs'):
                if metric_graph_config.get('show_for_model'):
                    data_return.append(
                        {
                            "graph_name": f"'Output_{metric_graph_config.get('output_idx')} - "
                                          f"График метрики {metric_graph_config.get('show_metric')} - "
                                          f"Эпоха №{self.log_history.get('epochs')[-1]}'",
                            "x_label": "Эпоха",
                            "y_label": "Значение",
                            "plot_data": [
                                {
                                    "label": "Тренировочная выборка",
                                    "epochs": self.log_history.get("epochs"),
                                    "values": self.log_history.get(
                                        f"{metric_graph_config.get('output_idx')}").get('metrics').get(
                                        metric_graph_config.get('show_metric')).get("train")
                                },
                                {
                                    "label": "Проверочная выборка",
                                    "epochs": self.log_history.get("epochs"),
                                    "values": self.log_history.get(
                                        f"{metric_graph_config.get('output_idx')}").get('metrics').get(
                                        metric_graph_config.get('show_metric')).get("val")
                                }
                            ],
                            "progress_state": {
                                "normal": self.log_history.get(f"{metric_graph_config.get('output_idx')}").get(
                                    "progress_state").get("metrics").get(
                                    metric_graph_config.get('show_metric')).get('normal_state')[-1],
                                "underfitting": self.log_history.get(f"{metric_graph_config.get('output_idx')}").get(
                                    "progress_state").get("metrics").get(
                                    metric_graph_config.get('show_metric')).get('underfittng')[-1],
                                "overfitting": self.log_history.get(f"{metric_graph_config.get('output_idx')}").get(
                                    "progress_state").get("metrics").get(
                                    metric_graph_config.get('show_metric')).get('overfitting')[-1]
                            }
                        }
                    )
                elif metric_graph_config.get('show_for_classes'):
                    data_return.append(
                        {
                            "graph_name": f"Output_{metric_graph_config.get('output_idx')} - "
                                          f"График метрики {metric_graph_config.get('show_metric')} по классам - "
                                          f"Эпоха №{self.log_history.get('epochs')[-1]}",
                            "x_label": "Эпоха",
                            "y_label": "Значение",
                            "plot_data": [
                                {
                                    'class_label': f'Класс {class_name}',
                                    'epochs': self.log_history.get("epochs"),
                                    'values': self.log_history.get(
                                        f"{metric_graph_config.get('output_idx')}").get('class_metrics').get(
                                        class_name).get(metric_graph_config.get('show_metric')).get("val")
                                } for class_name in self.dataset.data.classes_names.get(
                                    f"{metric_graph_config.get('output_idx')}")
                            ]
                        }
                    )
                else:
                    data_return.append(
                        {
                            "graph_name": "",
                            "x_label": "",
                            "y_label": "",
                            "plot_data": []
                        }
                    )
        else:
            return []

    def _get_intermediate_result_request(self):

        """
        'intermediate_result': {
            example_num: {
                'line_number': int
                'initial_data': [
                    {
                        'type': 'image',
                        'layer': f'Input_{layer_id}',
                        'data': '/content/file.webm'
                    },
                    {
                        'type': 'video',
                        'file_path': '/content/file.webm'
                        'layer': f'Input_{layer_id}',
                    },
                    {
                        'type': 'text',
                        'layer': f'Input_{layer_id}',
                        'data': smth in base64
                    },
                    {
                        'type': 'text',
                        'layer': f'Input_{layer_id}',
                        'data': smth in base64
                    }
                ],
                'true_value': [
                    {
                        'type': str,
                        'layer': f'Output_{layer_id}',
                        'data': smth in base64
                    }
                ],
                'predict_value': [
                    {
                        'type': str,
                        'layer': f'Output_{layer_id}',
                        'data': smth in base64
                        'color_mark': str
                    }
                ],
                'class_stat': {
                    'type': str,
                    'class name': {
                        "value": str,
                        'color_mark': str
                    },
                }
            },
        }
        """
        if self.interactive_config.get('intermediate_result').get('show_results'):
            return_data = {}
            for idx in range(self.interactive_config.get('intermediate_result').get('num_examples')):
                return_data[idx] = {
                    'line_number': idx + 1,
                    'initial_data': {},
                    'true_value': {},
                    'predict_value': {},
                    'class_stat': {}
                }
                for inp in self.dataset.X.get('train').keys():
                    return_data[idx]['initial_data'] = {
                        'layer': f'Input_{inp}',
                        'data': self._postprocess_initial_data(
                            input_id=inp,
                            example_idx=self.example_idx[idx],
                        )
                    }
                for out in self.dataset.Y.get('train').keys():
                    true_lbl, predict_lbl, color_mark, stat = self._postprocess_result_data(
                        output_id=out,
                        data_type=self.interactive_config.get('intermediate_result').get('data_for_calculation'),
                        example_idx=self.example_idx[idx],
                        show_stat=self.interactive_config.get('intermediate_result').get('show_statistic'),
                    )
                    return_data[idx]['true_value'] = {
                        "type": "class_name",
                        "layer": f"Output_{out}",
                        "data": true_lbl
                    }
                    return_data[idx]['predict_value'] = {
                        "type": "class_name",
                        "layer": f"Output_{out}",
                        "data": predict_lbl,
                        "color_mark": color_mark
                    }
                    if stat:
                        return_data[idx]['class_stat'] = {
                            'layer': f'Output_{out}',
                            'data': stat
                        }
        else:
            return {}

    def _update_progress_table(self, epoch, epoch_time):
        """
        'progress_table': {
            'epoch': {
                'time': int,
                'layer': {
                    'Output_{layer_id}: {
                        'loss/metric': value,
                        ...
                    }
                }
            }
        }
        """
        self.progress_table[epoch] = {
            "time": epoch_time,
            "layer": {}
        }
        for out in self.dataset.Y.get('train').keys():
            self.progress_table[epoch]["layer"] = {
                'loss': self.log_history.get(out).get('loss').get(self.losses.get(out)).get('train'),
                'val_loss': self.log_history.get(out).get('loss').get(self.losses.get(out)).get('val')
            }
            for metric in self.metrics.get(out).keys():
                self.progress_table[epoch]["layer"][metric] = self.log_history.get(out).get(
                    'metrics').get(self.metrics.get(out)).get('train')
                self.progress_table[epoch]["layer"][f"val_{metric}"] = self.log_history.get(out).get(
                    'metrics').get(self.metrics.get(out)).get('val')

    def _get_statistic_data_request(self):
        """
        'statistic_data': {
            f'Output_{layer_id}': {
                'id': 1,
                'task_type': 'Classification',
                'graph_name':  f'Output_{layer_id} - Confusion matrix',
                'x_label': 'Предсказание',
                'y_label': 'Истинное значение',
                'labels': [],
                'data_array': array
            }
        }
        """
        return_data = {}
        _id = 1
        for out in self.interactive_config.get("statistic_data").get("output_id"):
            if self.dataset.data.task_type.get(out) == TaskChoice.Classification:
                cm = self._get_confusion_matrix(
                    np.argmax(self.dataset.Y.get("val"), axis=-1) if self.dataset.data.encoding == "ohe"
                    else self.dataset.Y.get("val"),
                    np.argmax(self.Y_pred.get("val"), axis=-1)
                )
                return_data[out] = {
                    "id": _id,
                    "task_type": TaskChoice.Classification,
                    "graph_name": f"Output_{out} - Confusion matrix",
                    "x_label": "Предсказание",
                    "y_label": "Истинное значение",
                    "labels": self.dataset.data.classes_names.get(out),
                    "data_array": cm[0],
                    "data_percent_array": cm[1]
                }
                _id += 1
            else:
                return_data[out] = {}
        return return_data

    def _get_balance_data_request(self):
        """
        'data_balance': {
            output_id: [
                {
                    'id': 1,
                    'graph_name': 'Тренировочная выборка',
                    'x_label': 'Название класса',
                    'y_label': 'Значение',
                    'plot_data': [
                        {
                            'labels': []:
                            'values': []
                        },
                    ]
                },
                {
                    'id': 2,
                    'graph_name': 'Проверочная выборка',
                    'x_label': 'Название класса',
                    'y_label': 'Значение',
                    'plot_data': [
                        {
                            'labels': []:
                            'values': []
                        },
                    ]
                },
            ]
        }
        """
        return_data = {}
        _id = 1
        for out in self.dataset.Y.get('train').keys():
            if self.dataset.data.task_type.get(out) == TaskChoice.Classification:
                class_train_names, class_train_count = sort_dict(
                    self.dataset_balance.get(out).get('train'),
                    mode=self.interactive_config.get('data_balance').get('sorted')
                )
                class_val_names, class_val_count = sort_dict(
                    self.dataset_balance.get(out).get('val'),
                    mode=self.interactive_config.get('data_balance').get('sorted')
                )

                return_data[out] = [
                    {
                        'id': _id,
                        'graph_name': 'Тренировочная выборка',
                        'x_label': 'Название класса',
                        'y_label': 'Значение',
                        'plot_data': [
                            {
                                'labels': class_train_names,
                                'values': class_train_count
                            },
                        ]
                    },
                    {
                        'id': _id + 1,
                        'graph_name': 'Тренировочная выборка',
                        'x_label': 'Название класса',
                        'y_label': 'Значение',
                        'plot_data': [
                            {
                                'labels': class_val_names,
                                'values': class_val_count
                            },
                        ]
                    }
                ]
                _id += 2
        return return_data

    # def _get_classification_data(self):
    #     pass

    @staticmethod
    def _get_confusion_matrix(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

        cm_percent = np.zeros_like(cm).astype('float32')
        for i in range(len(cm)):
            total = np.sum(cm[i])
            for j in range(len(cm[i])):
                cm_percent[i][j] = round(cm[i][j] * 100 / total, 1)
        return (cm, cm_percent)

    def _prepare_example_idx_to_show(self):
        """
        example_idx = {
            output_id: []
        }
        """
        example_idx = {}
        count = self.interactive_config.get('intermediate_result').get('num_examples')
        choice_type = self.interactive_config.get('intermediate_result').get('example_choice_type')
        if choice_type == 'best' or choice_type == 'worst':
            y_true = self.dataset.Y.get(
                self.interactive_config.get('intermediate_result').get('data_for_calculation')).get(
                self.interactive_config.get('intermediate_result').get('main_output'))
            y_pred = self.y_pred.get(
                self.interactive_config.get('intermediate_result').get('data_for_calculation')).get(
                self.interactive_config.get('intermediate_result').get('main_output'))
            if (y_pred.shape[-1] == y_true.shape[-1]) \
                    and (self.dataset.data.encoding.get(
                self.interactive_config.get('intermediate_result').get('main_output')) == 'ohe') \
                    and (y_true.shape[-1] > 1):
                classes = np.argmax(y_true, axis=-1)
            elif (len(y_true.shape) == 1) \
                    and (self.dataset.data.encoding.get(
                self.interactive_config.get('intermediate_result').get('main_output')) != 'ohe') \
                    and (y_pred.shape[-1] > 1):
                classes = copy.copy(y_true)
            elif (len(y_true.shape) == 1) \
                    and (self.dataset.data.encoding.get(
                self.interactive_config.get('intermediate_result').get('main_output')) != 'ohe') \
                    and (y_pred.shape[-1] == 1):
                classes = copy.deepcopy(y_true)
            else:
                classes = copy.deepcopy(y_true)

            probs = np.array([pred[classes[i]] for i, pred in enumerate(y_pred)])
            sorted_args = np.argsort(probs)
            if choice_type == 'best':
                example_idx = sorted_args[-count:]
            elif choice_type == 'worst':
                example_idx = sorted_args[:count]
            else:
                example_idx = np.random.choice(len(probs), count, replace=False)

        elif choice_type == 'seed':
            example_idx = self.seed_idx.get(
                self.interactive_config.get('intermediate_result').get('data_for_calculation'))[
                          :self.interactive_config.get('intermediate_result').get('num_examples')]
        elif choice_type == 'random':
            example_idx = np.random.randint(
                0,
                len(self.dataset.X.get(self.interactive_config.get('intermediate_result').get('data_for_calculation'))),
                self.interactive_config.get('intermediate_result').get('num_examples')
            )
        else:
            pass
        return example_idx

    def _postprocess_initial_data(self, input_id: int, example_idx: int):
        """
        Видео в .webm
        import moviepy.editor as mp
        clip = mp.VideoFileClip("mygif.gif")
        clip.write_videofile("myvideo.webm")

        Image to .webm
        img = Image.open('image_path')
        img = img.convert('RGB')
        img.save('image.webp', 'webp')

        Audio to .webm
        from pydub import AudioSegment
        AudioSegment.from_file("audio_path").export("audio.webm", format="webm")
        """
        # temp_file = tempfile.NamedTemporaryFile(delete=False)
        initial_file_path = os.path.join(self.dataset.datasets_path, self.dataset.dataframe.get(
            self.interactive_config.get('intermediate_result').get('data_for_calculation')).iat[example_idx, 0])
        # TODO: посмотреть как реализовать '.iat[example_idx, 0]' для нескольких входов
        if self.dataset.data.inputs.get(input_id).task == LayerInputTypeChoice.Image:
            img = Image.open(initial_file_path)
            img = img.convert('RGB')
            save_path = f"/tmp/initial_data_image{example_idx}_input{input_id}.webp"
            img.save(save_path, 'webp')
            return save_path
        elif self.dataset.data.inputs.get(input_id).task == LayerInputTypeChoice.Text:
            text_str = open(initial_file_path, 'r')
            return text_str
        elif self.dataset.data.inputs.get(input_id).task == LayerInputTypeChoice.Video:
            clip = moviepy_editor.VideoFileClip(initial_file_path)
            save_path = f"/tmp/initial_data_video{example_idx}_input{input_id}.webp"
            clip.write_videofile(save_path)
            return save_path
        elif self.dataset.data.inputs.get(input_id).task == LayerInputTypeChoice.Audio:
            save_path = f"/tmp/initial_data_audio{example_idx}_input{input_id}.webp"
            AudioSegment.from_file(initial_file_path).export(save_path, format="webm")
            return save_path
        elif self.dataset.data.inputs.get(input_id).task == LayerInputTypeChoice.Dataframe:
            # TODO: обсудить как пересылать датафреймы на фронт
            return initial_file_path
        else:
            return initial_file_path

    def _postprocess_result_data(self, output_id: int, data_type: str, example_idx: int, show_stat=True):
        if self.dataset.data.task_type.get(output_id).task == LayerOutputTypeChoice.Classification:
            labels = self.dataset.data.classes_names.get(output_id)
            ohe = True if self.dataset.data.encoding.get(output_id) == 'ohe' else False

            y_true = np.argmax(self.dataset.Y.get(data_type).get(output_id)[example_idx]) if ohe \
                else self.dataset.Y.get(data_type).get(output_id)[example_idx]

            predict = self.y_pred.get(data_type).get(output_id)[example_idx]
            if y_true == np.argmax(predict):
                color_mark = 'green'
            else:
                color_mark = 'red'

            class_stat = {}
            if show_stat:
                for i, val in enumerate(predict):
                    if val == max(predict) and i == y_true:
                        class_color_mark = "green"
                    elif val == max(predict) and i != y_true:
                        class_color_mark = "red"
                    else:
                        class_color_mark = "white"
                    class_stat[labels[i]] = {
                        "value": f"{round(val * 100, 1)}%",
                        "color_mark": class_color_mark
                    }
            return labels[y_true], labels[np.argmax(predict)], color_mark, class_stat

        elif self.dataset.data.task_type.get(output_id).task == LayerOutputTypeChoice.Segmentation:
            pass
        elif self.dataset.data.task_type.get(output_id).task == LayerOutputTypeChoice.TextSegmentation:
            pass
        elif self.dataset.data.task_type.get(output_id).task == LayerOutputTypeChoice.Regression:
            pass
        elif self.dataset.data.task_type.get(output_id).task == LayerOutputTypeChoice.Timeseries:
            pass
        elif self.dataset.data.task_type.get(output_id).task == LayerOutputTypeChoice.ObjectDetection:
            pass


class FitCallback(keras.callbacks.Callback):
    """CustomCallback for all task type"""

    def __init__(self, dataset, exchange=Exchange(), batch_size: int = None, epochs: int = None,
                 save_model_path: str = "./", model_name: str = "noname"):
        super().__init__()
        self.usage_info = MemoryUsage(debug=False)
        self.Exch = exchange
        self.DTS = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.batch = 0
        self.num_batches = 0
        self.epoch = 0
        self.last_epoch = 1
        self.history = {}
        self._start_time = time.time()
        self._time_batch_step = time.time()
        self._time_first_step = time.time()
        self._sum_time = 0
        self.stop_training = False
        self.retrain_flag = False
        self.stop_flag = False
        self.retrain_epochs = 0
        self.save_model_path = save_model_path
        self.nn_name = model_name
        self.progress_name = progress.PoolName.training

    def save_lastmodel(self) -> None:
        """
        Saving last model on each epoch end

        Returns:
            None
        """
        model_name = f"model_{self.nn_name}_on_epoch_end.last.h5"
        file_path_model: str = os.path.join(
            self.save_model_path, f"{model_name}"
        )
        self.model.save(file_path_model)
        progress.pool(
            self.progress_name,
            percent=(self.last_epoch - 1) / self.epochs * 100,
            message=f"Обучение. Эпоха {self.last_epoch - 1} из {self.epochs}",
            data={'info': f"Последняя модель сохранена как {file_path_model}"},
            finished=False,
        )
        # print(("Инфо", f"Последняя модель сохранена как {file_path_model}"))
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
        return [self.eta_format(eta), int(eta)]

    def on_train_begin(self, logs=None):
        self.stop_training = False
        self._start_time = time.time()
        if not self.stop_flag:
            self.batch = 0
        self.num_batches = len(self.DTS.dataframe['train']) // self.batch_size

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self._time_first_step = time.time()

    def on_train_batch_end(self, batch, logs=None):
        stop = False
        if stop:
            self.model.stop_training = True
            self.stop_training = True
            self.stop_flag = True
            msg = f'ожидайте остановку...'
            self.batch += 1
            print(('Обучение остановлено пользователем', msg))
        else:
            msg_batch = f'Батч {batch}/{self.num_batches}'
            msg_epoch = f'Эпоха {self.last_epoch}/{self.epochs}:' \
                        f'{self.update_progress(self.num_batches, batch, self._time_first_step)[0]}, '
            time_start = \
                self.update_progress(self.num_batches * self.epochs + 1, self.batch, self._start_time, finalize=True)[1]
            if self.retrain_flag:
                msg_progress_end = f'Расчетное время окончания:' \
                                   f'{self.update_progress(self.num_batches * self.retrain_epochs + 1, self.batch, self._start_time)[0]}, '
                msg_progress_start = f'Время выполнения дообучения:' \
                                     f'{self.eta_format(time_start)}, '
            elif self.stop_flag:
                msg_progress_end = f'Расчетное время окончания после остановки:' \
                                   f'{self.update_progress(self.num_batches * self.epochs + 1, self.batch, self._start_time, stop_current=batch, stop_flag=True)[0]}'
                msg_progress_start = f'Время выполнения:' \
                                     f'{self.eta_format(self._sum_time + time_start)}, '
            else:
                msg_progress_end = f'Расчетное время окончания:' \
                                   f'{self.update_progress(self.num_batches * self.epochs + 1, self.batch, self._start_time)[0]}, '
                msg_progress_start = f'Время выполнения:' \
                                     f'{self.eta_format(time_start)}, '
            self.batch += 1
            progress.pool(
                self.progress_name,
                percent=(self.last_epoch - 1) / self.epochs * 100,
                message=f"Обучение. Эпоха {self.last_epoch} из {self.epochs}",
                data={'info': f"{msg_progress_start + msg_progress_end + msg_epoch + msg_batch}",
                      'usage': self.usage_info.get_usage()},
                finished=False,
            )
            # print(('Прогресс обучения', msg_progress_start +
            #                              msg_progress_end + msg_epoch + msg_batch))
            # print(self.usage_info.get_usage())

    def on_epoch_end(self, epoch, logs=None):
        """
        Returns:
            {}:
        """
        progress.pool(
            self.progress_name,
            percent=(self.last_epoch - 1) / self.epochs * 100,
            message=f"Обучение. Эпоха {self.last_epoch} из {self.epochs}",
            data={'info': logs,
                  'usage': self.usage_info.get_usage()},
            finished=False,
        )
        # print(logs)
        # print(self.last_epoch)
        # print(self.model.get_weights())
        self.last_epoch += 1
        # self.Exch.last_epoch_model(self.model)

    def on_train_end(self, logs=None):
        self.save_lastmodel()
        time_end = self.update_progress(self.num_batches * self.epochs + 1,
                                        self.batch, self._start_time, finalize=True)[1]
        self._sum_time += time_end
        if self.model.stop_training:
            msg = f'Модель сохранена.'
            print(('Обучение остановлено пользователем!', msg))
            # self.Exch.out_data['stop_flag'] = True
        else:
            if self.retrain_flag:
                msg = f'Затрачено времени на обучение: ' \
                      f'{self.eta_format(time_end)} '
            else:
                msg = f'Затрачено времени на обучение: ' \
                      f'{self.eta_format(self._sum_time)} '
            progress.pool(
                self.progress_name,
                percent=(self.last_epoch - 1) / self.epochs * 100,
                message=f"Обучение завершено. Эпоха {self.last_epoch - 1} из {self.epochs}",
                data={'info': f"Обучение закончено. {msg}",
                      'usage': self.usage_info.get_usage()},
                finished=True,
            )
            # print(msg)


class BaseCallback:
    """Callback for callbacks"""

    def __init__(
            self,
            metrics=None,
            step=1,
            class_metrics=None,
            data_tag=None,
            num_classes=2,
            show_worst=False,
            show_best=True,
            # show_random=True,
            show_final=True,
            dataset=None,
            exchange=Exchange(),
    ):
        """
        Init for base callback
        Args:
            metrics (list):             список используемых метрик: по умолчанию [], что соответсвует 'loss'
            class_metrics:              вывод графиков метрик по каждому сегменту: по умолчанию []
            step int():                 шаг вывода хода обучения, по умолчанию step = 1
            show_worst (bool):          выводить ли экземпляры, худшие по метрикам, по умолчанию False
            show_best (bool):           выводить ли экземпляры, лучшие по метрикам, по умолчанию False
            show_final (bool):          выводить ли в конце обучения график, по умолчанию True
            dataset (DTS):              экземпляр класса DTS
            exchange (Exchange)         экземпляр класса Exchange  (для вывода текстовой и графической инф-ии)
        Returns:
            None
        """
        self.step = step
        if metrics is None:
            metrics = []
        if class_metrics is None:
            class_metrics = []
        if data_tag is None:
            data_tag = []
        self.clbck_metrics = metrics
        self.class_metrics = class_metrics
        self.exchange = exchange
        self.dataset = dataset
        self.show_final = show_final
        self.show_best = show_best
        self.show_worst = show_worst
        # self.show_random = show_random
        self.num_classes = num_classes
        self.data_tag = data_tag
        self.epoch = 0
        self.history = {}
        self.predict_cls = {}
        self.max_accuracy_value = 0
        self.predict_cls = (
            {}
        )  # словарь для сбора истории предикта по классам и метрикам
        self.batch_count = 0
        self.x_Val = {}
        self.y_true = []
        self.y_pred = []
        self.loss = ""
        self.dice = []

    def plot_result(self, output_key: str = None):
        """
        Returns: plot_data
        """
        plot_data = {}
        try:
            msg_epoch = f"Эпоха №{self.epoch + 1:03d}"
            if len(self.clbck_metrics) >= 1:
                for metric_name in self.clbck_metrics:
                    if not isinstance(metric_name, str):
                        metric_name = metric_name.name
                    if len(self.dataset.Y) > 1:
                        # определяем, что демонстрируем во 2м и 3м окне
                        metric_name = f"{output_key}_{metric_name}"
                        val_metric_name = f"val_{metric_name}"
                    else:
                        val_metric_name = f"val_{metric_name}"
                    metric_title = f"{metric_name} и {val_metric_name} {msg_epoch}"
                    xlabel = "Эпоха"
                    ylabel = "Значение"
                    labels = (metric_title, xlabel, ylabel)
                    plot_data[labels] = [
                        [
                            list(range(len(self.history[metric_name]))),
                            self.history[metric_name],
                            f"{metric_name}",
                        ],
                        [
                            list(range(len(self.history[val_metric_name]))),
                            self.history[val_metric_name],
                            f"{val_metric_name}",
                        ],
                    ]

                if len(self.class_metrics):
                    for metric_name in self.class_metrics:
                        if not isinstance(metric_name, str):
                            metric_name = metric_name.name
                        if len(self.dataset.Y) > 1:
                            metric_name = f'{output_key}_{metric_name}'
                            val_metric_name = f"val_{metric_name}"
                        else:
                            val_metric_name = f"val_{metric_name}"
                        xlabel = "Эпоха"
                        if metric_name.endswith("loss"):
                            classes_title = f"Ошибка {output_key} для {self.num_classes} классов. {msg_epoch}"
                            ylabel = "Значение"
                        else:
                            classes_title = f"Точность {output_key} для {self.num_classes} классов. {msg_epoch}"
                            ylabel = "Значение"
                        labels = (classes_title, xlabel, ylabel)
                        plot_data[labels] = [
                            [
                                list(range(len(self.predict_cls[val_metric_name][j]))),
                                self.predict_cls[val_metric_name][j],
                                f" класс {l}", ] for j, l in
                            enumerate(self.dataset.classes_names[output_key])]
        except Exception as e:
            print("Exception", e.__str__())
        return plot_data

    def image_indices(self, count=10, output_key: str = None) -> np.ndarray:
        """
        Computes indices of images based on instance mode ('worst', 'best', "random")
        Returns: array of best or worst predictions indices
        """
        if (self.y_pred.shape[-1] == self.y_true.shape[-1]) \
                and (self.dataset.one_hot_encoding[output_key]) \
                and (self.y_true.shape[-1] > 1):
            classes = np.argmax(self.y_true, axis=-1)
        elif (len(self.y_true.shape) == 1) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_pred.shape[-1] > 1):
            classes = copy.copy(self.y_true)
        elif (len(self.y_true.shape) == 1) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_pred.shape[-1] == 1):
            classes = copy.deepcopy(self.y_true)
        else:
            classes = copy.deepcopy(self.y_true)

        probs = np.array([pred[classes[i]]
                          for i, pred in enumerate(self.y_pred)])
        sorted_args = np.argsort(probs)
        if self.show_best:
            indices = sorted_args[-count:]
        elif self.show_worst:
            indices = sorted_args[:count]
        # elif self.show_random:
        #     indices = np.random.choice(len(probs), count, replace=False)
        else:
            indices = np.random.choice(len(probs), count, replace=False)
        print('indices', indices)
        return indices, probs

    def plot_images(self, input_key: str = 'input_1', output_key: str = None):
        """
        Plot images based on indices in dataset
        Returns:
            images
        """
        images = {"images": []}
        if self.show_best:
            img_title = "лучшее по метрике: "
        elif self.show_worst:
            img_title = "худшее по метрике: "
        else:
            img_title = "случайное: "
        img_indices, probs = self.image_indices(output_key=output_key)
        classes_labels = self.dataset.classes_names[output_key]
        if (self.y_pred.shape[-1] == self.y_true.shape[-1]) \
                and (self.dataset.one_hot_encoding[output_key]) \
                and (self.y_true.shape[-1] > 1):
            y_pred = np.argmax(self.y_pred, axis=-1)
            y_true = np.argmax(self.y_true, axis=-1)
        elif (len(self.y_true.shape) == 1) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_pred.shape[-1] > 1):
            y_pred = np.argmax(self.y_pred, axis=-1)
            y_true = copy.deepcopy(self.y_true)
        elif (len(self.y_true.shape) == 1) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_pred.shape[-1] == 1):
            y_pred = np.reshape(self.y_pred, (self.y_pred.shape[0]))
            y_true = copy.deepcopy(self.y_true)
        else:
            y_pred = np.reshape(self.y_pred, (self.y_pred.shape[0]))
            y_true = copy.deepcopy(self.y_true)

        for idx in img_indices:
            # TODO нужно как то определять тип входа по тэгу (images)
            image = self.x_Val[input_key][idx]
            true_idx = y_true[idx]
            pred_idx = y_pred[idx]

            # исходное изобаржение
            image = self.inverse_scaler(image, input_key)
            image_data = {
                "image": self.image_to_base64(image),
                "title": f"{img_title + str(round(probs[idx], 4))}",
                "info": [
                    {
                        "label": "Выход",
                        "value": output_key,
                    },
                    {
                        "label": "Распознано",
                        "value": classes_labels[pred_idx],
                    },
                    {
                        "label": "Верный ответ",
                        "value": classes_labels[true_idx],
                    }
                ]
            }
            images["images"].append(image_data)

        return images

    def evaluate_accuracy(self, output_key: str = None):
        """
        Compute accuracy for classes

        Returns:
            metric_classes
        """
        metric_classes = []
        if (self.y_pred.shape[-1] == self.y_true.shape[-1]) \
                and (self.dataset.one_hot_encoding[output_key]) \
                and (self.y_true.shape[-1] > 1):
            pred_classes = np.argmax(self.y_pred, axis=-1)
            true_classes = np.argmax(self.y_true, axis=-1)
        elif (len(self.y_true.shape) == 1) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_pred.shape[-1] > 1):
            pred_classes = np.argmax(self.y_pred, axis=-1)
            true_classes = copy.deepcopy(self.y_true)
        elif (len(self.y_true.shape) == 1) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_pred.shape[-1] == 1):
            pred_classes = np.reshape(self.y_pred, (self.y_pred.shape[0]))
            true_classes = copy.deepcopy(self.y_true)
        else:
            pred_classes = np.reshape(self.y_pred, (self.y_pred.shape[0]))
            true_classes = copy.deepcopy(self.y_true)
        for j in range(self.num_classes):
            y_true_count_sum = 0
            y_pred_count_sum = 0
            for i in range(self.y_true.shape[0]):
                y_true_diff = true_classes[i] - j
                if y_true_diff == 0:
                    y_pred_count_sum += 1
                y_pred_diff = pred_classes[i] - j
                if y_pred_diff == 0 and y_true_diff == 0:
                    y_true_count_sum += 1
            if y_pred_count_sum == 0:
                acc_val = 0
            else:
                acc_val = y_true_count_sum / y_pred_count_sum
            metric_classes.append(acc_val)

        return metric_classes

    def evaluate_f1(self, output_key: str = None):
        metric_classes = []

        if (self.y_pred.shape[-1] == self.y_true.shape[-1]) \
                and (self.dataset.one_hot_encoding[output_key]) \
                and (self.y_true.shape[-1] > 1):
            pred_classes = np.argmax(self.y_pred, axis=-1)
            true_classes = np.argmax(self.y_true, axis=-1)
        elif (len(self.y_true.shape) == 1) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_pred.shape[-1] > 1):
            pred_classes = np.argmax(self.y_pred, axis=-1)
            true_classes = copy.deepcopy(self.y_true)
        elif (len(self.y_true.shape) == 1) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_pred.shape[-1] == 1):
            pred_classes = np.reshape(self.y_pred, (self.y_pred.shape[0]))
            true_classes = copy.deepcopy(self.y_true)
        else:
            pred_classes = np.reshape(self.y_pred, (self.y_pred.shape[0]))
            true_classes = copy.deepcopy(self.y_true)
        for j in range(self.num_classes):
            tp = 0
            fp = 0
            fn = 0
            for i in range(self.y_true.shape[0]):
                cross = pred_classes[i][pred_classes[i] == true_classes[i]]
                tp += cross[cross == j].size

                pred_uncross = pred_classes[i][pred_classes[i] != true_classes[i]]
                fp += pred_uncross[pred_uncross == j].size

                true_uncross = true_classes[i][true_classes[i] != pred_classes[i]]
                fn += true_uncross[true_uncross == j].size

            recall = (tp + 1) / (tp + fp + 1)
            precision = (tp + 1) / (tp + fn + 1)
            f1 = 2 * precision * recall / (precision + recall)
            metric_classes.append(f1)

        return metric_classes

    def evaluate_loss(self, output_key: str = None):
        """
        Compute loss for classes

        Returns:
            metric_classes
        """
        metric_classes = []

        if (self.y_pred.shape[-1] == self.y_true.shape[-1]) \
                and (self.dataset.one_hot_encoding[output_key]) \
                and (self.y_true.shape[-1] > 1):
            cross_entropy = CategoricalCrossentropy()
            for i in range(self.num_classes):
                loss = cross_entropy(self.y_true[..., i], self.y_pred[..., i]).numpy()
                metric_classes.append(loss)
        elif (len(self.y_true.shape) == 1) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_pred.shape[-1] > 1):
            y_true = tf.keras.utils.to_categorical(self.y_true, num_classes=self.num_classes)
            cross_entropy = CategoricalCrossentropy()
            for i in range(self.num_classes):
                loss = cross_entropy(y_true[..., i], self.y_pred[..., i]).numpy()
                metric_classes.append(loss)
        elif (len(self.y_true.shape) == 1) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_pred.shape[-1] == 1):
            y_true = tf.keras.utils.to_categorical(self.y_true, num_classes=self.num_classes)
            y_pred = tf.keras.utils.to_categorical(np.reshape(self.y_pred, (self.y_pred.shape[0])),
                                                   num_classes=self.num_classes)
            cross_entropy = CategoricalCrossentropy()
            for i in range(self.num_classes):
                loss = cross_entropy(y_true[..., i], y_pred[..., i]).numpy()
                metric_classes.append(loss)
        else:
            bce = BinaryCrossentropy()
            for i in range(self.num_classes):
                loss = bce(self.y_true[..., i], self.y_pred[..., i]).numpy()
                metric_classes.append(loss)

        return metric_classes

    def plot_text(self, input_key: str = 'input_1', output_key: str = None):
        """
        Plot sample text based on indices in dataset
        Returns:
            images
        """
        text = {"text": []}

        indices, probs = self.image_indices(output_key=output_key)
        if self.show_best:
            text_title = "лучшее по метрике: "
        elif self.show_worst:
            text_title = "худшее по метрике: "
        else:
            text_title = "случайное: "
        classes_labels = self.dataset.classes_names.get(output_key)
        if self.dataset.task_type.get(output_key) != "segmentation":
            if (self.y_pred.shape[-1] == self.y_true.shape[-1]) \
                    and (self.dataset.one_hot_encoding[output_key]) \
                    and (self.y_true.shape[-1] > 1):
                y_pred = np.argmax(self.y_pred, axis=-1)
                y_true = np.argmax(self.y_true, axis=-1)
            elif (len(self.y_true.shape) == 1) \
                    and (not self.dataset.one_hot_encoding[output_key]) \
                    and (self.y_pred.shape[-1] > 1):
                y_pred = np.argmax(self.y_pred, axis=-1)
                y_true = copy.deepcopy(self.y_true)
            elif (len(self.y_true.shape) == 1) \
                    and (not self.dataset.one_hot_encoding[output_key]) \
                    and (self.y_pred.shape[-1] == 1):
                y_pred = np.reshape(self.y_pred, (self.y_pred.shape[0]))
                y_true = copy.deepcopy(self.y_true)
            else:
                y_pred = np.reshape(self.y_pred, (self.y_pred.shape[0]))
                y_true = copy.deepcopy(self.y_true)

            for idx in indices:
                # TODO нужно как то определять тип входа по тэгу (images)
                sample = self.x_Val[input_key][idx]
                true_idx = y_true[idx]
                pred_idx = y_pred[idx]

                # исходный формат примера
                sample = self.inverse_scaler(sample, input_key)
                text_data = {
                    "text": self.dataset.inverse_data(input_key, sample),
                    "title": f"{text_title + str(round(probs[idx], 4))}",
                    "info": [
                        {
                            "label": "Выход",
                            "value": output_key,
                        },
                        {
                            "label": "Распознано",
                            "value": classes_labels[pred_idx],
                        },
                        {
                            "label": "Верный ответ",
                            "value": classes_labels[true_idx],
                        }
                    ]
                }
                text["text"].append(text_data)

        else:
            y_pred = copy.deepcopy(self.y_pred)
            y_true = copy.deepcopy(self.y_true)

            #######################
            # Функция, выводящая точность распознавания каждой категории отдельно
            #######################
            def recognizeSet(tagI, pred, tags, length, value, num_classes):
                total = 0

                for j in range(num_classes):  # общее количество тегов
                    correct = 0
                    for i in range(len(tagI)):  # проходимся по каждому списку списка тегов
                        for k in range(length):  # проходимся по каждому тегу
                            if tagI[i][k][j] == (pred[i][k][j] > value).astype(
                                    int):  # если соответствующие индексы совпадают, значит сеть распознала верно
                                correct += 1
                    print("Сеть распознала категорию '{}' на {}%".format(tags[j], 100 * correct / (len(tagI) * length)))
                    total += 100 * correct / (len(tagI) * length)
                print("средняя точность {}%".format(total / num_classes))

            recognizeSet(y_true, y_pred, classes_labels, y_true.shape[1], 0.999,
                         self.dataset.num_classes.get(output_key))
        return text

    # def image_to_gif(image_directory: pathlib.Path, frames_per_second: float, **kwargs):
    #     """
    #     import imageio
    #     import pathlib
    #     from datetime import datetime
    #
    #     Makes a .gif which shows many images at a given frame rate.
    #     All images should be in order (don't know how this works) in the image directory
    #
    #     Only tested with .png images but may work with others.
    #
    #     :param image_directory:
    #     :type image_directory: pathlib.Path
    #     :param frames_per_second:
    #     :type frames_per_second: float
    #     :param kwargs: image_type='png' or other
    #     :return: nothing
    #     Example:
    #         fps = 5
    #         png_dir = pathlib.Path('/content/valid')
    #         image_to_gif(png_dir, fps)
    #     """
    #     assert isinstance(image_directory, pathlib.Path), "input must be a pathlib object"
    #     image_type = kwargs.get('type', 'jpg')
    #
    #     timestampStr = datetime.now().strftime("%y%m%d_%H%M%S")
    #     gif_dir = image_directory.joinpath(timestampStr + "_GIF.gif")
    #
    #     print('Started making GIF')
    #     print('Please wait... ')
    #
    #     images = []
    #     for file_name in image_directory.glob('*.' + image_type):
    #         images.append(imageio.imread(image_directory.joinpath(file_name)))
    #     imageio.mimsave(gif_dir.as_posix(), images, fps=frames_per_second)
    #
    #     print('Finished making GIF!')
    #     print('GIF can be found at: ' + gif_dir.as_posix())

    # # build gif
    # with imageio.get_writer('mygif.gif', mode='I') as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)

    def image_to_base64(self, image_as_array):
        if image_as_array.dtype != 'uint8':
            image_as_array = image_as_array.astype(np.uint8)
        temp_image = tempfile.NamedTemporaryFile(prefix='image_', suffix='tmp.png', delete=False)
        try:
            plt.imsave(temp_image.name, image_as_array, cmap='Greys')
        except Exception as e:
            plt.imsave(temp_image.name, image_as_array.reshape(image_as_array.shape[:-1]), cmap='gray')
        with open(temp_image.name, 'rb') as img:
            output_image = base64.b64encode(img.read()).decode('utf-8')
        temp_image.close()
        os.remove(temp_image.name)
        return output_image

    def inverse_scaler(self, x, key):
        x_shape = x.shape
        if self.dataset.scaler.get(key) is not None:
            if len(x_shape) > 2:
                x_inverse = self.dataset.scaler.get(key).inverse_transform(x.reshape((-1, 1)))
                x_inverse = x_inverse.reshape(x_shape)
            else:
                x_inverse = self.dataset.scaler.get(key).inverse_transform(x)
            return x_inverse
        else:
            return x


class CustomCallback(keras.callbacks.Callback):
    """CustomCallback for all task type"""

    def __init__(
            self,
            params: dict = None,
            step=1,
            show_final=True,
            dataset=None,
            exchange=Exchange(),
            samples_x: dict = None,
            samples_y: dict = None,
            batch_size: int = None,
            epochs: int = None,
            save_model_path: str = "./",
            model_name: str = "noname",
    ):

        """
        Init for Custom callback
        Args:
            params (list):              список используемых параметров
            step int():                 шаг вывода хода обучения, по умолчанию step = 1
            show_final (bool):          выводить ли в конце обучения график, по умолчанию True
            dataset (DTS):              экземпляр класса DTS
            exchange (Exchange)         экземпляр класса Exchange
        Returns:
            None
        """
        super().__init__()
        if params is None:
            params = {}
        if samples_y is None:
            samples_y = {}
        if samples_x is None:
            samples_x = {}
        self.step = step
        self.clbck_params = params
        self.show_final = show_final
        self.Exch = exchange
        self.DTS = dataset
        self.save_model_path = save_model_path
        self.nn_name = model_name
        self.metrics = []
        self.loss = []
        self.callbacks = []
        self.callbacks_name = []
        self.task_name = []
        self.num_classes = []
        self.y_Scaler = []
        self.tokenizer = []
        self.one_hot_encoding = []
        self.data_tag = []
        self.x_Val = samples_x
        self.y_true = samples_y
        self.batch_size = batch_size
        self.epochs = epochs
        self.last_epoch = 0
        self.batch = 0
        self.num_batches = 0
        self.msg_epoch = ""
        self.y_pred = []
        self.epoch = 0
        self.history = {}
        self._start_time = time.time()
        self._time_batch_step = time.time()
        self._time_first_step = time.time()
        self._sum_time = 0
        self.out_table_data = {}
        self.stop_training = False
        self.retrain_flag = False
        self.stop_flag = False
        self.retrain_epochs = 0
        self.task_type_defaults_dict = {
            "classification": {
                "optimizer_name": "Adam",
                "loss": "categorical_crossentropy",
                "metrics": ["accuracy"],
                "batch_size": 32,
                "epochs": 20,
                "shuffle": True,
                "clbck_object": ClassificationCallback,
                "callback_kwargs": {
                    "metrics": ["loss", "accuracy"],
                    "step": 1,
                    "class_metrics": [],
                    "num_classes": 2,
                    "data_tag": "images",
                    "show_best": True,
                    "show_worst": False,
                    # "show_random": False,
                    "show_final": True,
                    "dataset": self.DTS,
                    "exchange": self.Exch,
                },
            },
            "segmentation": {
                "optimizer_name": "Adam",
                "loss": "categorical_crossentropy",
                "metrics": ["dice_coef"],
                "batch_size": 16,
                "epochs": 20,
                "shuffle": True,
                "clbck_object": SegmentationCallback,
                "callback_kwargs": {
                    "metrics": ["dice_coef"],
                    "step": 1,
                    "class_metrics": [],
                    "num_classes": 2,
                    "data_tag": "images",
                    "show_best": True,
                    "show_worst": False,
                    "show_final": True,
                    "dataset": self.DTS,
                    "exchange": self.Exch,
                },
            },
            "regression": {
                "optimizer_name": "Adam",
                "loss": "mse",
                "metrics": ["mae"],
                "batch_size": 32,
                "epochs": 20,
                "shuffle": True,
                "clbck_object": RegressionCallback,
                "callback_kwargs": {
                    "metrics": ["loss", "mse"],
                    "step": 1,
                    "plot_scatter": True,
                    "show_final": True,
                    "dataset": self.DTS,
                    "exchange": self.Exch,
                },
            },
            "timeseries": {
                "optimizer_name": "Adam",
                "loss": "mse",
                "metrics": ["mae"],
                "batch_size": 32,
                "epochs": 20,
                "shuffle": True,
                "clbck_object": TimeseriesCallback,
                "callback_kwargs": {
                    "metrics": ["loss", "mse"],
                    "step": 1,
                    "corr_step": 50,
                    "plot_pred_and_true": True,
                    "show_final": True,
                    "dataset": self.DTS,
                    "exchange": self.Exch,
                },
            },
            "object_detection": {
                "optimizer_name": "Adam",
                "loss": "yolo_loss",
                "metrics": [],
                "batch_size": 8,
                "epochs": 20,
                "shuffle": True,
                "clbck_object": ObjectdetectionCallback,
                "callback_kwargs": {
                    "metrics": ["loss"],
                    "step": 1,
                    "class_metrics": [],
                    "num_classes": 2,
                    "data_tag": "images",
                    "show_best": False,
                    "show_worst": False,
                    "show_final": True,
                    "dataset": self.DTS,
                    "exchange": self.Exch,
                },
            },
        }
        self.callback_kwargs = []
        self.clbck_object = []
        self.prepare_params()

    def save_lastmodel(self) -> None:
        """
        Saving last model on each epoch end

        Returns:
            None
        """
        model_name = f"model_{self.nn_name}_on_epoch_end.last.h5"
        file_path_model: str = os.path.join(
            self.save_model_path, f"{model_name}"
        )
        self.model.save(file_path_model)
        self.Exch.print_2status_bar(
            ("Инфо", f"Последняя модель сохранена как {file_path_model}")
        )
        pass

    def prepare_callbacks(
            self, task_type: str = "", metrics: list = None, num_classes: int = None,
            clbck_options: dict = {}, tags: dict = {}):
        """
        if terra in raw mode  - setting callback if its set
        if terra with django - checking switches and set callback options from switches

        Returns:
            initialized_callback
        """
        _task_type_defaults_kwargs = self.task_type_defaults_dict.get(task_type)
        callback_kwargs = _task_type_defaults_kwargs["callback_kwargs"]
        if metrics:
            callback_kwargs["metrics"] = copy.deepcopy(metrics)
        for option_name, option_value in clbck_options.items():
            if option_name == "show_every_epoch":
                if option_value:
                    callback_kwargs["step"] = 1
                else:
                    callback_kwargs["step"] = 0
            elif option_name == "plot_loss_metric":
                if option_value:
                    if not ("loss" in callback_kwargs["metrics"]):
                        callback_kwargs["metrics"].append("loss")
                else:
                    if "loss" in callback_kwargs["metrics"]:
                        callback_kwargs["metrics"].remove("loss")
            elif option_name == "plot_metric":
                if option_value:
                    if not (metrics[0] in callback_kwargs["metrics"]):
                        callback_kwargs["metrics"].append(metrics[0])
                else:
                    if metrics[0] in callback_kwargs["metrics"]:
                        callback_kwargs["metrics"].remove(metrics[0])
            elif option_name == "plot_final":
                if option_value:
                    callback_kwargs["show_final"] = True
                else:
                    callback_kwargs["show_final"] = False

        if (task_type == "classification") or (task_type == "segmentation") or (task_type == "object_detection"):
            callback_kwargs["num_classes"] = copy.deepcopy(num_classes)
            if tags["input_1"]:
                callback_kwargs["data_tag"] = tags["input_1"]
            callback_kwargs["class_metrics"] = []
            for option_name, option_value in clbck_options.items():
                if option_name == "plot_loss_for_classes":
                    if option_value:
                        if not ("loss" in callback_kwargs["class_metrics"]):
                            callback_kwargs["class_metrics"].append("loss")
                    else:
                        if "loss" in callback_kwargs["class_metrics"]:
                            callback_kwargs["class_metrics"].remove("loss")
                elif option_name == "plot_metric_for_classes":
                    if option_value:
                        if not (metrics[0] in callback_kwargs["class_metrics"]):
                            callback_kwargs["class_metrics"].append(metrics[0])
                    else:
                        if metrics[0] in callback_kwargs["class_metrics"]:
                            callback_kwargs["class_metrics"].remove(metrics[0])
                elif option_name == "show_worst_images":
                    if option_value:
                        callback_kwargs["show_worst"] = True
                    else:
                        callback_kwargs["show_worst"] = False
                        # callback_kwargs['show_random'] = False
                elif option_name == 'show_best_images':
                    if option_value:
                        callback_kwargs['show_best'] = True
                    else:
                        callback_kwargs['show_best'] = False
                        # callback_kwargs['show_random'] = False
                # elif option_name == 'show_random_images':
                #     if option_value:
                #         callback_kwargs['show_random'] = True
                #     else:
                #         callback_kwargs['show_best'] = False
                #         callback_kwargs['show_worst'] = False
                else:
                    continue

        if task_type == 'regression':
            for option_name, option_value in clbck_options.items():
                if option_name == 'plot_scatter':
                    if option_value:
                        callback_kwargs['plot_scatter'] = True
                    else:
                        callback_kwargs['plot_scatter'] = False

        if task_type == 'timeseries':
            for option_name, option_value in clbck_options.items():
                if option_name == 'plot_autocorrelation':
                    if option_value:
                        callback_kwargs['corr_step'] = 10
                    else:
                        callback_kwargs['corr_step'] = 0
                elif option_name == 'plot_pred_and_true':
                    if option_value:
                        callback_kwargs['plot_pred_and_true'] = True
                    else:
                        callback_kwargs['plot_pred_and_true'] = False
        print("callback_kwargs", callback_kwargs)
        clbck_object = _task_type_defaults_kwargs['clbck_object']
        initialized_callback = clbck_object(**callback_kwargs)
        return initialized_callback

    def prepare_params(self):
        print("self.clbck_params", self.clbck_params)
        print("self.DTS.tags", self.DTS.tags)
        for _key in self.clbck_params.keys():
            if (_key == 'output_1' and self.clbck_params[_key]["task"] == 'object_detection') \
                    or (self.clbck_params[_key]["task"] != 'object_detection'):
                self.metrics.append(self.clbck_params[_key]["metrics"])
                self.loss.append(self.clbck_params[_key]["loss"])
                self.task_name.append(self.clbck_params[_key]["task"])
                self.num_classes.append(self.clbck_params.setdefault(_key)["num_classes"])
                self.y_Scaler.append(self.DTS.scaler.setdefault(_key))
                self.tokenizer.append(self.DTS.tokenizer.setdefault(_key))
                self.one_hot_encoding.append(self.DTS.one_hot_encoding.setdefault(_key))
                initialized_callback = self.prepare_callbacks(
                    task_type=self.clbck_params[_key]["task"].value,
                    metrics=self.clbck_params[_key]["metrics"],
                    num_classes=self.clbck_params.setdefault(_key)["num_classes"],
                    clbck_options=self.clbck_params[_key]["callbacks"],
                    tags=self.DTS.tags,
                )
                self.callbacks.append(initialized_callback)

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

        info = ' %s' % eta_format
        return info

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
        info = self.eta_format(eta)

        return [info, int(eta)]

    def predict_yolo(self, x):
        out1 = self.model.get_layer(name='output_1').output
        out2 = self.model.get_layer(name='output_2').output
        out3 = self.model.get_layer(name='output_3').output
        model = keras.models.Model(self.model.input, [out1, out2, out3])
        return model.predict(x, batch_size=self.batch_size)

    def on_train_begin(self, logs=None):
        # self.model.stop_training = False
        self.stop_training = False
        self._start_time = time.time()
        if not self.stop_flag:
            self.batch = 0
        self.num_batches = self.DTS.X['input_1']['data'][0].shape[0] // self.batch_size

        self.Exch.show_current_epoch(self.last_epoch)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self._time_first_step = time.time()

    def on_train_batch_end(self, batch, logs=None):
        stop = self.Exch.get_stop_training_flag()
        if stop:
            self.model.stop_training = True
            self.stop_training = True
            self.stop_flag = True
            msg = f'ожидайте остановку...'
            self.batch += 1
            self.Exch.print_2status_bar(('Обучение остановлено пользователем', msg))
        else:
            msg_batch = f'Батч {batch}/{self.num_batches}'
            msg_epoch = f'Эпоха {self.last_epoch + 1}/{self.epochs}:' \
                        f'{self.update_progress(self.num_batches, batch, self._time_first_step)[0]}, '
            time_start = \
                self.update_progress(self.num_batches * self.epochs + 1, self.batch, self._start_time, finalize=True)[1]
            if self.retrain_flag:
                msg_progress_end = f'Расчетное время окончания:' \
                                   f'{self.update_progress(self.num_batches * self.retrain_epochs + 1, self.batch, self._start_time)[0]}, '
                msg_progress_start = f'Время выполнения дообучения:' \
                                     f'{self.eta_format(time_start)}, '
            elif self.stop_flag:
                msg_progress_end = f'Расчетное время окончания после остановки:' \
                                   f'{self.update_progress(self.num_batches * self.epochs + 1, self.batch, self._start_time, stop_current=batch, stop_flag=True)[0]}'
                msg_progress_start = f'Время выполнения:' \
                                     f'{self.eta_format(self._sum_time + time_start)}, '
            else:
                msg_progress_end = f'Расчетное время окончания:' \
                                   f'{self.update_progress(self.num_batches * self.epochs + 1, self.batch, self._start_time)[0]}, '
                msg_progress_start = f'Время выполнения:' \
                                     f'{self.eta_format(time_start)}, '
            self.batch += 1
            self.Exch.print_2status_bar(('Прогресс обучения', msg_progress_start +
                                         msg_progress_end + msg_epoch + msg_batch))

    def on_epoch_end(self, epoch, logs=None):
        """
        Returns:
            {}:
        """
        out_table_data = {
            "epoch": {
                "number": self.last_epoch + 1,
                "time": self.update_progress(self.num_batches, self.batch, self._time_first_step, finalize=True)[1],
                "data": {},
            },
            "summary": "",
        }
        out_plots_data = {}
        if self.x_Val.get("input_1") is not None:
            if self.clbck_params['output_1']["task"] == 'object_detection':
                self.y_pred = self.predict_yolo(self.x_Val)
            else:
                self.y_pred = self.model.predict(self.x_Val, batch_size=self.batch_size)
        else:
            self.y_pred = copy.copy(self.y_true)
        if isinstance(self.y_pred, list):
            for i, output_key in enumerate(self.clbck_params.keys()):
                if output_key == 'output_1' and self.clbck_params[output_key]["task"] == 'object_detection':
                    callback_out_data = self.callbacks[i].epoch_end(
                        self.last_epoch,
                        logs=logs,
                        output_key=output_key,
                        y_pred=self.y_pred,
                        y_true=self.y_true,
                        loss=self.loss[i],
                        msg_epoch=self.msg_epoch
                    )
                elif self.clbck_params[output_key]["task"] != 'object_detection':
                    callback_out_data = self.callbacks[i].epoch_end(
                        self.last_epoch,
                        logs=logs,
                        output_key=output_key,
                        y_pred=self.y_pred[i],
                        y_true=self.y_true.get(output_key),
                        loss=self.loss[i],
                        msg_epoch=self.msg_epoch
                    )
                else:
                    callback_out_data = {}
                if len(callback_out_data) != 0:
                    for key in callback_out_data.keys():
                        if key == "table":
                            out_table_data["epoch"]["data"].update({output_key: callback_out_data[key][output_key]})
                        elif key == "plots":
                            out_plots_data.update(callback_out_data[key])

        else:
            for i, output_key in enumerate(self.clbck_params.keys()):
                callback_out_data = self.callbacks[i].epoch_end(
                    self.last_epoch,
                    logs=logs,
                    output_key=output_key,
                    y_pred=self.y_pred,
                    y_true=self.y_true.get(output_key),
                    loss=self.loss[i],
                    msg_epoch=self.msg_epoch
                )
                if len(callback_out_data) != 0:
                    for key in callback_out_data.keys():
                        if key == "table":
                            out_table_data["epoch"]["data"].update({output_key: callback_out_data[key][output_key]})
                        elif key == "plots":
                            out_plots_data.update(callback_out_data[key])

        self.Exch.show_current_epoch(self.last_epoch)
        self.last_epoch += 1
        if len(out_table_data) != 0:
            self.Exch.show_text_data(out_table_data)
        if len(out_plots_data) != 0:
            self.Exch.show_plot_data(out_plots_data)
        self.save_lastmodel()

    def on_train_end(self, logs=None):
        out_table_data = {
            "epoch": {},
            "summary": "",
        }
        out_plots_data = {}
        out_images_data = {"images": {
            "title": "Исходное изображение",
            "values": []
        },
            "ground_truth_masks": {
                "title": "Маска сегментации",
                "values": []
            },
            "predicted_mask": {
                "title": "Результат работы модели",
                "values": []
            },
            "ground_truth_bbox": {
                "title": "Правильный bbox",
                "values": []
            },
            "predicted_bbox": {
                "title": "Результат работы модели",
                "values": []
            },
        }
        for i, output_key in enumerate(self.clbck_params.keys()):
            if (output_key == 'output_1' and self.clbck_params[output_key]["task"] == 'object_detection') \
                    or (self.clbck_params[output_key]["task"] != 'object_detection'):
                callback_out_data = self.callbacks[i].train_end(output_key=output_key, x_val=self.x_Val)
                if len(callback_out_data) != 0:
                    for key in callback_out_data.keys():
                        if key == "plots":
                            out_plots_data.update(callback_out_data[key])
                        elif key == "images":
                            for im_key in callback_out_data[key].keys():
                                if im_key == "images":
                                    out_images_data["images"]["values"].extend(callback_out_data[key][im_key])
                                elif im_key == "ground_truth_masks":
                                    out_images_data["ground_truth_masks"]["values"].extend(
                                        callback_out_data[key][im_key])
                                elif im_key == "predicted_mask":
                                    out_images_data["predicted_mask"]["values"].extend(
                                        callback_out_data[key][im_key])
                                elif im_key == "ground_truth_bbox":
                                    out_images_data["ground_truth_bbox"]["values"].extend(
                                        callback_out_data[key][im_key])
                                elif im_key == "predicted_bbox":
                                    out_images_data["predicted_bbox"]["values"].extend(
                                        callback_out_data[key][im_key])
        self.save_lastmodel()
        time_end = self.update_progress(self.num_batches * self.epochs + 1,
                                        self.batch, self._start_time, finalize=True)[1]
        self._sum_time += time_end
        if self.model.stop_training:
            out_table_data["summary"] = f'Затрачено времени на обучение: ' \
                                        f'{self.eta_format(self._sum_time)} '
            self.Exch.show_text_data(self.out_table_data)
            msg = f'Модель сохранена.'
            self.Exch.print_2status_bar(('Обучение остановлено пользователем!', msg))
            self.Exch.out_data['stop_flag'] = True
        else:
            if self.retrain_flag:
                out_table_data["summary"] = f'Затрачено времени на обучение: ' \
                                            f'{self.eta_format(time_end)} '
            else:
                out_table_data["summary"] = f'Затрачено времени на обучение: ' \
                                            f'{self.eta_format(self._sum_time)} '
            self.Exch.show_text_data(out_table_data)
        if (len(out_images_data["images"]["values"]) or len(out_images_data["ground_truth_bbox"]["values"])) != 0:
            self.Exch.show_image_data(out_images_data)
        if len(out_plots_data) != 0:
            self.Exch.show_plot_data(out_plots_data)


class ClassificationCallback(BaseCallback):
    """Callback for classification"""

    def __init__(
            self,
            metrics=None,
            step=1,
            class_metrics=None,
            data_tag=None,
            num_classes=2,
            show_worst=False,
            show_best=True,
            show_final=True,
            # show_random=True,
            dataset=None,
            exchange=Exchange(),
    ):
        """
        Init for classification callback
        Args:
            metrics (list):             список используемых метрик: по умолчанию [], что соответсвует 'loss'
            class_metrics:              вывод графиков метрик по каждому сегменту: по умолчанию []
            step int():                 шаг вывода хода обучения, по умолчанию step = 1
            show_worst (bool):          выводить ли справа отдельно экземпляры, худшие по метрикам, по умолчанию False
            show_best (bool):           выводить ли справа отдельно экземпляры, лучшие по метрикам, по умолчанию False
            show_final (bool):          выводить ли в конце обучения график, по умолчанию True
            dataset (DTS):              экземпляр класса DTS
        Returns:
            None
        """
        super().__init__()
        if class_metrics is None:
            class_metrics = []
        if metrics is None:
            metrics = []
        self.__name__ = "Callback for classification"
        self.step = step
        self.clbck_metrics = metrics
        self.class_metrics = class_metrics
        self.show_worst = show_worst
        self.show_best = show_best
        # self.show_random = show_random
        self.show_final = show_final
        self.dataset = dataset
        self.Exch = exchange
        self.data_tag = data_tag

        self.accuracy_metric = [[] for i in range(len(self.clbck_metrics))]
        self.accuracy_val_metric = [[] for i in range(len(self.clbck_metrics))]
        self.num_classes = num_classes
        self.acls_lst = [
            [[] for i in range(self.num_classes + 1)]
            for i in range(len(self.clbck_metrics))
        ]

        pass

    def epoch_end(
            self,
            epoch,
            logs: dict = None,
            output_key: str = None,
            y_pred: list = None,
            y_true: dict = None,
            loss: str = None,
            msg_epoch: str = None,
    ):
        """
        Returns:
            {}:
        """
        self.epoch = epoch
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss = loss
        epoch_table_data = {
            output_key: {}
        }
        out_data = {}
        for metric_idx in range(len(self.clbck_metrics)):
            # # проверяем есть ли метрика заданная функцией
            if not isinstance(self.clbck_metrics[metric_idx], str):
                metric_name = self.clbck_metrics[metric_idx].name
                self.clbck_metrics[metric_idx] = metric_name

            if len(self.dataset.Y) > 1:
                metric_name = f'{output_key}_{self.clbck_metrics[metric_idx]}'
                val_metric_name = f"val_{metric_name}"
            else:
                metric_name = f"{self.clbck_metrics[metric_idx]}"
                val_metric_name = f"val_{metric_name}"
            # определяем лучшую метрику для вывода данных при class_metrics='best'
            if logs[val_metric_name] > self.max_accuracy_value:
                self.max_accuracy_value = logs[val_metric_name]
            # собираем в словарь по метрикам
            self.accuracy_metric[metric_idx].append(logs[metric_name])
            self.accuracy_val_metric[metric_idx].append(logs[val_metric_name])
            dm = {metric_name: self.accuracy_metric[metric_idx]}
            self.history.update(dm)
            dv = {val_metric_name: self.accuracy_val_metric[metric_idx]}
            self.history.update(dv)

            epoch_table_data[output_key].update({metric_name: self.history[metric_name][-1]})
            epoch_table_data[output_key].update({val_metric_name: self.history[val_metric_name][-1]})
            out_data.update({"table": epoch_table_data})

            if self.y_pred is not None:
                # распознаем и выводим результат по классам
                # TODO считаем каждую метрику на каждом выходе
                if metric_name.endswith("accuracy"):
                    metric_classes = self.evaluate_accuracy(output_key=output_key)
                elif metric_name.endswith('loss'):
                    metric_classes = self.evaluate_loss(output_key=output_key)
                else:
                    metric_classes = self.evaluate_f1(output_key=output_key)

                # собираем в словарь по метрикам и классам
                if len(metric_classes):
                    dclsup = {}
                    for j in range(self.num_classes):
                        self.acls_lst[metric_idx][j].append(metric_classes[j])
                    dcls = {val_metric_name: self.acls_lst[metric_idx]}
                    dclsup.update(dcls)
                    self.predict_cls.update(dclsup)
        if self.step:
            if (self.epoch % self.step == 0) and (self.step >= 1):
                plot_data = self.plot_result(output_key)
                out_data.update({"plots": plot_data})

        return out_data

    def train_end(self, output_key: str = None, x_val: dict = None):
        self.x_Val = x_val
        out_data = {}
        if self.show_final:
            plot_data = self.plot_result(output_key)
            out_data.update({"plots": plot_data})
            if self.data_tag == 'images':
                if self.show_best or self.show_worst:
                    images = self.plot_images(output_key=output_key)
                    out_data.update({"images": images})
            elif self.data_tag == 'text':
                text = self.plot_text(output_key=output_key)
                print("text", text)

        return out_data


class SegmentationCallback(BaseCallback):
    """Callback for segmentation"""

    def __init__(
            self,
            metrics=None,
            step=1,
            num_classes=2,
            class_metrics=None,
            data_tag=None,
            show_worst=False,
            show_best=True,
            show_final=True,
            dataset=None,
            exchange=Exchange(),
    ):
        """
        Init for classification callback
        Args:
            metrics (list):             список используемых метрик
            class_metrics:              вывод графиков метрик по каждому сегменту
            step int(list):             шаг вывода хода обучения, по умолчанию step = 1
            show_worst bool():          выводить ли справа отдельно, плохие метрики, по умолчанию False
            show_final bool ():         выводить ли в конце обучения график, по умолчанию True
            dataset (trds.DTS):         instance of DTS class
            exchange:                   экземпляр Exchange (для вывода текстовой и графической инф-ии)
        Returns:
            None
        """
        super().__init__()
        if class_metrics is None:
            class_metrics = []
        if metrics is None:
            metrics = []
        self.__name__ = "Callback for segmentation"
        self.step = step
        self.clbck_metrics = metrics
        self.class_metrics = class_metrics
        self.show_worst = show_worst
        self.show_best = show_best
        self.show_final = show_final
        self.dataset = dataset
        self.Exch = exchange
        self.data_tag = data_tag
        self.accuracy_metric = [[] for i in range(len(self.clbck_metrics))]
        self.accuracy_val_metric = [[] for i in range(len(self.clbck_metrics))]
        self.num_classes = num_classes  # количество классов
        self.acls_lst = [
            [[] for i in range(self.num_classes + 1)]
            for i in range(len(self.clbck_metrics))
        ]

        pass

    def _get_colored_mask(self, mask, input_key: str = None, output_key: str = None):
        """
        Transforms prediction mask to colored mask

        Parameters:
        mask : numpy array                 segmentation mask

        Returns:
        colored_mask : numpy array         mask with colors by classes
        """

        def index2color(pix, num_classes, classes_colors):
            index = np.argmax(pix)
            color = []
            for i in range(num_classes):
                if index == i:
                    color = classes_colors[i]
            return color

        colored_mask = []
        mask = mask.reshape(-1, self.num_classes)
        for pix in range(len(mask)):
            colored_mask.append(
                index2color(mask[pix], self.num_classes, self.dataset.classes_colors[output_key])
            )
        colored_mask = np.array(colored_mask).astype(np.uint8)
        self.colored_mask = colored_mask.reshape(self.dataset.input_shape[input_key])

    def _dice_coef(self, smooth=1.0):
        """
        Compute dice coefficient for each mask

        Parameters:
        smooth : float     to avoid division by zero

        Returns:
        -------
        None
        """

        intersection = np.sum(self.y_true * self.y_pred, axis=(1, 2, 3))
        union = np.sum(self.y_true, axis=(1, 2, 3)) + np.sum(
            self.y_pred, axis=(1, 2, 3)
        )
        self.dice = (2.0 * intersection + smooth) / (union + smooth)

    def image_indices(self, count=5, output_key: str = None) -> np.ndarray:
        """
        Computes indices of mask based on instance mode ('worst', 'best', 'random')
        Returns: array of best or worst predictions indices
        """
        self._dice_coef()
        # выбираем count лучших либо худших или случайных результатов сегментации
        if self.show_best:
            indices = np.argsort(self.dice)[-count:]
        elif self.show_worst:
            indices = np.argsort(self.dice)[:count]
        else:
            indices = np.random.choice(len(self.dice), count, replace=False)

        return indices

    def plot_images(self, input_key: str = None, output_key: str = None):
        """
        Returns:
            images
        """
        images = {"images": [],
                  "ground_truth_masks": [],
                  "predicted_mask": [],
                  }
        if self.show_best:
            img_title = "лучшее по метрике: "
        elif self.show_worst:
            img_title = "худшее по метрике: "
        else:
            img_title = "случайное: "
        indexes = self.image_indices(output_key=output_key)

        for idx in indexes:
            # исходное изобаржение
            image_data = {
                "image": None,
                "title": None,
                "info": [
                    {
                        "label": "Выход",
                        "value": output_key,
                    }
                ]
            }

            image = np.squeeze(
                self.x_Val[input_key][idx].reshape(self.dataset.input_shape[input_key])
            )
            image = self.inverse_scaler(image, input_key)
            image_data["image"] = self.image_to_base64(image)
            images["images"].append(image_data)

            # истинная маска
            truth_masks_data = {
                "image": None,
                "title": None,
                "info": [
                    {
                        "label": "Выход",
                        "value": output_key,
                    }
                ]
            }
            self._get_colored_mask(mask=self.y_true[idx], input_key=input_key, output_key=output_key)
            image = np.squeeze(self.colored_mask)

            truth_masks_data["image"] = self.image_to_base64(image)
            images["ground_truth_masks"].append(truth_masks_data)

            # предсказанная маска
            predicted_mask_data = {
                "image": None,
                "title": f'{img_title + str(round(self.dice[idx], 4))}',
                "info": [
                    {
                        "label": "Выход",
                        "value": output_key,
                    }
                ]
            }
            self._get_colored_mask(mask=self.y_pred[idx], input_key=input_key, output_key=output_key)
            image = np.squeeze(self.colored_mask)
            predicted_mask_data["image"] = self.image_to_base64(image)
            images["predicted_mask"].append(predicted_mask_data)

        return images

    def evaluate_accuracy(self, smooth=1.0, output_key: str = None):
        """
        Compute accuracy for classes

        Parameters:
        smooth : float     to avoid division by zero

        Returns:
            metric_classes
        """
        metric_classes = []
        if (self.y_pred.shape[-1] == self.y_true.shape[-1]) \
                and (self.dataset.one_hot_encoding[output_key]) \
                and (self.y_true.shape[-1] > 1):
            predsegments = np.argmax(self.y_pred, axis=-1)
            truesegments = np.argmax(self.y_true, axis=-1)
        elif (self.y_pred.shape[-1] > self.y_true.shape[-1]) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_true.shape[-1] == 1):
            predsegments = np.argmax(self.y_pred, axis=-1)
            truesegments = np.reshape(self.y_true, (self.y_true.shape[0]))
        elif (self.y_pred.shape[-1] == self.y_true.shape[-1]) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_true.shape[-1] == 1):
            predsegments = np.reshape(self.y_pred, (self.y_pred.shape[0]))
            truesegments = np.reshape(self.y_true, (self.y_true.shape[0]))
        else:
            predsegments = np.reshape(self.y_pred, (self.y_pred.shape[0]))
            truesegments = np.reshape(self.y_true, (self.y_true.shape[0]))

        for j in range(self.num_classes):
            summ_val = 0
            for i in range(self.y_true.shape[0]):
                # делаем сегмент класса для сверки
                testsegment = np.ones_like(predsegments[0]) * j
                truezero = np.abs(truesegments[i] - testsegment)
                predzero = np.abs(predsegments[i] - testsegment)
                summ_val += (
                                    testsegment.size - np.count_nonzero(truezero + predzero)
                            ) / (testsegment.size - np.count_nonzero(predzero) + smooth)
            acc_val = summ_val / self.y_true.shape[0]
            metric_classes.append(acc_val)

        return metric_classes

    def evaluate_dice_coef(self, input_key: str = "input_1", smooth=1.0):
        """
        Compute dice coefficient for classes

        Parameters:
        smooth : float     to avoid division by zero

        Returns:
            dice
        """
        # TODO сделать для нескольких входов
        if self.dataset.tags[input_key] == "images":
            axis = (1, 2)
        elif self.dataset.tags[input_key] == "text":
            axis = 1
        intersection = np.sum(self.y_true * self.y_pred, axis=axis)
        union = np.sum(self.y_true, axis=axis) + np.sum(self.y_pred, axis=axis)
        dice = np.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)

        return dice

        # def evaluate_mean_io_u(self, input_key: str = "input_1", smooth=1.0):
        #     """
        #     Compute dice coefficient for classes
        #
        #     Parameters:
        #     smooth : float     to avoid division by zero
        #
        #     Returns:
        #     -------
        #     None
        #     """
        #     # TODO сделать для нескольких входов
        #     if self.dataset.tags[input_key] == "images":
        #         axis = (1, 2)
        #     elif self.dataset.tags[input_key] == "text":
        #         axis = 1
        #     intersection = np.sum(self.y_true * self.y_pred, axis=axis)
        #     union = np.sum(self.y_true, axis=axis) + np.sum(self.y_pred, axis=axis)
        #     dice = np.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)
        #     self.metric_classes = dice

    def epoch_end(
            self,
            epoch: int = None,
            logs: dict = None,
            output_key: str = None,
            y_pred: list = None,
            y_true: dict = None,
            loss: str = None,
            msg_epoch: str = None,
    ):
        """
        Returns:
            {}:
        """
        self.epoch = epoch
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss = loss
        epoch_table_data = {
            output_key: {}
        }
        out_data = {}
        try:
            for metric_idx in range(len(self.clbck_metrics)):
                # проверяем есть ли метрика заданная функцией
                if not isinstance(self.clbck_metrics[metric_idx], str):
                    metric_name = self.clbck_metrics[metric_idx].name
                    self.clbck_metrics[metric_idx] = metric_name
                if len(self.dataset.Y) > 1:
                    metric_name = f'{output_key}_{self.clbck_metrics[metric_idx]}'
                    val_metric_name = f"val_{metric_name}"
                else:
                    metric_name = f"{self.clbck_metrics[metric_idx]}"
                    val_metric_name = f"val_{metric_name}"

                if logs[val_metric_name] > self.max_accuracy_value:
                    self.max_accuracy_value = logs[val_metric_name]

                # собираем в словарь по метрикам
                self.accuracy_metric[metric_idx].append(logs[metric_name])
                self.accuracy_val_metric[metric_idx].append(logs[val_metric_name])
                dm = {str(metric_name): self.accuracy_metric[metric_idx]}
                self.history.update(dm)
                dv = {str(val_metric_name): self.accuracy_val_metric[metric_idx]}
                self.history.update(dv)

                epoch_table_data[output_key].update({metric_name: self.history[metric_name][-1]})
                epoch_table_data[output_key].update({val_metric_name: self.history[val_metric_name][-1]})
                out_data.update({"table": epoch_table_data})

                if self.y_pred is not None:
                    # TODO Добавить другие варианты по используемым метрикам
                    # вычисляем результат по классам
                    if metric_name.endswith("accuracy"):
                        metric_classes = self.evaluate_accuracy(output_key=output_key)
                    elif metric_name.endswith("dice_coef"):
                        metric_classes = self.evaluate_dice_coef(input_key="input_1")
                    elif metric_name.endswith("loss"):
                        metric_classes = self.evaluate_loss(output_key=output_key)
                    else:
                        metric_classes = self.evaluate_f1(output_key=output_key)
                    # собираем в словарь по метрикам и классам
                    if len(metric_classes):
                        dclsup = {}
                        for j in range(self.num_classes):
                            self.acls_lst[metric_idx][j].append(metric_classes[j])
                        dcls = {val_metric_name: self.acls_lst[metric_idx]}
                        dclsup.update(dcls)
                        self.predict_cls.update(dclsup)
        except Exception as e:
            print("Exception epoch_end", e.__str__())

        if self.step > 0:
            if self.epoch % self.step == 0:
                plot_data = self.plot_result(output_key=output_key)
                out_data.update({"plots": plot_data})

        return out_data

    def train_end(self, output_key: str = None, x_val: dict = None, input_key="input_1"):
        self.x_Val = x_val
        out_data = {}
        if self.show_final:

            plot_data = self.plot_result(output_key=output_key)
            out_data.update({"plots": plot_data})

            if self.data_tag == 'images':
                if self.show_best or self.show_worst:
                    images = self.plot_images(input_key=input_key, output_key=output_key)
                    out_data.update({"images": images})

            elif self.data_tag == 'text':

                try:
                    text = self.plot_text(output_key=output_key)
                    print("text", text)
                except Exception as e:
                    print("Exception train_end", e.__str__())

        return out_data


class TimeseriesCallback(BaseCallback):
    def __init__(
            self,
            metrics=None,
            step=1,
            corr_step=50,
            show_final=True,
            plot_pred_and_true=True,
            dataset=None,
            exchange=Exchange(),
    ):
        """
        Init for timeseries callback
        Args:
            metrics (list):             список используемых метрик (по умолчанию clbck_metrics = list()), что соответсвует 'loss'
            step int():                 шаг вывода хода обучения, по умолчанию step = 1
            show_final (bool):          выводить ли в конце обучения график, по умолчанию True
            plot_pred_and_true (bool):  выводить ли графики реальных и предсказанных рядов
            dataset (DTS):              экземпляр класса DTS
            corr_step (int):            количество шагов для отображения корреляции (при <= 0 не отображается)
        Returns:
            None
        """
        super().__init__()
        self.__name__ = "Callback for timeseries"
        if metrics is None:
            metrics = []
        self.metrics = metrics
        self.step = step
        self.show_final = show_final
        self.plot_pred_and_true = plot_pred_and_true
        self.dataset = dataset
        self.Exch = exchange
        self.corr_step = corr_step

        self.losses = (
            self.metrics if "loss" in self.metrics else self.metrics + ["loss"]
        )
        self.met = [[] for _ in range(len(self.losses))]
        self.valmet = [[] for _ in range(len(self.losses))]

    def plot_result(self, input_key: str = "input_1", output_key: str = None):
        """
        Returns: plot_data
        """
        plot_data = {}
        for i in range(len(self.losses)):
            # проверяем есть ли метрика заданная функцией
            if not isinstance(self.losses[i], str):
                metric_name = self.losses[i].name
                self.losses[i] = metric_name
            if len(self.dataset.Y) > 1:
                showmet = f'{output_key}_{self.losses[i]}'
                vshowmet = f"val_{showmet}"
            else:
                showmet = f'{self.losses[i]}'
                vshowmet = f"val_{showmet}"
            epochcomment = f" эпоха {self.epoch + 1}"
            loss_len = len(self.history[showmet])

            metric_title = (
                f"{showmet} и {vshowmet}{epochcomment}",
                "эпоха",
                f"{showmet}",
            )
            plot_data.update(
                {
                    metric_title: [
                        [list(range(loss_len)), self.history[showmet], showmet],
                        [list(range(loss_len)), self.history[vshowmet], vshowmet],
                    ]
                }
            )

        if self.plot_pred_and_true and self.y_true is not None:
            x_val = self.inverse_scaler(self.dataset.X.get(input_key)['data'][1], input_key)
            print('x_val[0]', x_val[0].shape)
            y_true = self.inverse_scaler(self.y_true, output_key)
            y_pred = self.inverse_scaler(self.y_pred, output_key)
            if len(y_true.shape) == 2:
                for ch in range(y_true.shape[1]):
                    pred_title = (
                        f"Предикт канала {self.dataset.classes_names.get(output_key)[ch]}", "Время", "Значение")
                    plot_data.update(
                        {
                            pred_title: [
                                [list(range(len(x_val[0]))), x_val[0][:, 0], "Исходный"],
                                [list(range(len(y_true))), y_true[:60, ch], "Истина"],
                                [list(range(len(y_pred))), y_pred[60:, ch], "Предикт"],
                            ]
                        }
                    )

        return plot_data

    @staticmethod
    def autocorr(a, b):
        ma = a.mean()
        mb = b.mean()
        mab = (a * b).mean()
        sa = a.std()
        sb = b.std()
        corr = 0
        if (sa > 0) & (sb > 0):
            corr = (mab - ma * mb) / (sa * sb)
        return corr

    @staticmethod
    def collect_correlation_data(y_pred, y_true, channel, ch_label, corr_steps=10):
        corr_list = []
        autocorr_list = []
        yLen = y_true.shape[0]
        for i in range(corr_steps):
            corr_list.append(
                TimeseriesCallback.autocorr(
                    y_true[: yLen - i, channel], y_pred[i:, channel]
                )
            )
            autocorr_list.append(
                TimeseriesCallback.autocorr(
                    y_true[: yLen - i, channel], y_true[i:, channel]
                )
            )
        corr_label = f"Предсказание на {corr_steps} шаг"
        autocorr_label = "Эталон"
        title = (f"Автокорреляция канала {ch_label[channel]}", '', '')
        correlation_data = {
            title: [
                [list(range(corr_steps)), corr_list, corr_label],
                [list(range(corr_steps)), autocorr_list, autocorr_label],
            ]
        }
        return correlation_data

    def epoch_end(
            self,
            epoch: int = None,
            logs: dict = None,
            output_key: str = None,
            y_pred: list = None,
            y_true: dict = None,
            loss: str = None,
            msg_epoch: str = None,
    ):
        self.epoch = epoch
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss = loss
        epoch_table_data = {
            output_key: {}
        }
        out_data = {}
        for i in range(len(self.losses)):
            # проверяем есть ли метрика заданная функцией
            if type(self.losses[i]) == types.FunctionType:
                metric_name = self.losses[i].name
                self.losses[i] = metric_name
            if len(self.dataset.Y) > 1:
                metric_name = f'{output_key}_{self.losses[i]}'
                val_metric_name = f"val_{metric_name}"
            else:
                metric_name = f'{self.losses[i]}'
                val_metric_name = f"val_{metric_name}"

            # собираем в словарь по метрикам
            self.met[i].append(logs[metric_name])
            self.valmet[i].append(logs[val_metric_name])
            self.history[metric_name] = self.met[i]
            self.history[val_metric_name] = self.valmet[i]

            epoch_table_data[output_key].update({metric_name: self.history[metric_name][-1]})
            epoch_table_data[output_key].update({val_metric_name: self.history[val_metric_name][-1]})
            out_data.update({"table": epoch_table_data})

        if self.step:
            if (self.epoch % self.step == 0) and (self.step >= 1):
                plot_data = self.plot_result(output_key=output_key)
                out_data.update({"plots": plot_data})

        return out_data

    def train_end(self, output_key: str = None, x_val: dict = None):
        self.x_Val = x_val
        out_data = {}
        if self.show_final and self.y_true is not None:
            plot_data = self.plot_result(output_key=output_key)
            out_data.update({"plots": plot_data})
            # Plot correlation and autocorrelation graphics
            if self.corr_step > 0:
                if len(self.y_true.shape) == 2:
                    for ch in range(self.y_true.shape[1]):
                        corr_data = TimeseriesCallback.collect_correlation_data(
                            self.y_pred, self.y_true, ch, self.dataset.classes_names.get(output_key), self.corr_step
                        )
                        out_data["plots"].update(corr_data)

        return out_data


class RegressionCallback(BaseCallback):
    def __init__(
            self,
            metrics=None,
            step=1,
            show_final=True,
            plot_scatter=False,
            dataset=None,
            exchange=Exchange(),
    ):
        """
        Init for regression callback
        Args:
            metrics (list):         список используемых метрик
            step int():             шаг вывода хода обучения, по умолчанию step = 1
            plot_scatter (bool):    вывод граыика скатера по умолчанию True
            show_final (bool):      выводить ли в конце обучения график, по умолчанию True
            exchange:               экземпляр Exchange (для вывода текстовой и графической инф-ии)
        Returns:
            None
        """
        super().__init__()
        self.__name__ = "Callback for regression"
        if metrics is None:
            metrics = []
        self.step = step
        self.metrics = metrics
        self.show_final = show_final
        self.plot_scatter = plot_scatter
        self.dataset = dataset
        self.Exch = exchange
        self.losses = (
            self.metrics if "loss" in self.metrics else self.metrics + ["loss"]
        )
        self.met = [[] for _ in range(len(self.losses))]
        self.valmet = [[] for _ in range(len(self.losses))]

    def plot_result(self, output_key=None):
        """
        Returns: plot_data
        """
        plot_data = {}
        for i in range(len(self.losses)):
            # проверяем есть ли метрика заданная функцией
            if not isinstance(self.losses[i], str):
                metric_name = self.losses[i].name
                self.losses[i] = metric_name
            if len(self.dataset.Y) > 1:
                showmet = f'{output_key}_{self.losses[i]}'
                vshowmet = f"val_{showmet}"
            else:
                showmet = f'{self.losses[i]}'
                vshowmet = f"val_{showmet}"
            epochcomment = f" эпоха {self.epoch + 1}"
            loss_len = len(self.history[showmet])

            metric_title = f"метрика: {showmet} и {vshowmet}{epochcomment}"
            xlabel = "эпоха"
            ylabel = f"{showmet}"
            labels = (metric_title, xlabel, ylabel)
            plot_data[labels] = [
                (list(range(loss_len)), self.history[showmet], showmet),
                (list(range(loss_len)), self.history[vshowmet], vshowmet),
            ]

        if self.plot_scatter and self.y_true is not None:
            y_true = self.inverse_scaler(self.y_true, output_key)
            y_pred = self.inverse_scaler(self.y_pred, output_key)
            data = {}
            scatter_title = "Scatter"
            xlabel = "Истинные значения"
            ylabel = "Предсказанные значения"
            key = (scatter_title, xlabel, ylabel)
            value = [(y_true.reshape(-1), y_pred.reshape(-1), "Регрессия")]
            data.update({key: value})
            self.Exch.show_scatter_data(data)

        return plot_data

    def epoch_end(
            self,
            epoch: int = None,
            logs: dict = None,
            output_key: str = None,
            y_pred: list = None,
            y_true: dict = None,
            loss: str = None,
            msg_epoch: str = None,
    ):

        self.epoch = epoch
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss = loss
        epoch_table_data = {
            output_key: {}
        }
        out_data = {}
        for i in range(len(self.losses)):
            # проверяем есть ли метрика заданная функцией
            if type(self.losses[i]) == types.FunctionType:
                metric_name = self.losses[i].name
                self.losses[i] = metric_name
            if len(self.dataset.Y) > 1:
                metric_name = f'{output_key}_{self.losses[i]}'
                val_metric_name = f"val_{metric_name}"
            else:
                metric_name = f'{self.losses[i]}'
                val_metric_name = f"val_{metric_name}"
            # собираем в словарь по метрикам
            self.met[i].append(logs[metric_name])
            self.valmet[i].append(logs[val_metric_name])
            self.history[metric_name] = self.met[i]
            self.history[val_metric_name] = self.valmet[i]

            epoch_table_data[output_key].update({metric_name: self.history[metric_name][-1]})
            epoch_table_data[output_key].update({val_metric_name: self.history[val_metric_name][-1]})
            out_data.update({"table": epoch_table_data})

        if self.step > 0:
            if self.epoch % self.step == 0:
                plot_data = self.plot_result(output_key=output_key)
                out_data.update({"plots": plot_data})

        return out_data

    def train_end(self, output_key: str = None, x_val: dict = None):
        self.x_Val = x_val
        out_data = {}
        if self.show_final:
            plot_data = self.plot_result(output_key=output_key)
            out_data.update({"plots": plot_data})

        return out_data


class ObjectdetectionCallback(BaseCallback):
    """Callback for object_detection"""

    def __init__(
            self,
            metrics=None,
            step=1,
            class_metrics=None,
            data_tag=None,
            num_classes=2,
            show_worst=False,
            show_best=True,
            show_final=True,
            dataset=None,
            exchange=Exchange(),
    ):
        """
        Init for ObjectdetectionCallback callback
        Args:
            metrics (list):             список используемых метрик: по умолчанию [], что соответсвует 'loss'
            class_metrics:              вывод графиков метрик по каждому сегменту: по умолчанию []
            step int():                 шаг вывода хода обучения, по умолчанию step = 1
            show_worst (bool):          выводить ли справа отдельно экземпляры, худшие по метрикам, по умолчанию False
            show_best (bool):           выводить ли справа отдельно экземпляры, лучшие по метрикам, по умолчанию False
            show_final (bool):          выводить ли в конце обучения график, по умолчанию True
            dataset (DTS):              экземпляр класса DTS
        Returns:
            None
        """
        super().__init__()
        if class_metrics is None:
            class_metrics = []
        if metrics is None:
            metrics = []
        self.__name__ = "Callback for object_detection"
        self.step = step
        self.clbck_metrics = metrics
        self.class_metrics = class_metrics
        self.show_worst = show_worst
        self.show_best = show_best
        self.show_final = show_final
        self.dataset = dataset
        self.Exch = exchange
        self.data_tag = data_tag

        self.accuracy_metric = [[] for i in range(len(self.clbck_metrics))]
        self.accuracy_val_metric = [[] for i in range(len(self.clbck_metrics))]
        self.num_classes = num_classes
        self.acls_lst = [
            [[] for i in range(self.num_classes + 1)]
            for i in range(len(self.clbck_metrics))
        ]

        pass

    def epoch_end(
            self,
            epoch,
            logs: dict = None,
            output_key: str = None,
            y_pred: list = None,
            y_true: dict = None,
            loss: str = None,
            msg_epoch: str = None,
    ):
        """
        Returns:
            {}:
        """
        self.epoch = epoch
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss = loss
        epoch_table_data = {
            output_key: {}
        }
        out_data = {}
        for metric_idx in range(len(self.clbck_metrics)):
            # # проверяем есть ли метрика заданная функцией
            if not isinstance(self.clbck_metrics[metric_idx], str):
                metric_name = self.clbck_metrics[metric_idx].name
                self.clbck_metrics[metric_idx] = metric_name

            # if len(self.dataset.Y) > 1:
            #     metric_name = f'{output_key}_{self.clbck_metrics[metric_idx]}'
            #     val_metric_name = f"val_{metric_name}"
            # else:
            metric_name = f"{self.clbck_metrics[metric_idx]}"
            val_metric_name = f"val_{metric_name}"

            # определяем лучшую метрику для вывода данных при class_metrics='best'
            if logs[val_metric_name] > self.max_accuracy_value:
                self.max_accuracy_value = logs[val_metric_name]
            # собираем в словарь по метрикам
            self.accuracy_metric[metric_idx].append(logs[metric_name])
            self.accuracy_val_metric[metric_idx].append(logs[val_metric_name])
            dm = {metric_name: self.accuracy_metric[metric_idx]}
            self.history.update(dm)
            dv = {val_metric_name: self.accuracy_val_metric[metric_idx]}
            self.history.update(dv)

            epoch_table_data[output_key].update({metric_name: self.history[metric_name][-1]})
            epoch_table_data[output_key].update({val_metric_name: self.history[val_metric_name][-1]})
            out_data.update({"table": epoch_table_data})

        if self.step:
            if (self.epoch % self.step == 0) and (self.step >= 1):
                plot_data = self.plot_result(output_key)
                out_data.update({"plots": plot_data})

        return out_data

    def train_end(self, output_key: str = None, x_val: dict = None):
        self.x_Val = x_val
        out_data = {}
        if self.show_final:
            plot_data = self.plot_result(output_key)
            out_data.update({"plots": plot_data})
            print("self.data_tag", self.data_tag)
            if self.data_tag == 'images':
                print("self.show_best or self.show_worst", self.show_best, self.show_worst)
                if self.show_best or self.show_worst:
                    images = self.plot_images(input_key="input_1", output_key=output_key)
                    out_data.update({"images": images})

        return out_data

    def plot_result(self, output_key: str = None):
        """
        Returns: plot_data
        """
        plot_data = {}
        msg_epoch = f"Эпоха №{self.epoch + 1:03d}"
        if len(self.clbck_metrics) >= 1:
            for metric_name in self.clbck_metrics:
                if not isinstance(metric_name, str):
                    metric_name = metric_name.name
                # if len(self.dataset.Y) > 1:
                #     # определяем, что демонстрируем во 2м и 3м окне
                #     metric_name = f"{output_key}_{metric_name}"
                #     val_metric_name = f"val_{metric_name}"
                # else:
                val_metric_name = f"val_{metric_name}"
                metric_title = f"{metric_name} и {val_metric_name} {msg_epoch}"
                xlabel = "эпоха"
                ylabel = f"{metric_name}"
                labels = (metric_title, xlabel, ylabel)
                plot_data[labels] = [
                    [
                        list(range(len(self.history[metric_name]))),
                        self.history[metric_name],
                        f"{metric_name}",
                    ],
                    [
                        list(range(len(self.history[val_metric_name]))),
                        self.history[val_metric_name],
                        f"{val_metric_name}",
                    ],
                ]

        return plot_data

    def plot_images(self, input_key: str = None, output_key: str = None):
        """
        Returns:
            images
        """
        images = {"images": [],
                  "ground_truth_bbox": [],
                  "predicted_bbox": [],
                  }
        # self._dice_coef()
        # выбираем 5 лучших либо 5 худших результатов сегментации
        # if self.show_best:
        #     indexes = np.argsort(self.dice)[-5:]
        # elif self.show_worst:
        #     indexes = np.argsort(self.dice)[:5]
        print("self.x_Val.get(input_key).shape", self.x_Val.get(input_key).shape)
        indexes = np.random.choice((self.x_Val.get(input_key).shape[0]) - 1, 5, replace=False)
        print("indexes", indexes)
        for idx in indexes:
            # исходное изобаржение
            original_image = self.x_Val.get(input_key)[idx]
            image_shape = original_image.shape
            if self.dataset.scaler.get(input_key) is not None:
                original_image = self.dataset.scaler.get(input_key).inverse_transform(original_image.reshape((-1, 1)))
                original_image = original_image.reshape(image_shape)
                print("original_image scaler", original_image)
            image_data = {
                "image": None,
                "title": None,
                "info": [
                    {
                        "label": 'Выход',
                        "value": output_key,
                    }
                ]
            }

            # image = np.squeeze(
            #     self.x_Val.get(input_key)[idx])  # .reshape(self.dataset.input_shape.get(input_key)
            image_data["image"] = self.image_to_base64(original_image)
            images["images"].append(image_data)

            # истинная маска
            truth_bbox_data = {
                "image": None,
                "title": None,
                "info": [
                    {
                        "label": None,
                        "value": None,
                    }
                ]
            }
            # self._get_colored_mask(mask=self.y_true[idx], input_key=input_key, output_key=output_key)
            # image = np.squeeze(self.colored_mask)
            #
            # print("self.y_true", self.y_true[0].shape, self.y_true[1].shape, self.y_true[2].shape)
            image = self.get_frame(original_image,
                                   self.y_true, input_key=input_key,
                                   output_key=output_key)
            print("Posle images truth")
            truth_bbox_data["image"] = self.image_to_base64(image)
            images["ground_truth_bbox"].append(truth_bbox_data)

            # предсказанная маска
            predicted_bbox_data = {
                "image": None,
                "title": None,
                "info": [
                    {
                        "label": "Выход",
                        "value": output_key,
                    }
                ]
            }
            print("do self.get_frame")
            image = self.get_frame(original_image,
                                   self.y_pred, input_key=input_key,
                                   output_key=output_key)  # .reshape(self.dataset.input_shape.get(input_key)
            # self._get_colored_mask(mask=self.y_pred[idx], input_key=input_key, output_key=output_key)
            # image = np.squeeze(self.colored_mask)
            predicted_bbox_data["image"] = self.image_to_base64(image)
            # print("predicted_bbox_dat", predicted_bbox_data)
            images["predicted_bbox"].append(predicted_bbox_data)

        return images

    def get_frame(self, original_image, predict, input_key: str = None, output_key: str = None):
        ANCHORS = [[[10, 13], [16, 30], [33, 23]],
                   [[30, 61], [62, 45], [59, 119]],
                   [[116, 90], [156, 198], [373, 326]]]
        input_size = 416
        score_threshold = 0.3
        iou_threshold = 0.45
        pred_bbox = []
        for i in range(len(ANCHORS)):
            pred_bbox.append(self.decode(predict[i], i))
        print("pred_bbox", len(pred_bbox))
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        print("reshape predbox")
        pred_bbox = tf.concat(pred_bbox, axis=0)
        print("concat predbox")
        bboxes = self.postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
        print("bboxes postproces")
        bboxes = self.nms(bboxes, iou_threshold, method='nms')
        print("bboxes nms", bboxes)
        print("self.dataset.classes_names", self.dataset.classes_names)

        image = self.draw_bbox(original_image, bboxes, CLASS=self.dataset.classes_names[output_key])
        # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))
        print("image.shape", image.shape)
        print(image)
        return image

    def video_detection(self, yolo, video_path, output_path, classes, score_threshold=0.3, input_size=416,
                        iou_threshold=0.45, rectangle_colors='', input_key: str = None, output_key: str = None):

        times, times_2 = [], []

        vid = cv2.VideoCapture(video_path)  # detect on video

        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, codec, fps, (width, height))  # output_path must be .mp4

        while True:
            _, frame = vid.read()

            try:
                original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            except:
                break

            # TODO здесь должен быть метод датасета препроцессинга картинки.
            image_data = 0  #
            # image_data = image_preprocess(np.copy(original_frame), [input_size, input_size])
            # image_data = image_data[np.newaxis, ...].astype(np.float32)

            t1 = time.time()

            pred_bbox = yolo.predict(image_data)

            # t1 = time.time()
            # pred_bbox = Yolo.predict(image_data)
            t2 = time.time()

            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)

            bboxes = self.postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
            bboxes = self.nms(bboxes, iou_threshold, method='nms')

            # extract bboxes to boxes (x, y, width, height), scores and names
            boxes, scores, names = [], [], []
            for bbox in bboxes:
                boxes.append(
                    [bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int) - bbox[0].astype(int),
                     bbox[3].astype(int) - bbox[1].astype(int)])
                scores.append(bbox[4])
                names.append(classes[int(bbox[5])])

            t3 = time.time()
            times.append(t2 - t1)
            times_2.append(t3 - t1)

            times = times[-20:]
            times_2 = times_2[-20:]

            ms = sum(times) / len(times) * 1000
            fps = 1000 / ms
            fps2 = 1000 / (sum(times_2) / len(times_2) * 1000)

            image = cv2.putText(image, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (0, 0, 255), 2)

            # draw original yolo detection
            image = self.draw_bbox(image, bboxes, CLASS=classes, show_label=True, rectangle_colors=rectangle_colors)

            print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
            if output_path != '':
                out.write(image)

    @staticmethod
    def decode(conv_output, i=0):
        ANCHORS = [[[10, 13], [16, 30], [33, 23]],
                   [[30, 61], [62, 45], [59, 119]],
                   [[116, 90], [156, 198], [373, 326]]]
        STRIDES = [32, 16, 8]
        # where i = 0, 1 or 2 to correspond to the three grid scales
        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]

        # conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]  # offset of center position
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]  # Prediction box length and width offset
        conv_raw_conf = conv_output[:, :, :, :, 4:5]  # confidence of the prediction box
        conv_raw_prob = conv_output[:, :, :, :, 5:]  # category probability of the prediction box

        # next need Draw the grid. Where output_size is equal to 13, 26 or 52
        y = tf.range(output_size, dtype=tf.int32)
        y = tf.expand_dims(y, -1)
        y = tf.tile(y, [1, output_size])
        x = tf.range(output_size, dtype=tf.int32)
        x = tf.expand_dims(x, 0)
        x = tf.tile(x, [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        # Calculate the center position of the prediction box:
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
        # Calculate the length and width of the prediction box:
        pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]

        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        pred_conf = tf.sigmoid(conv_raw_conf)  # object box calculates the predicted confidence
        pred_prob = tf.sigmoid(conv_raw_prob)  # calculating the predicted probability category box object

        # calculating the predicted probability category box object
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    @staticmethod
    def bbox_iou(boxes1, boxes2):
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return 1.0 * inter_area / union_area

    @staticmethod
    def draw_bbox(image, bboxes, CLASS, show_label=True, show_confidence=True,
                  Text_colors=(255, 255, 0), rectangle_colors=''):

        num_classes = len(CLASS)
        print("CLASS", CLASS, num_classes)
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        # print("hsv_tuples", hsv_tuples)
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for i, bbox in enumerate(bboxes):
            coor = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            print("bbox[5]", bbox[5])
            class_ind = int(bbox[5])
            print("class_ind", class_ind)
            bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 1000)
            if bbox_thick < 1: bbox_thick = 1
            fontScale = 0.75 * bbox_thick
            (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

            # put object rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick * 2)

            if show_label:
                # get text label
                score_str = " {:.2f}".format(score) if show_confidence else ""

                try:
                    label = "{}".format(CLASS[class_ind]) + score_str
                except KeyError:
                    print("You received KeyError, this might be that you are trying to use yolo original weights")
                    print(
                        "while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")

                # get text size
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                      fontScale, thickness=bbox_thick)
                # put filled text rectangle
                cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color,
                              thickness=cv2.FILLED)

                # put text above rectangle
                cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

        return image

    @staticmethod
    def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
        """
        :param bboxes: (xmin, ymin, xmax, ymax, score, class)
        Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
              https://github.com/bharatsingh430/soft-nms
        """
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        def bboxes_iou(boxes1, boxes2):
            boxes1 = np.array(boxes1)
            boxes2 = np.array(boxes2)

            boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
            boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

            left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
            right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

            inter_section = np.maximum(right_down - left_up, 0.0)
            inter_area = inter_section[..., 0] * inter_section[..., 1]
            union_area = boxes1_area + boxes2_area - inter_area
            ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

            return ious

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]
            # Process 1: Determine whether the number of bounding boxes is greater than 0
            while len(cls_bboxes) > 0:
                # Process 2: Select the bounding box with the highest score according to socre order A
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                # Process 3: Calculate this bounding box A and
                # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold
                iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                assert method in ['nms', 'soft-nms']

                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0

                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes

    @staticmethod
    def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
        valid_scale = [0, np.inf]
        pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                    pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
        # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        org_h, org_w = original_image.shape[:2]
        resize_ratio = min(input_size / org_w, input_size / org_h)

        dw = (input_size - resize_ratio * org_w) / 2
        dh = (input_size - resize_ratio * org_h) / 2

        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # 3. clip some boxes those are out of range
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # 4. discard some invalid boxes
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # 5. discard boxes with low scores
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > score_threshold
        mask = np.logical_and(scale_mask, score_mask)
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

        return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)


if __name__ == "__main__":
    def model_func(input_shape, num_classes):
        input_1 = Input(shape=input_shape, name='input_1')
        x_1 = BatchNormalization()(input_1)
        x_2 = Conv2D(filters=16, kernel_size=(2, 2), padding='same', strides=(2, 2))(x_1)
        x_2 = InstanceNormalization()(x_2)
        x_3 = Conv2D(filters=16, kernel_size=(2, 2), padding='same', strides=(2, 2))(x_2)
        x_4 = MaxPooling2D(pool_size=(4, 4), strides=None, padding='same')(x_3)
        x_6 = Flatten()(x_4)
        x_7 = Dense(units=256, activation='relu')(x_6)
        output_1 = Dense(units=num_classes, activation='softmax', name='output_2')(x_7)
        model = Model([input_1], [output_1])
        return model


    # dataset_variants = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'boston_housing',
    #                     'reuters', 'трейдинг', 'умный_дом', 'квартиры', 'автомобили', 'автомобили_3',
    #                     'заболевания', 'договоры', 'самолеты', 'губы', 'sber']
    # with open('G:/Мой диск/Colab Notebooks/Стажировка/terra_gui/TerraAI/datasets/dataset автомобили 3/config.json',
    #           'r') as cfg:
    #     ddd = json.load(cfg)
    # print('\nconfig.json\n', ddd)
    #
    # dataset_data = DatasetData(**ddd)
    # dtts = PrepareDTS(dataset_data)
    # dtts.prepare_dataset()
    # print(dtts.X.get('train').get(1).shape, dtts.Y.get('train').get(2).shape)
    # # print(dtts.data.encoding.get(2))
    # # print(dtts.Y.get('train').get(2))
    #
    # model = model_func(dtts.data.inputs.get(1).shape, dtts.data.num_classes.get(2))
    # model.compile(optimizer='adam', loss="CategoricalCrossentropy")
    # model.fit(dtts.X.get('train').get(1), dtts.Y.get('train').get(2), epochs=2)

    # start = time.time()
    # clb = InteractiveCallback(model, dtts, log_gap=5,
    #                           metrics={2: ['Accuracy', 'CategoricalAccuracy']},
    #                           losses={2: ["CategoricalCrossentropy"]})
    # print(clb.log_history)
    # clb.update_state(current_epoch=1, current_weights=model.get_weights())
    # print(clb.Y_pred['train'].keys(), clb.Y['train'].keys())
    # clb.get_prediction()
    # print(clb.Y_pred)
    # print(clb.log_history)
    # print(time.time() - start)
    # to_json = json.dumps(clb.log_history)
    # print(to_json)

    # print(TasksGroups[0]['task'].value)
    # print(TaskChoice.Classification.value)
    # print(TasksGroups[0]['task'].value == dtts.data.task_type.get(2).value)
    # print(TasksGroups[0]['metrics'])
    # print(getattr(tf.keras.metrics, TasksGroups[0]['metrics'][0])
    # )

    MemoryUsage(debug=True).get_usage()
