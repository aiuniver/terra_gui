import copy
import importlib
import math
import os
import random
import re
import string
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow
from PIL import Image, ImageDraw, ImageFont  # Модули работы с изображениями
from pandas import DataFrame

from tensorflow.keras.utils import to_categorical
from pydub import AudioSegment
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import moviepy.editor as moviepy_editor
from tensorflow.python.keras.api.keras.preprocessing import image

from terra_ai import progress
from terra_ai.data.datasets.dataset import DatasetOutputsData, DatasetData
from terra_ai.data.datasets.extra import LayerInputTypeChoice, LayerOutputTypeChoice, DatasetGroupChoice, \
    LayerEncodingChoice
from terra_ai.data.presets.training import Metric
from terra_ai.data.training.extra import ExampleChoiceTypeChoice, LossGraphShowChoice
from terra_ai.data.training.train import InteractiveData
from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.utils import camelize, decamelize

__version__ = 0.083


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
    y_array = np.argmax(y_array, axis=-1) if ohe else np.squeeze(y_array)
    for y in y_array:
        class_dict[classes_names[y]] += 1
    return class_dict


loss_metric_config = {
    "loss": {
        "BinaryCrossentropy": {
            "log_name": "binary_crossentropy",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "CategoricalCrossentropy": {
            "log_name": "categorical_crossentropy",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "CategoricalHinge": {
            "log_name": "categorical_hinge",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        'ContrastiveLoss': {
            "log_name": "contrastive_loss",
            "mode": "min",
            "module": "tensorflow_addons.losses"
        },
        "CosineSimilarity": {
            "log_name": "cosine_similarity",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },  # min if loss, max if metric
        "Hinge": {
            "log_name": "hinge",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "Huber": {
            "log_name": "huber",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "KLDivergence": {
            "log_name": "kullback_leibler_divergence",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "LogCosh": {
            "log_name": "logcosh",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "MeanAbsoluteError": {
            "log_name": "mean_absolute_error",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "MeanAbsolutePercentageError": {
            "log_name": "mean_absolute_percentage_error",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "MeanSquaredError": {
            "log_name": "mean_squared_error",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "MeanSquaredLogarithmicError": {
            "log_name": "mean_squared_logarithmic_error",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "Poisson": {
            "log_name": "poisson",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "SparseCategoricalCrossentropy": {
            "log_name": "sparse_categorical_crossentropy",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
        "SquaredHinge": {
            "log_name": "squared_hinge",
            "mode": "min",
            "module": "tensorflow.keras.losses"
        },
    },
    "metric": {
        "AUC": {
            "log_name": "auc",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "Accuracy": {
            "log_name": "accuracy",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "BinaryAccuracy": {
            "log_name": "binary_accuracy",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "BinaryCrossentropy": {
            "log_name": "binary_crossentropy",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "CategoricalAccuracy": {
            "log_name": "categorical_accuracy",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "CategoricalCrossentropy": {
            "log_name": "categorical_crossentropy",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "CategoricalHinge": {
            "log_name": "categorical_hinge",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "CosineSimilarity": {
            "log_name": "cosine_similarity",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },  # min if loss, max if metric
        "DiceCoef": {
            "log_name": "dice_coef",
            "mode": "max",
            "module": "terra_ai.training.customlosses"
        },
        "FalseNegatives": {
            "log_name": "false_negatives",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "FalsePositives": {
            "log_name": "false_positives",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "Hinge": {
            "log_name": "hinge",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "KLDivergence": {
            "log_name": "kullback_leibler_divergence",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "LogCoshError": {
            "log_name": "logcosh",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "MeanAbsoluteError": {
            "log_name": "mean_absolute_error",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "MeanAbsolutePercentageError": {
            "log_name": "mean_absolute_percentage_error",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "MeanIoU": {
            "log_name": "mean_io_u",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "MeanSquaredError": {
            "log_name": "mean_squared_error",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "MeanSquaredLogarithmicError": {
            "log_name": "mean_squared_logarithmic_error",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "Poisson": {
            "log_name": "poisson",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "Precision": {
            "log_name": "precision",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "Recall": {
            "log_name": "recall",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "RootMeanSquaredError": {
            "log_name": "root_mean_squared_error",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "SparseCategoricalAccuracy": {
            "log_name": "sparse_categorical_accuracy",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "SparseCategoricalCrossentropy": {
            "log_name": "sparse_categorical_crossentropy",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "SparseTopKCategoricalAccuracy": {
            "log_name": "sparse_top_k_categorical_accuracy",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "SquaredHinge": {
            "log_name": "squared_hinge",
            "mode": "min",
            "module": "tensorflow.keras.metrics"
        },
        "TopKCategoricalAccuracy": {
            "log_name": "top_k_categorical_accuracy",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "TrueNegatives": {
            "log_name": "true_negatives",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
        "TruePositives": {
            "log_name": "true_positives",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },
    }
}


class InteractiveCallback:
    """Callback for interactive requests"""

    def __init__(self):
        self.losses = None
        self.metrics = None
        self.loss_obj = None
        self.metrics_obj = None
        self.options: PrepareDataset = None
        self.class_colors = []
        self.dataset_path = None
        self.x_val = None
        self.inverse_x_val = None
        self.y_true = {}
        self.inverse_y_true = {}
        self.y_pred = {}
        self.inverse_y_pred = {}
        self.current_epoch = None

        # overfitting params
        self.log_gap = 5
        self.progress_threashold = 3

        self.current_logs = {}
        self.log_history = {}
        self.progress_table = {}
        self.dataset_balance = None
        self.class_idx = None
        self.class_graphics = {}

        self.show_examples = 10
        self.ex_type_choice = 'seed'
        self.seed_idx = None
        self.example_idx = []
        self.intermediate_result = {}
        self.statistic_result = {}
        self.train_progress = {}
        self.progress_name = "training"
        self.preset_path = ""

        self.urgent_predict = False
        self.deploy_presets_data = None

        self.train_states = {
            "status": "no_train",  # training, trained, stopped, addtrain
            "buttons": {
                "train": {
                    "title": "Обучить",  # Возобновить, Дообучить
                    "visible": True
                },
                "stop": {
                    "title": "Остановить",
                    "visible": False
                },
                "clear": {
                    "title": "Сбросить",
                    "visible": False
                },
                "save": {
                    "title": "Сохранить",
                    "visible": False
                }
            }
        }

        self.interactive_config: InteractiveData = InteractiveData(**{})
        pass

    def set_attributes(self, dataset: PrepareDataset,
                       metrics: dict,
                       losses: dict,
                       dataset_path: str,
                       training_path: str,
                       initial_config: InteractiveData):

        self.preset_path = os.path.join(training_path, "presets")
        if not os.path.exists(self.preset_path):
            os.mkdir(self.preset_path)
        self.losses = losses
        self.metrics = self._reformat_metrics(metrics)
        self.loss_obj = self._prepare_loss_obj(losses)
        self.metrics_obj = self._prepare_metric_obj(metrics)
        self.interactive_config = initial_config

        # self._prepare_dataset_config(dataset, dataset_path)
        self.options = dataset
        self.dataset_path = dataset_path
        self._get_classes_colors()
        self.x_val, self.inverse_x_val = self._prepare_x_val(dataset)
        self.y_true, self.inverse_y_true = self._prepare_y_true(dataset)
        self._class_metric_list()

        if not self.log_history:
            self._prepare_null_log_history_template()
        self.dataset_balance = self._prepare_dataset_balance()
        self.class_idx = self._prepare_class_idx()
        self.seed_idx = self._prepare_seed()
        # self.example_idx = self._prepare_example_idx_to_show()

    def set_status(self, status):
        self.train_states["status"] = status
        if status in ["training", "addtrain"]:
            self.train_states["buttons"]["train"]["title"] = "Возобновить"
            self.train_states["buttons"]["train"]["visible"] = False
            self.train_states["buttons"]["stop"]["visible"] = True
            self.train_states["buttons"]["clear"]["visible"] = False
            self.train_states["buttons"]["save"]["visible"] = False
        elif status == "trained":
            self.train_states["buttons"]["train"]["title"] = "Дообучить"
            self.train_states["buttons"]["train"]["visible"] = True
            self.train_states["buttons"]["stop"]["visible"] = False
            self.train_states["buttons"]["clear"]["visible"] = True
            self.train_states["buttons"]["save"]["visible"] = True
        elif status == "stopped":
            self.train_states["buttons"]["train"]["title"] = "Возобновить"
            self.train_states["buttons"]["train"]["visible"] = True
            self.train_states["buttons"]["stop"]["visible"] = False
            self.train_states["buttons"]["clear"]["visible"] = True
            self.train_states["buttons"]["save"]["visible"] = True
        else:
            self.clear_history()
            self.train_states["buttons"]["train"]["title"] = "Обучить"
            self.train_states["buttons"]["train"]["visible"] = True
            self.train_states["buttons"]["stop"]["visible"] = False
            self.train_states["buttons"]["clear"]["visible"] = False
            self.train_states["buttons"]["save"]["visible"] = False

    def clear_history(self):
        self.log_history = {}
        self.current_logs = {}
        self.progress_table = {}
        self.intermediate_result = {}
        self.statistic_result = {}
        self.train_progress = {}
        self.deploy_presets_data = None

    def get_states(self):
        return self.train_states

    def get_presets(self):
        return self.deploy_presets_data

    def update_train_progress(self, data: dict):
        self.train_progress = data

    def update_state(self, y_pred, fit_logs=None, current_epoch_time=None, on_epoch_end_flag=False) -> dict:
        if self.log_history:
            if y_pred is not None:
                self._reformat_y_pred(y_pred)
                if self.interactive_config.intermediate_result.show_results:
                    self.example_idx = self._prepare_example_idx_to_show()
                if on_epoch_end_flag:
                    self.current_epoch = fit_logs.get('epoch')
                    self.current_logs = self._reformat_fit_logs(fit_logs)
                    self._update_log_history()
                    self._update_progress_table(current_epoch_time)
                    if self.interactive_config.intermediate_result.autoupdate:
                        self.intermediate_result = self._get_intermediate_result_request()
                    if self.interactive_config.statistic_data.output_id \
                            and self.interactive_config.statistic_data.autoupdate:
                        self.statistic_result = self._get_statistic_data_request()
                else:
                    self.intermediate_result = self._get_intermediate_result_request()
                    if self.interactive_config.statistic_data.output_id:
                        self.statistic_result = self._get_statistic_data_request()
                self.urgent_predict = False

            return {
                "class_graphics": self.class_graphics,
                'loss_graphs': self._get_loss_graph_data_request(),
                'metric_graphs': self._get_metric_graph_data_request(),
                'intermediate_result': self.intermediate_result,
                'progress_table': self.progress_table,
                'statistic_data': self.statistic_result,
                'data_balance': self._get_balance_data_request(),
            }
        else:
            return {}

    def get_train_results(self, config: InteractiveData) -> Union[dict, None]:
        """Return dict with data for current interactive request"""
        self.interactive_config = config if config else self.interactive_config
        if self.log_history and self.log_history.get("epochs", {}):
            if self.interactive_config.intermediate_result.show_results:
                self.example_idx = self._prepare_example_idx_to_show()
            if config.intermediate_result.show_results or config.statistic_data.output_id:
                self.urgent_predict = True
                self.intermediate_result = self._get_intermediate_result_request()
                if self.interactive_config.statistic_data.output_id:
                    self.statistic_result = self._get_statistic_data_request()

            self.train_progress['train_data'] = {
                "class_graphics": self.class_graphics,
                'loss_graphs': self._get_loss_graph_data_request(),
                'metric_graphs': self._get_metric_graph_data_request(),
                'intermediate_result': self.intermediate_result,
                'progress_table': self.progress_table,
                'statistic_data': self.statistic_result,
                'data_balance': self._get_balance_data_request(),
            }
            progress.pool(
                self.progress_name,
                data=self.train_progress,
                finished=False,
            )
            return self.train_progress

    # Методы для set_attributes()
    @staticmethod
    def _reformat_metrics(metrics: dict) -> dict:
        output = {}
        for out, out_metrics in metrics.items():
            output[out] = []
            for metric in out_metrics:
                metric_name = metric.name
                if re.search(r'_\d+$', metric_name):
                    end = len(f"_{metric_name.split('_')[-1]}")
                    metric_name = metric_name[:-end]
                output[out].append(camelize(metric_name))
        return output

    @staticmethod
    def _prepare_loss_obj(losses: dict) -> dict:
        loss_obj = {}
        for out in losses.keys():
            loss_obj[out] = getattr(
                importlib.import_module(loss_metric_config.get("loss").get(losses.get(out)).get("module")),
                losses.get(out)
            )
        return loss_obj

    @staticmethod
    def _prepare_metric_obj(metrics: dict) -> dict:
        metrics_obj = {}
        for out in metrics.keys():
            metrics_obj[out] = {}
            for metric in metrics.get(out):
                metric_name = metric.name
                if re.search(r'_\d+$', metric_name):
                    end = len(f"_{metric_name.split('_')[-1]}")
                    metric_name = metric_name[:-end]
                metrics_obj[out][camelize(metric_name)] = metric
        return metrics_obj

    def _get_classes_colors(self):
        colors = []
        for out in self.options.data.outputs.keys():
            task = self.options.data.outputs.get(out).task
            classes_colors = self.options.data.outputs.get(out).classes_colors
            if task == LayerOutputTypeChoice.TextSegmentation and classes_colors:
                self.class_colors = [color.as_rgb_tuple() for color in classes_colors]
            elif task == LayerOutputTypeChoice.TextSegmentation and not classes_colors:
                for _ in self.options.data.outputs.get(out).classes_names:
                    colors.append(tuple(np.random.randint(256, size=3).astype('int').tolist()))
                self.class_colors = colors
            elif task == LayerOutputTypeChoice.Segmentation:
                self.class_colors = [color.as_rgb_tuple() for color in classes_colors]
            else:
                pass

    @staticmethod
    def _prepare_x_val(dataset: PrepareDataset):
        x_val = None
        inverse_x_val = None
        if dataset.data.group == DatasetGroupChoice.keras:
            x_val = dataset.X.get("val")
        dataframe = False
        for inp in dataset.data.inputs.keys():
            if dataset.data.inputs.get(inp).task == LayerInputTypeChoice.Dataframe:
                dataframe = True
                break
        ts = False
        for out in dataset.data.outputs.keys():
            if dataset.data.outputs.get(out).task == LayerOutputTypeChoice.Timeseries or \
                    dataset.data.outputs.get(out).task == LayerOutputTypeChoice.Timeseries_trend:
                ts = True
                break
        if dataframe and not dataset.data.use_generator:
            x_val = dataset.X.get("val")

        elif dataframe and dataset.data.use_generator:
            x_val = {}
            for inp in dataset.dataset['val'].keys():
                x_val[inp] = []
                for x_val_, _ in dataset.dataset['val'].batch(1):
                    x_val[inp].extend(x_val_.get(f'{inp}').numpy())
                x_val[inp] = np.array(x_val[inp])
        else:
            pass

        if ts:
            inverse_x_val = {}
            for input in x_val.keys():
                preprocess_dict = dataset.preprocessing.preprocessing.get(int(input))
                inverse_x = np.zeros_like(x_val.get(input)[:, 0:1, :])
                for i, column in enumerate(preprocess_dict.keys()):
                    if type(preprocess_dict.get(column)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                        _options = {
                            int(input): {
                                column: x_val.get(input)[:, i:i + 1, :]
                            }
                        }
                        inverse_col = dataset.preprocessing.inverse_data(_options).get(int(input)).get(column)
                    else:
                        inverse_col = x_val.get(input)[:, i:i + 1, :]
                    inverse_x = np.concatenate([inverse_x, inverse_col], axis=1)
                inverse_x_val[input] = inverse_x[:, 1:, :]
        return x_val, inverse_x_val

    @staticmethod
    def _prepare_y_true(dataset: PrepareDataset):
        y_true = {
            "train": {},
            "val": {}
        }
        inverse_y_true = {
            "train": {},
            "val": {}
        }
        for data_type in y_true.keys():
            for out in dataset.data.outputs.keys():
                task = dataset.data.outputs.get(out).task
                if not dataset.data.use_generator:
                    y_true[data_type][f"{out}"] = dataset.Y.get(data_type).get(f"{out}")
                else:
                    y_true[data_type][f"{out}"] = []
                    for _, y_val in dataset.dataset[data_type].batch(1):
                        y_true[data_type][f"{out}"].extend(y_val.get(f'{out}').numpy())
                    y_true[data_type][f"{out}"] = np.array(y_true[data_type][f"{out}"])

                if task == LayerOutputTypeChoice.Regression or task == LayerOutputTypeChoice.Dataframe:
                    preprocess_dict = dataset.preprocessing.preprocessing.get(out)
                    inverse_y = np.zeros_like(y_true.get(data_type).get(f"{out}")[:, 0:1])
                    for i, column in enumerate(preprocess_dict.keys()):
                        if type(preprocess_dict.get(column)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                            _options = {int(out): {column: y_true.get(data_type).get(f"{out}")[:, i:i + 1]}}
                            inverse_col = dataset.preprocessing.inverse_data(_options).get(out).get(column)
                        else:
                            inverse_col = y_true.get(data_type).get(f"{out}")[:, i:i + 1]
                        inverse_y = np.concatenate([inverse_y, inverse_col], axis=1)
                    inverse_y_true[data_type][f"{out}"] = inverse_y[:, 1:]

                if task == LayerOutputTypeChoice.Timeseries:
                    preprocess_dict = dataset.preprocessing.preprocessing.get(int(out))
                    inverse_y = np.zeros_like(y_true.get(data_type).get(f"{out}")[:, 0:1, :])
                    for i, column in enumerate(preprocess_dict.keys()):
                        if type(preprocess_dict.get(column)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                            _options = {int(out): {column: y_true.get(data_type).get(f"{out}")[:, i:i + 1, :]}}
                            inverse_col = dataset.preprocessing.inverse_data(_options).get(int(out)).get(column)
                        else:
                            inverse_col = y_true.get(data_type).get(f"{out}")[:, i:i + 1, :]
                        inverse_y = np.concatenate([inverse_y, inverse_col], axis=1)
                    inverse_y_true[data_type][f"{out}"] = inverse_y[:, 1:, :]
        return y_true, inverse_y_true

    def _class_metric_list(self):
        self.class_graphics = {}
        for out in self.losses.keys():
            out_task = self.options.data.outputs.get(int(out)).task
            if out_task == LayerOutputTypeChoice.Classification or \
                    out_task == LayerOutputTypeChoice.Segmentation or \
                    out_task == LayerOutputTypeChoice.TextSegmentation or \
                    out_task == LayerOutputTypeChoice.Timeseries_trend:
                self.class_graphics[out] = True
            else:
                self.class_graphics[out] = False

    def _prepare_null_log_history_template(self):
        self.log_history["epochs"] = []
        for out in self.losses.keys():
            task = self.options.data.outputs.get(int(out)).task
            self.log_history[out] = {
                "loss": {},
                "metrics": {},
                "progress_state": {
                    "loss": {},
                    "metrics": {}
                }
            }
            if self.metrics.get(out) and isinstance(self.metrics.get(out), str):
                self.metrics[out] = [self.metrics.get(out)]

            self.log_history[out]["loss"][self.losses.get(out)] = {"train": [], "val": []}
            self.log_history[out]["progress_state"]["loss"][self.losses.get(out)] = {
                "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []
            }
            for metric in self.metrics.get(out):
                self.log_history[out]["metrics"][f"{metric}"] = {"train": [], "val": []}
                self.log_history[out]["progress_state"]["metrics"][f"{metric}"] = {
                    "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []
                }

            if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.Segmentation or \
                    task == LayerOutputTypeChoice.TextSegmentation or task == LayerOutputTypeChoice.Timeseries_trend:
                self.log_history[out]["class_loss"] = {}
                self.log_history[out]["class_metrics"] = {}
                for class_name in self.options.data.outputs.get(int(out)).classes_names:
                    self.log_history[out]["class_metrics"][f"{class_name}"] = {}
                    self.log_history[out]["class_loss"][f"{class_name}"] = {self.losses.get(out): []}
                    for metric in self.metrics.get(out):
                        self.log_history[out]["class_metrics"][f"{class_name}"][f"{metric}"] = []

    def _prepare_dataset_balance(self) -> dict:
        dataset_balance = {}
        for out in self.options.data.outputs.keys():
            task = self.options.data.outputs.get(out).task
            encoding = self.options.data.outputs.get(out).encoding
            dataset_balance[f"{out}"] = {}
            if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.Timeseries_trend:
                for data_type in self.y_true.keys():
                    dataset_balance[f"{out}"][data_type] = class_counter(
                        self.y_true.get(data_type).get(f"{out}"),
                        self.options.data.outputs.get(out).classes_names,
                        encoding == 'ohe'
                    )

            if task == LayerOutputTypeChoice.Segmentation and encoding == LayerEncodingChoice.ohe:
                for data_type in self.y_true.keys():
                    dataset_balance[f"{out}"][data_type] = {
                        "presence_balance": {},
                        "percent_balance": {}
                    }
                    classes_names = self.options.data.outputs.get(out).classes_names
                    classes = np.arange(self.options.data.outputs.get(out).num_classes)
                    class_percent = {}
                    class_count = {}
                    for cl in classes:
                        class_percent[classes_names[cl]] = np.round(
                            np.sum(
                                self.y_true.get(data_type).get(f"{out}")[:, :, :, cl]) * 100
                            / np.prod(self.y_true.get(data_type).get(f"{out}")[:, :, :, 0].shape)
                        ).astype("float").tolist()
                        class_count[classes_names[cl]] = 0

                    for img_array in np.argmax(self.y_true.get(data_type).get(f"{out}"), axis=-1):
                        for cl in classes:
                            if cl in img_array:
                                class_count[classes_names[cl]] += 1
                    dataset_balance[f"{out}"][data_type]["presence_balance"] = class_count
                    dataset_balance[f"{out}"][data_type]["square_balance"] = class_percent

            if task == LayerOutputTypeChoice.TextSegmentation and encoding == LayerEncodingChoice.ohe or \
                    encoding == LayerEncodingChoice.multi:
                for data_type in self.y_true.keys():
                    dataset_balance[f"{out}"][data_type] = {
                        "presence_balance": {},
                        "percent_balance": {}
                    }
                    classes_names = self.options.data.outputs.get(out).classes_names
                    classes = np.arange(self.options.data.outputs.get(out).num_classes)
                    class_count = {}
                    class_percent = {}
                    for cl in classes:
                        class_count[classes_names[cl]] = \
                            np.sum(self.y_true.get(data_type).get(f"{out}")[:, :, cl]).item()
                        class_percent[self.options.data.outputs.get(out).classes_names[cl]] = np.round(
                            np.sum(self.y_true.get(data_type).get(f"{out}")[:, :, cl]) * 100
                            / np.prod(self.y_true.get(data_type).get(f"{out}")[:, :, cl].shape)).item()
                    dataset_balance[f"{out}"][data_type]["presence_balance"] = class_count
                    dataset_balance[f"{out}"][data_type]["percent_balance"] = class_percent

            if task == LayerOutputTypeChoice.Timeseries:
                for data_type in self.y_true.keys():
                    dataset_balance[f"{out}"][data_type] = {}
                    for output_channel in self.options.data.columns.get(out).keys():
                        dataset_balance[f"{out}"][data_type][output_channel] = {
                            'graphic': {},
                            'dense_histogram': {}
                        }
                        dataset_balance[f"{out}"][data_type][output_channel]['graphic'] = {
                            "type": "graphic",
                            "x": np.array(self.options.dataframe.get(data_type).index).astype('float').tolist(),
                            "y": np.array(self.options.dataframe.get(data_type)[output_channel]).astype(
                                'float').tolist()
                        }
                        x, y = self._get_distribution_histogram(
                            list(self.options.dataframe.get(data_type)[output_channel]),
                            bins=25,
                            categorical=False
                        )
                        dataset_balance[f"{out}"][data_type][output_channel]['dense_histogram'] = {
                            "type": "histogram",
                            "x": x,
                            "y": y
                        }

            if task == LayerOutputTypeChoice.Regression:
                for data_type in self.y_true.keys():
                    dataset_balance[f"{out}"][data_type] = {
                        'histogram': [],
                        'correlation': {}
                    }
                    for column in list(self.options.dataframe.get(data_type).columns):
                        column_id = int(column.split("_")[0])
                        column_data = list(self.options.dataframe.get(data_type)[column])
                        column_task = self.options.data.columns.get(column_id).get(column).task
                        if column_task == LayerInputTypeChoice.Text:
                            continue
                        elif column_task == LayerInputTypeChoice.Classification:
                            x, y = self._get_distribution_histogram(column_data, bins=25, categorical=True)
                            hist_type = "histogram"
                        else:
                            x, y = self._get_distribution_histogram(column_data, bins=25, categorical=False)
                            hist_type = "bar"
                        dataset_balance[f"{out}"][data_type]['histogram'].append(
                            {
                                "name": column.split("_", 1)[-1],
                                "type": hist_type,
                                "x": x,
                                "y": y
                            }
                        )
                    labels, matrix = self._get_correlation_matrix(
                        pd.DataFrame(self.options.dataframe.get(data_type))
                    )
                    dataset_balance[f"{out}"][data_type]['correlation'] = {
                        "labels": labels,
                        "matrix": matrix
                    }
        return dataset_balance

    def _prepare_class_idx(self) -> dict:
        class_idx = {}
        for data_type in self.y_true.keys():
            class_idx[data_type] = {}
            for out in self.y_true.get(data_type).keys():
                class_idx[data_type][out] = {}
                task = self.options.data.outputs.get(int(out)).task
                if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.Timeseries_trend:
                    ohe = self.options.data.outputs.get(int(out)).encoding == LayerEncodingChoice.ohe
                    for name in self.options.data.outputs.get(int(out)).classes_names:
                        class_idx[data_type][out][name] = []
                    y_true = np.argmax(self.y_true.get(data_type).get(out), axis=-1) if ohe \
                        else np.squeeze(self.y_true.get(data_type).get(out))
                    for idx in range(len(y_true)):
                        class_idx[data_type][out][
                            self.options.data.outputs.get(int(out)).classes_names[y_true[idx]]
                        ].append(idx)
        return class_idx

    def _prepare_seed(self):
        if self.options.data.group == DatasetGroupChoice.keras or self.x_val:
            data_lenth = np.arange(len(self.y_true.get("val").get(list(self.y_true.get("val").keys())[0])))
        else:
            data_lenth = np.arange(len(self.options.dataframe.get("val")))
        np.random.shuffle(data_lenth)
        return data_lenth

    # Методы для update_state()
    def _reformat_fit_logs(self, logs) -> dict:
        interactive_log = {}
        update_logs = {}
        for log, val in logs.items():
            if re.search(r"_\d+$", log):
                end = len(f"_{log.split('_')[-1]}")
                log = log[:-end]
            update_logs[re.sub("__", "_", decamelize(log))] = val

        for out in self.metrics.keys():
            interactive_log[out] = {}
            if len(self.metrics.keys()) == 1:
                train_loss = update_logs.get('loss')
                val_loss = update_logs.get('val_loss')
            else:
                train_loss = update_logs.get(f'{out}_loss')
                val_loss = update_logs.get(f'val_{out}_loss')
            interactive_log[out]['loss'] = {
                self.losses.get(out): {
                    'train': float(train_loss) if not math.isnan(float(train_loss)) else None,
                    'val': float(val_loss) if not math.isnan(float(val_loss)) else None,
                }
            }

            interactive_log[out]['metrics'] = {}
            for metric_name in self.metrics.get(out):
                interactive_log[out]['metrics'][metric_name] = {}
                if len(self.metrics.keys()) == 1:
                    train_metric = update_logs.get(loss_metric_config.get('metric').get(metric_name).get('log_name'))
                    val_metric = update_logs.get(
                        f"val_{loss_metric_config.get('metric').get(metric_name).get('log_name')}")
                else:
                    train_metric = update_logs.get(
                        f"{out}_{loss_metric_config.get('metric').get(metric_name).get('log_name')}")
                    val_metric = update_logs.get(
                        f"val_{out}_{loss_metric_config.get('metric').get(metric_name).get('log_name')}")
                interactive_log[out]['metrics'][metric_name] = {
                    'train': float(train_metric) if not math.isnan(float(train_metric)) else None,
                    'val': float(val_metric) if not math.isnan(float(val_metric)) else None
                }
        return interactive_log

    def _reformat_y_pred(self, y_pred):
        self.y_pred = {}
        self.inverse_y_pred = {}
        for idx, out in enumerate(self.y_true.get('val').keys()):
            task = self.options.data.outputs.get(int(out)).task
            if len(self.y_true.get('val').keys()) == 1:
                self.y_pred[out] = y_pred
            else:
                self.y_pred[out] = y_pred[idx]

            if task == LayerOutputTypeChoice.Regression or task == LayerOutputTypeChoice.Dataframe:
                preprocess_dict = self.options.preprocessing.preprocessing.get(int(out))
                inverse_y = np.zeros_like(self.y_pred.get(out)[:, 0:1])
                for i, column in enumerate(preprocess_dict.keys()):
                    if type(preprocess_dict.get(column)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                        _options = {int(out): {column: self.y_pred.get(out)[:, i:i + 1]}}
                        inverse_col = self.options.preprocessing.inverse_data(_options).get(int(out)).get(column)
                    else:
                        inverse_col = self.y_pred.get(out)[:, i:i + 1]
                    inverse_y = np.concatenate([inverse_y, inverse_col], axis=1)
                self.inverse_y_pred[out] = inverse_y[:, 1:]

            if task == LayerOutputTypeChoice.Regression.Timeseries:
                preprocess_dict = self.options.preprocessing.preprocessing.get(int(out))
                inverse_y = np.zeros_like(self.y_pred.get(out)[:, 0:1, :])
                for i, column in enumerate(preprocess_dict.keys()):
                    if type(preprocess_dict.get(column)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                        _options = {int(out): {column: self.y_pred.get(out)[:, i:i + 1, :]}}
                        inverse_col = self.options.preprocessing.inverse_data(_options).get(int(out)).get(column)
                    else:
                        inverse_col = self.y_pred.get(out)[:, i:i + 1, :]
                    inverse_y = np.concatenate([inverse_y, inverse_col], axis=1)
                self.inverse_y_pred[out] = inverse_y[:, 1:, :]

    def _prepare_example_idx_to_show(self) -> dict:
        example_idx = {}
        out = f"{self.interactive_config.intermediate_result.main_output}"
        ohe = self.options.data.outputs.get(int(out)).encoding == LayerEncodingChoice.ohe
        count = self.interactive_config.intermediate_result.num_examples
        choice_type = self.interactive_config.intermediate_result.example_choice_type
        task = self.options.data.outputs.get(int(out)).task

        if choice_type == ExampleChoiceTypeChoice.best or choice_type == ExampleChoiceTypeChoice.worst:
            if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.Timeseries_trend:
                y_true = self.y_true.get("val").get(out)
                y_pred = self.y_pred.get(out)
                if y_pred.shape[-1] == y_true.shape[-1] and ohe and y_true.shape[-1] > 1:
                    classes = np.argmax(y_true, axis=-1)
                elif len(y_true.shape) == 1 and not ohe and y_pred.shape[-1] > 1:
                    classes = copy.deepcopy(y_true)
                elif len(y_true.shape) == 1 and not ohe and y_pred.shape[-1] == 1:
                    classes = copy.deepcopy(y_true)
                else:
                    classes = copy.deepcopy(y_true)
                probs = np.array([pred[classes[i]] for i, pred in enumerate(y_pred)])
                sorted_args = np.argsort(probs)
                if choice_type == ExampleChoiceTypeChoice.best:
                    example_idx = sorted_args[::-1][:count]
                if choice_type == ExampleChoiceTypeChoice.worst:
                    example_idx = sorted_args[:count]

            elif task == LayerOutputTypeChoice.Segmentation or task == LayerOutputTypeChoice.TextSegmentation:
                y_true = self.y_true.get("val").get(out)
                y_pred = to_categorical(
                    np.argmax(self.y_pred.get(out), axis=-1),
                    num_classes=self.options.data.outputs.get(int(out)).num_classes
                )
                dice_val = self._dice_coef(y_true, y_pred, batch_mode=True)
                dice_dict = dict(zip(np.arange(0, len(dice_val)), dice_val))
                if choice_type == ExampleChoiceTypeChoice.best:
                    example_idx, _ = sort_dict(dice_dict, mode="descending")
                    example_idx = example_idx[:count]
                if choice_type == ExampleChoiceTypeChoice.worst:
                    example_idx, _ = sort_dict(dice_dict, mode="ascending")
                    example_idx = example_idx[:count]

            elif task == LayerOutputTypeChoice.Timeseries or task == LayerOutputTypeChoice.Regression:
                delta = np.abs(
                    (self.inverse_y_true.get('val').get(out) - self.inverse_y_pred.get(out)) * 100 /
                    self.inverse_y_true.get('val').get(out)
                )
                while len(delta.shape) != 1:
                    delta = np.mean(delta, axis=-1)
                delta_dict = dict(zip(np.arange(0, len(delta)), delta))
                if choice_type == ExampleChoiceTypeChoice.best:
                    example_idx, _ = sort_dict(delta_dict, mode="ascending")
                    example_idx = example_idx[:count]
                if choice_type == ExampleChoiceTypeChoice.worst:
                    example_idx, _ = sort_dict(delta_dict, mode="descending")
                    example_idx = example_idx[:count]
            else:
                pass

        elif choice_type == ExampleChoiceTypeChoice.seed:
            example_idx = self.seed_idx[:self.interactive_config.intermediate_result.num_examples]

        elif choice_type == ExampleChoiceTypeChoice.random:
            example_idx = np.random.randint(
                0, len(self.y_true.get("val").get(list(self.y_true.get('val').keys())[0])),
                self.interactive_config.intermediate_result.num_examples
            )
        else:
            pass
        return example_idx

    def _update_log_history(self):
        data_idx = None
        if self.log_history:
            if self.current_epoch in self.log_history['epochs']:
                data_idx = self.log_history['epochs'].index(self.current_epoch)
            else:
                self.log_history['epochs'].append(self.current_epoch)
            for out in self.options.data.outputs.keys():
                out_task = self.options.data.outputs.get(out).task
                classes_names = self.options.data.outputs.get(out).classes_names
                for loss_name in self.log_history.get(f"{out}").get('loss').keys():
                    for data_type in ['train', 'val']:
                        # fill losses
                        if data_idx or data_idx == 0:
                            self.log_history[f"{out}"]['loss'][loss_name][data_type][data_idx] = \
                                self.current_logs.get(f"{out}").get('loss').get(loss_name).get(data_type)
                        else:
                            self.log_history[f"{out}"]['loss'][loss_name][data_type].append(
                                self.current_logs.get(f"{out}").get('loss').get(loss_name).get(data_type)
                            )
                    # fill loss progress state
                    if data_idx or data_idx == 0:
                        self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['mean_log_history'][data_idx] = \
                            self._get_mean_log(self.log_history.get(f"{out}").get('loss').get(loss_name).get('val'))
                    else:
                        self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['mean_log_history'].append(
                            self._get_mean_log(self.log_history.get(f"{out}").get('loss').get(loss_name).get('val'))
                        )
                    # get progress state data
                    loss_underfitting = self._evaluate_underfitting(
                        loss_name,
                        self.log_history[f"{out}"]['loss'][loss_name]['train'][-1],
                        self.log_history[f"{out}"]['loss'][loss_name]['val'][-1],
                        metric_type='loss'
                    )
                    loss_overfitting = self._evaluate_overfitting(
                        loss_name,
                        self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['mean_log_history'],
                        metric_type='loss'
                    )
                    if loss_underfitting or loss_overfitting:
                        normal_state = False
                    else:
                        normal_state = True

                    if data_idx or data_idx == 0:
                        self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['underfitting'][data_idx] = \
                            loss_underfitting
                        self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['overfitting'][data_idx] = \
                            loss_overfitting
                        self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['normal_state'][data_idx] = \
                            normal_state
                    else:
                        self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['underfitting'].append(
                            loss_underfitting)
                        self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['overfitting'].append(
                            loss_overfitting)
                        self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['normal_state'].append(
                            normal_state)

                    if out_task == LayerOutputTypeChoice.Classification or \
                            out_task == LayerOutputTypeChoice.Segmentation or \
                            out_task == LayerOutputTypeChoice.TextSegmentation or \
                            out_task == LayerOutputTypeChoice.Timeseries_trend:
                        for cls in self.log_history.get(f"{out}").get('class_loss').keys():
                            class_loss = 0.
                            if out_task == LayerOutputTypeChoice.Classification or \
                                    out_task == LayerOutputTypeChoice.Timeseries_trend:
                                class_loss = self._get_loss_calculation(
                                    loss_obj=self.loss_obj.get(f"{out}"),
                                    out=f"{out}",
                                    y_true=self.y_true.get('val').get(f"{out}")[
                                        self.class_idx.get('val').get(f"{out}").get(cls)],
                                    y_pred=self.y_pred.get(f"{out}")[self.class_idx.get('val').get(f"{out}").get(cls)],
                                )
                            if out_task == LayerOutputTypeChoice.Segmentation:
                                class_idx = classes_names.index(cls)
                                class_loss = self._get_loss_calculation(
                                    loss_obj=self.loss_obj.get(f"{out}"),
                                    out=f"{out}",
                                    y_true=self.y_true.get('val').get(f"{out}")[:, :, :, class_idx],
                                    y_pred=self.y_pred.get(f"{out}")[:, :, :, class_idx],
                                )
                            if out_task == LayerOutputTypeChoice.TextSegmentation:
                                class_idx = classes_names.index(cls)
                                class_loss = self._get_loss_calculation(
                                    loss_obj=self.loss_obj.get(f"{out}"),
                                    out=f"{out}",
                                    y_true=self.y_true.get('val').get(f"{out}")[:, :, class_idx],
                                    y_pred=self.y_pred.get(f"{out}")[:, :, class_idx],
                                )
                            if data_idx or data_idx == 0:
                                self.log_history[f"{out}"]['class_loss'][cls][loss_name][data_idx] = class_loss
                            else:
                                self.log_history[f"{out}"]['class_loss'][cls][loss_name].append(class_loss)

                for metric_name in self.log_history.get(f"{out}").get('metrics').keys():
                    for data_type in ['train', 'val']:
                        # fill metrics
                        if data_idx or data_idx == 0:
                            if self.current_logs:
                                self.log_history[f"{out}"]['metrics'][metric_name][data_type][data_idx] = \
                                    self.current_logs.get(f"{out}").get('metrics').get(metric_name).get(data_type)
                        else:
                            if self.current_logs:
                                self.log_history[f"{out}"]['metrics'][metric_name][data_type].append(
                                    self.current_logs.get(f"{out}").get('metrics').get(metric_name).get(data_type)
                                )

                    if data_idx or data_idx == 0:
                        self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['mean_log_history'][
                            data_idx] = \
                            self._get_mean_log(self.log_history[f"{out}"]['metrics'][metric_name]['val'])
                    else:
                        self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['mean_log_history'].append(
                            self._get_mean_log(self.log_history[f"{out}"]['metrics'][metric_name]['val'])
                        )
                    metric_underfittng = self._evaluate_underfitting(
                        metric_name,
                        self.log_history[f"{out}"]['metrics'][metric_name]['train'][-1],
                        self.log_history[f"{out}"]['metrics'][metric_name]['val'][-1],
                        metric_type='metric'
                    )
                    metric_overfittng = self._evaluate_overfitting(
                        metric_name,
                        self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['mean_log_history'],
                        metric_type='metric'
                    )
                    if metric_underfittng or metric_overfittng:
                        normal_state = False
                    else:
                        normal_state = True

                    if data_idx or data_idx == 0:
                        self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['underfitting'][data_idx] = \
                            metric_underfittng
                        self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['overfitting'][data_idx] = \
                            metric_overfittng
                        self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['normal_state'][data_idx] = \
                            normal_state
                    else:
                        self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['underfitting'].append(
                            metric_underfittng)
                        self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['overfitting'].append(
                            metric_overfittng)
                        self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['normal_state'].append(
                            normal_state)

                    if out_task == LayerOutputTypeChoice.Classification or \
                            out_task == LayerOutputTypeChoice.Segmentation or \
                            out_task == LayerOutputTypeChoice.TextSegmentation or \
                            out_task == LayerOutputTypeChoice.Timeseries_trend:
                        for cls in self.log_history.get(f"{out}").get('class_metrics').keys():
                            class_metric = 0.
                            if out_task == LayerOutputTypeChoice.Classification or \
                                    out_task == LayerOutputTypeChoice.Timeseries_trend:
                                class_metric = self._get_metric_calculation(
                                    metric_name=metric_name,
                                    metric_obj=self.metrics_obj.get(f"{out}").get(metric_name),
                                    out=f"{out}",
                                    y_true=self.y_true.get('val').get(f"{out}")[
                                        self.class_idx.get('val').get(f"{out}").get(cls)],
                                    y_pred=self.y_pred.get(f"{out}")[self.class_idx.get('val').get(f"{out}").get(cls)],
                                )
                            if out_task == LayerOutputTypeChoice.Segmentation:
                                class_idx = classes_names.index(cls)
                                class_metric = self._get_metric_calculation(
                                    metric_name=metric_name,
                                    metric_obj=self.metrics_obj.get(f"{out}").get(metric_name),
                                    out=f"{out}",
                                    y_true=self.y_true.get('val').get(f"{out}")[:, :, :, class_idx],
                                    y_pred=self.y_pred.get(f"{out}")[:, :, :, class_idx],
                                )
                            if out_task == LayerOutputTypeChoice.TextSegmentation:
                                class_idx = classes_names.index(cls)
                                class_metric = self._get_metric_calculation(
                                    metric_name=metric_name,
                                    metric_obj=self.metrics_obj.get(f"{out}").get(metric_name),
                                    out=f"{out}",
                                    y_true=self.y_true.get('val').get(f"{out}")[:, :, class_idx],
                                    y_pred=self.y_pred.get(f"{out}")[:, :, class_idx],
                                )
                            if data_idx or data_idx == 0:
                                self.log_history[f"{out}"]['class_metrics'][cls][metric_name][data_idx] = class_metric
                            else:
                                self.log_history[f"{out}"]['class_metrics'][cls][metric_name].append(class_metric)

    def _update_progress_table(self, epoch_time: float):
        self.progress_table[self.current_epoch] = {
            "time": epoch_time,
            "data": {}
        }
        for out in list(self.log_history.keys())[1:]:
            self.progress_table[self.current_epoch]["data"][f"Выходной слой «{out}»"] = {
                'loss': {},
                'metrics': {}
            }
            self.progress_table[self.current_epoch]["data"][f"Выходной слой «{out}»"]["loss"] = {
                'loss': self.log_history.get(out).get('loss').get(self.losses.get(out)).get('train')[-1],
                'val_loss': self.log_history.get(out).get('loss').get(self.losses.get(out)).get('val')[-1]
            }
            for metric in self.metrics.get(out):
                self.progress_table[self.current_epoch]["data"][f"Выходной слой «{out}»"]["metrics"][metric] = \
                    self.log_history.get(out).get('metrics').get(metric).get('train')[-1]
                self.progress_table[self.current_epoch]["data"][f"Выходной слой «{out}»"]["metrics"][f"val_{metric}"] = \
                    self.log_history.get(out).get('metrics').get(metric).get('val')[-1]

    def _get_loss_calculation(self, loss_obj, out: str, y_true, y_pred):
        encoding = self.options.data.outputs.get(int(out)).encoding
        task = self.options.data.outputs.get(int(out)).task
        num_classes = self.options.data.outputs.get(int(out)).num_classes
        if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.Timeseries_trend:
            # if loss_name == Loss.SparseCategoricalCrossentropy:
            #     return float(loss_obj()(np.argmax(y_true, axis=-1) if ohe else np.squeeze(y_true), y_pred).numpy())
            # else:
            # print(y_true.shape, y_pred.shape)
            loss_value = float(loss_obj()(y_true if encoding == LayerEncodingChoice.ohe
                                          else to_categorical(y_true, num_classes), y_pred).numpy())
        elif task == LayerOutputTypeChoice.Segmentation or \
                (task == LayerOutputTypeChoice.TextSegmentation and encoding == LayerEncodingChoice.ohe):
            # if loss_name == Loss.SparseCategoricalCrossentropy:
            #     return float(loss_obj()(
            #         np.expand_dims(np.argmax(y_true, axis=-1), axis=-1) if ohe else np.squeeze(y_true), y_pred
            #     ).numpy())
            # else:
            loss_value = float(loss_obj()(
                y_true if encoding == LayerEncodingChoice.ohe else to_categorical(y_true, num_classes), y_pred
            ).numpy())
        elif task == LayerOutputTypeChoice.TextSegmentation and encoding == LayerEncodingChoice.multi:
            # if loss_name == Loss.SparseCategoricalCrossentropy:
            #     return 0.
            # else:
            loss_value = float(loss_obj()(y_true, y_pred).numpy())
        elif task == LayerOutputTypeChoice.Regression or task == LayerOutputTypeChoice.Timeseries:
            loss_value = float(loss_obj()(y_true, y_pred).numpy())
        else:
            loss_value = 0.
        return round(loss_value, 6) if not math.isnan(loss_value) else None

    def _get_metric_calculation(self, metric_name, metric_obj, out: str, y_true, y_pred):
        encoding = self.options.data.outputs.get(int(out)).encoding
        task = self.options.data.outputs.get(int(out)).task
        num_classes = self.options.data.outputs.get(int(out)).num_classes
        if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.Timeseries_trend:
            if metric_name == Metric.Accuracy:
                metric_obj.update_state(
                    np.argmax(y_true, axis=-1) if encoding == LayerEncodingChoice.ohe else y_true,
                    np.argmax(y_pred, axis=-1)
                )
            # elif metric_name == Metric.SparseCategoricalAccuracy or \
            #         metric_name == Metric.SparseTopKCategoricalAccuracy or \
            #         metric_name == Metric.SparseCategoricalCrossentropy:
            #     metric_obj.update_state(np.argmax(y_true, axis=-1) if ohe else np.squeeze(y_true), y_pred)
            else:
                metric_obj.update_state(
                    y_true if encoding == LayerEncodingChoice.ohe else to_categorical(y_true, num_classes),
                    y_pred
                )
            metric_value = float(metric_obj.result().numpy())
        elif task == LayerOutputTypeChoice.Segmentation or \
                (task == LayerOutputTypeChoice.TextSegmentation and encoding == LayerEncodingChoice.ohe):
            # if metric_name == Metric.SparseCategoricalAccuracy or \
            #         metric_name == Metric.SparseTopKCategoricalAccuracy or \
            #         metric_name == Metric.SparseCategoricalCrossentropy:
            #     metric_obj.update_state(
            #         np.expand_dims(np.argmax(y_true, axis=-1), axis=-1) if ohe else np.squeeze(y_true), y_pred
            #     )
            # else:
            metric_obj.update_state(
                y_true if encoding == LayerEncodingChoice.ohe else to_categorical(y_true, num_classes),
                y_pred
            )
            metric_value = float(metric_obj.result().numpy())
        elif task == LayerOutputTypeChoice.TextSegmentation and encoding == LayerEncodingChoice.multi:
            # if metric_name == Metric.SparseCategoricalAccuracy or \
            #         metric_name == Metric.SparseTopKCategoricalAccuracy or \
            #         metric_name == Metric.SparseCategoricalCrossentropy:
            #     return 0.
            # else:
            metric_obj.update_state(y_true, y_pred)
            metric_value = float(metric_obj.result().numpy())
        elif task == LayerOutputTypeChoice.Regression or task == LayerOutputTypeChoice.Timeseries:
            metric_obj.update_state(y_true, y_pred)
            metric_value = float(metric_obj.result().numpy())
        else:
            metric_value = 0.
        return round(metric_value, 6) if not math.isnan(metric_value) else None

    def _get_mean_log(self, logs):
        try:
            if len(logs) < self.log_gap:
                return float(np.mean(logs))
            else:
                return float(np.mean(logs[-self.log_gap:]))
        except TypeError:
            return 0.

    @staticmethod
    def _evaluate_overfitting(metric_name: str, mean_log: list, metric_type: str):
        mode = loss_metric_config.get(metric_type).get(metric_name).get("mode")
        if min(mean_log) or max(mean_log):
            if mode == 'min' and mean_log[-1] > min(mean_log) and \
                    (mean_log[-1] - min(mean_log)) * 100 / min(mean_log) > 2:
                return True
            elif mode == 'max' and mean_log[-1] < max(mean_log) and \
                    (max(mean_log) - mean_log[-1]) * 100 / max(mean_log) > 2:
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def _evaluate_underfitting(metric_name: str, train_log: float, val_log: float, metric_type: str):
        mode = loss_metric_config.get(metric_type).get(metric_name).get("mode")
        if train_log:
            if mode == 'min' and val_log < 1 and train_log < 1 and (val_log - train_log) > 0.05:
                return True
            elif mode == 'min' and (val_log >= 1 or train_log >= 1) \
                    and (val_log - train_log) / train_log * 100 > 5:
                return True
            elif mode == 'max' and (train_log - val_log) / train_log * 100 > 3:
                return True
            else:
                return False
        else:
            return False

    # Методы для конечных данных для вывода
    @staticmethod
    def _fill_graph_plot_data(x: list, y: list, label=None):
        return {'label': label, 'x': x, 'y': y}

    @staticmethod
    def _fill_graph_front_structure(_id: int, _type: str, graph_name: str, short_name: str,
                                    x_label: str, y_label: str, plot_data: list,
                                    type_data: str = None, progress_state: str = None):
        return {
            'id': _id,
            'type': _type,
            'type_data': type_data,
            'graph_name': graph_name,
            'short_name': short_name,
            'x_label': x_label,
            'y_label': y_label,
            'plot_data': plot_data,
            'progress_state': progress_state
        }

    @staticmethod
    def _fill_heatmap_front_structure(_id: int, _type: str, graph_name: str, short_name: str,
                                      x_label: str, y_label: str, labels: list, data_array: list,
                                      type_data: str = None, data_percent_array: list = None,
                                      progress_state: str = None):
        return {
            'id': _id,
            'type': _type,
            'type_data': type_data,
            'graph_name': graph_name,
            'short_name': short_name,
            'x_label': x_label,
            'y_label': y_label,
            'labels': labels,
            'data_array': data_array,
            'data_percent_array': data_percent_array,
            'progress_state': progress_state
        }

    @staticmethod
    def _fill_table_front_structure(_id: int, graph_name: str, plot_data: list):
        return {'id': _id, 'type': 'table', 'graph_name': graph_name, 'plot_data': plot_data}

    def _get_loss_graph_data_request(self) -> list:
        data_return = []
        if not self.interactive_config.loss_graphs or not self.log_history.get("epochs"):
            return data_return
        for loss_graph_config in self.interactive_config.loss_graphs:
            if loss_graph_config.show == LossGraphShowChoice.model:
                if sum(self.log_history.get(f"{loss_graph_config.output_idx}").get("progress_state").get(
                        "loss").get(self.losses.get(f"{loss_graph_config.output_idx}")).get(
                    'overfitting')[-self.log_gap:]) >= self.progress_threashold:
                    progress_state = "overfitting"
                elif sum(self.log_history.get(f"{loss_graph_config.output_idx}").get("progress_state").get(
                        "loss").get(self.losses.get(f"{loss_graph_config.output_idx}")).get(
                    'underfitting')[-self.log_gap:]) >= self.progress_threashold:
                    progress_state = "underfitting"
                else:
                    progress_state = "normal"

                train_plot = self._fill_graph_plot_data(
                    x=self.log_history.get("epochs"),
                    y=self.log_history.get(f"{loss_graph_config.output_idx}").get('loss').get(
                        self.losses.get(f"{loss_graph_config.output_idx}")).get('train'),
                    label="Тренировочная выборка"
                )
                val_plot = self._fill_graph_plot_data(
                    x=self.log_history.get("epochs"),
                    y=self.log_history.get(f"{loss_graph_config.output_idx}").get('loss').get(
                        self.losses.get(f"{loss_graph_config.output_idx}")).get("val"),
                    label="Проверочная выборка"
                )
                data_return.append(
                    self._fill_graph_front_structure(
                        _id=loss_graph_config.id,
                        _type='graphic',
                        graph_name=f"Выходной слой «{loss_graph_config.output_idx}» - "
                                   f"График ошибки обучения - Эпоха №{self.log_history.get('epochs')[-1]}",
                        short_name=f"{loss_graph_config.output_idx} - График ошибки обучения",
                        x_label="Эпоха",
                        y_label="Значение",
                        plot_data=[train_plot, val_plot],
                        progress_state=progress_state
                    )
                )
            if loss_graph_config.show == LossGraphShowChoice.classes and \
                    self.class_graphics.get(str(loss_graph_config.output_idx)):
                data_return.append(
                    self._fill_graph_front_structure(
                        _id=loss_graph_config.id,
                        _type='graphic',
                        graph_name=f"Выходной слой «{loss_graph_config.output_idx}» - График ошибки обучения по классам"
                                   f" - Эпоха №{self.log_history.get('epochs')[-1]}",
                        short_name=f"{loss_graph_config.output_idx} - График ошибки обучения по классам",
                        x_label="Эпоха",
                        y_label="Значение",
                        plot_data=[
                            self._fill_graph_plot_data(
                                x=self.log_history.get("epochs"),
                                y=self.log_history.get(f"{loss_graph_config.output_idx}").get('class_loss').get(
                                    class_name).get(self.losses.get(f"{loss_graph_config.output_idx}")),
                                label=f"Класс {class_name}"
                            ) for class_name in
                            self.options.data.outputs.get(loss_graph_config.output_idx).classes_names
                        ],
                    )
                )
        return data_return

    def _get_metric_graph_data_request(self) -> list:
        data_return = []
        if not self.interactive_config.metric_graphs or not self.log_history.get("epochs"):
            return data_return

        for metric_graph_config in self.interactive_config.metric_graphs:
            if metric_graph_config.show == LossGraphShowChoice.model:
                if sum(self.log_history.get(f"{metric_graph_config.output_idx}").get("progress_state").get(
                        "metrics").get(metric_graph_config.show_metric.name).get(
                    'overfitting')[-self.log_gap:]) >= self.progress_threashold:
                    progress_state = 'overfitting'
                elif sum(self.log_history.get(f"{metric_graph_config.output_idx}").get("progress_state").get(
                        "metrics").get(metric_graph_config.show_metric.name).get(
                    'underfitting')[-self.log_gap:]) >= self.progress_threashold:
                    progress_state = 'underfitting'
                else:
                    progress_state = 'normal'

                train_plot = self._fill_graph_plot_data(
                    x=self.log_history.get("epochs"),
                    y=self.log_history.get(f"{metric_graph_config.output_idx}").get('metrics').get(
                        metric_graph_config.show_metric.name).get("train"),
                    label="Тренировочная выборка"
                )
                val_plot = self._fill_graph_plot_data(
                    x=self.log_history.get("epochs"),
                    y=self.log_history.get(f"{metric_graph_config.output_idx}").get('metrics').get(
                        metric_graph_config.show_metric.name).get("val"),
                    label="Проверочная выборка"
                )
                data_return.append(
                    self._fill_graph_front_structure(
                        _id=metric_graph_config.id,
                        _type='graphic',
                        graph_name=f"Выходной слой «{metric_graph_config.output_idx}» - График метрики "
                                   f"{metric_graph_config.show_metric.name} - Эпоха №{self.log_history.get('epochs')[-1]}",
                        short_name=f"{metric_graph_config.output_idx} - {metric_graph_config.show_metric.name}",
                        x_label="Эпоха",
                        y_label="Значение",
                        plot_data=[train_plot, val_plot],
                        progress_state=progress_state
                    )
                )

            if metric_graph_config.show == LossGraphShowChoice.classes and \
                    self.class_graphics.get(str(metric_graph_config.output_idx)):
                data_return.append(
                    self._fill_graph_front_structure(
                        _id=metric_graph_config.id,
                        _type='graphic',
                        graph_name=f"Выходной слой «{metric_graph_config.output_idx}» - График метрики "
                                   f"{metric_graph_config.show_metric.name} по классам - Эпоха №{self.log_history.get('epochs')[-1]}",
                        short_name=f"{metric_graph_config.output_idx} - {metric_graph_config.show_metric.name} по классам",
                        x_label="Эпоха",
                        y_label="Значение",
                        plot_data=[
                            self._fill_graph_plot_data(
                                x=self.log_history.get("epochs"),
                                y=self.log_history.get(f"{metric_graph_config.output_idx}").get('class_loss').get(
                                    class_name).get(self.losses.get(f"{metric_graph_config.output_idx}")),
                                label=f"Класс {class_name}"
                            ) for class_name in
                            self.options.data.outputs.get(metric_graph_config.output_idx).classes_names
                        ],
                    )
                )

        return data_return

    def _get_intermediate_result_request(self) -> dict:
        return_data = {}
        if self.interactive_config.intermediate_result.show_results:
            for idx in range(self.interactive_config.intermediate_result.num_examples):
                return_data[f"{idx + 1}"] = {
                    'initial_data': {},
                    'true_value': {},
                    'predict_value': {},
                    'tags_color': {},
                    'statistic_values': {}
                }
                if not (
                        len(self.options.data.outputs.keys()) == 1 and
                        self.options.data.outputs.get(list(self.options.data.outputs.keys())[0]).task ==
                        LayerOutputTypeChoice.TextSegmentation
                ):
                    for inp in self.options.data.inputs.keys():
                        data, type_choice = CreateArray().postprocess_initial_source(
                            options=self.options,
                            input_id=inp,
                            save_id=idx + 1,
                            example_id=self.example_idx[idx],
                            dataset_path=self.dataset_path,
                            preset_path=self.preset_path,
                            x_array=self.x_val.get(f"{inp}") if self.x_val else None,
                            inverse_x_array=self.inverse_x_val.get(f"{inp}") if self.inverse_x_val else None,
                            return_mode='callback'
                        )
                        random_key = ''.join(random.sample(string.ascii_letters + string.digits, 16))
                        return_data[f"{idx + 1}"]['initial_data'][f"Входной слой «{inp}»"] = {
                            'update': random_key,
                            'type': type_choice,
                            'data': data,
                        }
                for out in self.options.data.outputs.keys():
                    task = self.options.data.outputs.get(out).task

                    if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.Timeseries_trend:
                        data = CreateArray().postprocess_classification(
                            predict_array=self.y_pred.get(f'{out}')[self.example_idx[idx]],
                            true_array=self.y_true.get('val').get(f'{out}')[self.example_idx[idx]],
                            options=self.options.data.outputs.get(out),
                            show_stat=self.interactive_config.intermediate_result.show_statistic,
                            return_mode='callback'
                        )

                    elif task == LayerOutputTypeChoice.Segmentation:
                        data = CreateArray().postprocess_segmentation(
                            predict_array=self.y_pred.get(f'{out}')[self.example_idx[idx]],
                            true_array=self.y_true.get('val').get(f'{out}')[self.example_idx[idx]],
                            options=self.options.data.outputs.get(out),
                            colors=self.class_colors,
                            output_id=out,
                            image_id=idx,
                            save_path=self.preset_path,
                            return_mode='callback',
                            show_stat=self.interactive_config.intermediate_result.show_statistic
                        )

                    elif task == LayerOutputTypeChoice.TextSegmentation:
                        # TODO: пока исходим что для сегментации текста есть только один вход с текстом, если будут сложные модели
                        #  на сегментацию текста на несколько входов то придется искать решения
                        output_col = list(self.options.instructions.get(out).keys())[0]
                        data = CreateArray().postprocess_text_segmentation(
                            pred_array=self.y_pred.get(f'{out}')[self.example_idx[idx]],
                            true_array=self.y_true.get('val').get(f'{out}')[self.example_idx[idx]],
                            options=self.options.data.outputs.get(out),
                            dataframe=self.options.dataframe.get('val'),
                            example_id=self.example_idx[idx],
                            dataset_params=self.options.instructions.get(out).get(output_col),
                            return_mode='callback',
                            class_colors=self.class_colors,
                            show_stat=self.interactive_config.intermediate_result.show_statistic
                        )

                    elif task == LayerOutputTypeChoice.Regression:
                        data = CreateArray().postprocess_regression(
                            column_names=list(self.options.data.columns.get(out).keys()),
                            inverse_y_true=self.inverse_y_true.get('val').get(f"{out}")[self.example_idx[idx]],
                            inverse_y_pred=self.inverse_y_pred.get(f"{out}")[self.example_idx[idx]],
                            show_stat=self.interactive_config.intermediate_result.show_statistic
                        )

                    elif task == LayerOutputTypeChoice.Timeseries:
                        input = list(self.inverse_x_val.keys())[0]
                        data = CreateArray().postprocess_time_series(
                            options=self.options.data,
                            real_x=self.inverse_x_val.get(f"{input}")[self.example_idx[idx]],
                            inverse_y_true=self.inverse_y_true.get("val").get(f"{out}")[self.example_idx[idx]],
                            inverse_y_pred=self.inverse_y_pred.get(f"{out}")[self.example_idx[idx]],
                            output_id=out,
                            depth=self.inverse_y_true.get("val").get(f"{out}")[self.example_idx[idx]].shape[-1],
                            show_stat=self.interactive_config.intermediate_result.show_statistic,
                            templates=[self._fill_graph_plot_data, self._fill_graph_front_structure]
                        )

                    elif task == LayerOutputTypeChoice.Dataframe:
                        data = {
                            "y_true": {},
                            "y_pred": {},
                            "stat": {}
                        }
                        pass

                    elif task == LayerOutputTypeChoice.ObjectDetection:
                        data = {
                            "y_true": {},
                            "y_pred": {},
                            "stat": {}
                        }
                        # image with bb
                        # accuracy, correlation bb for classes
                        pass

                    else:
                        data = {
                            "y_true": {},
                            "y_pred": {},
                            "stat": {}
                        }

                    if data.get('y_true'):
                        return_data[f"{idx + 1}"]['true_value'][f"Выходной слой «{out}»"] = data.get('y_true')
                    return_data[f"{idx + 1}"]['predict_value'][f"Выходной слой «{out}»"] = data.get('y_pred')

                    if self.options.data.outputs.get(out).task == LayerOutputTypeChoice.TextSegmentation:
                        return_data[f"{idx + 1}"]['tags_color'][f"Выходной слой «{out}»"] = data.get('tags_color')
                    else:
                        return_data[f"{idx + 1}"]['tags_color'] = None

                    if data.get('stat'):
                        return_data[f"{idx + 1}"]['statistic_values'][f"Выходной слой «{out}»"] = data.get('stat')
                    else:
                        return_data[f"{idx + 1}"]['statistic_values'] = {}
        return return_data

    def _get_statistic_data_request(self) -> list:
        return_data = []
        _id = 1
        for out in self.interactive_config.statistic_data.output_id:
            task = self.options.data.outputs.get(out).task
            encoding = self.options.data.outputs.get(out).encoding
            if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.Timeseries_trend and \
                    encoding != LayerEncodingChoice.multi:
                cm, cm_percent = self._get_confusion_matrix(
                    np.argmax(self.y_true.get("val").get(f'{out}'), axis=-1) if encoding == LayerEncodingChoice.ohe
                    else self.y_true.get("val").get(f'{out}'),
                    np.argmax(self.y_pred.get(f'{out}'), axis=-1),
                    get_percent=True
                )
                return_data.append(
                    self._fill_heatmap_front_structure(
                        _id=_id,
                        _type="heatmap",
                        graph_name=f"Выходной слой «{out}» - Confusion matrix",
                        short_name=f"{out} - Confusion matrix",
                        x_label="Предсказание",
                        y_label="Истинное значение",
                        labels=self.options.data.outputs.get(out).classes_names,
                        data_array=cm,
                        data_percent_array=cm_percent,
                    )
                )
                _id += 1

            elif task == LayerOutputTypeChoice.Segmentation or \
                    (task == LayerOutputTypeChoice.TextSegmentation and encoding == LayerEncodingChoice.ohe):
                cm, cm_percent = self._get_confusion_matrix(
                    np.argmax(self.y_true.get("val").get(f"{out}"), axis=-1).reshape(
                        np.prod(np.argmax(self.y_true.get("val").get(f"{out}"), axis=-1).shape)).astype('int'),
                    np.argmax(self.y_pred.get(f'{out}'), axis=-1).reshape(
                        np.prod(np.argmax(self.y_pred.get(f'{out}'), axis=-1).shape)).astype('int'),
                    get_percent=True
                )
                return_data.append(
                    self._fill_heatmap_front_structure(
                        _id=_id,
                        _type="heatmap",
                        graph_name=f"Выходной слой «{out}» - Confusion matrix",
                        short_name=f"{out} - Confusion matrix",
                        x_label="Предсказание",
                        y_label="Истинное значение",
                        labels=self.options.data.outputs.get(out).classes_names,
                        data_array=cm,
                        data_percent_array=cm_percent,
                    )
                )
                _id += 1

            elif (task == LayerOutputTypeChoice.TextSegmentation or task == LayerOutputTypeChoice.Classification) \
                    and encoding == LayerEncodingChoice.multi:

                report = self._get_classification_report(
                    y_true=self.y_true.get("val").get(f"{out}").reshape(
                        (np.prod(self.y_true.get("val").get(f"{out}").shape[:-1]),
                         self.y_true.get("val").get(f"{out}").shape[-1])
                    ),
                    y_pred=np.where(self.y_pred.get(f"{out}") >= 0.9, 1, 0).reshape(
                        (np.prod(self.y_pred.get(f"{out}").shape[:-1]), self.y_pred.get(f"{out}").shape[-1])
                    ),
                    labels=self.options.data.outputs.get(out).classes_names
                )
                return_data.append(
                    self._fill_table_front_structure(
                        _id=_id,
                        graph_name=f"Выходной слой «{out}» - Отчет по классам",
                        plot_data=report
                    )
                )
                _id += 1

            elif task == LayerOutputTypeChoice.Regression:
                y_true = self.inverse_y_true.get("val").get(f'{out}').squeeze()
                y_pred = self.inverse_y_pred.get(f'{out}').squeeze()
                x_scatter, y_scatter = self._get_scatter(y_true, y_pred)
                return_data.append(
                    self._fill_graph_front_structure(
                        _id=_id,
                        _type='scatter',
                        graph_name=f"Выходной слой «{out}» - Скаттер",
                        short_name=f"{out} - Скаттер",
                        x_label="Истинные значения",
                        y_label="Предсказанные значения",
                        plot_data=[self._fill_graph_plot_data(x=x_scatter, y=y_scatter)],
                    )
                )
                _id += 1
                deviation = (y_pred - y_true) * 100 / y_true
                x_mae, y_mae = self._get_distribution_histogram(np.abs(deviation), bins=25, categorical=False)
                return_data.append(
                    self._fill_graph_front_structure(
                        _id=_id,
                        _type='bar',
                        graph_name=f'Выходной слой «{out}» - Распределение абсолютной ошибки',
                        short_name=f"{out} - Распределение MAE",
                        x_label="Абсолютная ошибка",
                        y_label="Значение",
                        plot_data=[self._fill_graph_plot_data(x=x_mae, y=y_mae)],
                    )
                )
                _id += 1
                x_me, y_me = self._get_distribution_histogram(deviation, bins=25, categorical=False)
                return_data.append(
                    self._fill_graph_front_structure(
                        _id=_id,
                        _type='bar',
                        graph_name=f'Выходной слой «{out}» - Распределение ошибки',
                        short_name=f"{out} - Распределение ME",
                        x_label="Ошибка",
                        y_label="Значение",
                        plot_data=[self._fill_graph_plot_data(x=x_me, y=y_me)],
                    )
                )
                _id += 1

            elif task == LayerOutputTypeChoice.Timeseries:
                for i, channel_name in enumerate(self.options.data.columns.get(out).keys()):
                    for step in range(self.y_true.get("val").get(f'{out}').shape[-1]):
                        y_true = self.inverse_y_true.get("val").get(f"{out}")[:, i, step].astype('float')
                        y_pred = self.inverse_y_pred.get(f"{out}")[:, i, step].astype('float')

                        return_data.append(
                            self._fill_graph_front_structure(
                                _id=_id,
                                _type='graphic',
                                graph_name=f"Выходной слой «{out}» - Предсказание канала "
                                           f"«{channel_name.split('_', 1)[-1]}» на {step + 1} "
                                           f"шаг{'ов' if step else ''} вперед",
                                short_name=f"{out} - «{channel_name.split('_', 1)[-1]}» на {step + 1} "
                                           f"шаг{'ов' if step else ''}",
                                x_label="Время",
                                y_label="Значение",
                                plot_data=[
                                    self._fill_graph_plot_data(
                                        x=np.arange(len(y_true)).astype('int').tolist(),
                                        y=y_true.tolist(),
                                        label="Истинное значение"
                                    ),
                                    self._fill_graph_plot_data(
                                        x=np.arange(len(y_true)).astype('int').tolist(),
                                        y=y_pred.tolist(),
                                        label="Предсказанное значение"
                                    )
                                ],
                            )
                        )
                        _id += 1
                        x_axis, auto_corr_true, auto_corr_pred = self._get_autocorrelation_graphic(
                            y_true, y_pred, depth=10
                        )
                        return_data.append(
                            self._fill_graph_front_structure(
                                _id=_id,
                                _type='graphic',
                                graph_name=f"Выходной слой «{out}» - Автокорреляция канала "
                                           f"«{channel_name.split('_', 1)[-1]}» на {step + 1} шаг"
                                           f"{'а' if step else ''} вперед",
                                short_name=f"{out} - Автокорреляция канала «{channel_name.split('_', 1)[-1]}»",
                                x_label="Время",
                                y_label="Значение",
                                plot_data=[
                                    self._fill_graph_plot_data(x=x_axis, y=auto_corr_true, label="Истинное значение"),
                                    self._fill_graph_plot_data(x=x_axis, y=auto_corr_pred,
                                                               label="Предсказанное значение")
                                ],
                            )
                        )
                        _id += 1
                        deviation = (y_pred - y_true) * 100 / y_true
                        x_mae, y_mae = self._get_distribution_histogram(np.abs(deviation), bins=25, categorical=False)
                        return_data.append(
                            self._fill_graph_front_structure(
                                _id=_id,
                                _type='bar',
                                graph_name=f"Выходной слой «{out}» - Распределение абсолютной ошибки канала "
                                           f"«{channel_name.split('_', 1)[-1]}» на {step + 1} шаг"
                                           f"{'ов' if step + 1 == 1 else ''} вперед",
                                short_name=f"{out} - Распределение MAE канала «{channel_name.split('_', 1)[-1]}»",
                                x_label="Абсолютная ошибка",
                                y_label="Значение",
                                plot_data=[self._fill_graph_plot_data(x=x_mae, y=y_mae)],
                            )
                        )
                        _id += 1
                        x_me, y_me = self._get_distribution_histogram(deviation, bins=25, categorical=False)
                        return_data.append(
                            self._fill_graph_front_structure(
                                _id=_id,
                                _type='bar',
                                graph_name=f"Выходной слой «{out}» - Распределение ошибки канала "
                                           f"«{channel_name.split('_', 1)[-1]}» на {step + 1} шаг"
                                           f"{'ов' if step + 1 == 1 else ''} вперед",
                                short_name=f"{out} - Распределение ME канала «{channel_name.split('_', 1)[-1]}»",
                                x_label="Ошибка",
                                y_label="Значение",
                                plot_data=[self._fill_graph_plot_data(x=x_me, y=y_me)],
                            )
                        )
                        _id += 1

            elif task == LayerOutputTypeChoice.Dataframe:
                pass

            elif task == LayerOutputTypeChoice.ObjectDetection:
                # accuracy for classes? smth else?
                pass

            else:
                pass
        return return_data

    def _get_balance_data_request(self) -> list:
        return_data = []
        _id = 1
        for out in self.options.data.outputs.keys():
            task = self.options.data.outputs.get(out).task
            if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.Timeseries_trend:
                class_train_names, class_train_count = sort_dict(
                    self.dataset_balance.get(f"{out}").get('train'),
                    mode=self.interactive_config.data_balance.sorted.name
                )
                class_val_names, class_val_count = sort_dict(
                    self.dataset_balance.get(f"{out}").get('val'),
                    mode=self.interactive_config.data_balance.sorted.name
                )
                return_data.append(
                    self._fill_graph_front_structure(
                        _id=_id,
                        _type='histogram',
                        type_data="train",
                        graph_name=f"Тренировочная выборка",
                        short_name=f"Тренировочная выборка",
                        x_label="Название класса",
                        y_label="Значение",
                        plot_data=[self._fill_graph_plot_data(x=class_train_names, y=class_train_count)],
                    )
                )
                _id += 1
                return_data.append(
                    self._fill_graph_front_structure(
                        _id=_id,
                        _type='histogram',
                        type_data="val",
                        graph_name=f"Проверчная выборка",
                        short_name=f"Проверчная выборка",
                        x_label="Название класса",
                        y_label="Значение",
                        plot_data=[self._fill_graph_plot_data(x=class_val_names, y=class_val_count)],
                    )
                )
                _id += 1

            elif task == LayerOutputTypeChoice.Segmentation:
                presence_train_names, presence_train_count = sort_dict(
                    self.dataset_balance.get(f"{out}").get('train').get('presence_balance'),
                    mode=self.interactive_config.data_balance.sorted.name
                )
                presence_val_names, presence_val_count = sort_dict(
                    self.dataset_balance.get(f"{out}").get('val').get('presence_balance'),
                    mode=self.interactive_config.data_balance.sorted.name
                )
                square_train_names, square_train_count = sort_dict(
                    self.dataset_balance.get(f"{out}").get('train').get('square_balance'),
                    mode=self.interactive_config.data_balance.sorted.name
                )
                square_val_names, square_val_count = sort_dict(
                    self.dataset_balance.get(f"{out}").get('val').get('square_balance'),
                    mode=self.interactive_config.data_balance.sorted.name
                )
                return_data.append(
                    self._fill_graph_front_structure(
                        _id=_id,
                        _type='histogram',
                        type_data="train",
                        graph_name=f"Тренировочная выборка - баланс присутсвия",
                        short_name=f"Тренировочная - присутсвие",
                        x_label="Название класса",
                        y_label="Значение",
                        plot_data=[self._fill_graph_plot_data(x=presence_train_names, y=presence_train_count)],
                    )
                )
                return_data.append(
                    self._fill_graph_front_structure(
                        _id=_id + 1,
                        _type='histogram',
                        type_data="val",
                        graph_name=f"Проверочная выборка - баланс присутсвия",
                        short_name=f"Проверочная - присутсвие",
                        x_label="Название класса",
                        y_label="Значение",
                        plot_data=[self._fill_graph_plot_data(x=presence_val_names, y=presence_val_count)],
                    )
                )
                return_data.append(
                    self._fill_graph_front_structure(
                        _id=_id + 2,
                        _type='histogram',
                        type_data="train",
                        graph_name=f"Тренировочная выборка - процент пространства",
                        short_name=f"Тренировочная - пространство",
                        x_label="Название класса",
                        y_label="Значение",
                        plot_data=[self._fill_graph_plot_data(x=square_train_names, y=square_train_count)],
                    )
                )
                return_data.append(
                    self._fill_graph_front_structure(
                        _id=_id + 3,
                        _type='histogram',
                        type_data="val",
                        graph_name=f"Проверочная выборка - процент пространства",
                        short_name=f"Проверочная - пространство",
                        x_label="Название класса",
                        y_label="Значение",
                        plot_data=[self._fill_graph_plot_data(x=square_val_names, y=square_val_count)],
                    )
                )
                _id += 4

            elif task == LayerOutputTypeChoice.TextSegmentation:
                presence_train_names, presence_train_count = sort_dict(
                    self.dataset_balance.get(f"{out}").get('train').get('presence_balance'),
                    mode=self.interactive_config.data_balance.sorted.name
                )
                presence_val_names, presence_val_count = sort_dict(
                    self.dataset_balance.get(f"{out}").get('val').get('presence_balance'),
                    mode=self.interactive_config.data_balance.sorted.name
                )
                return_data.append(
                    self._fill_graph_front_structure(
                        _id=_id,
                        _type='histogram',
                        type_data="train",
                        graph_name=f"Тренировочная выборка - баланс присутсвия",
                        short_name=f"Тренировочная - присутсвие",
                        x_label="Название класса",
                        y_label="Значение",
                        plot_data=[self._fill_graph_plot_data(x=presence_train_names, y=presence_train_count)],
                    )
                )
                return_data.append(
                    self._fill_graph_front_structure(
                        _id=_id + 1,
                        _type='histogram',
                        type_data="val",
                        graph_name=f"Проверочная выборка - баланс присутсвия",
                        short_name=f"Проверочная - присутсвие",
                        x_label="Название класса",
                        y_label="Значение",
                        plot_data=[self._fill_graph_plot_data(x=presence_val_names, y=presence_val_count)],
                    )
                )
                _id += 2

            elif task == LayerOutputTypeChoice.Regression:
                for data_type in ["train", "val"]:
                    data_type_name = "Тренировочная" if data_type == "train" else "Проверочная"
                    for histogram in self.dataset_balance[f"{out}"][data_type]['histogram']:
                        return_data.append(
                            self._fill_graph_front_structure(
                                _id=_id,
                                _type=histogram.get("type"),
                                type_data=data_type,
                                graph_name=f"{data_type_name} выборка - "
                                           f"Гистограмма распределения колонки «{histogram['name']}»",
                                short_name=histogram['name'],
                                x_label="Значение",
                                y_label="Количество",
                                plot_data=[self._fill_graph_plot_data(x=histogram.get("x"), y=histogram.get("y"))],
                            )
                        )
                        _id += 1
                    return_data.append(
                        self._fill_heatmap_front_structure(
                            _id=_id,
                            _type="correlation_heatmap",
                            type_data=data_type,
                            graph_name=f"{data_type_name} выборка - Матрица корреляций",
                            short_name=f"Матрица корреляций",
                            x_label="Колонка",
                            y_label="Колонка",
                            labels=self.dataset_balance[f"{out}"][data_type]['correlation']["labels"],
                            data_array=self.dataset_balance[f"{out}"][data_type]['correlation']["matrix"],
                        )
                    )
                    _id += 1

            elif task == LayerOutputTypeChoice.Timeseries:
                _id += 1
                for channel_name in list(self.options.dataframe.get('train').columns):
                    for data_type in ["train", "val"]:
                        data_type_name = "Тренировочная" if data_type == "train" else "Проверочная"
                        y_true = self.options.dataframe.get(data_type)[channel_name].to_list()
                        x_graph_axis = np.arange(len(y_true)).astype('float').tolist()
                        x_hist, y_hist = self._get_distribution_histogram(y_true, bins=25, categorical=False)
                        return_data.append(
                            self._fill_graph_front_structure(
                                _id=_id,
                                _type="graphic",
                                type_data=data_type,
                                graph_name=f'{data_type_name} выборка - График канала «{channel_name}»',
                                short_name=f'{data_type_name} - «{channel_name}»',
                                x_label="Время",
                                y_label="Количество",
                                plot_data=[self._fill_graph_plot_data(x=x_graph_axis, y=y_true)],
                            )
                        )
                        return_data.append(
                            self._fill_graph_front_structure(
                                _id=_id+1,
                                _type="bar",
                                type_data=data_type,
                                graph_name=f'{data_type_name} выборка - Гистограмма плотности канала «{channel_name}»',
                                short_name=f'{data_type_name} - Гистограмма «{channel_name}»',
                                x_label="Значение",
                                y_label="Количество",
                                plot_data=[self._fill_graph_plot_data(x=x_hist, y=y_hist)],
                            )
                        )
                        _id += 2

            elif task == LayerOutputTypeChoice.ObjectDetection:
                # frequency of classes, like with segmentation
                pass

            else:
                pass

        return return_data

    @staticmethod
    def _get_confusion_matrix(y_true, y_pred, get_percent=True) -> tuple:
        cm = confusion_matrix(y_true, y_pred)
        cm_percent = None
        if get_percent:
            cm_percent = np.zeros_like(cm).astype('float')
            for i in range(len(cm)):
                total = np.sum(cm[i])
                for j in range(len(cm[i])):
                    cm_percent[i][j] = round(cm[i][j] * 100 / total, 1)
        return cm.astype('float').tolist(), cm_percent.astype('float').tolist()

    @staticmethod
    def _get_classification_report(y_true, y_pred, labels):
        cr = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        return_stat = []
        for lbl in labels:
            return_stat.append(
                {
                    'Класс': lbl,
                    "Точность": round(float(cr.get(lbl).get('precision')) * 100, 2),
                    "Чувствительность": round(float(cr.get(lbl).get('recall')) * 100, 2),
                    "F1-мера": round(float(cr.get(lbl).get('f1-score')) * 100, 2),
                    "Количество": int(cr.get(lbl).get('support'))
                }
            )
        for i in ['macro avg', 'micro avg', 'samples avg', 'weighted avg']:
            return_stat.append(
                {
                    'Класс': i,
                    "Точность": round(float(cr.get(i).get('precision')) * 100, 2),
                    "Чувствительность": round(float(cr.get(i).get('recall')) * 100, 2),
                    "F1-мера": round(float(cr.get(i).get('f1-score')) * 100, 2),
                    "Количество": int(cr.get(i).get('support'))
                }
            )
        return return_stat

    @staticmethod
    def _get_error_distribution(y_true, y_pred, bins=25, absolute=True):
        error = (y_true - y_pred)  # "* 100 / y_true
        if absolute:
            error = np.abs(error)
        return InteractiveCallback()._get_distribution_histogram(error, bins=bins, categorical=False)

    @staticmethod
    def _get_time_series_graphic(data):
        return np.arange(len(data)).astype('int').tolist(), np.array(data).astype('float').tolist()

    @staticmethod
    def _get_correlation_matrix(data_frame: DataFrame):
        corr = data_frame.corr()
        labels = []
        for lbl in list(corr.columns):
            labels.append(lbl.split("_", 1)[-1])
        return labels, np.array(np.round(corr, 2)).astype('float').tolist()

    @staticmethod
    def _get_scatter(y_true, y_pred):
        return InteractiveCallback().clean_data_series([y_true, y_pred], mode="duo")

    @staticmethod
    def _get_distribution_histogram(data_series, bins=25, categorical=True):
        if categorical:
            hist_data = pd.Series(data_series).value_counts()
            return hist_data.index.to_list(), hist_data.to_list()
        else:
            data_series = InteractiveCallback().clean_data_series([data_series], mode="mono")
            bar_values, x_labels = np.histogram(data_series, bins=bins)
            return x_labels[:-1].astype('float').tolist(), bar_values.astype('int').tolist()

    @staticmethod
    def clean_data_series(data_series: list, mode="mono"):
        if mode == "mono":
            sort_x = pd.Series(data_series[0])
            sort_x = sort_x[sort_x > sort_x.quantile(0.02)]
            sort_x = sort_x[sort_x < sort_x.quantile(0.98)]
            data_series = np.array(sort_x)
            return data_series
        elif mode == "duo":
            sort = pd.DataFrame({
                'y_true': np.array(data_series[0]).squeeze(),
                'y_pred': np.array(data_series[1]).squeeze(),
            })
            sort = sort[sort['y_true'] > sort['y_true'].quantile(0.05)]
            sort = sort[sort['y_true'] < sort['y_true'].quantile(0.95)]
            return sort['y_true'].to_list(), sort['y_pred'].to_list()
        else:
            return None

    @staticmethod
    def _get_autocorrelation_graphic(y_true, y_pred, depth=10) -> (list, list, list):

        def get_auto_corr(y_true, y_pred, k):
            l = len(y_true)
            time_series_1 = y_pred[:-k]
            time_series_2 = y_true[k:]
            time_series_mean = np.mean(y_true)
            time_series_var = np.array([i ** 2 for i in y_true - time_series_mean]).sum()
            auto_corr = 0
            for i in range(l - k):
                temp = (time_series_1[i] - time_series_mean) * (time_series_2[i] - time_series_mean) / time_series_var
                auto_corr = auto_corr + temp
            return auto_corr

        x_axis = np.arange(depth).astype('int').tolist()

        auto_corr_true = []
        for i in range(depth):
            auto_corr_true.append(get_auto_corr(y_true, y_true, i + 1))
        auto_corr_pred = []
        for i in range(depth):
            auto_corr_pred.append(get_auto_corr(y_true, y_pred, i + 1))
        return x_axis, auto_corr_true, auto_corr_pred

    @staticmethod
    def _dice_coef(y_true, y_pred, batch_mode=True, smooth=1.0):
        return CreateArray().dice_coef(y_true, y_pred, batch_mode=batch_mode, smooth=smooth)

    # def _postprocess_initial_data(self, input_id: str, example_idx: int, save_id: int = None):
    #     column_idx = []
    #     input_task = self.dataset_config.get("inputs").get(input_id).get("task")
    #     if self.dataset_config.get("data").group != DatasetGroupChoice.keras:
    #         for column_name in self.dataset_config.get("dataframe").get('val').columns:
    #             if column_name.split('_')[0] == input_id:
    #                 column_idx.append(
    #                     self.dataset_config.get("dataframe").get('val').columns.tolist().index(column_name)
    #                 )
    #         if input_task == LayerInputTypeChoice.Text or input_task == LayerInputTypeChoice.Dataframe:
    #             initial_file_path = ""
    #         else:
    #             initial_file_path = os.path.join(
    #                 self.dataset_config.get("dataset_path"),
    #                 self.dataset_config.get("dataframe").get('val').iat[example_idx, column_idx[0]]
    #             )
    #         if not save_id:
    #             return str(os.path.abspath(initial_file_path))
    #     else:
    #         initial_file_path = ""
    #
    #     data = []
    #     data_type = ""
    #     if task == LayerInputTypeChoice.Image:
    #         if self.dataset_config.get("group") != DatasetGroupChoice.keras:
    #             img = Image.open(initial_file_path)
    #             img = img.resize(
    #                 self.dataset_config.get("inputs").get(input_id).get("input_shape")[0:2][::-1],
    #                 Image.ANTIALIAS
    #             )
    #         else:
    #             img = image.array_to_img(self.x_val.get(input_id)[example_idx])
    #         img = img.convert('RGB')
    #         save_path = os.path.join(
    #             self.preset_path, f"initial_data_image_{save_id}_input_{input_id}.webp"
    #         )
    #         img.save(save_path, 'webp')
    #         data_type = LayerInputTypeChoice.Image.name
    #         data = [
    #             {
    #                 "title": "Изображение",
    #                 "value": save_path,
    #                 "color_mark": None
    #             }
    #         ]
    #
    #     elif task == LayerInputTypeChoice.Text:
    #         regression_task = False
    #         for out in self.dataset_config.get("outputs").keys():
    #             if self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Regression:
    #                 regression_task = True
    #         for column in column_idx:
    #             text_str = self.dataset_config.get("dataframe").get('val').iat[example_idx, column]
    #             data_type = LayerInputTypeChoice.Text.name
    #             title = "Текст"
    #             if regression_task:
    #                 title = list(self.dataset_config.get("dataframe").get('val').columns)[column].split("_", 1)[-1]
    #             data = [
    #                 {
    #                     "title": title,
    #                     "value": text_str,
    #                     "color_mark": None
    #                 }
    #             ]
    #
    #     elif task == LayerInputTypeChoice.Video:
    #         clip = moviepy_editor.VideoFileClip(initial_file_path)
    #         save_path = os.path.join(
    #             self.preset_path, f"initial_data_video_{save_id}_input_{input_id}.webm"
    #         )
    #         clip.write_videofile(save_path)
    #         data_type = LayerInputTypeChoice.Video.name
    #         data = [
    #             {
    #                 "title": "Видео",
    #                 "value": save_path,
    #                 "color_mark": None
    #             }
    #         ]
    #
    #     elif task == LayerInputTypeChoice.Audio:
    #         save_path = os.path.join(
    #             self.preset_path, f"initial_data_audio_{save_id}_input_{input_id}.webp"
    #         )
    #         AudioSegment.from_file(initial_file_path).export(save_path, format="webm")
    #         data_type = LayerInputTypeChoice.Audio.name
    #         data = [
    #             {
    #                 "title": "Аудио",
    #                 "value": save_path,
    #                 "color_mark": None
    #             }
    #         ]
    #
    #     elif task == LayerInputTypeChoice.Dataframe:
    #         time_series_choice = False
    #         for out in self.dataset_config.get("outputs").keys():
    #             if self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Timeseries or \
    #                     self.dataset_config.get("outputs").get(out).get(
    #                         "task") == LayerOutputTypeChoice.Timeseries_trend:
    #                 time_series_choice = True
    #                 break
    #
    #         if time_series_choice:
    #             graphics_data = []
    #             names = ""
    #             multi = False
    #             for i, channel in enumerate(self.dataset_config.get("columns").get(int(input_id)).keys()):
    #                 multi = True if i > 0 else False
    #                 names += f"«{channel.split('_', 1)[-1]}», "
    #                 graphics_data.append(
    #                     {
    #                         'id': i + 1,
    #                         'graph_name': f"График канала «{channel.split('_', 1)[-1]}»",
    #                         'x_label': 'Время',
    #                         'y_label': 'Значение',
    #                         'plot_data': {
    #                             'x': np.arange(self.inverse_x_val.get(input_id)[example_idx].shape[-1]).astype(
    #                                 'int').tolist(),
    #                             'y': self.inverse_x_val.get(input_id)[example_idx][i].astype('float').tolist()
    #                         },
    #                     }
    #                 )
    #             data_type = "graphic"
    #             data = [
    #                 {
    #                     "title": f"График{'и' if multi else ''} по канал{'ам' if multi else 'у'} {names[:-2]}",
    #                     "value": graphics_data,
    #                     "color_mark": None
    #                 }
    #             ]
    #         else:
    #             # data_type = LayerInputTypeChoice.Dataframe.name
    #             data_type = "str"
    #             for col_name in self.dataset_config.get('columns').get(int(input_id)).keys():
    #                 value = self.dataset_config.get('dataframe').get('val')[col_name].to_list()[example_idx]
    #                 data.append(
    #                     {
    #                         "title": col_name.split("_", 1)[-1],
    #                         "value": value,
    #                         "color_mark": None
    #                     }
    #                 )
    #
    #     return data, data_type.lower()
    #
    # def _postprocess_result_data(self, output_id: str, data_type: str, save_id: int, example_idx: int, show_stat=True):
    #
    #     def add_tags_to_word(word: str, tag: str):
    #         if tag:
    #             return f"<{tag[1:-1]}>{word}</{tag[1:-1]}>"
    #         else:
    #             return word
    #
    #     def color_mixer(colors: list):
    #         if colors:
    #             result = np.zeros((3,))
    #             for color in colors:
    #                 result += np.array(color)
    #             result = result / len(colors)
    #             return tuple(result.astype('int').tolist())
    #
    #     def tag_mixer(tags: list, colors: dict):
    #         tags = list(set(sorted(tags, reverse=True)))
    #         mix_tag = f"<{tags[0][1:-1]}"
    #         for tag in tags[1:]:
    #             mix_tag += f"+{tag[1:-1]}"
    #         mix_tag = f"{mix_tag}>"
    #         if mix_tag not in colors.keys():
    #             colors[mix_tag] = color_mixer([colors[tag] for tag in tags])
    #         return mix_tag
    #
    #     def reformat_tags(y_array, classes_names: list, colors: dict, sensitivity: float = 0.9):
    #         norm_array = np.where(y_array >= sensitivity, 1, 0).astype('int')
    #         reformat_tags = []
    #         for word_tag in norm_array:
    #             if np.sum(word_tag) == 0:
    #                 reformat_tags.append(None)
    #             elif np.sum(word_tag) == 1:
    #                 reformat_tags.append(classes_names[np.argmax(word_tag, axis=-1)])
    #             else:
    #                 mix_tag = []
    #                 for i, tag in enumerate(word_tag):
    #                     if tag == 1:
    #                         mix_tag.append(classes_names[i])
    #                 reformat_tags.append(tag_mixer(mix_tag, colors))
    #         return reformat_tags
    #
    #     def text_colorization(text: str, labels: list, classes_names: list, colors: dict):
    #         text = text.split(" ")
    #         labels = reformat_tags(labels, classes_names, colors)
    #         colored_text = []
    #         for i, word in enumerate(text):
    #             colored_text.append(add_tags_to_word(word, labels[i]))
    #         return ' '.join(colored_text)
    #
    #     data = {
    #         "y_true": {},
    #         "y_pred": {},
    #         "stat": {}
    #     }
    #     task = self.dataset_config.get("outputs").get(output_id).get("task")
    #
    #     if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.Timeseries_trend:
    #         labels = self.dataset_config.get("outputs").get(output_id).get("classes_names")
    #         ohe = True if self.dataset_config.get("outputs").get(output_id).get("encoding") == 'ohe' else False
    #         y_true = np.argmax(self.y_true.get(data_type).get(output_id)[example_idx]) if ohe \
    #             else np.squeeze(self.y_true.get(data_type).get(output_id)[example_idx])
    #         data["y_true"] = {
    #             "type": "str",
    #             "data": [
    #                 {
    #                     "title": "Класс",
    #                     "value": labels[y_true],
    #                     "color_mark": None
    #                 }
    #             ]
    #         }
    #         predict = self.y_pred.get(output_id)[example_idx]
    #         if y_true == np.argmax(predict):
    #             color_mark = 'success'
    #         else:
    #             color_mark = 'wrong'
    #         data["y_pred"] = {
    #             "type": "str",
    #             "data": [
    #                 {
    #                     "title": "Класс",
    #                     "value": labels[np.argmax(predict)],
    #                     "color_mark": color_mark
    #                 }
    #             ]
    #         }
    #         if show_stat:
    #             data["stat"] = {
    #                 "type": "str",
    #                 "data": []
    #             }
    #             for i, val in enumerate(predict):
    #                 if val == max(predict) and i == y_true:
    #                     class_color_mark = "success"
    #                 elif val == max(predict) and i != y_true:
    #                     class_color_mark = "wrong"
    #                 else:
    #                     class_color_mark = None
    #                 data["stat"]["data"].append(
    #                     dict(title=labels[i], value=f"{round(val * 100, 1)}%", color_mark=class_color_mark)
    #                 )
    #
    #     elif task == LayerOutputTypeChoice.Segmentation:
    #         labels = self.dataset_config.get("outputs").get(output_id).get("classes_names")
    #
    #         y_true = np.expand_dims(
    #             np.argmax(self.y_true.get(data_type).get(output_id)[example_idx], axis=-1), axis=-1) * 512
    #         for i, color in enumerate(self.dataset_config.get("outputs").get(output_id).get("classes_colors")):
    #             y_true = np.where(y_true == i * 512, np.array(color), y_true)
    #         y_true = y_true.astype("uint8")
    #         y_true_save_path = os.path.join(
    #             self.preset_path, f"true_segmentation_data_image_{save_id}_output_{output_id}.webp"
    #         )
    #         matplotlib.image.imsave(y_true_save_path, y_true)
    #         data["y_true"] = {
    #             "type": "image",
    #             "data": [
    #                 {
    #                     "title": "Изображение",
    #                     "value": y_true_save_path,
    #                     "color_mark": None
    #                 }
    #             ]
    #         }
    #
    #         y_pred = np.expand_dims(np.argmax(self.y_pred.get(output_id)[example_idx], axis=-1), axis=-1) * 512
    #         for i, color in enumerate(self.dataset_config.get("outputs").get(output_id).get("classes_colors")):
    #             y_pred = np.where(y_pred == i * 512, np.array(color), y_pred)
    #         y_pred = y_pred.astype("uint8")
    #         y_pred_save_path = os.path.join(
    #             self.preset_path, f"predict_segmentation_data_image_{save_id}_output_{output_id}.webp"
    #         )
    #         matplotlib.image.imsave(y_pred_save_path, y_pred)
    #         data["y_pred"] = {
    #             "type": "image",
    #             "data": [
    #                 {
    #                     "title": "Изображение",
    #                     "value": y_pred_save_path,
    #                     "color_mark": None
    #                 }
    #             ]
    #         }
    #         if show_stat:
    #             data["stat"] = {
    #                 "type": "str",
    #                 "data": []
    #             }
    #             y_true = np.array(self.y_true.get(data_type).get(output_id)[example_idx]).astype('int')
    #             y_pred = to_categorical(
    #                 np.argmax(self.y_pred.get(output_id)[example_idx], axis=-1),
    #                 self.dataset_config.get("outputs").get(output_id).get("num_classes")).astype('int')
    #             count = 0
    #             mean_val = 0
    #             for idx, cls in enumerate(labels):
    #                 dice_val = np.round(self._dice_coef(y_true[:, :, idx], y_pred[:, :, idx], batch_mode=False) * 100,
    #                                     1)
    #                 count += 1
    #                 mean_val += dice_val
    #                 data["stat"]["data"].append(
    #                     {
    #                         'title': cls,
    #                         'value': f"{dice_val}%",
    #                         'color_mark': 'success' if dice_val >= 90 else 'wrong'
    #                     }
    #                 )
    #             data["stat"]["data"].insert(
    #                 0,
    #                 {
    #                     'title': "Средняя точность",
    #                     'value': f"{round(mean_val / count, 2)}%",
    #                     'color_mark': 'success' if mean_val / count >= 90 else 'wrong'
    #                 }
    #             )
    #
    #     elif task == LayerOutputTypeChoice.TextSegmentation:
    #         # TODO: пока исходим что для сегментации текста есть только один вход с текстом, если будут сложные модели
    #         #  на сегментацию текста на несколько входов то придется искать решения
    #
    #         classes_names = self.dataset_config.get("outputs").get(output_id).get("classes_names")
    #         text_for_preparation = self.dataset_config.get('dataframe').get('val').iat[example_idx, 0]
    #         true_text_segmentation = text_colorization(
    #             text_for_preparation,
    #             self.y_true.get(data_type).get(output_id)[example_idx],
    #             classes_names,
    #             self.dataset_config.get("outputs").get(output_id).get('classes_colors')
    #         )
    #         data["y_true"] = {
    #             "type": "segmented_text",
    #             "data": [
    #                 {
    #                     "title": "Текст",
    #                     "value": true_text_segmentation,
    #                     "color_mark": None
    #                 }
    #             ]
    #         }
    #         pred_text_segmentation = text_colorization(
    #             text_for_preparation,
    #             self.y_pred.get(output_id)[example_idx],
    #             classes_names,
    #             self.dataset_config.get("outputs").get(output_id).get('classes_colors')
    #         )
    #         data["y_pred"] = {
    #             "type": "segmented_text",
    #             "data": [
    #                 {
    #                     "title": "Текст",
    #                     "value": pred_text_segmentation,
    #                     "color_mark": None
    #                 }
    #             ]
    #         }
    #         if show_stat:
    #             data["stat"] = {
    #                 "type": "str",
    #                 "data": []
    #             }
    #             y_true = np.array(self.y_true.get(data_type).get(output_id)[example_idx]).astype('int')
    #             y_pred = np.where(self.y_pred.get(output_id)[example_idx] >= 0.9, 1., 0.)
    #             count = 0
    #             mean_val = 0
    #             for idx, cls in enumerate(classes_names):
    #                 if np.sum(y_true[:, idx]) == 0 and np.sum(y_pred[:, idx]) == 0:
    #                     data["stat"]["data"].append(
    #                         {
    #                             'title': cls,
    #                             'value': "-",
    #                             'color_mark': None
    #                         }
    #                     )
    #                 else:
    #                     dice_val = np.round(self._dice_coef(y_true[:, idx], y_pred[:, idx], batch_mode=False) * 100, 1)
    #                     data["stat"]["data"].append(
    #                         {
    #                             'title': cls,
    #                             'value': f"{dice_val}%",
    #                             'color_mark': 'success' if dice_val >= 90 else 'wrong'}
    #                     )
    #                     count += 1
    #                     mean_val += dice_val
    #             if count and mean_val / count >= 90:
    #                 mean_color_mark = "success"
    #             elif count and mean_val / count < 90:
    #                 mean_color_mark = "wrong"
    #             else:
    #                 mean_color_mark = None
    #             data["stat"]["data"].insert(
    #                 0,
    #                 {
    #                     'title': "Средняя точность",
    #                     'value': f"{round(mean_val / count, 2)}%" if count else "-",
    #                     'color_mark': mean_color_mark
    #                 }
    #             )
    #
    #     elif task == LayerOutputTypeChoice.Regression:
    #         column_names = list(self.dataset_config["columns"][int(output_id)].keys())
    #         y_true = self.inverse_y_true.get(data_type).get(output_id)[example_idx]
    #         y_pred = self.inverse_y_pred.get(output_id)[example_idx]
    #         data["y_true"] = {
    #             "type": "str",
    #             "data": []
    #         }
    #         for i, name in enumerate(column_names):
    #             data["y_true"]["data"].append(
    #                 {
    #                     "title": name.split('_', 1)[-1],
    #                     "value": f"{y_true[i]: .2f}",
    #                     "color_mark": None
    #                 }
    #             )
    #         deviation = np.abs((y_pred - y_true) * 100 / y_true)
    #         data["y_pred"] = {
    #             "type": "str",
    #             "data": []
    #         }
    #         for i, name in enumerate(column_names):
    #             color_mark = 'success' if deviation[i] < 2 else "wrong"
    #             data["y_pred"]["data"].append(
    #                 {
    #                     "title": name.split('_', 1)[-1],
    #                     "value": f"{y_pred[i]: .2f}",
    #                     "color_mark": color_mark
    #                 }
    #             )
    #         if show_stat:
    #             data["stat"] = {
    #                 "type": "str",
    #                 "data": []
    #             }
    #             for i, name in enumerate(column_names):
    #                 color_mark = 'success' if deviation[i] < 2 else "wrong"
    #                 data["stat"]["data"].append(
    #                     {
    #                         'title': f"Отклонение - «{name.split('_', 1)[-1]}»",
    #                         'value': f"{np.round(deviation[i], 2)} %",
    #                         'color_mark': color_mark
    #                     }
    #                 )
    #
    #     elif task == LayerOutputTypeChoice.Timeseries:
    #         graphics = []
    #         real_x = np.arange(
    #             self.inverse_x_val.get(list(self.inverse_x_val.keys())[0]).shape[-1]).astype('float').tolist()
    #         depth = self.inverse_y_true.get("val").get(output_id)[example_idx].shape[-1]
    #
    #         _id = 1
    #         for i, channel in enumerate(self.dataset_config["columns"][int(output_id)].keys()):
    #             for input in self.dataset_config.get('inputs').keys():
    #                 for input_column in self.dataset_config["columns"][int(input)].keys():
    #                     if channel.split("_", 1)[-1] == input_column.split("_", 1)[-1]:
    #                         init_column = list(self.dataset_config["columns"][int(input)].keys()).index(input_column)
    #                         graphics.append(
    #                             {
    #                                 'id': _id + 1,
    #                                 'graph_name': f'График канала «{channel.split("_", 1)[-1]}»',
    #                                 'x_label': 'Время',
    #                                 'y_label': 'Значение',
    #                                 'plot_data': [
    #                                     {
    #                                         'label': "Исходное значение",
    #                                         'x': real_x,
    #                                         'y': np.array(
    #                                             self.inverse_x_val.get(f"{input}")[example_idx][init_column]
    #                                         ).astype('float').tolist()
    #                                     },
    #                                     {
    #                                         'label': "Истинное значение",
    #                                         'x': np.arange(len(real_x), len(real_x) + depth).astype('int').tolist(),
    #                                         'y': self.inverse_y_true.get("val").get(
    #                                             output_id)[example_idx][i].astype('float').tolist()
    #                                     },
    #                                     {
    #                                         'label': "Предсказанное значение",
    #                                         'x': np.arange(len(real_x), len(real_x) + depth).astype('float').tolist(),
    #                                         'y': self.inverse_y_pred.get(output_id)[
    #                                             example_idx][i].astype('float').tolist()
    #                                     },
    #                                 ]
    #                             }
    #                         )
    #                         _id += 1
    #                         break
    #         data["y_pred"] = {
    #             "type": "graphic",
    #             "data": [
    #                 {
    #                     "title": "Графики",
    #                     "value": graphics,
    #                     "color_mark": None
    #                 }
    #             ]
    #         }
    #         if show_stat:
    #             data["stat"]["data"] = []
    #             for i, channel in enumerate(self.dataset_config["columns"][int(output_id)].keys()):
    #                 data["stat"]["data"].append(
    #                     dict(title=channel.split("_", 1)[-1], value={"type": "table", "data": {}}, color_mark=None)
    #                 )
    #                 for step in range(self.inverse_y_true.get("val").get(output_id)[example_idx].shape[-1]):
    #                     deviation = (self.inverse_y_pred.get(output_id)[example_idx, i, step] -
    #                                  self.inverse_y_true.get("val").get(output_id)[example_idx, i, step]) * 100 / \
    #                                 self.inverse_y_true.get("val").get(output_id)[example_idx, i, step]
    #                     data["stat"]["data"][-1]["value"]["data"][f"{step + 1}"] = [
    #                         {
    #                             "title": "Истина",
    #                             "value": f"{round(self.inverse_y_true.get('val').get(output_id)[example_idx][i, step].astype('float'), 2)}",
    #                             'color_mark': None
    #                         },
    #                         {
    #                             "title": "Предсказание",
    #                             "value": f"{round(self.inverse_y_pred.get(output_id)[example_idx][i, step].astype('float'), 2)}",
    #                             'color_mark': "success" if abs(deviation) < 2 else "wrong"
    #                         },
    #                         {
    #                             "title": "Отклонение",
    #                             "value": f"{round(deviation, 2)} %",
    #                             'color_mark': "success" if abs(deviation) < 2 else "wrong"
    #                         }
    #                     ]
    #
    #     elif task == LayerOutputTypeChoice.Dataframe:
    #         pass
    #
    #     elif task == LayerOutputTypeChoice.ObjectDetection:
    #         # image with bb
    #         # accuracy, correlation bb for classes
    #         pass
    #
    #     return data
