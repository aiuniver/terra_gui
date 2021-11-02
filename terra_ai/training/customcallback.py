import colorsys
import copy
import importlib
import math
import os
import random
import re
import string
from typing import Union, Optional

import matplotlib
import pandas as pd
from pandas import DataFrame

from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from terra_ai import progress
from terra_ai.data.datasets.extra import LayerInputTypeChoice, LayerOutputTypeChoice, DatasetGroupChoice, \
    LayerEncodingChoice
from terra_ai.data.presets.training import Metric
from terra_ai.data.training.extra import LossGraphShowChoice, MetricGraphShowChoice, MetricChoice, ArchitectureChoice
from terra_ai.data.training.train import InteractiveData
from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.training.customlosses import UnscaledMAE, BalancedRecall, BalancedPrecision, BalancedFScore, FScore
from terra_ai.utils import camelize, decamelize

__version__ = 0.085

MAX_TS_GRAPH_COUNT = 200
MAX_HISTOGRAM_BINS = 50
MAX_INTERMEDIATE_GRAPH_LENGTH = 50


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
            "module": "tensorflow.keras.losses",
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
        "giou_loss": {
            "log_name": "giou_loss",
            "mode": "min",
            "module": ""
        },
        "conf_loss": {
            "log_name": "conf_loss",
            "mode": "min",
            "module": ""
        },
        "prob_loss": {
            "log_name": "prob_loss",
            "mode": "min",
            "module": ""
        },
        "total_loss": {
            "log_name": "total_loss",
            "mode": "min",
            "module": ""
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
        "BalancedDiceCoef": {
            "log_name": "balanced_dice_coef",
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
        "RecallPercent": {
            "log_name": "recall_percent",
            "mode": "max",
            "module": "terra_ai.training.customlosses"
        },
        "BalancedRecall": {
            "log_name": "balanced_recall",
            "mode": "max",
            "module": "terra_ai.training.customlosses"
        },
        "BalancedPrecision": {
            "log_name": "balanced_precision",
            "mode": "max",
            "module": "terra_ai.training.customlosses"
        },
        "BalancedFScore": {
            "log_name": "balanced_f_score",
            "mode": "max",
            "module": "terra_ai.training.customlosses"
        },
        "FScore": {
            "log_name": "f_score",
            "mode": "max",
            "module": "terra_ai.training.customlosses"
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
        "UnscaledMAE": {
            "log_name": "unscaled_mae",
            "mode": "min",
            "module": "terra_ai.training.customlosses"
        },
        "mAP50": {
            "log_name": "mAP50",
            "mode": "max",
            "module": ""
        },
        # "mAP95": {
        #     "log_name": "mAP95",
        #     "mode": "max",
        #     "module": ""
        # },
    }
}


def print_error(class_name: str, method_name: str, message: Exception):
    return print(f'\n_________________________________________________\n'
                 f'Error in class {class_name} method {method_name}: {message}'
                 f'\n_________________________________________________\n')


class InteractiveCallbackOld:
    """Callback for interactive requests"""

    def __init__(self):
        self.name = 'InteractiveCallback'
        self.losses = None
        self.metrics = None
        self.loss_obj = None
        self.metrics_obj = None
        self.options: Optional[PrepareDataset] = None
        self.class_colors = []
        self.dataset_path = None
        self.x_val = None
        self.inverse_x_val = None
        self.y_true = {}
        self.inverse_y_true = {}
        self.y_pred = {}
        self.raw_y_pred = None
        self.raw_y_true = None
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

        self.seed_idx = None
        self.example_idx = []
        self.intermediate_result = {}
        self.statistic_result = {}
        self.train_progress = {}
        self.addtrain_epochs = []
        self.progress_name = "training"
        self.preset_path = ""
        self.basic_architecture = [ArchitectureChoice.Basic, ArchitectureChoice.ImageClassification,
                                   ArchitectureChoice.ImageSegmentation, ArchitectureChoice.TextSegmentation,
                                   ArchitectureChoice.TextClassification, ArchitectureChoice.AudioClassification,
                                   ArchitectureChoice.VideoClassification, ArchitectureChoice.DataframeClassification,
                                   ArchitectureChoice.DataframeRegression, ArchitectureChoice.Timeseries,
                                   ArchitectureChoice.TimeseriesTrend]
        self.yolo_architecture = [ArchitectureChoice.YoloV3, ArchitectureChoice.YoloV4]

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
        self.random_key = ''

        self.interactive_config: InteractiveData = InteractiveData(**{})
        pass

    def set_attributes(self, dataset: PrepareDataset, metrics: dict, losses: dict, dataset_path: str,
                       training_path: str, initial_config: InteractiveData):
        # print('\ndataset.architecture', dataset.data.architecture)
        # print('\ndataset.data.outputs', dataset.data.outputs)
        # print('\ndataset.data.inputs', dataset.data.inputs)
        self.preset_path = os.path.join(training_path, "presets")
        self.interactive_config = initial_config
        if not os.path.exists(self.preset_path):
            os.mkdir(self.preset_path)
        if dataset.data.architecture in self.basic_architecture:
            self.losses = losses
            self.metrics = self._reformat_metrics(metrics)
            self.loss_obj = self._prepare_loss_obj(losses)
            self.metrics_obj = self._prepare_metric_obj(metrics)

        self.options = dataset
        self._class_metric_list()
        # print('\nset_attributes self._class_metric_list', self.class_graphics)
        self.dataset_path = dataset_path
        # print('\nset_attributes self.dataset_path', self.dataset_path)
        self._get_classes_colors()
        # print('\nset_attributes self._get_classes_colors', self.class_colors)
        self.x_val, self.inverse_x_val = self._prepare_x_val(dataset)
        self.y_true, self.inverse_y_true = self._prepare_y_true(dataset)

        if not self.log_history:
            self._prepare_null_log_history_template()
        self.dataset_balance = self._prepare_dataset_balance()
        # print('\nself.dataset_balance', self.dataset_balance)
        self.class_idx = self._prepare_class_idx()
        self.seed_idx = self._prepare_seed()
        # print('\nset_attributes self.seed_idx', self.seed_idx)
        self.random_key = ''.join(random.sample(string.ascii_letters + string.digits, 16))
        # print('\nset_attributes self.random_key', self.random_key)

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
        self.addtrain_epochs = []
        self.deploy_presets_data = None

    def get_states(self):
        return self.train_states

    def get_presets(self):
        return self.deploy_presets_data

    def update_train_progress(self, data: dict):
        self.train_progress = data

    def update_state(self, y_pred, y_true=None, fit_logs=None, current_epoch_time=None,
                     on_epoch_end_flag=False) -> dict:
        # if y_pred is not None:
            # print('\nupdate_state', fit_logs, len(y_pred), 'on_epoch_end_flag', on_epoch_end_flag)
        if self.log_history:
            if y_pred is not None:
                if self.options.data.architecture in self.basic_architecture:
                    self._reformat_y_pred(y_pred)
                    if self.interactive_config.intermediate_result.show_results:
                        out = f"{self.interactive_config.intermediate_result.main_output}"
                        self.example_idx = CreateArray().prepare_example_idx_to_show(
                            array=self.y_pred.get(out),
                            true_array=self.y_true.get("val").get(out),
                            options=self.options,
                            output=int(out),
                            count=self.interactive_config.intermediate_result.num_examples,
                            choice_type=self.interactive_config.intermediate_result.example_choice_type,
                            seed_idx=self.seed_idx[:self.interactive_config.intermediate_result.num_examples]
                        )
                if self.options.data.architecture in self.yolo_architecture:
                    # print('\nupdate_state self.seed_idx', self.seed_idx)
                    # print('\nupdate_state num_examples', self.yolo_interactive_config.intermediate_result.num_examples)
                    self.raw_y_pred = y_pred
                    self.raw_y_true = y_true
                    if self.interactive_config.intermediate_result.show_results:
                        self.example_idx, _ = CreateArray().prepare_yolo_example_idx_to_show(
                            array=copy.deepcopy(self.y_pred),
                            true_array=copy.deepcopy(self.y_true),
                            name_classes=self.options.data.outputs.get(
                                list(self.options.data.outputs.keys())[0]).classes_names,
                            box_channel=self.interactive_config.intermediate_result.box_channel,
                            count=self.interactive_config.intermediate_result.num_examples,
                            choice_type=self.interactive_config.intermediate_result.example_choice_type,
                            seed_idx=self.seed_idx,
                            sensitivity=self.interactive_config.intermediate_result.sensitivity,
                        )
                if on_epoch_end_flag:
                    self.current_epoch = fit_logs.get('epoch')
                    self.current_logs = self._reformat_fit_logs(fit_logs)
                    # print('\nupdate_state self.current_logs', self.current_epoch, self.current_logs)
                    self._update_log_history()
                    # print('\nupdate_state self._update_log_history', self.log_history)
                    self._update_progress_table(current_epoch_time)
                    # print('\nupdate_state self._update_progress_table', self.progress_table)
                    if self.interactive_config.intermediate_result.autoupdate:
                        self.intermediate_result = self._get_intermediate_result_request()
                    if self.options.data.architecture in self.basic_architecture and \
                            self.interactive_config.statistic_data.output_id \
                            and self.interactive_config.statistic_data.autoupdate:
                        self.statistic_result = self._get_statistic_data_request()
                    if self.options.data.architecture in self.yolo_architecture and  \
                            self.interactive_config.statistic_data.box_channel \
                                and self.interactive_config.statistic_data.autoupdate:
                        self.statistic_result = self._get_statistic_data_request()
                        # print('\nupdate_state self.statistic_result', self.statistic_result)
                    # print('\nupdate_state self._get_loss_graph_data_request()', self._get_loss_graph_data_request())
                    # print('\nupdate_state self._get_metric_graph_data_request()', self._get_metric_graph_data_request())
                else:
                    self.intermediate_result = self._get_intermediate_result_request()
                    # print('\nupdate_state self.intermediate_result', self.intermediate_result)
                    if self.options.data.architecture in self.basic_architecture and \
                            self.interactive_config.statistic_data.output_id:
                        self.statistic_result = self._get_statistic_data_request()
                    if self.options.data.architecture in self.yolo_architecture and \
                            self.interactive_config.statistic_data.box_channel:
                        self.statistic_result = self._get_statistic_data_request()
                self.urgent_predict = False
                self.random_key = ''.join(random.sample(string.ascii_letters + string.digits, 16))
                # print('\nupdate_state self.random_key', self.random_key)
            return {
                'update': self.random_key,
                "class_graphics": self.class_graphics,
                'loss_graphs': self._get_loss_graph_data_request(),
                'metric_graphs': self._get_metric_graph_data_request(),
                'intermediate_result': self.intermediate_result,
                'progress_table': self.progress_table,
                'statistic_data': self.statistic_result,
                'data_balance': self._get_balance_data_request(),
                'addtrain_epochs': self.addtrain_epochs,
            }
        else:
            return {}

    def get_train_results(self, config) -> Union[dict, None]:
        """Return dict with data for current interactive request"""
        self.interactive_config = config if config else self.interactive_config
        if self.log_history and self.log_history.get("epochs", {}):
            if self.options.data.architecture in self.basic_architecture:
                if self.interactive_config.intermediate_result.show_results:
                    out = f"{self.interactive_config.intermediate_result.main_output}"
                    self.example_idx = CreateArray().prepare_example_idx_to_show(
                        array=self.y_true.get("val").get(out),
                        true_array=self.y_true.get("val").get(out),
                        options=self.options,
                        output=int(out),
                        count=self.interactive_config.intermediate_result.num_examples,
                        choice_type=self.interactive_config.intermediate_result.example_choice_type,
                        seed_idx=self.seed_idx[:self.interactive_config.intermediate_result.num_examples]
                    )
                if config.intermediate_result.show_results or config.statistic_data.output_id:
                    self.urgent_predict = True
                    self.intermediate_result = self._get_intermediate_result_request()
                    if self.interactive_config.statistic_data.output_id:
                        self.statistic_result = self._get_statistic_data_request()

            if self.options.data.architecture in self.yolo_architecture:
                if self.interactive_config.intermediate_result.show_results:
                    self.example_idx, _ = CreateArray().prepare_yolo_example_idx_to_show(
                        array=copy.deepcopy(self.y_pred),
                        true_array=copy.deepcopy(self.y_true),
                        name_classes=self.options.data.outputs.get(
                            list(self.options.data.outputs.keys())[0]).classes_names,
                        box_channel=self.interactive_config.intermediate_result.box_channel,
                        count=self.interactive_config.intermediate_result.num_examples,
                        choice_type=self.interactive_config.intermediate_result.example_choice_type,
                        seed_idx=self.seed_idx[:self.interactive_config.intermediate_result.num_examples],
                        sensitivity=self.interactive_config.intermediate_result.sensitivity,
                    )
                if config.intermediate_result.show_results or config.statistic_data.box_channel:
                    self.urgent_predict = True
                    self.intermediate_result = self._get_intermediate_result_request()
                    if self.interactive_config.statistic_data.box_channel:
                        self.statistic_result = self._get_statistic_data_request()

            self.random_key = ''.join(random.sample(string.ascii_letters + string.digits, 16))
            self.train_progress['train_data'] = {
                'update': self.random_key,
                "class_graphics": self.class_graphics,
                'loss_graphs': self._get_loss_graph_data_request(),
                'metric_graphs': self._get_metric_graph_data_request(),
                'intermediate_result': self.intermediate_result,
                'progress_table': self.progress_table,
                'statistic_data': self.statistic_result,
                'data_balance': self._get_balance_data_request(),
                'addtrain_epochs': self.addtrain_epochs,
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
        method_name = '_reformat_metrics'
        try:
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
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    @staticmethod
    def _prepare_loss_obj(losses: dict) -> dict:
        method_name = '_prepare_loss_obj'
        try:
            loss_obj = {}
            for out in losses.keys():
                loss_obj[out] = getattr(
                    importlib.import_module(loss_metric_config.get("loss").get(losses.get(out)).get("module")),
                    losses.get(out)
                )
            return loss_obj
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    @staticmethod
    def _prepare_metric_obj(metrics: dict) -> dict:
        method_name = '_prepare_metric_obj'
        try:
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
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _get_classes_colors(self):
        method_name = '_get_classes_colors'
        try:
            colors = []
            for out in self.options.data.outputs.keys():
                task = self.options.data.outputs.get(out).task
                classes_colors = self.options.data.outputs.get(out).classes_colors
                if task == LayerOutputTypeChoice.TextSegmentation and classes_colors:
                    self.class_colors = [color.as_rgb_tuple() for color in classes_colors]
                elif task == LayerOutputTypeChoice.TextSegmentation or task == LayerOutputTypeChoice.ObjectDetection \
                        and not classes_colors:
                    name_classes = self.options.data.outputs.get(out).classes_names
                    hsv_tuples = [(x / len(name_classes), 1., 1.) for x in range(len(name_classes))]
                    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
                    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
                    self.class_colors = colors
                elif task == LayerOutputTypeChoice.Segmentation:
                    self.class_colors = [color.as_rgb_tuple() for color in classes_colors]
                else:
                    self.class_colors = colors
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    @staticmethod
    def _prepare_x_val(dataset: PrepareDataset):
        method_name = '_prepare_x_val'
        try:
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
                        dataset.data.outputs.get(out).task == LayerOutputTypeChoice.TimeseriesTrend:
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
                    inverse_x = np.zeros_like(x_val.get(input)[:, :, 0:1])
                    for i, column in enumerate(preprocess_dict.keys()):
                        if type(preprocess_dict.get(column)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                            _options = {
                                int(input): {
                                    column: x_val.get(input)[:, :, i]
                                }
                            }
                            inverse_col = np.expand_dims(
                                dataset.preprocessing.inverse_data(_options).get(int(input)).get(column), axis=-1)
                        else:
                            inverse_col = x_val.get(input)[:, :, i:i + 1]
                        inverse_x = np.concatenate([inverse_x, inverse_col], axis=-1)
                    inverse_x_val[input] = inverse_x[:, :, 1:]
            return x_val, inverse_x_val
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _prepare_y_true(self, dataset: PrepareDataset):
        method_name = '_prepare_y_true'
        try:
            y_true = {
                "train": {},
                "val": {}
            }
            inverse_y_true = {
                "train": {},
                "val": {}
            }
            for data_type in y_true.keys():
                if dataset.data.architecture in self.basic_architecture:
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
                                inverse_y = np.concatenate([inverse_y, inverse_col], axis=-1)
                            inverse_y_true[data_type][f"{out}"] = inverse_y[:, 1:]

                        if task == LayerOutputTypeChoice.Timeseries:
                            preprocess_dict = dataset.preprocessing.preprocessing.get(int(out))
                            inverse_y = np.zeros_like(y_true.get(data_type).get(f"{out}")[:, :, 0:1])
                            for i, column in enumerate(preprocess_dict.keys()):
                                if type(preprocess_dict.get(column)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                                    _options = {int(out): {column: y_true.get(data_type).get(f"{out}")[:, :, i]}}
                                    inverse_col = np.expand_dims(
                                        dataset.preprocessing.inverse_data(_options).get(int(out)).get(column), axis=-1)
                                else:
                                    inverse_col = y_true.get(data_type).get(f"{out}")[:, :, i:i + 1]
                                inverse_y = np.concatenate([inverse_y, inverse_col], axis=-1)
                            inverse_y_true[data_type][f"{out}"] = inverse_y[:, :, 1:]

            if dataset.data.architecture in self.yolo_architecture:
                y_true = CreateArray().get_yolo_y_true(options=dataset, dataset_path=self.dataset_path)

            return y_true, inverse_y_true
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _class_metric_list(self):
        method_name = '_class_metric_list'
        try:
            self.class_graphics = {}
            for out in self.options.data.outputs.keys():
                out_task = self.options.data.outputs.get(out).task
                if out_task == LayerOutputTypeChoice.Classification or \
                        out_task == LayerOutputTypeChoice.Segmentation or \
                        out_task == LayerOutputTypeChoice.TextSegmentation or \
                        out_task == LayerOutputTypeChoice.TimeseriesTrend or \
                        out_task == LayerOutputTypeChoice.ObjectDetection:
                    self.class_graphics[out] = True
                else:
                    self.class_graphics[out] = False
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _prepare_null_log_history_template(self):
        method_name = '_prepare_null_log_history_template'
        try:
            self.log_history["epochs"] = []
            if self.options.data.architecture in self.basic_architecture:
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
                            task == LayerOutputTypeChoice.TextSegmentation or task == LayerOutputTypeChoice.TimeseriesTrend:
                        self.log_history[out]["class_loss"] = {}
                        self.log_history[out]["class_metrics"] = {}
                        for class_name in self.options.data.outputs.get(int(out)).classes_names:
                            self.log_history[out]["class_metrics"][f"{class_name}"] = {}
                            self.log_history[out]["class_loss"][f"{class_name}"] = {self.losses.get(out): []}
                            for metric in self.metrics.get(out):
                                self.log_history[out]["class_metrics"][f"{class_name}"][f"{metric}"] = []

            if self.options.data.architecture in self.yolo_architecture:
                self.log_history['learning_rate'] = []
                self.log_history['output'] = {
                    "loss": {
                        'giou_loss': {"train": [], "val": []},
                        'conf_loss': {"train": [], "val": []},
                        'prob_loss': {"train": [], "val": []},
                        'total_loss': {"train": [], "val": []}
                    },
                    "class_loss": {
                        'prob_loss': {},
                    },
                    "metrics": {
                        'mAP50': [],
                        # 'mAP95': [],
                    },
                    "class_metrics": {
                        'mAP50': {},
                        # 'mAP95': {},
                    },
                    "progress_state": {
                        "loss": {
                            'giou_loss': {
                                "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []
                            },
                            'conf_loss': {
                                "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []
                            },
                            'prob_loss': {
                                "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []
                            },
                            'total_loss': {
                                "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []
                            }
                        },
                        "metrics": {
                            'mAP50': {
                                "mean_log_history": [], "normal_state": [], "overfitting": []
                            },
                            # 'mAP95': {
                            #     "mean_log_history": [], "normal_state": [], "overfitting": []
                            # },
                        }
                    }
                }
                out = list(self.options.data.outputs.keys())[0]
                for class_name in self.options.data.outputs.get(out).classes_names:
                    self.log_history['output']["class_loss"]['prob_loss'][class_name] = []
                    self.log_history['output']["class_metrics"]['mAP50'][class_name] = []
                    # self.log_history['output']["class_metrics"]['mAP95'][class_name] = []
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _prepare_dataset_balance(self) -> dict:
        method_name = '_prepare_dataset_balance'
        try:
            dataset_balance = {}
            for out in self.options.data.outputs.keys():
                task = self.options.data.outputs.get(out).task
                print('self.options.data.outputs.get(out)', self.options.data.outputs.get(out))
                encoding = self.options.data.outputs.get(out).encoding

                if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.TimeseriesTrend:
                    print('self.y_true.get(data_type)', self.y_true.get('train').keys(), self.options.data.outputs.get(out).classes_names)
                    print('self.y_true.get(data_type).get(f"{out}")', self.y_true.get('train').get(f"{out}").shape)
                    dataset_balance[f"{out}"] = {'class_histogramm': {}}
                    for data_type in ['train', 'val']:
                        dataset_balance[f"{out}"]['class_histogramm'][data_type] = class_counter(
                            y_array=self.y_true.get(data_type).get(f"{out}"),
                            classes_names=self.options.data.outputs.get(out).classes_names,
                            ohe=encoding == LayerEncodingChoice.ohe
                        )

                if task == LayerOutputTypeChoice.Segmentation and encoding == LayerEncodingChoice.ohe:
                    dataset_balance[f"{out}"] = {
                        "presence_balance": {},
                        "square_balance": {},
                        "colormap": {}
                    }
                    for data_type in ['train', 'val']:
                        dataset_balance[f"{out}"]["colormap"][data_type] = {}
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
                            colormap_path = os.path.join(
                                self.preset_path,
                                f"balance_segmentation_colormap_{data_type}_class_{classes_names[cl]}.webp"
                            )
                            self._get_image_class_colormap(
                                array=self.y_true.get(data_type).get(f"{out}"),
                                colors=self.class_colors,
                                class_id=cl,
                                save_path=colormap_path
                            )
                            dataset_balance[f"{out}"]["colormap"][data_type][classes_names[cl]] = colormap_path

                        for img_array in np.argmax(self.y_true.get(data_type).get(f"{out}"), axis=-1):
                            for cl in classes:
                                if cl in img_array:
                                    class_count[classes_names[cl]] += 1
                        dataset_balance[f"{out}"]["presence_balance"][data_type] = class_count
                        dataset_balance[f"{out}"]["square_balance"][data_type] = class_percent

                if task == LayerOutputTypeChoice.TextSegmentation and encoding == LayerEncodingChoice.ohe or \
                        encoding == LayerEncodingChoice.multi:
                    dataset_balance[f"{out}"] = {
                        "presence_balance": {},
                        "percent_balance": {}
                    }
                    for data_type in ['train', 'val']:
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
                        dataset_balance[f"{out}"]["presence_balance"][data_type] = class_count
                        dataset_balance[f"{out}"]["percent_balance"][data_type] = class_percent

                if task == LayerOutputTypeChoice.Timeseries:
                    dataset_balance[f"{out}"] = {
                        'graphic': {},
                        'dense_histogram': {}
                    }
                    for output_channel in self.options.data.columns.get(out).keys():
                        dataset_balance[f"{out}"]['graphic'][output_channel] = {}
                        dataset_balance[f"{out}"]['dense_histogram'][output_channel] = {}
                        for data_type in ['train', 'val']:
                            dataset_balance[f"{out}"]['graphic'][output_channel][data_type] = {
                                "type": "graphic",
                                "x": np.array(self.options.dataframe.get(data_type).index).astype('float').tolist(),
                                "y": np.array(self.options.dataframe.get(data_type)[output_channel]).astype(
                                    'float').tolist()
                            }
                            x, y = self._get_distribution_histogram(
                                list(self.options.dataframe.get(data_type)[output_channel]),
                                categorical=False
                            )
                            dataset_balance[f"{out}"]['dense_histogram'][output_channel][data_type] = {
                                "type": "bar",
                                "x": x,
                                "y": y
                            }

                if task == LayerOutputTypeChoice.Regression:
                    dataset_balance[f"{out}"] = {'histogram': {}, 'correlation': {}}
                    # print('\nself.options.dataframe.keys()', list(self.options.dataframe.get('train').columns))
                    for data_type in ['train', 'val']:
                        dataset_balance[f"{out}"]['histogram'][data_type] = {}
                        for column in list(self.options.dataframe.get('train').columns):
                            column_id = int(column.split("_")[0])
                            column_task = self.options.data.columns.get(column_id).get(column).get('task')
                            column_data = list(self.options.dataframe.get(data_type)[column])
                            # print('\n--column', column, column_task, column_data)
                            if column_task == LayerInputTypeChoice.Text:
                                continue
                            elif column_task == LayerInputTypeChoice.Classification or \
                                    len(set(column_data)) < MAX_HISTOGRAM_BINS:
                                x, y = self._get_distribution_histogram(column_data, categorical=True)
                                hist_type = "histogram"
                            else:
                                x, y = self._get_distribution_histogram(column_data, categorical=False)
                                hist_type = "bar"
                            # print('\n--column', column, column_task, '\n', x, '\n', y)
                            dataset_balance[f"{out}"]['histogram'][data_type][column] = {
                                "name": column.split("_", 1)[-1],
                                "type": hist_type,
                                "x": x,
                                "y": y
                            }
                    for data_type in ['train', 'val']:
                        labels, matrix = self._get_correlation_matrix(
                            pd.DataFrame(self.options.dataframe.get(data_type))
                        )
                        dataset_balance[f"{out}"]['correlation'][data_type] = {
                            "labels": labels,
                            "matrix": matrix
                        }

                if task == LayerOutputTypeChoice.ObjectDetection:
                    name_classes = self.options.data.outputs.get(
                        list(self.options.data.outputs.keys())[0]).classes_names
                    imsize = self.options.data.inputs.get(list(self.options.data.inputs.keys())[0]).shape
                    class_bb = {}
                    dataset_balance["output"] = {
                        'class_count': {},
                        'class_square': {},
                        'colormap': {}
                    }
                    for data_type in ["train", "val"]:
                        class_bb[data_type] = {}
                        for cl in range(len(name_classes)):
                            class_bb[data_type][cl] = []
                        for index in range(len(self.options.dataframe[data_type])):
                            y_true = self.options.dataframe.get(data_type).iloc[index, 1].split(' ')
                            bbox_data_gt = np.array([list(map(int, box.split(','))) for box in y_true])
                            bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                            bboxes_gt = np.concatenate(
                                [bboxes_gt[:, 1:2], bboxes_gt[:, 0:1], bboxes_gt[:, 3:4], bboxes_gt[:, 2:3]], axis=-1)
                            for i, cl in enumerate(classes_gt):
                                class_bb[data_type][cl].append(bboxes_gt[i].tolist())

                        dataset_balance["output"]['class_count'][data_type] = {}
                        dataset_balance["output"]['class_square'][data_type] = {}
                        for key, item in class_bb[data_type].items():
                            dataset_balance["output"]['class_count'][data_type][name_classes[key]] = len(item)
                            dataset_balance["output"]['class_square'][data_type][name_classes[key]] = \
                                self._round_loss_metric(self._get_box_square(item, imsize=(imsize[0], imsize[1])))

                        dataset_balance["output"]['colormap'][data_type] = self._plot_bb_colormap(
                            class_bb=class_bb,
                            colors=self.class_colors,
                            name_classes=name_classes,
                            data_type=data_type,
                            save_path=self.preset_path,
                            imgsize=(imsize[0], imsize[1])
                        )
                    break

            return dataset_balance
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    @staticmethod
    def _get_box_square(bbs, imsize=(416, 416)):
        method_name = '_get_box_square'
        try:
            if len(bbs):
                square = 0
                for bb in bbs:
                    square += (bb[2] - bb[0]) * (bb[3] - bb[1])
                return square / len(bbs) / np.prod(imsize) * 100
            else:
                return 0.
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    @staticmethod
    def _plot_bb_colormap(class_bb: dict, colors: list, name_classes: list, data_type: str,
                          save_path: str, imgsize=(416, 416)):
        method_name = '_plot_bb_colormap'
        try:
            template = np.zeros((imgsize[0], imgsize[1], 3))
            link_dict = {}
            total_len = 0
            for class_idx in class_bb[data_type].keys():
                total_len += len(class_bb[data_type][class_idx])
                class_template = np.zeros((imgsize[0], imgsize[1], 3))
                for box in class_bb[data_type][class_idx]:
                    template[box[0]:box[2], box[1]:box[3], :] += np.array(colors[class_idx])
                    class_template[box[0]:box[2], box[1]:box[3], :] += np.array(colors[class_idx])
                class_template = class_template / len(class_bb[data_type][class_idx])
                class_template = (class_template * 255 / class_template.max()).astype("uint8")
                img_save_path = os.path.join(
                    save_path, f"image_{data_type}_od_balance_colormap_class_{name_classes[class_idx]}.webp"
                )
                link_dict[name_classes[class_idx]] = img_save_path
                matplotlib.image.imsave(img_save_path, class_template)

            template = template / total_len
            template = (template * 255 / template.max()).astype('uint8')
            img_save_path = os.path.join(save_path, f"image_{data_type}_od_balance_colormap_all_classes.webp")
            link_dict['all_classes'] = img_save_path
            matplotlib.image.imsave(img_save_path, template)
            return link_dict
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _prepare_class_idx(self) -> dict:
        method_name = '_prepare_class_idx'
        try:
            class_idx = {}
            if self.options.data.architecture in self.basic_architecture:
                for data_type in self.y_true.keys():
                    class_idx[data_type] = {}
                    for out in self.y_true.get(data_type).keys():
                        class_idx[data_type][out] = {}
                        task = self.options.data.outputs.get(int(out)).task
                        if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.TimeseriesTrend:
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
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _prepare_seed(self):
        method_name = '_prepare_seed'
        try:
            if self.options.data.architecture in self.yolo_architecture:
                # output = self.yolo_interactive_config.intermediate_result.box_channel
                example_idx = np.arange(len(self.options.dataframe.get("val")))
                np.random.shuffle(example_idx)
            elif self.options.data.architecture in self.basic_architecture:
                output = self.interactive_config.intermediate_result.main_output
                # print('self.options.data.outputs.get(output)', self.options.data.outputs.get(output))
                task = self.options.data.outputs.get(output).task
                example_idx = []

                if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.TimeseriesTrend:
                    y_true = np.argmax(self.y_true.get('val').get(f"{output}"), axis=-1)
                    class_idx = {}
                    for _id in range(self.options.data.outputs.get(output).num_classes):
                        class_idx[_id] = []
                    for i, _id in enumerate(y_true):
                        class_idx[_id].append(i)
                    for key in class_idx.keys():
                        np.random.shuffle(class_idx[key])
                    num_ex = 25
                    while num_ex:
                        key = np.random.choice(list(class_idx.keys()))
                        if not class_idx.get(key):
                            class_idx.pop(key)
                            key = np.random.choice(list(class_idx.keys()))
                        example_idx.append(class_idx[key][0])
                        class_idx[key].pop(0)
                        num_ex -= 1
                else:
                    if self.options.data.group == DatasetGroupChoice.keras or self.x_val:
                        example_idx = np.arange(len(self.y_true.get("val").get(list(self.y_true.get("val").keys())[0])))
                    else:
                        example_idx = np.arange(len(self.options.dataframe.get("val")))
                    np.random.shuffle(example_idx)
            else:
                example_idx = np.arange(len(self.options.dataframe.get("val")))
                np.random.shuffle(example_idx)
            return example_idx
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    # Методы для update_state()
    @staticmethod
    def _round_loss_metric(x):
        method_name = '_round_loss_metric'
        try:
            if not x:
                return x
            elif math.isnan(float(x)):
                return None
            elif x > 1000:
                return np.round(x, 0).item()
            elif x > 1:
                return np.round(x, -int(math.floor(math.log10(abs(x))) - 3)).item()
            else:
                return np.round(x, -int(math.floor(math.log10(abs(x))) - 2)).item()
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _reformat_fit_logs(self, logs) -> dict:
        method_name = '_reformat_fit_logs'
        try:
            interactive_log = {}
            if self.options.data.architecture in self.basic_architecture:
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
                            'train': self._round_loss_metric(train_loss) if not math.isnan(float(train_loss)) else None,
                            'val': self._round_loss_metric(val_loss) if not math.isnan(float(val_loss)) else None,
                        }
                    }

                    interactive_log[out]['metrics'] = {}
                    for metric_name in self.metrics.get(out):
                        interactive_log[out]['metrics'][metric_name] = {}
                        if len(self.metrics.keys()) == 1:
                            train_metric = update_logs.get(
                                loss_metric_config.get('metric').get(metric_name).get('log_name'))
                            val_metric = update_logs.get(
                                f"val_{loss_metric_config.get('metric').get(metric_name).get('log_name')}")
                        else:
                            train_metric = update_logs.get(
                                f"{out}_{loss_metric_config.get('metric').get(metric_name).get('log_name')}")
                            val_metric = update_logs.get(
                                f"val_{out}_{loss_metric_config.get('metric').get(metric_name).get('log_name')}")

                        if metric_name == MetricChoice.UnscaledMAE:
                            train_metric, val_metric = UnscaledMAE().unscale_result(
                                [train_metric, val_metric], int(out), self.options.preprocessing
                            )
                        if metric_name == MetricChoice.BalancedRecall:
                            m = BalancedRecall()
                            m.update_state(y_true=self.y_true.get('val').get(out), y_pred=self.y_pred.get(out))
                            val_metric = m.result().numpy().item()
                            # print('\nBalancedRecall', val_metric, update_logs.get(
                            #     f"val_{loss_metric_config.get('metric').get(metric_name).get('log_name')}"))
                        if metric_name == MetricChoice.BalancedPrecision:
                            m = BalancedPrecision()
                            m.update_state(y_true=self.y_true.get('val').get(out), y_pred=self.y_pred.get(out))
                            val_metric = m.result().numpy().item()
                            # print('BalancedPrecision', val_metric, update_logs.get(
                            #     f"val_{loss_metric_config.get('metric').get(metric_name).get('log_name')}"))
                        if metric_name == MetricChoice.BalancedFScore:
                            m = BalancedFScore()
                            m.update_state(y_true=self.y_true.get('val').get(out), y_pred=self.y_pred.get(out))
                            val_metric = m.result().numpy().item()
                            # print('BalancedFScore', val_metric, update_logs.get(
                            #     f"val_{loss_metric_config.get('metric').get(metric_name).get('log_name')}"))
                        if metric_name == MetricChoice.FScore:
                            m = FScore()
                            m.update_state(y_true=self.y_true.get('val').get(out), y_pred=self.y_pred.get(out))
                            val_metric = m.result().numpy().item()
                            # print('FScore', val_metric, update_logs.get(
                            #     f"val_{loss_metric_config.get('metric').get(metric_name).get('log_name')}"))
                        interactive_log[out]['metrics'][metric_name] = {
                            'train': self._round_loss_metric(train_metric) if not math.isnan(
                                float(train_metric)) else None,
                            'val': self._round_loss_metric(val_metric) if not math.isnan(float(val_metric)) else None
                        }

            if self.options.data.architecture in self.yolo_architecture:
                # self._round_loss_metric(train_loss) if not math.isnan(float(train_loss)) else None
                interactive_log['learning_rate'] = self._round_loss_metric(logs.get('optimizer.lr'))
                interactive_log['output'] = {
                    "train": {
                        "loss": {
                            'giou_loss': self._round_loss_metric(logs.get('giou_loss')),
                            'conf_loss': self._round_loss_metric(logs.get('conf_loss')),
                            'prob_loss': self._round_loss_metric(logs.get('prob_loss')),
                            'total_loss': self._round_loss_metric(logs.get('total_loss'))
                        },
                        "metrics": {
                            'mAP50': self._round_loss_metric(logs.get('mAP50')),
                            # 'mAP95': logs.get('mAP95'),
                        }
                    },
                    "val": {
                        "loss": {
                            'giou_loss': self._round_loss_metric(logs.get('val_giou_loss')),
                            'conf_loss': self._round_loss_metric(logs.get('val_conf_loss')),
                            'prob_loss': self._round_loss_metric(logs.get('val_prob_loss')),
                            'total_loss': self._round_loss_metric(logs.get('val_total_loss'))
                        },
                        "class_loss": {
                            'prob_loss': {},
                        },
                        "metrics": {
                            'mAP50': self._round_loss_metric(logs.get('val_mAP50')),
                            # 'mAP95': logs.get('val_mAP95'),
                        },
                        "class_metrics": {
                            'mAP50': {},
                            # 'mAP95': {},
                        }
                    }
                }
                for name in self.options.data.outputs.get(list(self.options.data.outputs.keys())[0]).classes_names:
                    interactive_log['output']['val']["class_loss"]['prob_loss'][name] = self._round_loss_metric(
                        logs.get(
                            f'val_prob_loss_{name}'))
                    interactive_log['output']['val']["class_metrics"]['mAP50'][name] = self._round_loss_metric(logs.get(
                        f'val_mAP50_class_{name}'))
                    # interactive_log['output']['val']["class_metrics"]['mAP95'][name] = logs.get(f'val_mAP95_class_{name}')

            return interactive_log

        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _reformat_y_pred(self, y_pred, sensitivity: float = 0.15, threashold: float = 0.1):
        method_name = '_reformat_y_pred'
        try:
            self.y_pred = {}
            self.inverse_y_pred = {}
            if self.options.data.architecture in self.basic_architecture:
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
                                inverse_col = self.options.preprocessing.inverse_data(_options).get(int(out)).get(
                                    column)
                            else:
                                inverse_col = self.y_pred.get(out)[:, i:i + 1]
                            inverse_y = np.concatenate([inverse_y, inverse_col], axis=-1)
                        self.inverse_y_pred[out] = inverse_y[:, 1:]

                    if task == LayerOutputTypeChoice.Regression.Timeseries:
                        preprocess_dict = self.options.preprocessing.preprocessing.get(int(out))
                        inverse_y = np.zeros_like(self.y_pred.get(out)[:, :, 0:1])
                        for i, column in enumerate(preprocess_dict.keys()):
                            if type(preprocess_dict.get(column)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                                _options = {int(out): {column: self.y_pred.get(out)[:, :, i]}}
                                inverse_col = np.expand_dims(
                                    self.options.preprocessing.inverse_data(_options).get(int(out)).get(column),
                                    axis=-1)
                            else:
                                inverse_col = self.y_pred.get(out)[:, :, i:i + 1]
                            inverse_y = np.concatenate([inverse_y, inverse_col], axis=-1)
                        self.inverse_y_pred[out] = inverse_y[:, :, 1:]

            if self.options.data.architecture in self.yolo_architecture:
                self.y_pred = CreateArray().get_yolo_y_pred(
                    array=y_pred,
                    options=self.options,
                    sensitivity=sensitivity,
                    threashold=threashold
                )
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _update_log_history(self):
        method_name = '_update_log_history'
        try:
            data_idx = None
            if self.log_history:
                if self.current_epoch in self.log_history['epochs']:
                    data_idx = self.log_history['epochs'].index(self.current_epoch)
                else:
                    self.log_history['epochs'].append(self.current_epoch)

                if self.options.data.architecture in self.basic_architecture:
                    for out in self.options.data.outputs.keys():
                        out_task = self.options.data.outputs.get(out).task
                        classes_names = self.options.data.outputs.get(out).classes_names
                        for loss_name in self.log_history.get(f"{out}").get('loss').keys():
                            for data_type in ['train', 'val']:
                                # fill losses
                                if data_idx or data_idx == 0:
                                    self.log_history[f"{out}"]['loss'][loss_name][data_type][data_idx] = \
                                        self._round_loss_metric(
                                            self.current_logs.get(f"{out}").get('loss').get(loss_name).get(data_type)
                                        )
                                else:
                                    self.log_history[f"{out}"]['loss'][loss_name][data_type].append(
                                        self._round_loss_metric(
                                            self.current_logs.get(f"{out}").get('loss').get(loss_name).get(data_type)
                                        )
                                    )
                            # fill loss progress state
                            if data_idx or data_idx == 0:
                                self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['mean_log_history'][
                                    data_idx] = \
                                    self._get_mean_log(
                                        self.log_history.get(f"{out}").get('loss').get(loss_name).get('val'))
                            else:
                                self.log_history[f"{out}"]['progress_state']['loss'][loss_name][
                                    'mean_log_history'].append(
                                    self._get_mean_log(
                                        self.log_history.get(f"{out}").get('loss').get(loss_name).get('val'))
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
                                self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['underfitting'][
                                    data_idx] = \
                                    loss_underfitting
                                self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['overfitting'][
                                    data_idx] = \
                                    loss_overfitting
                                self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['normal_state'][
                                    data_idx] = \
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
                                    out_task == LayerOutputTypeChoice.TimeseriesTrend:
                                for cls in self.log_history.get(f"{out}").get('class_loss').keys():
                                    class_loss = 0.
                                    if out_task == LayerOutputTypeChoice.Classification or \
                                            out_task == LayerOutputTypeChoice.TimeseriesTrend:
                                        class_loss = self._get_loss_calculation(
                                            loss_obj=self.loss_obj.get(f"{out}"),
                                            out=f"{out}",
                                            y_true=self.y_true.get('val').get(f"{out}")[
                                                self.class_idx.get('val').get(f"{out}").get(cls)],
                                            y_pred=self.y_pred.get(f"{out}")[
                                                self.class_idx.get('val').get(f"{out}").get(cls)],
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
                                        self.log_history[f"{out}"]['class_loss'][cls][loss_name][data_idx] = \
                                            self._round_loss_metric(class_loss)
                                    else:
                                        self.log_history[f"{out}"]['class_loss'][cls][loss_name].append(
                                            self._round_loss_metric(class_loss)
                                        )

                        for metric_name in self.log_history.get(f"{out}").get('metrics').keys():
                            for data_type in ['train', 'val']:
                                # fill metrics
                                if data_idx or data_idx == 0:
                                    if self.current_logs:
                                        self.log_history[f"{out}"]['metrics'][metric_name][data_type][data_idx] = \
                                            self._round_loss_metric(
                                                self.current_logs.get(f"{out}").get('metrics').get(metric_name).get(
                                                    data_type)
                                            )
                                else:
                                    if self.current_logs:
                                        self.log_history[f"{out}"]['metrics'][metric_name][data_type].append(
                                            self._round_loss_metric(
                                                self.current_logs.get(f"{out}").get('metrics').get(metric_name).get(
                                                    data_type)
                                            )
                                        )

                            if data_idx or data_idx == 0:
                                self.log_history[f"{out}"]['progress_state']['metrics'][metric_name][
                                    'mean_log_history'][
                                    data_idx] = \
                                    self._get_mean_log(self.log_history[f"{out}"]['metrics'][metric_name]['val'])
                            else:
                                self.log_history[f"{out}"]['progress_state']['metrics'][metric_name][
                                    'mean_log_history'].append(
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
                                self.log_history[f"{out}"]['progress_state']['metrics'][metric_name][
                                    'mean_log_history'],
                                metric_type='metric'
                            )
                            if metric_underfittng or metric_overfittng:
                                normal_state = False
                            else:
                                normal_state = True

                            if data_idx or data_idx == 0:
                                self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['underfitting'][
                                    data_idx] = \
                                    metric_underfittng
                                self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['overfitting'][
                                    data_idx] = \
                                    metric_overfittng
                                self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['normal_state'][
                                    data_idx] = \
                                    normal_state
                            else:
                                self.log_history[f"{out}"]['progress_state']['metrics'][metric_name][
                                    'underfitting'].append(
                                    metric_underfittng)
                                self.log_history[f"{out}"]['progress_state']['metrics'][metric_name][
                                    'overfitting'].append(
                                    metric_overfittng)
                                self.log_history[f"{out}"]['progress_state']['metrics'][metric_name][
                                    'normal_state'].append(
                                    normal_state)

                            if out_task == LayerOutputTypeChoice.Classification or \
                                    out_task == LayerOutputTypeChoice.Segmentation or \
                                    out_task == LayerOutputTypeChoice.TextSegmentation or \
                                    out_task == LayerOutputTypeChoice.TimeseriesTrend:
                                for cls in self.log_history.get(f"{out}").get('class_metrics').keys():
                                    class_metric = 0.
                                    if out_task == LayerOutputTypeChoice.Classification or \
                                            out_task == LayerOutputTypeChoice.TimeseriesTrend:
                                        class_metric = self._get_metric_calculation(
                                            metric_name=metric_name,
                                            metric_obj=self.metrics_obj.get(f"{out}").get(metric_name),
                                            out=f"{out}",
                                            y_true=self.y_true.get('val').get(f"{out}")[
                                                self.class_idx.get('val').get(f"{out}").get(cls)],
                                            y_pred=self.y_pred.get(f"{out}")[
                                                self.class_idx.get('val').get(f"{out}").get(cls)],
                                            show_class=True
                                        )
                                    if out_task == LayerOutputTypeChoice.Segmentation or \
                                            out_task == LayerOutputTypeChoice.TextSegmentation:
                                        class_idx = classes_names.index(cls)
                                        class_metric = self._get_metric_calculation(
                                            metric_name=metric_name,
                                            metric_obj=self.metrics_obj.get(f"{out}").get(metric_name),
                                            out=f"{out}",
                                            y_true=self.y_true.get('val').get(f"{out}")[..., class_idx:class_idx + 1],
                                            y_pred=self.y_pred.get(f"{out}")[..., class_idx:class_idx + 1],
                                        )
                                    # if out_task == LayerOutputTypeChoice.TextSegmentation:
                                    #     class_idx = classes_names.index(cls)
                                    #     class_metric = self._get_metric_calculation(
                                    #         metric_name=metric_name,
                                    #         metric_obj=self.metrics_obj.get(f"{out}").get(metric_name),
                                    #         out=f"{out}",
                                    #         y_true=self.y_true.get('val').get(f"{out}")[:, :, class_idx],
                                    #         y_pred=self.y_pred.get(f"{out}")[:, :, class_idx],
                                    #     )
                                    if data_idx or data_idx == 0:
                                        self.log_history[f"{out}"]['class_metrics'][cls][metric_name][data_idx] = \
                                            self._round_loss_metric(class_metric)
                                    else:
                                        self.log_history[f"{out}"]['class_metrics'][cls][metric_name].append(
                                            self._round_loss_metric(class_metric)
                                        )

                if self.options.data.architecture in self.yolo_architecture:
                    self.log_history['learning_rate'] = self.current_logs.get('learning_rate')
                    out = list(self.options.data.outputs.keys())[0]
                    classes_names = self.options.data.outputs.get(out).classes_names
                    for key in self.log_history['output']["loss"].keys():
                        for data_type in ['train', 'val']:
                            self.log_history['output']["loss"][key][data_type].append(
                                self._round_loss_metric(self.current_logs.get('output').get(
                                    data_type).get('loss').get(key))
                            )
                    for key in self.log_history['output']["metrics"].keys():
                        self.log_history['output']["metrics"][key].append(
                            self._round_loss_metric(self.current_logs.get('output').get(
                                'val').get('metrics').get(key))
                        )
                    for name in classes_names:
                        self.log_history['output']["class_loss"]['prob_loss'][name].append(
                            self._round_loss_metric(self.current_logs.get('output').get("val").get(
                                'class_loss').get("prob_loss").get(name))
                        )
                        self.log_history['output']["class_metrics"]['mAP50'][name].append(
                            self._round_loss_metric(self.current_logs.get('output').get("val").get(
                                'class_metrics').get("mAP50").get(name))
                        )
                        # self.log_history['output']["class_metrics"]['mAP95'][name].append(
                        #     self._round_loss_metric(self.current_logs.get('output').get("val").get(
                        #         'class_metrics').get("mAP95").get(name))
                        # )
                    for loss_name in self.log_history['output']["loss"].keys():
                        # fill loss progress state
                        if data_idx or data_idx == 0:
                            self.log_history['output']['progress_state']['loss'][loss_name]['mean_log_history'][
                                data_idx] = \
                                self._get_mean_log(self.log_history.get('output').get('loss').get(loss_name).get('val'))
                        else:
                            self.log_history['output']['progress_state']['loss'][loss_name]['mean_log_history'].append(
                                self._get_mean_log(self.log_history.get('output').get('loss').get(loss_name).get('val'))
                            )
                        # get progress state data
                        loss_underfitting = self._evaluate_underfitting(
                            loss_name,
                            self.log_history['output']['loss'][loss_name]['train'][-1],
                            self.log_history['output']['loss'][loss_name]['val'][-1],
                            metric_type='loss'
                        )
                        loss_overfitting = self._evaluate_overfitting(
                            loss_name,
                            self.log_history['output']['progress_state']['loss'][loss_name]['mean_log_history'],
                            metric_type='loss'
                        )
                        if loss_underfitting or loss_overfitting:
                            normal_state = False
                        else:
                            normal_state = True
                        if data_idx or data_idx == 0:
                            self.log_history['output']['progress_state']['loss'][loss_name]['underfitting'][data_idx] = \
                                loss_underfitting
                            self.log_history['output']['progress_state']['loss'][loss_name]['overfitting'][data_idx] = \
                                loss_overfitting
                            self.log_history['output']['progress_state']['loss'][loss_name]['normal_state'][data_idx] = \
                                normal_state
                        else:
                            self.log_history['output']['progress_state']['loss'][loss_name]['underfitting'].append(
                                loss_underfitting)
                            self.log_history['output']['progress_state']['loss'][loss_name]['overfitting'].append(
                                loss_overfitting)
                            self.log_history['output']['progress_state']['loss'][loss_name]['normal_state'].append(
                                normal_state)
                    for metric_name in self.log_history.get('output').get('metrics').keys():
                        if data_idx or data_idx == 0:
                            self.log_history['output']['progress_state']['metrics'][metric_name]['mean_log_history'][
                                data_idx] = self._get_mean_log(self.log_history['output']['metrics'][metric_name])
                        else:
                            self.log_history['output']['progress_state']['metrics'][metric_name][
                                'mean_log_history'].append(
                                self._get_mean_log(self.log_history['output']['metrics'][metric_name])
                            )
                        metric_overfitting = self._evaluate_overfitting(
                            metric_name,
                            self.log_history['output']['progress_state']['metrics'][metric_name]['mean_log_history'],
                            metric_type='metric'
                        )
                        if metric_overfitting:
                            normal_state = False
                        else:
                            normal_state = True
                        if data_idx or data_idx == 0:
                            self.log_history['output']['progress_state']['metrics'][metric_name]['overfitting'][
                                data_idx] = metric_overfitting
                            self.log_history['output']['progress_state']['metrics'][metric_name]['normal_state'][
                                data_idx] = normal_state
                        else:
                            self.log_history['output']['progress_state']['metrics'][metric_name]['overfitting'].append(
                                metric_overfitting)
                            self.log_history['output']['progress_state']['metrics'][metric_name]['normal_state'].append(
                                normal_state)
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _update_progress_table(self, epoch_time: float):
        method_name = '_update_progress_table'
        try:
            if self.options.data.architecture in self.basic_architecture:
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
                        'loss': f"{self.log_history.get(out).get('loss').get(self.losses.get(out)).get('train')[-1]}",
                        'val_loss': f"{self.log_history.get(out).get('loss').get(self.losses.get(out)).get('val')[-1]}"
                    }
                    for metric in self.metrics.get(out):
                        self.progress_table[self.current_epoch]["data"][f"Выходной слой «{out}»"]["metrics"][metric] = \
                            f"{self.log_history.get(out).get('metrics').get(metric).get('train')[-1]}"
                        self.progress_table[self.current_epoch]["data"][f"Выходной слой «{out}»"]["metrics"][
                            f"val_{metric}"] = \
                            f"{self.log_history.get(out).get('metrics').get(metric).get('val')[-1]}"

            if self.options.data.architecture in self.yolo_architecture:
                self.progress_table[self.current_epoch] = {
                    "time": epoch_time,
                    "learning_rate": self.current_logs.get("learning_rate"),
                    "data": {f"Прогресс обучения": {'loss': {}, 'metrics': {}}}
                }
                # print('\n_update_progress_table self.log_history', self.log_history['output']["loss"])
                for loss in self.log_history['output']["loss"].keys():
                    self.progress_table[self.current_epoch]["data"]["Прогресс обучения"]["loss"][f'{loss}'] = \
                        f"{self.log_history.get('output').get('loss').get(loss).get('train')[-1]}"
                    self.progress_table[self.current_epoch]["data"]["Прогресс обучения"]["loss"][f'val_{loss}'] = \
                        f"{self.log_history.get('output').get('loss').get(loss).get('val')[-1]}"
                # print('\n self.progress_table[self.current_epoch]', self.progress_table[self.current_epoch])
                # print('\n_update_progress_table self.log_history', self.log_history['output']["metrics"])
                for metric in self.log_history['output']["metrics"].keys():
                    # print(metric)
                    # self.progress_table[self.current_epoch]["data"][f"Прогресс обучения»"]["metrics"][metric] = \
                    #     f"{self.log_history.get('output').get('metrics').get(metric)[-1]}"
                    self.progress_table[self.current_epoch]["data"]["Прогресс обучения"]["metrics"][f"{metric}"] = \
                        f"{self.log_history.get('output').get('metrics').get(metric)[-1]}"
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _get_loss_calculation(self, loss_obj, out: str, y_true, y_pred):
        method_name = '_get_loss_calculation'
        try:
            encoding = self.options.data.outputs.get(int(out)).encoding
            task = self.options.data.outputs.get(int(out)).task
            num_classes = self.options.data.outputs.get(int(out)).num_classes
            if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.TimeseriesTrend:

                loss_value = float(loss_obj()(
                    y_true if encoding == LayerEncodingChoice.ohe else to_categorical(y_true, num_classes), y_pred
                ).numpy())
            elif task == LayerOutputTypeChoice.Segmentation or \
                    (task == LayerOutputTypeChoice.TextSegmentation and encoding == LayerEncodingChoice.ohe):
                loss_value = float(loss_obj()(
                    y_true if encoding == LayerEncodingChoice.ohe else to_categorical(y_true, num_classes), y_pred
                ).numpy())
            elif task == LayerOutputTypeChoice.TextSegmentation and encoding == LayerEncodingChoice.multi:
                loss_value = float(loss_obj()(y_true, y_pred).numpy())
            elif task == LayerOutputTypeChoice.Regression or task == LayerOutputTypeChoice.Timeseries:
                loss_value = float(loss_obj()(y_true, y_pred).numpy())
            else:
                loss_value = 0.
            return loss_value if not math.isnan(loss_value) else None
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _get_metric_calculation(self, metric_name, metric_obj, out: str, y_true, y_pred, show_class=False):
        method_name = '_get_metric_calculation'
        try:
            encoding = self.options.data.outputs.get(int(out)).encoding
            task = self.options.data.outputs.get(int(out)).task
            num_classes = self.options.data.outputs.get(int(out)).num_classes
            if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.TimeseriesTrend:
                if metric_name == Metric.Accuracy:
                    metric_obj.update_state(
                        np.argmax(y_true, axis=-1) if encoding == LayerEncodingChoice.ohe else y_true,
                        np.argmax(y_pred, axis=-1)
                    )
                elif metric_name in [Metric.BalancedRecall, Metric.BalancedPrecision, Metric.BalancedFScore]:
                    metric_obj.update_state(
                        y_true if encoding == LayerEncodingChoice.ohe else to_categorical(y_true, num_classes),
                        y_pred,
                        show_class=show_class
                    )
                else:
                    metric_obj.update_state(
                        y_true if encoding == LayerEncodingChoice.ohe else to_categorical(y_true, num_classes),
                        y_pred
                    )
                metric_value = float(metric_obj.result().numpy())
            elif task == LayerOutputTypeChoice.Segmentation or task == LayerOutputTypeChoice.TextSegmentation:
                if metric_name == Metric.BalancedDiceCoef:
                    metric_obj.encoding = None
                    metric_obj.update_state(
                        y_true if (encoding == LayerEncodingChoice.ohe or encoding == LayerEncodingChoice.multi)
                        else to_categorical(y_true, num_classes),
                        y_pred
                    )
                else:
                    metric_obj.update_state(
                        y_true if (encoding == LayerEncodingChoice.ohe or encoding == LayerEncodingChoice.multi)
                        else to_categorical(y_true, num_classes),
                        y_pred
                    )
                metric_value = float(metric_obj.result().numpy())
            # elif task == LayerOutputTypeChoice.TextSegmentation and encoding == LayerEncodingChoice.multi:
            #     metric_obj.update_state(y_true, y_pred)
            #     metric_value = float(metric_obj.result().numpy())
            elif task == LayerOutputTypeChoice.Regression or task == LayerOutputTypeChoice.Timeseries:
                metric_obj.update_state(y_true, y_pred)
                metric_value = float(metric_obj.result().numpy())
            else:
                metric_value = 0.
            return round(metric_value, 6) if not math.isnan(metric_value) else None
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _get_mean_log(self, logs):
        method_name = '_get_mean_log'
        # print('\n_get_mean_log', logs)
        try:
            copy_logs = copy.deepcopy(logs)
            for i, x in enumerate(copy_logs):
                if not x:
                    copy_logs[i] = 0.
            if len(copy_logs) < self.log_gap:
                return float(np.mean(copy_logs))
            else:
                return float(np.mean(copy_logs[-self.log_gap:]))
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    @staticmethod
    def _evaluate_overfitting(metric_name: str, mean_log: list, metric_type: str):
        method_name = '_evaluate_overfitting'
        try:
            mode = loss_metric_config.get(metric_type).get(metric_name).get("mode")
            overfitting = False
            if mode == 'min':
                if min(mean_log) and mean_log[-1] and mean_log[-1] > min(mean_log) and \
                        (mean_log[-1] - min(mean_log)) * 100 / min(mean_log) > 2:
                    overfitting = True
            if mode == 'max':
                if max(mean_log) and mean_log[-1] and mean_log[-1] < max(mean_log) and \
                        (max(mean_log) - mean_log[-1]) * 100 / max(mean_log) > 2:
                    overfitting = True
            return overfitting
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    @staticmethod
    def _evaluate_underfitting(metric_name: str, train_log: float, val_log: float, metric_type: str):
        method_name = '_evaluate_underfitting'
        try:
            mode = loss_metric_config.get(metric_type).get(metric_name).get("mode")
            if train_log and val_log:
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
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    # Методы для конечных данных для вывода
    @staticmethod
    def _fill_graph_plot_data(x: list, y: list, label=None):
        return {'label': label, 'x': x, 'y': y}

    @staticmethod
    def _fill_graph_front_structure(_id: int, _type: str, graph_name: str, short_name: str,
                                    x_label: str, y_label: str, plot_data: list, best: list = None,
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
            'best': best,
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
        self.first = True
        method_name = '_get_loss_graph_data_request'
        try:
            data_return = []
            # config = self.interactive_config
            if self.options.data.architecture in self.basic_architecture:
                if not self.interactive_config.loss_graphs or not self.log_history.get("epochs"):
                    return data_return
                for loss_graph_config in self.interactive_config.loss_graphs:
                    loss = self.losses.get(f"{loss_graph_config.output_idx}")
                    if self.options.data.architecture in self.yolo_architecture:
                        loss_graph_config.output_idx = 'output'
                    if loss_graph_config.show == LossGraphShowChoice.model:
                        if sum(self.log_history.get(f"{loss_graph_config.output_idx}").get("progress_state").get(
                                "loss").get(loss).get('overfitting')[-self.log_gap:]) >= self.progress_threashold:
                            progress_state = "overfitting"
                        elif sum(self.log_history.get(f"{loss_graph_config.output_idx}").get("progress_state").get(
                                "loss").get(loss).get('underfitting')[-self.log_gap:]) >= self.progress_threashold:
                            progress_state = "underfitting"
                        else:
                            progress_state = "normal"

                        train_list = self.log_history.get(f"{loss_graph_config.output_idx}").get('loss').get(
                            self.losses.get(f"{loss_graph_config.output_idx}")).get('train')
                        no_none_train = []
                        for x in train_list:
                            if x is not None:
                                no_none_train.append(x)
                        best_train_value = min(no_none_train) if no_none_train else None
                        best_train = self._fill_graph_plot_data(
                            x=[self.log_history.get("epochs")[train_list.index(best_train_value)]
                               if best_train_value is not None else None],
                            y=[best_train_value],
                            label="Лучший результат на тренировочной выборке"
                        )
                        train_plot = self._fill_graph_plot_data(
                            x=self.log_history.get("epochs"),
                            y=train_list,
                            label="Тренировочная выборка"
                        )

                        val_list = self.log_history.get(f"{loss_graph_config.output_idx}").get('loss').get(
                            self.losses.get(f"{loss_graph_config.output_idx}")).get("val")
                        no_none_val = []
                        for x in val_list:
                            if x is not None:
                                no_none_val.append(x)
                        best_val_value = min(no_none_val) if no_none_val else None
                        # best_val_value = min(val_list)
                        best_val = self._fill_graph_plot_data(
                            x=[self.log_history.get("epochs")[val_list.index(best_val_value)]
                               if best_val_value is not None else None],
                            y=[best_val_value],
                            label="Лучший результат на проверочной выборке"
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
                                best=[best_train, best_val],
                                progress_state=progress_state
                            )
                        )
                    if loss_graph_config.show == LossGraphShowChoice.classes and \
                            self.class_graphics.get(loss_graph_config.output_idx):
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
                                    self.options.data.outputs.get(int(loss_graph_config.output_idx)).classes_names
                                ],
                            )
                        )

            if self.options.data.architecture in self.yolo_architecture:
                if not self.interactive_config.loss_graphs or not self.log_history.get("epochs"):
                    return data_return
                _id = 1
                for loss_graph_config in self.interactive_config.loss_graphs:
                    # print('\nloss_graph_config', loss_graph_config)
                    if loss_graph_config.show == LossGraphShowChoice.model:
                        # print('self.log_history.get(output).get(loss).keys()', self.log_history.get('output').get('loss').keys())
                        for loss in self.log_history.get('output').get('loss').keys():
                            # print(loss)
                            if sum(self.log_history.get("output").get("progress_state").get("loss").get(loss).get(
                                    'overfitting')[-self.log_gap:]) >= self.progress_threashold:
                                progress_state = "overfitting"
                            elif sum(self.log_history.get("output").get("progress_state").get("loss").get(
                                    loss).get('underfitting')[-self.log_gap:]) >= self.progress_threashold:
                                progress_state = "underfitting"
                            else:
                                progress_state = "normal"
                            train_list = self.log_history.get("output").get('loss').get(loss).get('train')
                            no_none_train = []
                            for x in train_list:
                                if x is not None:
                                    no_none_train.append(x)
                            best_train_value = min(no_none_train) if no_none_train else None
                            # print('train_list', train_list, best_train_value, no_none_train)
                            best_train = self._fill_graph_plot_data(
                                x=[self.log_history.get("epochs")[train_list.index(best_train_value)]
                                   if best_train_value is not None else None],
                                y=[best_train_value],
                                label="Лучший результат на тренировочной выборке"
                            )
                            train_plot = self._fill_graph_plot_data(
                                x=self.log_history.get("epochs"),
                                y=train_list,
                                label="Тренировочная выборка"
                            )
                            # print('train_plot', train_plot)
                            val_list = self.log_history.get("output").get('loss').get(loss).get("val")
                            no_none_val = []
                            for x in val_list:
                                if x is not None:
                                    no_none_val.append(x)
                            best_val_value = min(no_none_val) if no_none_val else None
                            # print('val_list', val_list, best_val_value, no_none_val)
                            best_val = self._fill_graph_plot_data(
                                x=[self.log_history.get("epochs")[val_list.index(best_val_value)]
                                   if best_val_value is not None else None],
                                y=[best_val_value],
                                label="Лучший результат на проверочной выборке"
                            )
                            val_plot = self._fill_graph_plot_data(
                                x=self.log_history.get("epochs"),
                                y=self.log_history.get("output").get('loss').get(loss).get("val"),
                                label="Проверочная выборка"
                            )
                            # print('val_plot', val_plot)
                            data_return.append(
                                self._fill_graph_front_structure(
                                    _id=_id,
                                    _type='graphic',
                                    graph_name=f"График ошибки обучения «{loss}» - "
                                               f"Эпоха №{self.log_history.get('epochs')[-1]}",
                                    short_name=f"График «{loss}»",
                                    x_label="Эпоха",
                                    y_label="Значение",
                                    plot_data=[train_plot, val_plot],
                                    best=[best_train, best_val],
                                    progress_state=progress_state
                                )
                            )
                            _id += 1
                    if loss_graph_config.show == LossGraphShowChoice.classes:
                        output_idx = list(self.options.data.outputs.keys())[0]
                        # print('\nclasses_names', self.options.data.outputs.get(output_idx).classes_names)
                        data_return.append(
                            self._fill_graph_front_structure(
                                _id=_id,
                                _type='graphic',
                                graph_name=f"График ошибки обучения «prob_loss» по классам"
                                           f" - Эпоха №{self.log_history.get('epochs')[-1]}",
                                short_name=f"График ошибки обучения по классам",
                                x_label="Эпоха",
                                y_label="Значение",
                                plot_data=[
                                    self._fill_graph_plot_data(
                                        x=self.log_history.get("epochs"),
                                        y=self.log_history.get("output").get('class_loss').get('prob_loss').get(
                                            class_name),
                                        label=f"Класс {class_name}"
                                    ) for class_name in self.options.data.outputs.get(output_idx).classes_names
                                ],
                            )
                        )
            return data_return
        except Exception as e:
            if self.first:
                print_error(InteractiveCallback().name, method_name, e)
                self.first = False
            else:
                pass

    def _get_metric_graph_data_request(self) -> list:
        self.first = True
        method_name = '_get_metric_graph_data_request'
        try:
            data_return = []
            # config = self.interactive_config
            if self.options.data.architecture in self.basic_architecture:
                if not self.interactive_config.metric_graphs or not self.log_history.get("epochs"):
                    return data_return
                for metric_graph_config in self.interactive_config.metric_graphs:
                    if metric_graph_config.show == MetricGraphShowChoice.model:
                        min_max_mode = loss_metric_config.get("metric").get(metric_graph_config.show_metric.name).get(
                            "mode")
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
                        # print(1)
                        train_list = self.log_history.get(f"{metric_graph_config.output_idx}").get('metrics').get(
                            metric_graph_config.show_metric.name).get("train")
                        best_train_value = min(train_list) if min_max_mode == 'min' else max(train_list)
                        best_train = self._fill_graph_plot_data(
                            x=[self.log_history.get("epochs")[train_list.index(best_train_value)]],
                            y=[best_train_value],
                            label="Лучший результат на тренировочной выборке"
                        )
                        train_plot = self._fill_graph_plot_data(
                            x=self.log_history.get("epochs"),
                            y=self.log_history.get(f"{metric_graph_config.output_idx}").get('metrics').get(
                                metric_graph_config.show_metric.name).get("train"),
                            label="Тренировочная выборка"
                        )
                        # print(2)
                        val_list = self.log_history.get(f"{metric_graph_config.output_idx}").get('metrics').get(
                            metric_graph_config.show_metric.name).get("val")
                        best_val_value = min(val_list) if min_max_mode == 'min' else max(val_list)
                        best_val = self._fill_graph_plot_data(
                            x=[self.log_history.get("epochs")[val_list.index(best_val_value)]],
                            y=[best_val_value],
                            label="Лучший результат на проверочной выборке"
                        )
                        val_plot = self._fill_graph_plot_data(
                            x=self.log_history.get("epochs"),
                            y=self.log_history.get(f"{metric_graph_config.output_idx}").get('metrics').get(
                                metric_graph_config.show_metric.name).get("val"),
                            label="Проверочная выборка"
                        )
                        # print(3)
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
                                best=[best_train, best_val],
                                progress_state=progress_state
                            )
                        )
                    if metric_graph_config.show == MetricGraphShowChoice.classes and \
                            self.class_graphics.get(metric_graph_config.output_idx):
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
                                        y=self.log_history.get(f"{metric_graph_config.output_idx}").get(
                                            'class_metrics').get(
                                            class_name).get(metric_graph_config.show_metric),
                                        label=f"Класс {class_name}"
                                    ) for class_name in
                                    self.options.data.outputs.get(metric_graph_config.output_idx).classes_names
                                ],
                            )
                        )
                        # print(4)

            if self.options.data.architecture in self.yolo_architecture:
                if not self.interactive_config.metric_graphs or not self.log_history.get("epochs"):
                    return data_return
                _id = 1
                for metric_graph_config in self.interactive_config.metric_graphs:
                    if metric_graph_config.show == MetricGraphShowChoice.model:
                        min_max_mode = loss_metric_config.get("metric").get(metric_graph_config.show_metric.name).get(
                            "mode")
                        if sum(self.log_history.get("output").get("progress_state").get(
                                "metrics").get(metric_graph_config.show_metric.name).get(
                            'overfitting')[-self.log_gap:]) >= self.progress_threashold:
                            progress_state = 'overfitting'
                        # elif sum(self.log_history.get("output").get("progress_state").get(
                        #         "metrics").get(metric_graph_config.show_metric.name).get(
                        #     'underfitting')[-self.log_gap:]) >= self.progress_threashold:
                        #     progress_state = 'underfitting'
                        else:
                            progress_state = 'normal'
                        # train_list = self.log_history.get("output").get('metrics').get(
                        #     metric_graph_config.show_metric.name).get("train")
                        # best_train_value = min(train_list) if min_max_mode == 'min' else max(train_list)
                        # best_train = self._fill_graph_plot_data(
                        #     x=[self.log_history.get("epochs")[train_list.index(best_train_value)]],
                        #     y=[best_train_value],
                        #     label="Лучший результат на тренировочной выборке"
                        # )
                        # train_plot = self._fill_graph_plot_data(
                        #     x=self.log_history.get("epochs"),
                        #     y=self.log_history.get(f"{metric_graph_config.output_idx}").get('metrics').get(
                        #         metric_graph_config.show_metric.name).get("train"),
                        #     label="Тренировочная выборка"
                        # )
                        val_list = self.log_history.get("output").get('metrics').get(
                            metric_graph_config.show_metric.name)
                        best_val_value = min(val_list) if min_max_mode == 'min' else max(val_list)
                        best_val = self._fill_graph_plot_data(
                            x=[self.log_history.get("epochs")[val_list.index(best_val_value)]],
                            y=[best_val_value],
                            label="Лучший результат на проверочной выборке"
                        )
                        val_plot = self._fill_graph_plot_data(
                            x=self.log_history.get("epochs"),
                            y=self.log_history.get("output").get('metrics').get(
                                metric_graph_config.show_metric.name),
                            label="Проверочная выборка"
                        )
                        data_return.append(
                            self._fill_graph_front_structure(
                                _id=_id,
                                _type='graphic',
                                graph_name=f"График метрики {metric_graph_config.show_metric.name} - "
                                           f"Эпоха №{self.log_history.get('epochs')[-1]}",
                                short_name=f"{metric_graph_config.show_metric.name}",
                                x_label="Эпоха",
                                y_label="Значение",
                                plot_data=[val_plot],
                                best=[best_val],
                                progress_state=progress_state
                            )
                        )
                        _id += 1
                    if metric_graph_config.show == MetricGraphShowChoice.classes:
                        output_idx = list(self.options.data.outputs.keys())[0]
                        data_return.append(
                            self._fill_graph_front_structure(
                                _id=_id,
                                _type='graphic',
                                graph_name=f"График метрики {metric_graph_config.show_metric.name} по классам - "
                                           f"Эпоха №{self.log_history.get('epochs')[-1]}",
                                short_name=f"{metric_graph_config.show_metric.name} по классам",
                                x_label="Эпоха",
                                y_label="Значение",
                                plot_data=[
                                    self._fill_graph_plot_data(
                                        x=self.log_history.get("epochs"),
                                        y=self.log_history.get("output").get('class_metrics').get(
                                            metric_graph_config.show_metric.name).get(class_name),
                                        label=f"Класс {class_name}"
                                    ) for class_name in self.options.data.outputs.get(output_idx).classes_names
                                ],
                            )
                        )
                        _id += 1
            return data_return
        except Exception as e:
            if self.first:
                print_error(InteractiveCallback().name, method_name, e)
                self.first = False
            else:
                pass

    def _get_intermediate_result_request(self) -> dict:
        self.first = True
        method_name = '_get_intermediate_result_request'
        try:
            return_data = {}
            if self.options.data.architecture in self.basic_architecture and \
                    self.interactive_config.intermediate_result.show_results:
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
                                return_mode='callback',
                                max_lenth=MAX_INTERMEDIATE_GRAPH_LENGTH,
                                templates=[self._fill_graph_plot_data, self._fill_graph_front_structure]
                            )
                            # random_key = ''.join(random.sample(string.ascii_letters + string.digits, 16))
                            return_data[f"{idx + 1}"]['initial_data'][f"Входной слой «{inp}»"] = {
                                # 'update': random_key,
                                'type': type_choice,
                                'data': data,
                            }

                    for out in self.options.data.outputs.keys():
                        task = self.options.data.outputs.get(out).task

                        if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.TimeseriesTrend:
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
                                show_stat=self.interactive_config.intermediate_result.show_statistic,
                                return_mode='callback'
                            )

                        elif task == LayerOutputTypeChoice.Timeseries:
                            input = list(self.inverse_x_val.keys())[0]
                            data = CreateArray().postprocess_time_series(
                                options=self.options.data,
                                real_x=self.inverse_x_val.get(f"{input}")[self.example_idx[idx]],
                                inverse_y_true=self.inverse_y_true.get("val").get(f"{out}")[self.example_idx[idx]],
                                inverse_y_pred=self.inverse_y_pred.get(f"{out}")[self.example_idx[idx]],
                                output_id=out,
                                depth=self.inverse_y_true.get("val").get(f"{out}")[self.example_idx[idx]].shape[-2],
                                show_stat=self.interactive_config.intermediate_result.show_statistic,
                                templates=[self._fill_graph_plot_data, self._fill_graph_front_structure],
                                max_lenth=MAX_INTERMEDIATE_GRAPH_LENGTH
                            )

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

            elif self.options.data.architecture in self.yolo_architecture and \
                    self.interactive_config.intermediate_result.show_results:
                self._reformat_y_pred(
                    y_pred=self.raw_y_pred,
                    sensitivity=self.interactive_config.intermediate_result.sensitivity,
                    threashold=self.interactive_config.intermediate_result.threashold
                )
                for idx in range(self.interactive_config.intermediate_result.num_examples):
                    return_data[f"{idx + 1}"] = {
                        'initial_data': {},
                        'true_value': {},
                        'predict_value': {},
                        'tags_color': {},
                        'statistic_values': {}
                    }
                    image_path = os.path.join(
                        self.dataset_path, self.options.dataframe.get('val').iat[self.example_idx[idx], 0])
                    # print(image_path)
                    out = self.interactive_config.intermediate_result.box_channel
                    # print(out)
                    # print('self.example_idx[idx]', idx, self.example_idx[idx])
                    # print(out, len(self.y_pred.get(out)), len(self.y_true.get(out)))
                    data = CreateArray().postprocess_object_detection(
                        predict_array=copy.deepcopy(self.y_pred.get(out)[self.example_idx[idx]]),
                        true_array=self.y_true.get(out)[self.example_idx[idx]],
                        image_path=image_path,
                        colors=self.class_colors,
                        sensitivity=self.interactive_config.intermediate_result.sensitivity,
                        image_id=idx,
                        image_size=self.options.data.inputs.get(list(self.options.data.inputs.keys())[0]).shape[:2],
                        name_classes=self.options.data.outputs.get(
                            list(self.options.data.outputs.keys())[0]).classes_names,
                        save_path=self.preset_path,
                        return_mode='callback',
                        show_stat=self.interactive_config.intermediate_result.show_statistic
                    )
                    if data.get('y_true'):
                        return_data[f"{idx + 1}"]['true_value'][f"Выходной слой"] = data.get('y_true')
                    return_data[f"{idx + 1}"]['predict_value'][f"Выходной слой"] = data.get('y_pred')

                    if data.get('stat'):
                        return_data[f"{idx + 1}"]['statistic_values'] = data.get('stat')
                    else:
                        return_data[f"{idx + 1}"]['statistic_values'] = {}
            else:
                pass
            return return_data
        except Exception as e:
            if self.first:
                print_error(InteractiveCallback().name, method_name, e)
                self.first = False
            else:
                pass

    def _get_statistic_data_request(self) -> list:
        self.first = True
        method_name = '_get_statistic_data_request'
        try:
            return_data = []
            _id = 1
            if self.options.data.architecture in self.basic_architecture:
                for out in self.interactive_config.statistic_data.output_id:
                    task = self.options.data.outputs.get(out).task
                    encoding = self.options.data.outputs.get(out).encoding
                    if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.TimeseriesTrend and \
                            encoding != LayerEncodingChoice.multi:
                        cm, cm_percent = self._get_confusion_matrix(
                            np.argmax(self.y_true.get("val").get(f'{out}'),
                                      axis=-1) if encoding == LayerEncodingChoice.ohe
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

                    elif (
                            task == LayerOutputTypeChoice.TextSegmentation or task == LayerOutputTypeChoice.Classification) \
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
                        x_mae, y_mae = self._get_distribution_histogram(np.abs(deviation), categorical=False)
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
                        x_me, y_me = self._get_distribution_histogram(deviation, categorical=False)
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
                            for step in range(self.y_true.get("val").get(f'{out}').shape[-2]):
                                y_true = self.inverse_y_true.get("val").get(f"{out}")[:, step, i].astype('float')
                                y_pred = self.inverse_y_pred.get(f"{out}")[:, step, i].astype('float')
                                x_tr, y_tr = self._get_time_series_graphic(y_true, make_short=True)
                                x_pr, y_pr = self._get_time_series_graphic(y_pred, make_short=True)
                                return_data.append(
                                    self._fill_graph_front_structure(
                                        _id=_id,
                                        _type='graphic',
                                        graph_name=f"Выходной слой «{out}» - Предсказание канала "
                                                   f"«{channel_name.split('_', 1)[-1]}» на {step + 1} "
                                                   f"шаг{'ов' if step + 1 > 1 else ''} вперед",
                                        short_name=f"{out} - «{channel_name.split('_', 1)[-1]}» на {step + 1} "
                                                   f"шаг{'ов' if step + 1 > 1 else ''}",
                                        x_label="Время",
                                        y_label="Значение",
                                        plot_data=[
                                            self._fill_graph_plot_data(x=x_tr, y=y_tr, label="Истинное значение"),
                                            self._fill_graph_plot_data(x=x_pr, y=y_pr, label="Предсказанное значение")
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
                                            self._fill_graph_plot_data(x=x_axis, y=auto_corr_true,
                                                                       label="Истинное значение"),
                                            self._fill_graph_plot_data(x=x_axis, y=auto_corr_pred,
                                                                       label="Предсказанное значение")
                                        ],
                                    )
                                )
                                _id += 1
                                deviation = (y_pred - y_true) * 100 / y_true
                                x_mae, y_mae = self._get_distribution_histogram(np.abs(deviation), categorical=False)
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
                                x_me, y_me = self._get_distribution_histogram(deviation, categorical=False)
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

                    else:
                        pass

            elif self.options.data.architecture in self.yolo_architecture:
                box_channel = self.interactive_config.statistic_data.box_channel
                name_classes = self.options.data.outputs.get(list(self.options.data.outputs.keys())[0]).classes_names
                self._reformat_y_pred(
                    y_pred=self.raw_y_pred,
                    sensitivity=self.interactive_config.statistic_data.sensitivity,
                    threashold=self.interactive_config.statistic_data.threashold
                )
                object_tt = 0
                object_tf = 0
                object_ft = 0

                line_names = []
                class_accuracy_hist = {}
                class_loss_hist = {}
                class_coord_accuracy = {}
                for class_name in name_classes:
                    line_names.append(class_name)
                    class_accuracy_hist[class_name] = []
                    class_loss_hist[class_name] = []
                    class_coord_accuracy[class_name] = []
                line_names.append('empty')

                class_matrix = np.zeros((len(line_names), len(line_names)))
                for i in range(len(self.y_pred[box_channel])):
                    example_stat = CreateArray().get_yolo_example_statistic(
                        true_bb=self.y_true.get(box_channel)[i],
                        pred_bb=self.y_pred.get(box_channel)[i],
                        name_classes=name_classes,
                        sensitivity=self.interactive_config.statistic_data.sensitivity
                    )
                    object_ft += len(example_stat['recognize']['empty'])
                    object_tf += len(example_stat['recognize']['unrecognize'])
                    for class_name in line_names:
                        if class_name != 'empty':
                            object_tt += len(example_stat['recognize'][class_name])
                        for item in example_stat['recognize'][class_name]:
                            class_matrix[line_names.index(class_name)][line_names.index(item['pred_class'])] += 1
                            if class_name != 'empty':
                                if item['class_result']:
                                    class_accuracy_hist[class_name].append(item['class_conf'])
                                    class_coord_accuracy[class_name].append(item['overlap'])
                                else:
                                    class_loss_hist[class_name].append(item['class_conf'])
                    for item in example_stat['recognize']['unrecognize']:
                        class_matrix[line_names.index(item['class_name'])][-1] += 1

                for class_name in name_classes:
                    class_accuracy_hist[class_name] = np.round(np.mean(class_accuracy_hist[class_name]) * 100,
                                                               2).item() if \
                        class_accuracy_hist[class_name] else 0.
                    class_coord_accuracy[class_name] = np.round(np.mean(class_coord_accuracy[class_name]) * 100,
                                                                2).item() if \
                        class_coord_accuracy[class_name] else 0.
                    class_loss_hist[class_name] = np.round(np.mean(class_loss_hist[class_name]) * 100, 2).item() if \
                        class_loss_hist[
                            class_name] else 0.

                object_matrix = [[object_tt, object_tf], [object_ft, 0]]
                class_matrix_percent = []
                for i in class_matrix:
                    class_matrix_percent.append(i * 100 / np.sum(i) if np.sum(i) else np.zeros_like(i))
                class_matrix_percent = np.round(class_matrix_percent, 2).tolist()
                class_matrix = class_matrix.astype('int').tolist()
                # labels = copy.deepcopy(list(name_classes))
                # labels.append('Пустой бокс')
                # print('\nlabels', labels)
                return_data.append(
                    self._fill_heatmap_front_structure(
                        _id=1,
                        _type="heatmap",
                        graph_name=f"Бокс-канал «{box_channel}» - Матрица неточностей определения классов",
                        short_name=f"{box_channel} - Матрица классов",
                        x_label="Предсказание",
                        y_label="Истинное значение",
                        labels=line_names,
                        data_array=class_matrix,
                        data_percent_array=class_matrix_percent,
                    )
                )
                return_data.append(
                    self._fill_heatmap_front_structure(
                        _id=2,
                        _type="valheatmap",
                        graph_name=f"Бокс-канал «{box_channel}» - Матрица неточностей определения объектов",
                        short_name=f"{box_channel} - Матрица объектов",
                        x_label="Предсказание",
                        y_label="Истинное значение",
                        labels=['Объект', 'Отсутствие'],
                        data_array=object_matrix,
                        data_percent_array=None,
                    )
                )
                return_data.append(
                    self._fill_graph_front_structure(
                        _id=3,
                        _type='histogram',
                        graph_name=f'Бокс-канал «{box_channel}» - Средняя точность определеня  классов',
                        short_name=f"{box_channel} - точность классов",
                        x_label="Имя класса",
                        y_label="Средняя точность, %",
                        plot_data=[
                            self._fill_graph_plot_data(x=name_classes, y=[class_accuracy_hist[i] for i in name_classes])
                        ],
                    )
                )
                return_data.append(
                    self._fill_graph_front_structure(
                        _id=4,
                        _type='histogram',
                        graph_name=f'Бокс-канал «{box_channel}» - Средняя ошибка определеня  классов',
                        short_name=f"{box_channel} - ошибка классов",
                        x_label="Имя класса",
                        y_label="Средняя ошибка, %",
                        plot_data=[
                            self._fill_graph_plot_data(x=name_classes, y=[class_loss_hist[i] for i in name_classes])
                        ],
                    )
                )
                return_data.append(
                    self._fill_graph_front_structure(
                        _id=5,
                        _type='histogram',
                        graph_name=f'Бокс-канал «{box_channel}» - '
                                   f'Средняя точность определения  координат объекта класса (MeanIoU)',
                        short_name=f"{box_channel} - координаты классов",
                        x_label="Имя класса",
                        y_label="Средняя точность, %",
                        plot_data=[
                            self._fill_graph_plot_data(x=name_classes,
                                                       y=[class_coord_accuracy[i] for i in name_classes])
                        ],
                    )
                )

            else:
                pass
            # print('\n_get_statistic_data_request', return_data)
            return return_data
        except Exception as e:
            if self.first:
                print_error(InteractiveCallback().name, method_name, e)
                self.first = False
            else:
                pass

    def _get_balance_data_request(self) -> list:
        self.first = True
        method_name = '_get_balance_data_request'
        try:
            return_data = []
            _id = 0
            if self.options.data.architecture in self.basic_architecture:
                for out in self.options.data.outputs.keys():
                    # print('self.options.data.outputs', self.options.data.outputs)
                    task = self.options.data.outputs.get(out).task
                    # print('task', task)
                    if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.TimeseriesTrend:
                        for class_type in self.dataset_balance.get(f"{out}").keys():
                            preset = {}
                            for data_type in ['train', 'val']:
                                class_names, class_count = CreateArray().sort_dict(
                                    dict_to_sort=self.dataset_balance.get(f"{out}").get(class_type).get(data_type),
                                    mode=self.interactive_config.data_balance.sorted.name
                                )
                                preset[data_type] = self._fill_graph_front_structure(
                                    _id=_id,
                                    _type='histogram',
                                    type_data=data_type,
                                    graph_name=f"Выход {out} - "
                                               f"{'Тренировочная' if data_type == 'train' else 'Проверочная'} выборка",
                                    short_name=f"{out} - {'Тренировочная' if data_type == 'train' else 'Проверочная'}",
                                    x_label="Название класса",
                                    y_label="Значение",
                                    plot_data=[self._fill_graph_plot_data(x=class_names, y=class_count)],
                                )
                                _id += 1
                            return_data.append(preset)

                    elif task == LayerOutputTypeChoice.Segmentation:
                        for class_type in self.dataset_balance.get(f"{out}").keys():
                            preset = {}
                            if class_type in ["presence_balance", "square_balance"]:
                                for data_type in ['train', 'val']:
                                    names, count = CreateArray().sort_dict(
                                        dict_to_sort=self.dataset_balance.get(f"{out}").get(class_type).get(data_type),
                                        mode=self.interactive_config.data_balance.sorted.name
                                    )
                                    preset[data_type] = self._fill_graph_front_structure(
                                        _id=_id,
                                        _type='histogram',
                                        type_data=data_type,
                                        graph_name=f"Выход {out} - {'Тренировочная' if data_type == 'train' else 'Проверочная'} выборка - "
                                                   f"{'баланс присутсвия' if class_type == 'presence_balance' else 'процент пространства'}",
                                        short_name=f"{'Тренировочная' if data_type == 'train' else 'Проверочная'} - "
                                                   f"{'присутсвие' if class_type == 'presence_balance' else 'пространство'}",
                                        x_label="Название класса",
                                        y_label="Значение",
                                        plot_data=[self._fill_graph_plot_data(x=names, y=count)],
                                    )
                                    _id += 1
                                return_data.append(preset)

                            if class_type == "colormap":
                                for class_name, map_link in self.dataset_balance.get(f"{out}").get('colormap').get(
                                        'train').items():
                                    preset = {}
                                    for data_type in ['train', 'val']:
                                        preset[data_type] = self._fill_graph_front_structure(
                                            _id=_id,
                                            _type='colormap',
                                            type_data=data_type,
                                            graph_name=f"{'Тренировочная' if data_type == 'train' else 'Проверочная'} выборка "
                                                       f"- Цветовая карта класса {class_name}",
                                            short_name="",
                                            x_label="",
                                            y_label="",
                                            plot_data=map_link,
                                        )
                                        _id += 1
                                    return_data.append(preset)

                    elif task == LayerOutputTypeChoice.TextSegmentation:
                        for class_type in self.dataset_balance.get(f"{out}").keys():
                            preset = {}
                            if class_type in ["presence_balance", "percent_balance"]:
                                for data_type in ['train', 'val']:
                                    names, count = CreateArray().sort_dict(
                                        dict_to_sort=self.dataset_balance.get(f"{out}").get(class_type).get(data_type),
                                        mode=self.interactive_config.data_balance.sorted.name
                                    )
                                    preset[data_type] = self._fill_graph_front_structure(
                                        _id=_id,
                                        _type='histogram',
                                        type_data=data_type,
                                        graph_name=f"Выход {out} - {'Тренировочная' if data_type == 'train' else 'Проверочная'} выборка - "
                                                   f"{'баланс присутсвия' if class_type == 'presence_balance' else 'процент пространства'}",
                                        short_name=f"{'Тренировочная' if data_type == 'train' else 'Проверочная'} - "
                                                   f"{'присутсвие' if class_type == 'presence_balance' else 'процент'}",
                                        x_label="Название класса",
                                        y_label="Значение",
                                        plot_data=[self._fill_graph_plot_data(x=names, y=count)],
                                    )
                                    _id += 1
                            return_data.append(preset)

                    elif task == LayerOutputTypeChoice.Regression:
                        # print('-self.dataset_balance', self.dataset_balance.keys(), self.dataset_balance[f"{out}"])
                        for class_type in self.dataset_balance[f"{out}"].keys():
                            # print('--class_type', class_type)
                            if class_type == 'histogram':
                                for column in self.dataset_balance[f"{out}"][class_type]["train"].keys():
                                    # print('----column', column)
                                    preset = {}
                                    for data_type in ["train", "val"]:
                                        histogram = self.dataset_balance[f"{out}"][class_type][data_type][column]
                                        if histogram.get("type") == 'histogram':
                                            x, y = CreateArray().sort_dict(
                                                dict_to_sort=dict(zip(histogram.get("x"), histogram.get("y"))),
                                                mode=self.interactive_config.data_balance.sorted.name
                                            )
                                            # print('\n--histogram', histogram, x, y)
                                        else:
                                            x = histogram.get("x")
                                            y = histogram.get("y")
                                        data_type_name = "Тренировочная" if data_type == "train" else "Проверочная"
                                        preset[data_type] = self._fill_graph_front_structure(
                                            _id=_id,
                                            _type=histogram.get("type"),
                                            type_data=data_type,
                                            graph_name=f"Выход {out} - {data_type_name} выборка - "
                                                       f"Гистограмма распределения колонки «{histogram['name']}»",
                                            short_name=f"{data_type_name} - {histogram['name']}",
                                            x_label="Значение",
                                            y_label="Количество",
                                            plot_data=[
                                                self._fill_graph_plot_data(x=x, y=y)],
                                        )
                                        _id += 1
                                    return_data.append(preset)

                            if class_type == 'correlation':
                                preset = {}
                                for data_type in ["train", "val"]:
                                    data_type_name = "Тренировочная" if data_type == "train" else "Проверочная"
                                    preset[data_type] = self._fill_heatmap_front_structure(
                                        _id=_id,
                                        _type="corheatmap",
                                        type_data=data_type,
                                        graph_name=f"Выход {out} - {data_type_name} выборка - Матрица корреляций",
                                        short_name=f"Матрица корреляций",
                                        x_label="Колонка",
                                        y_label="Колонка",
                                        labels=self.dataset_balance[f"{out}"]['correlation'][data_type]["labels"],
                                        data_array=self.dataset_balance[f"{out}"]['correlation'][data_type]["matrix"],
                                    )
                                    _id += 1
                                return_data.append(preset)

                    elif task == LayerOutputTypeChoice.Timeseries:
                        for class_type in self.dataset_balance[f"{out}"].keys():
                            for channel_name in self.dataset_balance[f"{out}"][class_type].keys():
                                preset = {}
                                for data_type in ["train", "val"]:
                                    graph_type = self.dataset_balance[f"{out}"][class_type][channel_name][data_type][
                                        'type']
                                    data_type_name = "Тренировочная" if data_type == "train" else "Проверочная"
                                    y_true = self.options.dataframe.get(data_type)[channel_name].to_list()
                                    if class_type == 'graphic':
                                        # x_graph_axis = np.arange(len(y_true)).astype('float').tolist()
                                        x, y = self._get_time_series_graphic(y_true, make_short=True)
                                        plot_data = [self._fill_graph_plot_data(x=x, y=y)]
                                        graph_name = f'Выход {out} - {data_type_name} выборка - ' \
                                                     f'График канала «{channel_name.split("_", 1)[-1]}»'
                                        short_name = f'{data_type_name} - «{channel_name.split("_", 1)[-1]}»'
                                        x_label = "Время"
                                        y_label = "Величина"
                                    else:
                                        # if class_type == 'dense_histogram':
                                        x_hist, y_hist = self._get_distribution_histogram(y_true, categorical=False)
                                        plot_data = [self._fill_graph_plot_data(x=x_hist, y=y_hist)]
                                        graph_name = f'Выход {out} - {data_type_name} выборка - ' \
                                                     f'Гистограмма плотности канала «{channel_name.split("_", 1)[-1]}»'
                                        short_name = f'{data_type_name} - Гистограмма «{channel_name.split("_", 1)[-1]}»'
                                        x_label = "Значение"
                                        y_label = "Количество"
                                    preset[data_type] = self._fill_graph_front_structure(
                                        _id=_id,
                                        _type=graph_type,
                                        type_data=data_type,
                                        graph_name=graph_name,
                                        short_name=short_name,
                                        x_label=x_label,
                                        y_label=y_label,
                                        plot_data=plot_data,
                                    )
                                    _id += 1
                                return_data.append(preset)

                    else:
                        pass

            elif self.options.data.architecture in self.yolo_architecture:
                for class_type in self.dataset_balance.get("output").keys():
                    preset = {}
                    if class_type in ["class_count", "class_square"]:
                        for data_type in ['train', 'val']:
                            names, count = CreateArray().sort_dict(
                                dict_to_sort=self.dataset_balance.get("output").get(class_type).get(data_type),
                                mode=self.interactive_config.data_balance.sorted.name
                            )
                            preset[data_type] = self._fill_graph_front_structure(
                                _id=_id,
                                _type='histogram',
                                type_data=data_type,
                                graph_name=f"{'Тренировочная' if data_type == 'train' else 'Проверочная'} выборка - "
                                           f"{'баланс присутсвия' if class_type == 'class_count' else 'процент пространства'}",
                                short_name=f"{'Тренировочная' if data_type == 'train' else 'Проверочная'} - "
                                           f"{'присутсвие' if class_type == 'class_count' else 'пространство'}",
                                x_label="Название класса",
                                y_label="Значение",
                                plot_data=[self._fill_graph_plot_data(x=names, y=count)],
                            )
                            _id += 1
                        return_data.append(preset)

                    if class_type == "colormap":
                        classes_name = sorted(
                            list(self.dataset_balance.get("output").get('colormap').get('train').keys()))
                        for class_name in classes_name:
                            preset = {}
                            for data_type in ['train', 'val']:
                                _dict = self.dataset_balance.get("output").get('colormap').get(data_type)
                                preset[data_type] = self._fill_graph_front_structure(
                                    _id=_id,
                                    _type='colormap',
                                    type_data=data_type,
                                    graph_name=f"{'Тренировочная' if data_type == 'train' else 'Проверочная'} выборка "
                                               f"- Цветовая карта "
                                               f"{'всех классов' if class_name == 'all_classes' else 'класса'} "
                                               f"{'' if class_name == 'all_classes' else class_name}",
                                    short_name="",
                                    x_label="",
                                    y_label="",
                                    plot_data=_dict.get(class_name),
                                )
                                _id += 1
                            return_data.append(preset)
            else:
                pass

            return return_data
        except Exception as e:
            if self.first:
                print_error(InteractiveCallback().name, method_name, e)
                self.first = False
            else:
                pass

    @staticmethod
    def _get_confusion_matrix(y_true, y_pred, get_percent=True) -> tuple:
        method_name = '_get_confusion_matrix'
        try:
            cm = confusion_matrix(y_true, y_pred)
            cm_percent = None
            if get_percent:
                cm_percent = np.zeros_like(cm).astype('float')
                for i in range(len(cm)):
                    total = np.sum(cm[i])
                    for j in range(len(cm[i])):
                        cm_percent[i][j] = round(cm[i][j] * 100 / total, 1)
            return cm.astype('float').tolist(), cm_percent.astype('float').tolist()
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    @staticmethod
    def _get_classification_report(y_true, y_pred, labels):
        method_name = '_get_classification_report'
        try:
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
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    @staticmethod
    def _get_error_distribution(y_true, y_pred, bins=25, absolute=True):
        method_name = '_get_error_distribution'
        try:
            error = (y_true - y_pred)  # "* 100 / y_true
            if absolute:
                error = np.abs(error)
            return InteractiveCallback()._get_distribution_histogram(error, categorical=False)
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    @staticmethod
    def _get_time_series_graphic(data, make_short=False):
        method_name = '_get_time_series_graphic'
        try:
            if make_short and len(data) > MAX_TS_GRAPH_COUNT:
                union = int(len(data) // MAX_TS_GRAPH_COUNT)
                short_data = []
                for i in range(int(len(data) / union)):
                    short_data.append(
                        InteractiveCallback()._round_loss_metric(np.mean(data[union * i:union * i + union]).item())
                    )
                return np.arange(len(short_data)).astype('int').tolist(), np.array(short_data).astype('float').tolist()
            else:
                return np.arange(len(data)).astype('int').tolist(), np.array(data).astype('float').tolist()
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    @staticmethod
    def _get_correlation_matrix(data_frame: DataFrame):
        method_name = '_get_correlation_matrix'
        try:
            corr = data_frame.corr()
            labels = []
            for lbl in list(corr.columns):
                labels.append(lbl.split("_", 1)[-1])
            return labels, np.array(np.round(corr, 2)).astype('float').tolist()
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    @staticmethod
    def _get_scatter(y_true, y_pred):
        method_name = '_get_scatter'
        try:
            return InteractiveCallback().clean_data_series([y_true, y_pred], mode="duo")
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    @staticmethod
    def _get_distribution_histogram(data_series, categorical=True):
        method_name = '_get_distribution_histogram'
        try:
            if categorical:
                hist_data = pd.Series(data_series).value_counts()
                return hist_data.index.to_list(), hist_data.to_list()
            else:
                if len(InteractiveCallback().clean_data_series([data_series], mode="mono")) > 10:
                    data_series = InteractiveCallback().clean_data_series([data_series], mode="mono")
                if int(len(data_series) / 10) < MAX_HISTOGRAM_BINS:
                    bins = int(len(data_series) / 10)
                elif int(len(set(data_series))) < MAX_HISTOGRAM_BINS:
                    bins = int(len(set(data_series)))
                else:
                    bins = MAX_HISTOGRAM_BINS
                bar_values, x_labels = np.histogram(data_series, bins=bins)
                new_x = []
                for i in range(len(x_labels[:-1])):
                    new_x.append(np.mean([x_labels[i], x_labels[i + 1]]))
                new_x = np.array(new_x)
                return new_x.astype('float').tolist(), bar_values.astype('int').tolist()
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    @staticmethod
    def clean_data_series(data_series: list, mode="mono"):
        method_name = 'clean_data_series'
        try:
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
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    @staticmethod
    def _get_autocorrelation_graphic(y_true, y_pred, depth=10) -> (list, list, list):
        method_name = '_get_autocorrelation_graphic'
        try:
            def get_auto_corr(a, b):
                ma = a.mean()
                mb = b.mean()
                mab = (a * b).mean()
                sa = a.std()
                sb = b.std()

                val = 1
                if sa > 0 and sb > 0:
                    val = (mab - ma * mb) / (sa * sb)
                return val

            auto_corr_true = []
            for i in range(depth):
                auto_corr_true.append(get_auto_corr(y_true[:-(i + 1)], y_true[(i + 1):]))

            auto_corr_pred = []
            for i in range(depth):
                auto_corr_pred.append(get_auto_corr(y_true[:-(i + 1)], y_pred[(i + 1):]))

            x_axis = np.arange(depth).astype('int').tolist()
            return x_axis, auto_corr_true, auto_corr_pred
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    @staticmethod
    def _dice_coef(y_true, y_pred, batch_mode=True, smooth=1.0):
        method_name = '_dice_coef'
        try:
            return CreateArray().dice_coef(y_true, y_pred, batch_mode=batch_mode, smooth=smooth)
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    @staticmethod
    def _get_image_class_colormap(array: np.ndarray, colors: list, class_id: int, save_path: str):
        method_name = '_get_image_class_colormap'
        try:
            array = np.expand_dims(np.argmax(array, axis=-1), axis=-1) * 512
            array = np.where(array == class_id * 512,
                             np.array(colors[class_id]) if np.sum(np.array(colors[class_id])) > 50
                             else np.array((255, 255, 255)), np.array((0, 0, 0)))
            array = (np.sum(array, axis=0) / len(array)).astype("uint8")
            matplotlib.image.imsave(save_path, array)
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)
