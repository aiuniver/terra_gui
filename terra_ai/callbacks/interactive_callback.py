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

from tensorflow.keras.utils import to_categorical
import numpy as np

from terra_ai import progress
from terra_ai.callbacks.postprocess_results import PostprocessResults
from terra_ai.callbacks.utils import loss_metric_config, class_counter, get_image_class_colormap, \
    get_distribution_histogram, get_correlation_matrix, round_loss_metric, fill_graph_plot_data, \
    fill_graph_front_structure, get_confusion_matrix, fill_heatmap_front_structure, fill_table_front_structure, \
    get_scatter, get_classification_report, get_autocorrelation_graphic
from terra_ai.data.datasets.extra import LayerInputTypeChoice, LayerOutputTypeChoice, DatasetGroupChoice, \
    LayerEncodingChoice
from terra_ai.data.presets.training import Metric
from terra_ai.data.training.extra import LossGraphShowChoice, MetricGraphShowChoice, MetricChoice, ArchitectureChoice
from terra_ai.data.training.train import InteractiveData, YoloInteractiveData
from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.training.customlosses import UnscaledMAE
from terra_ai.utils import camelize, decamelize

__version__ = 0.085


class InteractiveCallback:
    """Callback for interactive requests"""

    def __init__(self):
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
        self.yolo_interactive_config: YoloInteractiveData = YoloInteractiveData(**{})
        pass

    def set_attributes(self, dataset: PrepareDataset,
                       metrics: dict,
                       losses: dict,
                       dataset_path: str,
                       training_path: str,
                       initial_config: InteractiveData,
                       yolo_initial_config: YoloInteractiveData = None):

        self.preset_path = os.path.join(training_path, "presets")
        if not os.path.exists(self.preset_path):
            os.mkdir(self.preset_path)
        if dataset.data.architecture in self.basic_architecture:
            self.losses = losses
            self.metrics = self._reformat_metrics(metrics)
            self.loss_obj = self._prepare_loss_obj(losses)
            self.metrics_obj = self._prepare_metric_obj(metrics)
            self.interactive_config = initial_config
        if dataset.data.architecture in self.yolo_architecture:
            self.yolo_interactive_config = yolo_initial_config

        self.options = dataset
        self._class_metric_list()
        self.dataset_path = dataset_path
        self._get_classes_colors()
        self.x_val, self.inverse_x_val = self._prepare_x_val(dataset)
        self.y_true, self.inverse_y_true = self._prepare_y_true(dataset)

        if not self.log_history:
            self._prepare_null_log_history_template()
        self.dataset_balance = self._prepare_dataset_balance()
        self.class_idx = self._prepare_class_idx()
        self.seed_idx = self._prepare_seed()
        self.random_key = ''.join(random.sample(string.ascii_letters + string.digits, 16))

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

    def update_state(self, y_pred, fit_logs=None, current_epoch_time=None, on_epoch_end_flag=False) -> dict:
        if self.log_history:
            if y_pred is not None:
                if self.options.data.architecture in self.basic_architecture:
                    self._reformat_y_pred(y_pred)
                    if self.interactive_config.intermediate_result.show_results:
                        out = f"{self.interactive_config.intermediate_result.main_output}"
                        self.example_idx = PostprocessResults().prepare_example_idx_to_show(
                            array=self.y_pred.get(out),
                            true_array=self.y_true.get("val").get(out),
                            options=self.options,
                            output=int(out),
                            count=self.interactive_config.intermediate_result.num_examples,
                            choice_type=self.interactive_config.intermediate_result.example_choice_type,
                            seed_idx=self.seed_idx[:self.interactive_config.intermediate_result.num_examples]
                        )
                if self.options.data.architecture in self.yolo_architecture:
                    self.raw_y_pred = y_pred
                    if self.yolo_interactive_config.intermediate_result.show_results:
                        self.example_idx, _ = CreateArray().prepare_yolo_example_idx_to_show(
                            array=copy.deepcopy(self.y_pred),
                            true_array=copy.deepcopy(self.y_true),
                            name_classes=self.options.data.outputs.get(
                                list(self.options.data.outputs.keys())[0]).classes_names,
                            box_channel=self.yolo_interactive_config.intermediate_result.box_channel,
                            count=self.yolo_interactive_config.intermediate_result.num_examples,
                            choice_type=self.yolo_interactive_config.intermediate_result.example_choice_type,
                            seed_idx=self.seed_idx,
                            sensitivity=self.yolo_interactive_config.intermediate_result.sensitivity,
                        )
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
                self.random_key = ''.join(random.sample(string.ascii_letters + string.digits, 16))
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
                if self.yolo_interactive_config.intermediate_result.show_results:
                    self.example_idx, _ = CreateArray().prepare_yolo_example_idx_to_show(
                        array=copy.deepcopy(self.y_pred),
                        true_array=copy.deepcopy(self.y_true),
                        name_classes=self.options.data.outputs.get(
                            list(self.options.data.outputs.keys())[0]).classes_names,
                        box_channel=self.yolo_interactive_config.intermediate_result.box_channel,
                        count=self.yolo_interactive_config.intermediate_result.num_examples,
                        choice_type=self.yolo_interactive_config.intermediate_result.example_choice_type,
                        seed_idx=self.seed_idx[:self.yolo_interactive_config.intermediate_result.num_examples],
                        sensitivity=self.yolo_interactive_config.intermediate_result.sensitivity,
                    )
                if config.intermediate_result.show_results or config.statistic_data.box_channel:
                    self.urgent_predict = True
                    self.intermediate_result = self._get_intermediate_result_request()
                    if self.yolo_interactive_config.statistic_data.box_channel:
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

    def _prepare_y_true(self, dataset: PrepareDataset):
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
            y_true = CreateArray().get_yolo_y_true(options=dataset)

        return y_true, inverse_y_true

    def _class_metric_list(self):
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

    def _prepare_null_log_history_template(self):
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
                    'mAP95': [],
                },
                "class_metrics": {
                    'mAP50': {},
                    'mAP95': {},
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
                        'mAP95': {
                            "mean_log_history": [], "normal_state": [], "overfitting": []
                        },
                    }
                }
            }
            out = list(self.options.data.outputs.keys())[0]
            for class_name in self.options.data.outputs.get(out).classes_names:
                self.log_history['output']["class_loss"]['prob_loss'][class_name] = []
                self.log_history['output']["class_metrics"]['mAP50'][class_name] = []
                self.log_history['output']["class_metrics"]['mAP95'][class_name] = []

    def _prepare_dataset_balance(self) -> dict:
        dataset_balance = {}
        for out in self.options.data.outputs.keys():
            task = self.options.data.outputs.get(out).task
            encoding = self.options.data.outputs.get(out).encoding

            if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.TimeseriesTrend:
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
                        get_image_class_colormap(
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
                        x, y = get_distribution_histogram(
                            list(self.options.dataframe.get(data_type)[output_channel]),
                            categorical=False
                        )
                        dataset_balance[f"{out}"]['dense_histogram'][output_channel][data_type] = {
                            "type": "bar",
                            "x": x,
                            "y": y
                        }

            if task == LayerOutputTypeChoice.Regression:
                dataset_balance[f"{out}"] = {
                    'histogram': {},
                    'correlation': {}
                }
                for data_type in ['train', 'val']:
                    dataset_balance[f"{out}"]['histogram'][data_type] = {}
                    for column in list(self.options.dataframe.get('train').columns):
                        column_id = int(column.split("_")[0])
                        column_task = self.options.data.columns.get(column_id).get(column).task
                        column_data = list(self.options.dataframe.get(data_type)[column])
                        if column_task == LayerInputTypeChoice.Text:
                            continue
                        elif column_task == LayerInputTypeChoice.Classification:
                            x, y = get_distribution_histogram(column_data, categorical=True)
                            hist_type = "histogram"
                        else:
                            x, y = get_distribution_histogram(column_data, categorical=False)
                            hist_type = "bar"
                        dataset_balance[f"{out}"]['histogram'][data_type][column] = {
                            "name": column.split("_", 1)[-1],
                            "type": hist_type,
                            "x": x,
                            "y": y
                        }
                for data_type in ['train', 'val']:
                    labels, matrix = get_correlation_matrix(pd.DataFrame(self.options.dataframe.get(data_type)))
                    dataset_balance[f"{out}"]['correlation'][data_type] = {
                        "labels": labels,
                        "matrix": matrix
                    }

            if task == LayerOutputTypeChoice.ObjectDetection:
                name_classes = self.options.data.outputs.get(list(self.options.data.outputs.keys())[0]).classes_names
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
                            round_loss_metric(self._get_box_square(item, imsize=(imsize[0], imsize[1])))

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

    @staticmethod
    def _get_box_square(bbs, imsize=(416, 416)):
        if len(bbs):
            square = 0
            for bb in bbs:
                square += (bb[2] - bb[0]) * (bb[3] - bb[1])
            return square / len(bbs) / np.prod(imsize) * 100
        else:
            return 0.

    @staticmethod
    def _plot_bb_colormap(class_bb: dict, colors: list, name_classes: list, data_type: str,
                          save_path: str, imgsize=(416, 416)):
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

    def _prepare_class_idx(self) -> dict:
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

    def _prepare_seed(self):
        if self.options.data.group == DatasetGroupChoice.keras or self.x_val:
            data_lenth = np.arange(len(self.y_true.get("val").get(list(self.y_true.get("val").keys())[0])))
        else:
            data_lenth = np.arange(len(self.options.dataframe.get("val")))
        np.random.shuffle(data_lenth)
        return data_lenth

    def _reformat_fit_logs(self, logs) -> dict:
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
                        'train': round_loss_metric(train_loss) if not math.isnan(float(train_loss)) else None,
                        'val': round_loss_metric(val_loss) if not math.isnan(float(val_loss)) else None,
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
                    interactive_log[out]['metrics'][metric_name] = {
                        'train': round_loss_metric(train_metric) if not math.isnan(float(train_metric)) else None,
                        'val': round_loss_metric(val_metric) if not math.isnan(float(val_metric)) else None
                    }

        if self.options.data.architecture in self.yolo_architecture:
            interactive_log['learning_rate'] = logs.get('optimizer.lr')
            interactive_log['output'] = {
                "train": {
                    "loss": {
                        'giou_loss': logs.get('giou_loss'),
                        'conf_loss': logs.get('conf_loss'),
                        'prob_loss': logs.get('prob_loss'),
                        'total_loss': logs.get('total_loss')
                    },
                    "metrics": {
                        'mAP50': logs.get('mAP50'),
                        'mAP95': logs.get('mAP95'),
                    }
                },
                "val": {
                    "loss": {
                        'giou_loss': logs.get('val_giou_loss'),
                        'conf_loss': logs.get('val_conf_loss'),
                        'prob_loss': logs.get('val_prob_loss'),
                        'total_loss': logs.get('val_total_loss')
                    },
                    "class_loss": {
                        'prob_loss': {},
                    },
                    "metrics": {
                        'mAP50': logs.get('val_mAP50'),
                        'mAP95': logs.get('val_mAP95'),
                    },
                    "class_metrics": {
                        'mAP50': {},
                        'mAP95': {},
                    }
                }
            }
            for name in self.options.data.outputs.get(list(self.options.data.outputs.keys())[0]).classes_names:
                interactive_log['output']['val']["class_loss"]['prob_loss'][name] = logs.get(f'val_prob_loss_{name}')
                interactive_log['output']['val']["class_metrics"]['mAP50'][name] = logs.get(f'val_mAP50_class_{name}')
                interactive_log['output']['val']["class_metrics"]['mAP95'][name] = logs.get(f'val_mAP95_class_{name}')

        return interactive_log

    def _reformat_y_pred(self, y_pred, sensitivity: float = 0.15, threashold: float = 0.1):
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
                            inverse_col = self.options.preprocessing.inverse_data(_options).get(int(out)).get(column)
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
                                self.options.preprocessing.inverse_data(_options).get(int(out)).get(column), axis=-1)
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

    def _update_log_history(self):
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
                                    round_loss_metric(
                                        self.current_logs.get(f"{out}").get('loss').get(loss_name).get(data_type)
                                    )
                            else:
                                self.log_history[f"{out}"]['loss'][loss_name][data_type].append(
                                    round_loss_metric(
                                        self.current_logs.get(f"{out}").get('loss').get(loss_name).get(data_type)
                                    )
                                )
                        # fill loss progress state
                        if data_idx or data_idx == 0:
                            self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['mean_log_history'][
                                data_idx] = \
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
                            self.log_history[f"{out}"]['progress_state']['loss'][loss_name][
                                'underfitting'][data_idx] = loss_underfitting
                            self.log_history[f"{out}"]['progress_state']['loss'][loss_name][
                                'overfitting'][data_idx] = loss_overfitting
                            self.log_history[f"{out}"]['progress_state']['loss'][loss_name][
                                'normal_state'][data_idx] = normal_state
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
                                        round_loss_metric(class_loss)
                                else:
                                    self.log_history[f"{out}"]['class_loss'][cls][loss_name].append(
                                        round_loss_metric(class_loss)
                                    )

                    for metric_name in self.log_history.get(f"{out}").get('metrics').keys():
                        for data_type in ['train', 'val']:
                            # fill metrics
                            if data_idx or data_idx == 0:
                                if self.current_logs:
                                    self.log_history[f"{out}"]['metrics'][metric_name][data_type][data_idx] = \
                                        round_loss_metric(
                                            self.current_logs.get(f"{out}").get('metrics').get(metric_name).get(
                                                data_type)
                                        )
                            else:
                                if self.current_logs:
                                    self.log_history[f"{out}"]['metrics'][metric_name][data_type].append(
                                        round_loss_metric(
                                            self.current_logs.get(f"{out}").get('metrics').get(metric_name).get(
                                                data_type)
                                        )
                                    )

                        if data_idx or data_idx == 0:
                            self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['mean_log_history'][
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
                            self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['mean_log_history'],
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
                            self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['underfitting'].append(
                                metric_underfittng)
                            self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['overfitting'].append(
                                metric_overfittng)
                            self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['normal_state'].append(
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
                                    self.log_history[f"{out}"]['class_metrics'][cls][metric_name][data_idx] = \
                                        round_loss_metric(class_metric)
                                else:
                                    self.log_history[f"{out}"]['class_metrics'][cls][metric_name].append(
                                        round_loss_metric(class_metric)
                                    )

            if self.options.data.architecture in self.yolo_architecture:
                out = list(self.options.data.outputs.keys())[0]
                classes_names = self.options.data.outputs.get(out).classes_names
                for key in self.log_history['output']["loss"].keys():
                    for data_type in ['train', 'val']:
                        self.log_history['output']["loss"][key][data_type].append(
                            round_loss_metric(self.current_logs.get('output').get(data_type).get('loss').get(key))
                        )
                for key in self.log_history['output']["metrics"].keys():
                    for data_type in ['train', 'val']:
                        self.log_history['output']["metrics"][key][data_type].append(
                            round_loss_metric(self.current_logs.get('output').get(data_type).get('metrics').get(key))
                        )
                for name in classes_names:
                    self.log_history['output']["class_loss"]['prob_loss'][name].append(
                        round_loss_metric(self.current_logs.get('output').get("val").get(
                            'class_loss').get("prob_loss").get(name))
                    )
                    self.log_history['output']["class_metrics"]['mAP50'][name].append(
                        round_loss_metric(self.current_logs.get('output').get("val").get(
                            'class_metrics').get("mAP50").get(name))
                    )
                    self.log_history['output']["class_metrics"]['mAP95'][name].append(
                        round_loss_metric(self.current_logs.get('output').get("val").get(
                            'class_metrics').get("mAP95").get(name))
                    )

                for loss_name in self.log_history['output']["loss"].keys():
                    # fill loss progress state
                    if data_idx or data_idx == 0:
                        self.log_history['output']['progress_state']['loss'][loss_name]['mean_log_history'][data_idx] = \
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
                            data_idx] = self._get_mean_log(self.log_history['output']['metrics'][metric_name]['val'])
                    else:
                        self.log_history['output']['progress_state']['metrics'][metric_name][
                            'mean_log_history'].append(
                            self._get_mean_log(self.log_history['output']['metrics'][metric_name]['val'])
                        )
                    metric_overfittng = self._evaluate_overfitting(
                        metric_name,
                        self.log_history['output']['progress_state']['metrics'][metric_name]['mean_log_history'],
                        metric_type='metric'
                    )
                    if metric_overfittng:
                        normal_state = False
                    else:
                        normal_state = True

                    if data_idx or data_idx == 0:
                        self.log_history['output']['progress_state']['metrics'][metric_name]['overfitting'][
                            data_idx] = metric_overfittng
                        self.log_history['output']['progress_state']['metrics'][metric_name]['normal_state'][
                            data_idx] = normal_state
                    else:
                        self.log_history['output']['progress_state']['metrics'][metric_name]['overfitting'].append(
                            metric_overfittng)
                        self.log_history['output']['progress_state']['metrics'][metric_name]['normal_state'].append(
                            normal_state)

    def _update_progress_table(self, epoch_time: float):
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
                "data": {}
            }
            self.progress_table[self.current_epoch]["data"][f"Прогресс обучения"] = {
                'loss': {},
                'metrics': {}
            }
            for loss in self.log_history['output']["loss"].keys():
                self.progress_table[self.current_epoch]["data"][f"Прогресс обучения"]["loss"] = {
                    'loss': f"{self.log_history.get('output').get('loss').get(loss).get('train')[-1]}",
                    'val_loss': f"{self.log_history.get('output').get('loss').get(loss).get('val')[-1]}"
                }
            for metric in self.log_history['output']["metrics"].keys():
                self.progress_table[self.current_epoch]["data"][f"Прогресс обучения»"]["metrics"][metric] = \
                    f"{self.log_history.get('output').get('metrics').get(metric).get('train')[-1]}"
                self.progress_table[self.current_epoch]["data"][f"Прогресс обучения»"]["metrics"][f"val_{metric}"] = \
                    f"{self.log_history.get('output').get('metrics').get(metric).get('val')[-1]}"

    def _get_loss_calculation(self, loss_obj, out: str, y_true, y_pred):
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

    def _get_metric_calculation(self, metric_name, metric_obj, out: str, y_true, y_pred, show_class=False):
        encoding = self.options.data.outputs.get(int(out)).encoding
        task = self.options.data.outputs.get(int(out)).task
        num_classes = self.options.data.outputs.get(int(out)).num_classes
        if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.TimeseriesTrend:
            if metric_name == Metric.Accuracy:
                metric_obj.update_state(
                    np.argmax(y_true, axis=-1) if encoding == LayerEncodingChoice.ohe else y_true,
                    np.argmax(y_pred, axis=-1)
                )
            if metric_name == Metric.BalancedRecall:
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
        elif task == LayerOutputTypeChoice.Segmentation or \
                (task == LayerOutputTypeChoice.TextSegmentation and encoding == LayerEncodingChoice.ohe):
            metric_obj.update_state(
                y_true if encoding == LayerEncodingChoice.ohe else to_categorical(y_true, num_classes),
                y_pred
            )
            metric_value = float(metric_obj.result().numpy())
        elif task == LayerOutputTypeChoice.TextSegmentation and encoding == LayerEncodingChoice.multi:
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
        if min(mean_log) != 0 or max(mean_log) != 0:
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

                train_list = self.log_history.get(f"{loss_graph_config.output_idx}").get('loss').get(
                    self.losses.get(f"{loss_graph_config.output_idx}")).get('train')
                best_train_value = min(train_list)
                best_train = fill_graph_plot_data(
                    x=[self.log_history.get("epochs")[train_list.index(best_train_value)]],
                    y=[best_train_value],
                    label="Лучший результат на тренировочной выборке"
                )
                train_plot = fill_graph_plot_data(
                    x=self.log_history.get("epochs"),
                    y=train_list,
                    label="Тренировочная выборка"
                )

                val_list = self.log_history.get(f"{loss_graph_config.output_idx}").get('loss').get(
                    self.losses.get(f"{loss_graph_config.output_idx}")).get("val")
                best_val_value = min(val_list)
                best_val = fill_graph_plot_data(
                    x=[self.log_history.get("epochs")[val_list.index(best_val_value)]],
                    y=[best_val_value],
                    label="Лучший результат на проверочной выборке"
                )
                val_plot = fill_graph_plot_data(
                    x=self.log_history.get("epochs"),
                    y=self.log_history.get(f"{loss_graph_config.output_idx}").get('loss').get(
                        self.losses.get(f"{loss_graph_config.output_idx}")).get("val"),
                    label="Проверочная выборка"
                )

                data_return.append(
                    fill_graph_front_structure(
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
                    fill_graph_front_structure(
                        _id=loss_graph_config.id,
                        _type='graphic',
                        graph_name=f"Выходной слой «{loss_graph_config.output_idx}» - График ошибки обучения по классам"
                                   f" - Эпоха №{self.log_history.get('epochs')[-1]}",
                        short_name=f"{loss_graph_config.output_idx} - График ошибки обучения по классам",
                        x_label="Эпоха",
                        y_label="Значение",
                        plot_data=[
                            fill_graph_plot_data(
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
            if metric_graph_config.show == MetricGraphShowChoice.model:
                min_max_mode = loss_metric_config.get("metric").get(metric_graph_config.show_metric.name).get("mode")
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

                train_list = self.log_history.get(f"{metric_graph_config.output_idx}").get('metrics').get(
                    metric_graph_config.show_metric.name).get("train")
                best_train_value = min(train_list) if min_max_mode == 'min' else max(train_list)
                best_train = fill_graph_plot_data(
                    x=[self.log_history.get("epochs")[train_list.index(best_train_value)]],
                    y=[best_train_value],
                    label="Лучший результат на тренировочной выборке"
                )
                train_plot = fill_graph_plot_data(
                    x=self.log_history.get("epochs"),
                    y=self.log_history.get(f"{metric_graph_config.output_idx}").get('metrics').get(
                        metric_graph_config.show_metric.name).get("train"),
                    label="Тренировочная выборка"
                )

                val_list = self.log_history.get(f"{metric_graph_config.output_idx}").get('metrics').get(
                    metric_graph_config.show_metric.name).get("val")
                best_val_value = min(val_list) if min_max_mode == 'min' else max(val_list)
                best_val = fill_graph_plot_data(
                    x=[self.log_history.get("epochs")[val_list.index(best_val_value)]],
                    y=[best_val_value],
                    label="Лучший результат на проверочной выборке"
                )
                val_plot = fill_graph_plot_data(
                    x=self.log_history.get("epochs"),
                    y=self.log_history.get(f"{metric_graph_config.output_idx}").get('metrics').get(
                        metric_graph_config.show_metric.name).get("val"),
                    label="Проверочная выборка"
                )
                data_return.append(
                    fill_graph_front_structure(
                        _id=metric_graph_config.id,
                        _type='graphic',
                        graph_name=f"Выходной слой «{metric_graph_config.output_idx}» - График метрики "
                                   f"{metric_graph_config.show_metric.name} - "
                                   f"Эпоха №{self.log_history.get('epochs')[-1]}",
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
                    fill_graph_front_structure(
                        _id=metric_graph_config.id,
                        _type='graphic',
                        graph_name=f"Выходной слой «{metric_graph_config.output_idx}» - График метрики "
                                   f"{metric_graph_config.show_metric.name} по классам - "
                                   f"Эпоха №{self.log_history.get('epochs')[-1]}",
                        short_name=f"{metric_graph_config.output_idx} - "
                                   f"{metric_graph_config.show_metric.name} по классам",
                        x_label="Эпоха",
                        y_label="Значение",
                        plot_data=[
                            fill_graph_plot_data(
                                x=self.log_history.get("epochs"),
                                y=self.log_history.get(f"{metric_graph_config.output_idx}").get('class_metrics').get(
                                    class_name).get(metric_graph_config.show_metric),
                                label=f"Класс {class_name}"
                            ) for class_name in
                            self.options.data.outputs.get(metric_graph_config.output_idx).classes_names
                        ],
                    )
                )
        return data_return

    def _get_intermediate_result_request(self) -> dict:
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
                            return_mode='callback'
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
                            depth=self.inverse_y_true.get("val").get(f"{out}")[self.example_idx[idx]].shape[-1],
                            show_stat=self.interactive_config.intermediate_result.show_statistic,
                            templates=[fill_graph_plot_data, fill_graph_front_structure]
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

        elif self.options.data.architecture in self.yolo_architecture and \
                self.yolo_interactive_config.intermediate_result.show_results:
            self._reformat_y_pred(
                y_pred=self.raw_y_pred,
                sensitivity=self.yolo_interactive_config.intermediate_result.sensitivity,
                threashold=self.yolo_interactive_config.intermediate_result.threashold
            )
            for idx in range(self.yolo_interactive_config.intermediate_result.num_examples):
                return_data[f"{idx + 1}"] = {
                    'initial_data': {},
                    'true_value': {},
                    'predict_value': {},
                    'tags_color': {},
                    'statistic_values': {}
                }
                image_path = os.path.join(
                    self.dataset_path, self.options.dataframe.get('val').iat[self.example_idx[idx], 0])
                out = self.yolo_interactive_config.intermediate_result.box_channel
                data = CreateArray().postprocess_object_detection(
                    predict_array=copy.deepcopy(self.y_pred.get(out)[self.example_idx[idx]]),
                    true_array=self.y_true.get(out)[self.example_idx[idx]],
                    image_path=image_path,
                    colors=self.class_colors,
                    sensitivity=self.yolo_interactive_config.intermediate_result.sensitivity,
                    image_id=idx,
                    image_size=self.options.data.inputs.get(list(self.options.data.inputs.keys())[0]).shape[:2],
                    name_classes=self.options.data.outputs.get(list(self.options.data.outputs.keys())[0]).classes_names,
                    save_path=self.preset_path,
                    return_mode='callback',
                    show_stat=self.yolo_interactive_config.intermediate_result.show_statistic
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

    def _get_statistic_data_request(self) -> list:
        return_data = []
        _id = 1
        if self.options.data.architecture in self.basic_architecture:
            for out in self.interactive_config.statistic_data.output_id:
                task = self.options.data.outputs.get(out).task
                encoding = self.options.data.outputs.get(out).encoding
                if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.TimeseriesTrend and \
                        encoding != LayerEncodingChoice.multi:
                    cm, cm_percent = get_confusion_matrix(
                        np.argmax(self.y_true.get("val").get(f'{out}'), axis=-1) if encoding == LayerEncodingChoice.ohe
                        else self.y_true.get("val").get(f'{out}'),
                        np.argmax(self.y_pred.get(f'{out}'), axis=-1),
                        get_percent=True
                    )
                    return_data.append(
                        fill_heatmap_front_structure(
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
                    cm, cm_percent = get_confusion_matrix(
                        np.argmax(self.y_true.get("val").get(f"{out}"), axis=-1).reshape(
                            np.prod(np.argmax(self.y_true.get("val").get(f"{out}"), axis=-1).shape)).astype('int'),
                        np.argmax(self.y_pred.get(f'{out}'), axis=-1).reshape(
                            np.prod(np.argmax(self.y_pred.get(f'{out}'), axis=-1).shape)).astype('int'),
                        get_percent=True
                    )
                    return_data.append(
                        fill_heatmap_front_structure(
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

                    report = get_classification_report(
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
                        fill_table_front_structure(
                            _id=_id,
                            graph_name=f"Выходной слой «{out}» - Отчет по классам",
                            plot_data=report
                        )
                    )
                    _id += 1

                elif task == LayerOutputTypeChoice.Regression:
                    y_true = self.inverse_y_true.get("val").get(f'{out}').squeeze()
                    y_pred = self.inverse_y_pred.get(f'{out}').squeeze()
                    x_scatter, y_scatter = get_scatter(y_true, y_pred)
                    return_data.append(
                        fill_graph_front_structure(
                            _id=_id,
                            _type='scatter',
                            graph_name=f"Выходной слой «{out}» - Скаттер",
                            short_name=f"{out} - Скаттер",
                            x_label="Истинные значения",
                            y_label="Предсказанные значения",
                            plot_data=[fill_graph_plot_data(x=x_scatter, y=y_scatter)],
                        )
                    )
                    _id += 1
                    deviation = (y_pred - y_true) * 100 / y_true
                    x_mae, y_mae = get_distribution_histogram(np.abs(deviation), categorical=False)
                    return_data.append(
                        fill_graph_front_structure(
                            _id=_id,
                            _type='bar',
                            graph_name=f'Выходной слой «{out}» - Распределение абсолютной ошибки',
                            short_name=f"{out} - Распределение MAE",
                            x_label="Абсолютная ошибка",
                            y_label="Значение",
                            plot_data=[fill_graph_plot_data(x=x_mae, y=y_mae)],
                        )
                    )
                    _id += 1
                    x_me, y_me = get_distribution_histogram(deviation, categorical=False)
                    return_data.append(
                        fill_graph_front_structure(
                            _id=_id,
                            _type='bar',
                            graph_name=f'Выходной слой «{out}» - Распределение ошибки',
                            short_name=f"{out} - Распределение ME",
                            x_label="Ошибка",
                            y_label="Значение",
                            plot_data=[fill_graph_plot_data(x=x_me, y=y_me)],
                        )
                    )
                    _id += 1

                elif task == LayerOutputTypeChoice.Timeseries:
                    for i, channel_name in enumerate(self.options.data.columns.get(out).keys()):
                        for step in range(self.y_true.get("val").get(f'{out}').shape[-1]):
                            y_true = self.inverse_y_true.get("val").get(f"{out}")[:, step, i].astype('float')
                            y_pred = self.inverse_y_pred.get(f"{out}")[:, step, i].astype('float')

                            return_data.append(
                                fill_graph_front_structure(
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
                                        fill_graph_plot_data(
                                            x=np.arange(len(y_true)).astype('int').tolist(),
                                            y=y_true.tolist(),
                                            label="Истинное значение"
                                        ),
                                        fill_graph_plot_data(
                                            x=np.arange(len(y_true)).astype('int').tolist(),
                                            y=y_pred.tolist(),
                                            label="Предсказанное значение"
                                        )
                                    ],
                                )
                            )
                            _id += 1
                            x_axis, auto_corr_true, auto_corr_pred = get_autocorrelation_graphic(
                                y_true, y_pred, depth=10
                            )
                            return_data.append(
                                fill_graph_front_structure(
                                    _id=_id,
                                    _type='graphic',
                                    graph_name=f"Выходной слой «{out}» - Автокорреляция канала "
                                               f"«{channel_name.split('_', 1)[-1]}» на {step + 1} шаг"
                                               f"{'а' if step else ''} вперед",
                                    short_name=f"{out} - Автокорреляция канала «{channel_name.split('_', 1)[-1]}»",
                                    x_label="Время",
                                    y_label="Значение",
                                    plot_data=[
                                        fill_graph_plot_data(x=x_axis, y=auto_corr_true, label="Истинное значение"),
                                        fill_graph_plot_data(x=x_axis, y=auto_corr_pred, label="Предсказанное значение")
                                    ],
                                )
                            )
                            _id += 1
                            deviation = (y_pred - y_true) * 100 / y_true
                            x_mae, y_mae = get_distribution_histogram(np.abs(deviation), categorical=False)
                            return_data.append(
                                fill_graph_front_structure(
                                    _id=_id,
                                    _type='bar',
                                    graph_name=f"Выходной слой «{out}» - Распределение абсолютной ошибки канала "
                                               f"«{channel_name.split('_', 1)[-1]}» на {step + 1} шаг"
                                               f"{'ов' if step + 1 == 1 else ''} вперед",
                                    short_name=f"{out} - Распределение MAE канала «{channel_name.split('_', 1)[-1]}»",
                                    x_label="Абсолютная ошибка",
                                    y_label="Значение",
                                    plot_data=[fill_graph_plot_data(x=x_mae, y=y_mae)],
                                )
                            )
                            _id += 1
                            x_me, y_me = get_distribution_histogram(deviation, categorical=False)
                            return_data.append(
                                fill_graph_front_structure(
                                    _id=_id,
                                    _type='bar',
                                    graph_name=f"Выходной слой «{out}» - Распределение ошибки канала "
                                               f"«{channel_name.split('_', 1)[-1]}» на {step + 1} шаг"
                                               f"{'ов' if step + 1 == 1 else ''} вперед",
                                    short_name=f"{out} - Распределение ME канала «{channel_name.split('_', 1)[-1]}»",
                                    x_label="Ошибка",
                                    y_label="Значение",
                                    plot_data=[fill_graph_plot_data(x=x_me, y=y_me)],
                                )
                            )
                            _id += 1

                elif task == LayerOutputTypeChoice.Dataframe:
                    pass

                else:
                    pass

        elif self.options.data.architecture in self.yolo_architecture:
            box_channel = self.yolo_interactive_config.statistic_data.box_channel
            name_classes = self.options.data.outputs.get(list(self.options.data.outputs.keys())[0]).classes_names
            self._reformat_y_pred(
                y_pred=self.raw_y_pred,
                sensitivity=self.yolo_interactive_config.statistic_data.sensitivity,
                threashold=self.yolo_interactive_config.statistic_data.threashold
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
                    sensitivity=self.yolo_interactive_config.statistic_data.sensitivity
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
                class_accuracy_hist[class_name] = np.round(np.mean(class_accuracy_hist[class_name]) * 100, 2).item() if \
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

            return_data.append(
                fill_heatmap_front_structure(
                    _id=1,
                    _type="heatmap",
                    graph_name=f"Бокс-канал «{box_channel}» - Матрица неточностей определения классов",
                    short_name=f"{box_channel} - Матрица классов",
                    x_label="Предсказание",
                    y_label="Истинное значение",
                    labels=name_classes,
                    data_array=class_matrix,
                    data_percent_array=class_matrix_percent,
                )
            )
            return_data.append(
                fill_heatmap_front_structure(
                    _id=2,
                    _type="heatmap",
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
                fill_graph_front_structure(
                    _id=3,
                    _type='histogram',
                    graph_name=f'Бокс-канал «{box_channel}» - Средняя точность определеня  классов',
                    short_name=f"{box_channel} - точность классов",
                    x_label="Имя класса",
                    y_label="Средняя точность, %",
                    plot_data=[
                        fill_graph_plot_data(x=name_classes, y=[class_accuracy_hist[i] for i in name_classes])
                    ],
                )
            )
            return_data.append(
                fill_graph_front_structure(
                    _id=4,
                    _type='histogram',
                    graph_name=f'Бокс-канал «{box_channel}» - Средняя ошибка определеня  классов',
                    short_name=f"{box_channel} - ошибка классов",
                    x_label="Имя класса",
                    y_label="Средняя ошибка, %",
                    plot_data=[
                        fill_graph_plot_data(x=name_classes, y=[class_loss_hist[i] for i in name_classes])
                    ],
                )
            )
            return_data.append(
                fill_graph_front_structure(
                    _id=5,
                    _type='histogram',
                    graph_name=f'Бокс-канал «{box_channel}» - '
                               f'Средняя точность определения  координат объекта класса (MeanIoU)',
                    short_name=f"{box_channel} - координаты классов",
                    x_label="Имя класса",
                    y_label="Средняя точность, %",
                    plot_data=[
                        fill_graph_plot_data(x=name_classes, y=[class_coord_accuracy[i] for i in name_classes])
                    ],
                )
            )

        else:
            pass

        return return_data

    def _get_balance_data_request(self) -> list:
        return_data = []
        _id = 0
        if self.options.data.architecture in self.basic_architecture:
            for out in self.options.data.outputs.keys():
                task = self.options.data.outputs.get(out).task

                if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.TimeseriesTrend:
                    for class_type in self.dataset_balance.get(f"{out}").keys():
                        preset = {}
                        for data_type in ['train', 'val']:
                            class_names, class_count = CreateArray().sort_dict(
                                dict_to_sort=self.dataset_balance.get(f"{out}").get(class_type).get(data_type),
                                mode=self.interactive_config.data_balance.sorted.name
                            )
                            preset[data_type] = fill_graph_front_structure(
                                _id=_id,
                                _type='histogram',
                                type_data=data_type,
                                graph_name=f"Выход {out} - "
                                           f"{'Тренировочная' if data_type == 'train' else 'Проверочная'} выборка",
                                short_name=f"{out} - {'Тренировочная' if data_type == 'train' else 'Проверочная'}",
                                x_label="Название класса",
                                y_label="Значение",
                                plot_data=[fill_graph_plot_data(x=class_names, y=class_count)],
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
                                preset[data_type] = fill_graph_front_structure(
                                    _id=_id,
                                    _type='histogram',
                                    type_data=data_type,
                                    graph_name=f"Выход {out} - {'Тренировочная' if data_type == 'train' else 'Проверочная'} выборка - " 
                                               f"{'баланс присутсвия' if class_type == 'presence_balance' else 'процент пространства'}",
                                    short_name=f"{'Тренировочная' if data_type == 'train' else 'Проверочная'} - "
                                               f"{'присутсвие' if class_type == 'presence_balance' else 'пространство'}",
                                    x_label="Название класса",
                                    y_label="Значение",
                                    plot_data=[fill_graph_plot_data(x=names, y=count)],
                                )
                                _id += 1
                            return_data.append(preset)

                        if class_type == "colormap":
                            for class_name, map_link in self.dataset_balance.get(f"{out}").get('colormap').get(
                                    'train').items():
                                preset = {}
                                for data_type in ['train', 'val']:
                                    preset[data_type] = fill_graph_front_structure(
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
                                preset[data_type] = fill_graph_front_structure(
                                    _id=_id,
                                    _type='histogram',
                                    type_data=data_type,
                                    graph_name=f"Выход {out} - {'Тренировочная' if data_type == 'train' else 'Проверочная'} выборка - "
                                               f"{'баланс присутсвия' if class_type == 'presence_balance' else 'процент пространства'}",
                                    short_name=f"{'Тренировочная' if data_type == 'train' else 'Проверочная'} - "
                                               f"{'присутсвие' if class_type == 'presence_balance' else 'процент'}",
                                    x_label="Название класса",
                                    y_label="Значение",
                                    plot_data=[fill_graph_plot_data(x=names, y=count)],
                                )
                                _id += 1
                        return_data.append(preset)

                elif task == LayerOutputTypeChoice.Regression:
                    for class_type in self.dataset_balance[f"{out}"].keys():
                        if class_type == 'histogram':
                            for column in self.dataset_balance[f"{out}"][class_type]["train"].keys():
                                preset = {}
                                for data_type in ["train", "val"]:
                                    histogram = self.dataset_balance[f"{out}"][class_type][data_type][column]
                                    data_type_name = "Тренировочная" if data_type == "train" else "Проверочная"
                                    preset[data_type] = fill_graph_front_structure(
                                        _id=_id,
                                        _type=histogram.get("type"),
                                        type_data=data_type,
                                        graph_name=f"Выход {out} - {data_type_name} выборка - "
                                                   f"Гистограмма распределения колонки «{histogram['name']}»",
                                        short_name=f"{data_type_name} - {histogram['name']}",
                                        x_label="Значение",
                                        y_label="Количество",
                                        plot_data=[
                                            fill_graph_plot_data(x=histogram.get("x"), y=histogram.get("y"))],
                                    )
                                    _id += 1
                                return_data.append(preset)

                        if class_type == 'correlation':
                            preset = {}
                            for data_type in ["train", "val"]:
                                data_type_name = "Тренировочная" if data_type == "train" else "Проверочная"
                                preset[data_type] = fill_heatmap_front_structure(
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
                                graph_type = self.dataset_balance[f"{out}"][class_type][channel_name][data_type]['type']
                                data_type_name = "Тренировочная" if data_type == "train" else "Проверочная"
                                y_true = self.options.dataframe.get(data_type)[channel_name].to_list()
                                if class_type == 'graphic':
                                    x_graph_axis = np.arange(len(y_true)).astype('float').tolist()
                                    plot_data = [fill_graph_plot_data(x=x_graph_axis, y=y_true)]
                                    graph_name = f'Выход {out} - {data_type_name} выборка - ' \
                                                 f'График канала «{channel_name.split("_", 1)[-1]}»'
                                    short_name = f'{data_type_name} - «{channel_name.split("_", 1)[-1]}»'
                                    x_label = "Время"
                                    y_label = "Величина"
                                if class_type == 'dense_histogram':
                                    x_hist, y_hist = get_distribution_histogram(y_true, categorical=False)
                                    plot_data = [fill_graph_plot_data(x=x_hist, y=y_hist)]
                                    graph_name = f'Выход {out} - {data_type_name} выборка - ' \
                                                 f'Гистограмма плотности канала «{channel_name.split("_", 1)[-1]}»'
                                    short_name = f'{data_type_name} - Гистограмма «{channel_name.split("_", 1)[-1]}»'
                                    x_label = "Значение"
                                    y_label = "Количество"
                                preset[data_type] = fill_graph_front_structure(
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
                        preset[data_type] = fill_graph_front_structure(
                            _id=_id,
                            _type='histogram',
                            type_data=data_type,
                            graph_name=f"{'Тренировочная' if data_type == 'train' else 'Проверочная'} выборка - "
                                       f"{'баланс присутсвия' if class_type == 'class_count' else 'процент пространства'}",
                            short_name=f"{'Тренировочная' if data_type == 'train' else 'Проверочная'} - "
                                       f"{'присутсвие' if class_type == 'class_count' else 'пространство'}",
                            x_label="Название класса",
                            y_label="Значение",
                            plot_data=[fill_graph_plot_data(x=names, y=count)],
                        )
                        _id += 1
                    return_data.append(preset)

                if class_type == "colormap":
                    classes_name = sorted(list(self.dataset_balance.get("output").get('colormap').get('train').keys()))
                    for class_name in classes_name:
                        preset = {}
                        for data_type in ['train', 'val']:
                            _dict = self.dataset_balance.get("output").get('colormap').get(data_type)
                            preset[data_type] = fill_graph_front_structure(
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

