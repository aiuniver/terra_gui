import copy
import math
import os
import random
import re
import string
from typing import Union, Optional

from tensorflow.keras.utils import to_categorical
import numpy as np

from terra_ai import progress
from terra_ai.callbacks.classification_callbacks import ImageClassificationCallback, TextClassificationCallback, \
    AudioClassificationCallback, VideoClassificationCallback, DataframeClassificationCallback, TimeseriesTrendCallback
from terra_ai.callbacks.object_detection_callbacks import YoloV3Callback, YoloV4Callback
from terra_ai.callbacks.regression_callbacks import DataframeRegressionCallback
from terra_ai.callbacks.segmentation_callbacks import ImageSegmentationCallback, TextSegmentationCallback
from terra_ai.callbacks.time_series_callbacks import TimeseriesCallback
from terra_ai.callbacks.utils import loss_metric_config, round_loss_metric, fill_graph_plot_data, \
    fill_graph_front_structure, reformat_metrics, prepare_loss_obj, prepare_metric_obj, get_classes_colors, print_error
from terra_ai.data.datasets.extra import LayerOutputTypeChoice, DatasetGroupChoice, LayerEncodingChoice, \
    LayerInputTypeChoice
from terra_ai.data.presets.training import Metric
from terra_ai.data.training.extra import LossGraphShowChoice, MetricGraphShowChoice, MetricChoice, ArchitectureChoice
from terra_ai.data.training.train import InteractiveData
from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.training.customlosses import UnscaledMAE, FScore, BalancedFScore, BalancedRecall, BalancedPrecision
from terra_ai.utils import decamelize

__version__ = 0.085


class InteractiveCallback:
    """Callback for interactive requests"""

    def __init__(self):
        self.name = 'InteractiveCallback'
        self.first_error = True
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
        self.callback = None

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
        self.class_architecture = [ArchitectureChoice.ImageClassification, ArchitectureChoice.TimeseriesTrend,
                                   ArchitectureChoice.ImageSegmentation, ArchitectureChoice.TextSegmentation,
                                   ArchitectureChoice.TextClassification, ArchitectureChoice.AudioClassification,
                                   ArchitectureChoice.VideoClassification, ArchitectureChoice.DataframeClassification,
                                   ArchitectureChoice.YoloV3, ArchitectureChoice.YoloV4]
        self.classification_architecture = [
            ArchitectureChoice.ImageClassification, ArchitectureChoice.TimeseriesTrend,
            ArchitectureChoice.TextClassification, ArchitectureChoice.AudioClassification,
            ArchitectureChoice.VideoClassification, ArchitectureChoice.DataframeClassification,
        ]

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

        self.options = dataset
        self._callback_router(dataset)
        self._class_metric_list()
        print('\nself._class_metric_list()', self.class_graphics)
        print('set_attributes', dataset.data.architecture)
        self.preset_path = os.path.join(training_path, "presets")
        if not os.path.exists(self.preset_path):
            os.mkdir(self.preset_path)
        self.interactive_config = initial_config
        if dataset.data.architecture in self.basic_architecture:
            self.losses = losses
            self.metrics = reformat_metrics(metrics)
            self.loss_obj = prepare_loss_obj(losses)
            self.metrics_obj = prepare_metric_obj(metrics)
        self.dataset_path = dataset_path
        self.class_colors = get_classes_colors(dataset)
        self.x_val, self.inverse_x_val = self.callback.get_x_array(dataset)
        self.y_true, self.inverse_y_true = self.callback.get_y_true(dataset, dataset_path)
        if not self.log_history:
            self._prepare_null_log_history_template()
        self.dataset_balance = self.callback.dataset_balance(
            options=self.options, y_true=self.y_true, preset_path=self.preset_path, class_colors=self.class_colors
        )
        if dataset.data.architecture in self.classification_architecture:
            self.class_idx = self.callback.prepare_class_idx(self.y_true, self.options)
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

    def update_state(self, y_pred, y_true=None, fit_logs=None, current_epoch_time=None,
                     on_epoch_end_flag=False) -> dict:
        if self.log_history:
            if y_pred is not None:
                if self.options.data.architecture in self.basic_architecture:
                    self.y_pred, self.inverse_y_pred = self.callback.get_y_pred(self.y_true, y_pred, self.options)
                    out = f"{self.interactive_config.intermediate_result.main_output}"
                    self.example_idx = self.callback.prepare_example_idx_to_show(
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
                    self.y_pred = self.callback.get_y_pred(
                        y_pred=y_pred, options=self.options,
                        sensitivity=self.interactive_config.intermediate_result.sensitivity,
                        threashold=self.interactive_config.intermediate_result.threashold
                    )
                    self.raw_y_true = y_true
                    self.example_idx, _ = self.callback.prepare_example_idx_to_show(
                        array=self.y_pred,
                        true_array=self.y_true,
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
                    self._update_log_history()
                    self._update_progress_table(current_epoch_time)
                    if self.interactive_config.intermediate_result.autoupdate:
                        self.intermediate_result = self.callback.intermediate_result_request(
                            options=self.options,
                            interactive_config=self.interactive_config,
                            example_idx=self.example_idx,
                            dataset_path=self.dataset_path,
                            preset_path=self.preset_path,
                            x_val=self.x_val,
                            inverse_x_val=self.inverse_x_val,
                            y_pred=self.y_pred,
                            inverse_y_pred=self.inverse_y_pred,
                            y_true=self.y_true,
                            inverse_y_true=self.inverse_y_true,
                            class_colors=self.class_colors,
                        )
                    if self.options.data.architecture in self.basic_architecture and \
                            self.interactive_config.statistic_data.output_id \
                            and self.interactive_config.statistic_data.autoupdate:
                        self.statistic_result = self.callback.statistic_data_request(
                            interactive_config=self.interactive_config,
                            options=self.options,
                            y_true=self.y_true,
                            inverse_y_true=self.inverse_y_true,
                            y_pred=self.y_pred,
                            inverse_y_pred=self.inverse_y_pred,
                        )
                    if self.options.data.architecture in self.yolo_architecture and \
                            self.interactive_config.statistic_data.box_channel \
                            and self.interactive_config.statistic_data.autoupdate:
                        self.statistic_result = self.callback.statistic_data_request(
                            interactive_config=self.interactive_config,
                            options=self.options,
                            y_true=self.y_true,
                            y_pred=self.y_pred,
                            inverse_y_pred=self.inverse_y_pred,
                            inverse_y_true=self.inverse_y_true
                        )
                else:
                    self.intermediate_result = self.callback.intermediate_result_request(
                        options=self.options,
                        interactive_config=self.interactive_config,
                        example_idx=self.example_idx,
                        dataset_path=self.dataset_path,
                        preset_path=self.preset_path,
                        x_val=self.x_val,
                        inverse_x_val=self.inverse_x_val,
                        y_pred=self.y_pred,
                        inverse_y_pred=self.inverse_y_pred,
                        y_true=self.y_true,
                        inverse_y_true=self.inverse_y_true,
                        class_colors=self.class_colors,
                        # raw_y_pred=self.raw_y_pred
                    )
                    if self.options.data.architecture in self.basic_architecture and \
                            self.interactive_config.statistic_data.output_id:
                        self.statistic_result = self.callback.statistic_data_request(
                            interactive_config=self.interactive_config,
                            options=self.options,
                            y_true=self.y_true,
                            y_pred=self.y_pred,
                            inverse_y_pred=self.inverse_y_pred,
                            inverse_y_true=self.inverse_y_true,
                        )
                    if self.options.data.architecture in self.yolo_architecture and \
                            self.interactive_config.statistic_data.box_channel:
                        self.statistic_result = self.callback.statistic_data_request(
                            interactive_config=self.interactive_config,
                            options=self.options,
                            y_true=self.y_true,
                            y_pred=self.y_pred,
                            inverse_y_pred=self.inverse_y_pred,
                            inverse_y_true=self.inverse_y_true,
                        )
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
                'data_balance': self.callback.balance_data_request(
                    options=self.options,
                    dataset_balance=self.dataset_balance,
                    interactive_config=self.interactive_config
                ),
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
                    self.example_idx = self.callback.prepare_example_idx_to_show(
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
                    self.intermediate_result = self.callback.intermediate_result_request(
                        options=self.options,
                        interactive_config=self.interactive_config,
                        example_idx=self.example_idx,
                        dataset_path=self.dataset_path,
                        preset_path=self.preset_path,
                        x_val=self.x_val,
                        inverse_x_val=self.inverse_x_val,
                        y_pred=self.y_pred,
                        inverse_y_pred=self.inverse_y_pred,
                        y_true=self.y_true,
                        inverse_y_true=self.inverse_y_true,
                        class_colors=self.class_colors,
                        raw_y_pred=self.raw_y_pred
                    )
                    if self.interactive_config.statistic_data.output_id:
                        self.statistic_result = self.callback.statistic_data_request(
                            interactive_config=self.interactive_config,
                            options=self.options,
                            y_true=self.y_true,
                            y_pred=self.y_pred,
                            inverse_y_pred=self.inverse_y_pred,
                            inverse_y_true=self.inverse_y_true,
                        )

            if self.options.data.architecture in self.yolo_architecture:
                if self.interactive_config.intermediate_result.show_results:
                    self.y_pred = self.callback.get_y_pred(
                        y_pred=self.raw_y_pred, options=self.options,
                        sensitivity=self.interactive_config.intermediate_result.sensitivity,
                        threashold=self.interactive_config.intermediate_result.threashold
                    )
                    self.example_idx, _ = self.callback.prepare_example_idx_to_show(
                        array=self.y_pred,
                        true_array=self.y_true,
                        name_classes=self.options.data.outputs.get(
                            list(self.options.data.outputs.keys())[0]).classes_names,
                        box_channel=self.interactive_config.intermediate_result.box_channel,
                        count=self.interactive_config.intermediate_result.num_examples,
                        choice_type=self.interactive_config.intermediate_result.example_choice_type,
                        seed_idx=self.seed_idx,
                        sensitivity=self.interactive_config.intermediate_result.sensitivity,
                    )
                    self.intermediate_result = self.callback.intermediate_result_request(
                        options=self.options,
                        interactive_config=self.interactive_config,
                        example_idx=self.example_idx,
                        dataset_path=self.dataset_path,
                        preset_path=self.preset_path,
                        x_val=self.x_val,
                        inverse_x_val=self.inverse_x_val,
                        y_pred=self.y_pred,
                        inverse_y_pred=self.inverse_y_pred,
                        y_true=self.y_true,
                        inverse_y_true=self.inverse_y_true,
                        class_colors=self.class_colors,
                    )
                    self.statistic_result = self.callback.statistic_data_request(
                        interactive_config=self.interactive_config,
                        options=self.options,
                        y_true=self.y_true,
                        y_pred=self.y_pred,
                        inverse_y_pred=self.inverse_y_pred,
                        inverse_y_true=self.inverse_y_true,
                    )

            self.random_key = ''.join(random.sample(string.ascii_letters + string.digits, 16))
            self.train_progress['train_data'] = {
                'update': self.random_key,
                "class_graphics": self.class_graphics,
                'loss_graphs': self._get_loss_graph_data_request(),
                'metric_graphs': self._get_metric_graph_data_request(),
                'intermediate_result': self.intermediate_result,
                'progress_table': self.progress_table,
                'statistic_data': self.statistic_result,
                'data_balance': self.callback.balance_data_request(
                    options=self.options,
                    dataset_balance=self.dataset_balance,
                    interactive_config=self.interactive_config
                ),
                'addtrain_epochs': self.addtrain_epochs,
            }
            progress.pool(
                self.progress_name,
                data=self.train_progress,
                finished=False,
            )
            return self.train_progress

    def _callback_router(self, dataset: PrepareDataset):
        method_name = '_callback_router'
        try:
            if dataset.data.architecture == ArchitectureChoice.Basic:
                for out in dataset.data.outputs.keys():
                    if dataset.data.outputs.get(out).task == LayerOutputTypeChoice.Classification:
                        for inp in dataset.data.inputs.keys():
                            if dataset.data.inputs.get(inp).task == LayerInputTypeChoice.Image:
                                self.options.data.architecture = ArchitectureChoice.ImageClassification
                                self.callback = ImageClassificationCallback()
                            elif dataset.data.inputs.get(inp).task == LayerInputTypeChoice.Text:
                                self.options.data.architecture = ArchitectureChoice.TextClassification
                                self.callback = TextClassificationCallback()
                            elif dataset.data.inputs.get(inp).task == LayerInputTypeChoice.Audio:
                                self.options.data.architecture = ArchitectureChoice.AudioClassification
                                self.callback = AudioClassificationCallback()
                            elif dataset.data.inputs.get(inp).task == LayerInputTypeChoice.Video:
                                self.options.data.architecture = ArchitectureChoice.VideoClassification
                                self.callback = VideoClassificationCallback()
                            elif dataset.data.inputs.get(inp).task == LayerInputTypeChoice.Dataframe:
                                self.options.data.architecture = ArchitectureChoice.DataframeClassification
                                self.callback = DataframeClassificationCallback()
                            else:
                                pass
                    elif dataset.data.outputs.get(out).task == LayerOutputTypeChoice.TimeseriesTrend:
                        self.options.data.architecture = ArchitectureChoice.TimeseriesTrend
                        self.callback = TimeseriesTrendCallback()
                    elif dataset.data.outputs.get(out).task == LayerOutputTypeChoice.Regression:
                        self.options.data.architecture = ArchitectureChoice.DataframeRegression
                        self.callback = DataframeRegressionCallback()
                    elif dataset.data.outputs.get(out).task == LayerOutputTypeChoice.Timeseries:
                        self.options.data.architecture = ArchitectureChoice.Timeseries
                        self.callback = TimeseriesCallback()
                    elif dataset.data.outputs.get(out).task == LayerOutputTypeChoice.Segmentation:
                        self.options.data.architecture = ArchitectureChoice.ImageSegmentation
                        self.callback = ImageSegmentationCallback()
                    elif dataset.data.outputs.get(out).task == LayerOutputTypeChoice.TextSegmentation:
                        self.options.data.architecture = ArchitectureChoice.TextSegmentation
                        self.callback = TextSegmentationCallback()
                    else:
                        pass
            elif dataset.data.architecture == ArchitectureChoice.ImageClassification:
                self.callback = ImageClassificationCallback()
            elif dataset.data.architecture == ArchitectureChoice.ImageSegmentation:
                self.callback = ImageSegmentationCallback()
            elif dataset.data.architecture == ArchitectureChoice.TextSegmentation:
                self.callback = TextSegmentationCallback()
            elif dataset.data.architecture == ArchitectureChoice.TextClassification:
                self.callback = TextClassificationCallback()
            elif dataset.data.architecture == ArchitectureChoice.AudioClassification:
                self.callback = AudioClassificationCallback()
            elif dataset.data.architecture == ArchitectureChoice.VideoClassification:
                self.callback = VideoClassificationCallback()
            elif dataset.data.architecture == ArchitectureChoice.DataframeClassification:
                self.callback = DataframeClassificationCallback()
            elif dataset.data.architecture == ArchitectureChoice.DataframeRegression:
                self.callback = DataframeRegressionCallback()
            elif dataset.data.architecture == ArchitectureChoice.Timeseries:
                self.callback = TimeseriesCallback()
            elif dataset.data.architecture == ArchitectureChoice.TimeseriesTrend:
                self.callback = TimeseriesTrendCallback()
            elif dataset.data.architecture == ArchitectureChoice.YoloV3:
                self.callback = YoloV3Callback()
            elif dataset.data.architecture == ArchitectureChoice.YoloV4:
                self.callback = YoloV4Callback()
            else:
                pass
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _class_metric_list(self):
        method_name = '_class_metric_list'
        try:
            self.class_graphics = {}
            if self.options.data.architecture in self.class_architecture:
                for out in self.options.data.outputs.keys():
                    self.class_graphics[out] = True
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _prepare_null_log_history_template(self):
        method_name = '_prepare_null_log_history_template'
        try:
            self.log_history["epochs"] = []
            if self.options.data.architecture in self.basic_architecture:
                for out in self.losses.keys():
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

                    if self.options.data.architecture in self.class_architecture:
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
                        'giou_loss': {"train": [], "val": []}, 'conf_loss': {"train": [], "val": []},
                        'prob_loss': {"train": [], "val": []}, 'total_loss': {"train": [], "val": []}
                    },
                    "class_loss": {'prob_loss': {}},
                    "metrics": {'mAP50': []},
                    "class_metrics": {'mAP50': {}, 'mAP95': {}},
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
                            'mAP50': {"mean_log_history": [], "normal_state": [], "overfitting": []},
                            # 'mAP95': {"mean_log_history": [], "normal_state": [], "overfitting": []},
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

    def _prepare_seed(self):
        method_name = '_prepare_seed'
        try:
            if self.options.data.architecture in self.yolo_architecture:
                example_idx = np.arange(len(self.options.dataframe.get("val")))
                np.random.shuffle(example_idx)
            elif self.options.data.architecture in self.basic_architecture:
                output = self.interactive_config.intermediate_result.main_output
                example_idx = []
                if self.options.data.architecture in self.classification_architecture:
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
                        if metric_name == MetricChoice.BalancedRecall:
                            m = BalancedRecall()
                            m.update_state(y_true=self.y_true.get('val').get(out), y_pred=self.y_pred.get(out))
                            val_metric = m.result().numpy().item()
                        if metric_name == MetricChoice.BalancedPrecision:
                            m = BalancedPrecision()
                            m.update_state(y_true=self.y_true.get('val').get(out), y_pred=self.y_pred.get(out))
                            val_metric = m.result().numpy().item()
                        if metric_name == MetricChoice.BalancedFScore:
                            m = BalancedFScore()
                            m.update_state(y_true=self.y_true.get('val').get(out), y_pred=self.y_pred.get(out))
                            val_metric = m.result().numpy().item()
                        if metric_name == MetricChoice.FScore:
                            m = FScore()
                            m.update_state(y_true=self.y_true.get('val').get(out), y_pred=self.y_pred.get(out))
                            val_metric = m.result().numpy().item()
                        interactive_log[out]['metrics'][metric_name] = {
                            'train': round_loss_metric(train_metric) if not math.isnan(
                                float(train_metric)) else None,
                            'val': round_loss_metric(val_metric) if not math.isnan(float(val_metric)) else None
                        }

            if self.options.data.architecture in self.yolo_architecture:
                interactive_log['learning_rate'] = round_loss_metric(logs.get('optimizer.lr'))
                interactive_log['output'] = {
                    "train": {
                        "loss": {
                            'giou_loss': round_loss_metric(logs.get('giou_loss')),
                            'conf_loss': round_loss_metric(logs.get('conf_loss')),
                            'prob_loss': round_loss_metric(logs.get('prob_loss')),
                            'total_loss': round_loss_metric(logs.get('total_loss'))
                        },
                        "metrics": {'mAP50': round_loss_metric(logs.get('mAP50'))}
                    },
                    "val": {
                        "loss": {
                            'giou_loss': round_loss_metric(logs.get('val_giou_loss')),
                            'conf_loss': round_loss_metric(logs.get('val_conf_loss')),
                            'prob_loss': round_loss_metric(logs.get('val_prob_loss')),
                            'total_loss': round_loss_metric(logs.get('val_total_loss'))
                        },
                        "class_loss": {'prob_loss': {}},
                        "metrics": {'mAP50': round_loss_metric(logs.get('val_mAP50'))},
                        "class_metrics": {'mAP50': {}}
                    }
                }
                for name in self.options.data.outputs.get(list(self.options.data.outputs.keys())[0]).classes_names:
                    interactive_log['output']['val']["class_loss"]['prob_loss'][name] = round_loss_metric(
                        logs.get(f'val_prob_loss_{name}'))
                    interactive_log['output']['val']["class_metrics"]['mAP50'][name] = round_loss_metric(logs.get(
                        f'val_mAP50_class_{name}'))
            return interactive_log
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
                            metric_overfitting = self._evaluate_overfitting(
                                metric_name,
                                self.log_history[f"{out}"]['progress_state']['metrics'][metric_name][
                                    'mean_log_history'],
                                metric_type='metric'
                            )
                            if metric_underfittng or metric_overfitting:
                                normal_state = False
                            else:
                                normal_state = True

                            if data_idx or data_idx == 0:
                                self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['underfitting'][
                                    data_idx] = \
                                    metric_underfittng
                                self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['overfitting'][
                                    data_idx] = \
                                    metric_overfitting
                                self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['normal_state'][
                                    data_idx] = \
                                    normal_state
                            else:
                                self.log_history[f"{out}"]['progress_state']['metrics'][metric_name][
                                    'underfitting'].append(
                                    metric_underfittng)
                                self.log_history[f"{out}"]['progress_state']['metrics'][metric_name][
                                    'overfitting'].append(
                                    metric_overfitting)
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
                                            round_loss_metric(class_metric)
                                    else:
                                        self.log_history[f"{out}"]['class_metrics'][cls][metric_name].append(
                                            round_loss_metric(class_metric)
                                        )

                if self.options.data.architecture in self.yolo_architecture:
                    self.log_history['learning_rate'] = self.current_logs.get('learning_rate')
                    out = list(self.options.data.outputs.keys())[0]
                    classes_names = self.options.data.outputs.get(out).classes_names
                    for key in self.log_history['output']["loss"].keys():
                        for data_type in ['train', 'val']:
                            self.log_history['output']["loss"][key][data_type].append(
                                round_loss_metric(self.current_logs.get('output').get(
                                    data_type).get('loss').get(key))
                            )
                    for key in self.log_history['output']["metrics"].keys():
                        self.log_history['output']["metrics"][key].append(
                            round_loss_metric(self.current_logs.get('output').get(
                                'val').get('metrics').get(key))
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
                            self.log_history['output']['progress_state']['loss'][loss_name][
                                'underfitting'][data_idx] = loss_underfitting
                            self.log_history['output']['progress_state']['loss'][loss_name][
                                'overfitting'][data_idx] = loss_overfitting
                            self.log_history['output']['progress_state']['loss'][loss_name][
                                'normal_state'][data_idx] = normal_state
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
                for loss in self.log_history['output']["loss"].keys():
                    self.progress_table[self.current_epoch]["data"]["Прогресс обучения"]["loss"][f'{loss}'] = \
                        f"{self.log_history.get('output').get('loss').get(loss).get('train')[-1]}"
                    self.progress_table[self.current_epoch]["data"]["Прогресс обучения"]["loss"][f'val_{loss}'] = \
                        f"{self.log_history.get('output').get('loss').get(loss).get('val')[-1]}"
                for metric in self.log_history['output']["metrics"].keys():
                    self.progress_table[self.current_epoch]["data"]["Прогресс обучения"]["metrics"][f"{metric}"] = \
                        f"{self.log_history.get('output').get('metrics').get(metric)[-1]}"
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _get_loss_calculation(self, loss_obj, out: str, y_true, y_pred):
        method_name = '_get_loss_calculation'
        try:
            encoding = self.options.data.outputs.get(int(out)).encoding
            task = self.options.data.architecture
            num_classes = self.options.data.outputs.get(int(out)).num_classes
            if task in self.class_architecture:
                if encoding == LayerEncodingChoice.ohe or encoding == LayerEncodingChoice.multi:
                    loss_value = float(loss_obj()(y_true, y_pred).numpy())
                else:
                    loss_value = float(loss_obj()(to_categorical(y_true, num_classes), y_pred).numpy())
            else:
                loss_value = float(loss_obj()(y_true, y_pred).numpy())
            return loss_value if not math.isnan(loss_value) else None
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _get_metric_calculation(self, metric_name, metric_obj, out: str, y_true, y_pred, show_class=False):
        method_name = '_get_metric_calculation'
        try:
            encoding = self.options.data.outputs.get(int(out)).encoding
            task = self.options.data.architecture
            num_classes = self.options.data.outputs.get(int(out)).num_classes
            if task in self.class_architecture:
                if encoding == LayerEncodingChoice.ohe or encoding == LayerEncodingChoice.multi:
                    if metric_name == Metric.Accuracy:
                        metric_obj.update_state(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1))
                    elif metric_name in [Metric.BalancedRecall, Metric.BalancedPrecision, Metric.BalancedFScore]:
                        metric_obj.update_state(y_true, y_pred, show_class=show_class)
                    elif metric_name == Metric.BalancedDiceCoef:
                        metric_obj.encoding = 'multi' if encoding == 'multi' else None
                        metric_obj.update_state(y_true, y_pred)
                    else:
                        metric_obj.update_state(y_true, y_pred)
                else:
                    if metric_name == Metric.Accuracy:
                        metric_obj.update_state(y_true, np.argmax(y_pred, axis=-1))
                    else:
                        metric_obj.update_state(to_categorical(y_true, num_classes), y_pred)
            else:
                metric_obj.update_state(y_true, y_pred)
            metric_value = float(metric_obj.result().numpy())
            return round(metric_value, 6) if not math.isnan(metric_value) else None
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _get_mean_log(self, logs):
        method_name = '_get_mean_log'
        try:
            copy_logs = copy.deepcopy(logs)
            while None in copy_logs:
                copy_logs.pop(copy_logs.index(None))
            if len(copy_logs) < self.log_gap:
                return float(np.mean(copy_logs))
            else:
                return float(np.mean(copy_logs[-self.log_gap:]))
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)
            return 0.

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
            underfitting = False
            if mode == 'min' and train_log and val_log:
                if val_log < 1 and train_log < 1 and (val_log - train_log) > 0.05:
                    underfitting = True
                if (val_log >= 1 or train_log >= 1) and (val_log - train_log) / train_log * 100 > 5:
                    underfitting = True
            if mode == 'max' and train_log and val_log and (train_log - val_log) / train_log * 100 > 3:
                underfitting = True
            return underfitting
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _get_loss_graph_data_request(self) -> list:
        method_name = '_get_loss_graph_data_request'
        try:
            data_return = []
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
                        best_train = fill_graph_plot_data(
                            x=[self.log_history.get("epochs")[train_list.index(best_train_value)]
                               if best_train_value is not None else None],
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
                        no_none_val = []
                        for x in val_list:
                            if x is not None:
                                no_none_val.append(x)
                        best_val_value = min(no_none_val) if no_none_val else None
                        # best_val_value = min(val_list)
                        best_val = fill_graph_plot_data(
                            x=[self.log_history.get("epochs")[val_list.index(best_val_value)]
                               if best_val_value is not None else None],
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
                                graph_name=f"Выходной слой «{loss_graph_config.output_idx}» - "
                                           f"График ошибки обучения по классам"
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
                                    self.options.data.outputs.get(int(loss_graph_config.output_idx)).classes_names
                                ],
                            )
                        )

            if self.options.data.architecture in self.yolo_architecture:
                if not self.interactive_config.loss_graphs or not self.log_history.get("epochs"):
                    return data_return
                _id = 1
                for loss_graph_config in self.interactive_config.loss_graphs:
                    if loss_graph_config.show == LossGraphShowChoice.model:
                        for loss in self.log_history.get('output').get('loss').keys():
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
                            best_train = fill_graph_plot_data(
                                x=[self.log_history.get("epochs")[train_list.index(best_train_value)]
                                   if best_train_value is not None else None],
                                y=[best_train_value],
                                label="Лучший результат на тренировочной выборке"
                            )
                            train_plot = fill_graph_plot_data(
                                x=self.log_history.get("epochs"),
                                y=train_list,
                                label="Тренировочная выборка"
                            )
                            val_list = self.log_history.get("output").get('loss').get(loss).get("val")
                            no_none_val = []
                            for x in val_list:
                                if x is not None:
                                    no_none_val.append(x)
                            best_val_value = min(no_none_val) if no_none_val else None
                            best_val = fill_graph_plot_data(
                                x=[self.log_history.get("epochs")[val_list.index(best_val_value)]
                                   if best_val_value is not None else None],
                                y=[best_val_value],
                                label="Лучший результат на проверочной выборке"
                            )
                            val_plot = fill_graph_plot_data(
                                x=self.log_history.get("epochs"),
                                y=self.log_history.get("output").get('loss').get(loss).get("val"),
                                label="Проверочная выборка"
                            )
                            data_return.append(
                                fill_graph_front_structure(
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
                        data_return.append(
                            fill_graph_front_structure(
                                _id=_id,
                                _type='graphic',
                                graph_name=f"График ошибки обучения «prob_loss» по классам"
                                           f" - Эпоха №{self.log_history.get('epochs')[-1]}",
                                short_name=f"График ошибки обучения по классам",
                                x_label="Эпоха",
                                y_label="Значение",
                                plot_data=[
                                    fill_graph_plot_data(
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
            if self.first_error:
                self.first_error = False
                print_error(InteractiveCallback().name, method_name, e)
            else:
                pass

    def _get_metric_graph_data_request(self) -> list:
        method_name = '_get_metric_graph_data_request'
        try:
            data_return = []
            if self.options.data.architecture in self.basic_architecture:
                if not self.interactive_config.metric_graphs or not self.log_history.get("epochs"):
                    return data_return
                for metric_graph_config in self.interactive_config.metric_graphs:
                    if metric_graph_config.show == MetricGraphShowChoice.model:
                        min_max_mode = loss_metric_config.get("metric").get(metric_graph_config.show_metric.name).get(
                            "mode")
                        if sum(self.log_history.get(f"{metric_graph_config.output_idx}").get(
                                "progress_state").get("metrics").get(metric_graph_config.show_metric.name).get(
                            'overfitting')[-self.log_gap:]) >= self.progress_threashold:
                            progress_state = 'overfitting'
                        elif sum(self.log_history.get(f"{metric_graph_config.output_idx}").get(
                                "progress_state").get("metrics").get(metric_graph_config.show_metric.name).get(
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
                                        y=self.log_history.get(f"{metric_graph_config.output_idx}").get(
                                            'class_metrics').get(
                                            class_name).get(metric_graph_config.show_metric),
                                        label=f"Класс {class_name}"
                                    ) for class_name in
                                    self.options.data.outputs.get(metric_graph_config.output_idx).classes_names
                                ],
                            )
                        )

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
                        else:
                            progress_state = 'normal'
                        val_list = self.log_history.get("output").get('metrics').get(
                            metric_graph_config.show_metric.name)
                        best_val_value = min(val_list) if min_max_mode == 'min' else max(val_list)
                        best_val = fill_graph_plot_data(
                            x=[self.log_history.get("epochs")[val_list.index(best_val_value)]],
                            y=[best_val_value],
                            label="Лучший результат на проверочной выборке"
                        )
                        val_plot = fill_graph_plot_data(
                            x=self.log_history.get("epochs"),
                            y=self.log_history.get("output").get('metrics').get(
                                metric_graph_config.show_metric.name),
                            label="Проверочная выборка"
                        )
                        data_return.append(
                            fill_graph_front_structure(
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
                            fill_graph_front_structure(
                                _id=_id,
                                _type='graphic',
                                graph_name=f"График метрики {metric_graph_config.show_metric.name} по классам - "
                                           f"Эпоха №{self.log_history.get('epochs')[-1]}",
                                short_name=f"{metric_graph_config.show_metric.name} по классам",
                                x_label="Эпоха",
                                y_label="Значение",
                                plot_data=[
                                    fill_graph_plot_data(
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
            if self.first_error:
                print_error(InteractiveCallback().name, method_name, e)
                self.first_error = False
            else:
                pass
