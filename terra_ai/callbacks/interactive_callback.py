import os
import random
import string
from pathlib import Path
import time
from typing import Union, Optional
import numpy as np

from terra_ai import progress
from terra_ai.callbacks.classification_callbacks import ImageClassificationCallback, TextClassificationCallback, \
    AudioClassificationCallback, VideoClassificationCallback, DataframeClassificationCallback, TimeseriesTrendCallback
from terra_ai.callbacks.object_detection_callbacks import YoloV3Callback, YoloV4Callback
from terra_ai.callbacks.regression_callbacks import DataframeRegressionCallback
from terra_ai.callbacks.segmentation_callbacks import ImageSegmentationCallback, TextSegmentationCallback
from terra_ai.callbacks.time_series_callbacks import TimeseriesCallback
from terra_ai.callbacks.utils import loss_metric_config, round_loss_metric, fill_graph_plot_data, \
    fill_graph_front_structure, reformat_metrics, prepare_loss_obj, prepare_metric_obj, get_classes_colors, \
    print_error, BASIC_ARCHITECTURE, CLASSIFICATION_ARCHITECTURE, YOLO_ARCHITECTURE, CLASS_ARCHITECTURE, \
    class_metric_list, reformat_fit_array
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
        self.urgent_predict = False
        self.deploy_presets_data = None
        self.random_key = ''

        self.interactive_config: InteractiveData = InteractiveData(**{})
        pass

    def set_attributes(self, dataset: PrepareDataset, dataset_path: Path,
                       training_path: Path, initial_config: InteractiveData):

        self.options = dataset
        self._callback_router(dataset)
        self._class_metric_list()
        print('\nself._class_metric_list()', self.class_graphics)
        print('set_attributes', dataset.data.architecture)
        self.preset_path = os.path.join(training_path, "presets")
        if not os.path.exists(self.preset_path):
            os.mkdir(self.preset_path)
        self.interactive_config = initial_config
        self.dataset_path = dataset_path
        self.class_colors = get_classes_colors(dataset)
        self.x_val, self.inverse_x_val = self.callback.get_x_array(dataset)
        self.y_true, self.inverse_y_true = self.callback.get_y_true(dataset, dataset_path)
        self.dataset_balance = self.callback.dataset_balance(
            options=self.options, y_true=self.y_true, preset_path=self.preset_path, class_colors=self.class_colors
        )
        if dataset.data.architecture in CLASSIFICATION_ARCHITECTURE:
            self.class_idx = self.callback.prepare_class_idx(self.y_true, self.options)
        self.seed_idx = self._prepare_seed()
        self.random_key = ''.join(random.sample(string.ascii_letters + string.digits, 16))

    def clear_history(self):
        self.log_history = {}
        self.current_logs = {}
        self.progress_table = {}
        self.intermediate_result = {}
        self.statistic_result = {}
        self.train_progress = {}
        self.addtrain_epochs = []
        self.deploy_presets_data = None

    def get_presets(self):
        return self.deploy_presets_data

    def update_train_progress(self, data: dict):
        self.train_progress = data

    def update_state(self, arrays: dict = None, fit_logs=None, current_epoch_time=None,
                     on_epoch_end_flag=False, train_idx: list = None) -> dict:
        if arrays is None:
            arrays = {}
        if self.log_history:
            if arrays:
                if self.options.data.architecture in BASIC_ARCHITECTURE:
                    self.y_true = reformat_fit_array(
                        array={"train": arrays.get("train_true"), "val": arrays.get("val_true")},
                        options=self.options, train_idx=train_idx)
                    self.inverse_y_true = self.callback.get_inverse_array(self.y_true, self.options)
                    self.y_pred = reformat_fit_array(
                        array={"train": arrays.get("train_pred"), "val": arrays.get("val_pred")},
                        options=self.options, train_idx=train_idx)
                    self.inverse_y_pred = self.callback.get_inverse_array(self.y_pred, self.options)
                    # self.y_pred, self.inverse_y_pred = self.callback.get_y_pred(self.y_true, y_pred, self.options)
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
                if self.options.data.architecture in YOLO_ARCHITECTURE:
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
                    self.current_epoch = fit_logs.get('epochs')[-1]
                    self.log_history = fit_logs
                    # self._update_log_history()
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
                    if self.options.data.architecture in BASIC_ARCHITECTURE and \
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
                    if self.options.data.architecture in YOLO_ARCHITECTURE and \
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
                    if self.options.data.architecture in BASIC_ARCHITECTURE and \
                            self.interactive_config.statistic_data.output_id:
                        self.statistic_result = self.callback.statistic_data_request(
                            interactive_config=self.interactive_config,
                            options=self.options,
                            y_true=self.y_true,
                            y_pred=self.y_pred,
                            inverse_y_pred=self.inverse_y_pred,
                            inverse_y_true=self.inverse_y_true,
                        )
                    if self.options.data.architecture in YOLO_ARCHITECTURE and \
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
            if self.options.data.architecture in BASIC_ARCHITECTURE:
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

            if self.options.data.architecture in YOLO_ARCHITECTURE:
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
            return class_metric_list(self.options)
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _prepare_seed(self):
        method_name = '_prepare_seed'
        try:
            if self.options.data.architecture in YOLO_ARCHITECTURE:
                example_idx = np.arange(len(self.options.dataframe.get("val")))
                np.random.shuffle(example_idx)
            elif self.options.data.architecture in BASIC_ARCHITECTURE:
                output = self.interactive_config.intermediate_result.main_output
                example_idx = []
                if self.options.data.architecture in CLASSIFICATION_ARCHITECTURE:
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

    # def _update_log_history(self, log_history: dict, current_epoch: int, options: PrepareDataset):
    #     method_name = '_update_log_history'
    #     try:
    #         data_idx = None
    #         if log_history:
    #             if current_epoch in log_history['epochs']:
    #                 data_idx = log_history['epochs'].index(current_epoch)
    #             else:
    #                 log_history['epochs'].append(current_epoch)
    #
    #             if options.data.architecture in BASIC_ARCHITECTURE:
    #                 for out in options.data.outputs.keys():
    #                     out_task = self.options.data.outputs.get(out).task
    #                     classes_names = self.options.data.outputs.get(out).classes_names
    #                     for loss_name in self.log_history.get(f"{out}").get('loss').keys():
    #                         for data_type in ['train', 'val']:
    #                             # fill losses
    #                             if data_idx or data_idx == 0:
    #                                 self.log_history[f"{out}"]['loss'][loss_name][data_type][data_idx] = \
    #                                     round_loss_metric(
    #                                         self.current_logs.get(f"{out}").get('loss').get(loss_name).get(data_type)
    #                                     )
    #                             else:
    #                                 self.log_history[f"{out}"]['loss'][loss_name][data_type].append(
    #                                     round_loss_metric(
    #                                         self.current_logs.get(f"{out}").get('loss').get(loss_name).get(data_type)
    #                                     )
    #                                 )
    #                         # fill loss progress state
    #                         if data_idx or data_idx == 0:
    #                             self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['mean_log_history'][
    #                                 data_idx] = \
    #                                 self._get_mean_log(
    #                                     self.log_history.get(f"{out}").get('loss').get(loss_name).get('val'))
    #                         else:
    #                             self.log_history[f"{out}"]['progress_state']['loss'][loss_name][
    #                                 'mean_log_history'].append(
    #                                 self._get_mean_log(
    #                                     self.log_history.get(f"{out}").get('loss').get(loss_name).get('val'))
    #                             )
    #                         # get progress state data
    #                         loss_underfitting = self._evaluate_underfitting(
    #                             loss_name,
    #                             self.log_history[f"{out}"]['loss'][loss_name]['train'][-1],
    #                             self.log_history[f"{out}"]['loss'][loss_name]['val'][-1],
    #                             metric_type='loss'
    #                         )
    #                         loss_overfitting = self._evaluate_overfitting(
    #                             loss_name,
    #                             self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['mean_log_history'],
    #                             metric_type='loss'
    #                         )
    #                         if loss_underfitting or loss_overfitting:
    #                             normal_state = False
    #                         else:
    #                             normal_state = True
    #
    #                         if data_idx or data_idx == 0:
    #                             self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['underfitting'][
    #                                 data_idx] = \
    #                                 loss_underfitting
    #                             self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['overfitting'][
    #                                 data_idx] = \
    #                                 loss_overfitting
    #                             self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['normal_state'][
    #                                 data_idx] = \
    #                                 normal_state
    #                         else:
    #                             self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['underfitting'].append(
    #                                 loss_underfitting)
    #                             self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['overfitting'].append(
    #                                 loss_overfitting)
    #                             self.log_history[f"{out}"]['progress_state']['loss'][loss_name]['normal_state'].append(
    #                                 normal_state)
    #
    #                         if out_task == LayerOutputTypeChoice.Classification or \
    #                                 out_task == LayerOutputTypeChoice.Segmentation or \
    #                                 out_task == LayerOutputTypeChoice.TextSegmentation or \
    #                                 out_task == LayerOutputTypeChoice.TimeseriesTrend:
    #                             for cls in self.log_history.get(f"{out}").get('class_loss').keys():
    #                                 class_loss = 0.
    #                                 if out_task == LayerOutputTypeChoice.Classification or \
    #                                         out_task == LayerOutputTypeChoice.TimeseriesTrend:
    #                                     class_loss = self._get_loss_calculation(
    #                                         loss_obj=self.loss_obj.get(f"{out}"),
    #                                         out=f"{out}",
    #                                         y_true=self.y_true.get('val').get(f"{out}")[
    #                                             self.class_idx.get('val').get(f"{out}").get(cls)],
    #                                         y_pred=self.y_pred.get(f"{out}")[
    #                                             self.class_idx.get('val').get(f"{out}").get(cls)],
    #                                     )
    #                                 if out_task == LayerOutputTypeChoice.Segmentation:
    #                                     class_idx = classes_names.index(cls)
    #                                     class_loss = self._get_loss_calculation(
    #                                         loss_obj=self.loss_obj.get(f"{out}"),
    #                                         out=f"{out}",
    #                                         y_true=self.y_true.get('val').get(f"{out}")[:, :, :, class_idx],
    #                                         y_pred=self.y_pred.get(f"{out}")[:, :, :, class_idx],
    #                                     )
    #                                 if out_task == LayerOutputTypeChoice.TextSegmentation:
    #                                     class_idx = classes_names.index(cls)
    #                                     class_loss = self._get_loss_calculation(
    #                                         loss_obj=self.loss_obj.get(f"{out}"),
    #                                         out=f"{out}",
    #                                         y_true=self.y_true.get('val').get(f"{out}")[:, :, class_idx],
    #                                         y_pred=self.y_pred.get(f"{out}")[:, :, class_idx],
    #                                     )
    #                                 if data_idx or data_idx == 0:
    #                                     self.log_history[f"{out}"]['class_loss'][cls][loss_name][data_idx] = \
    #                                         round_loss_metric(class_loss)
    #                                 else:
    #                                     self.log_history[f"{out}"]['class_loss'][cls][loss_name].append(
    #                                         round_loss_metric(class_loss)
    #                                     )
    #
    #                     for metric_name in self.log_history.get(f"{out}").get('metrics').keys():
    #                         for data_type in ['train', 'val']:
    #                             # fill metrics
    #                             if data_idx or data_idx == 0:
    #                                 if self.current_logs:
    #                                     self.log_history[f"{out}"]['metrics'][metric_name][data_type][data_idx] = \
    #                                         round_loss_metric(
    #                                             self.current_logs.get(f"{out}").get('metrics').get(metric_name).get(
    #                                                 data_type)
    #                                         )
    #                             else:
    #                                 if self.current_logs:
    #                                     self.log_history[f"{out}"]['metrics'][metric_name][data_type].append(
    #                                         round_loss_metric(
    #                                             self.current_logs.get(f"{out}").get('metrics').get(metric_name).get(
    #                                                 data_type)
    #                                         )
    #                                     )
    #
    #                         if data_idx or data_idx == 0:
    #                             self.log_history[f"{out}"]['progress_state']['metrics'][metric_name][
    #                                 'mean_log_history'][
    #                                 data_idx] = \
    #                                 self._get_mean_log(self.log_history[f"{out}"]['metrics'][metric_name]['val'])
    #                         else:
    #                             self.log_history[f"{out}"]['progress_state']['metrics'][metric_name][
    #                                 'mean_log_history'].append(
    #                                 self._get_mean_log(self.log_history[f"{out}"]['metrics'][metric_name]['val'])
    #                             )
    #                         metric_underfittng = self._evaluate_underfitting(
    #                             metric_name,
    #                             self.log_history[f"{out}"]['metrics'][metric_name]['train'][-1],
    #                             self.log_history[f"{out}"]['metrics'][metric_name]['val'][-1],
    #                             metric_type='metric'
    #                         )
    #                         metric_overfitting = self._evaluate_overfitting(
    #                             metric_name,
    #                             self.log_history[f"{out}"]['progress_state']['metrics'][metric_name][
    #                                 'mean_log_history'],
    #                             metric_type='metric'
    #                         )
    #                         if metric_underfittng or metric_overfitting:
    #                             normal_state = False
    #                         else:
    #                             normal_state = True
    #
    #                         if data_idx or data_idx == 0:
    #                             self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['underfitting'][
    #                                 data_idx] = \
    #                                 metric_underfittng
    #                             self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['overfitting'][
    #                                 data_idx] = \
    #                                 metric_overfitting
    #                             self.log_history[f"{out}"]['progress_state']['metrics'][metric_name]['normal_state'][
    #                                 data_idx] = \
    #                                 normal_state
    #                         else:
    #                             self.log_history[f"{out}"]['progress_state']['metrics'][metric_name][
    #                                 'underfitting'].append(
    #                                 metric_underfittng)
    #                             self.log_history[f"{out}"]['progress_state']['metrics'][metric_name][
    #                                 'overfitting'].append(
    #                                 metric_overfitting)
    #                             self.log_history[f"{out}"]['progress_state']['metrics'][metric_name][
    #                                 'normal_state'].append(
    #                                 normal_state)
    #
    #                         if out_task == LayerOutputTypeChoice.Classification or \
    #                                 out_task == LayerOutputTypeChoice.Segmentation or \
    #                                 out_task == LayerOutputTypeChoice.TextSegmentation or \
    #                                 out_task == LayerOutputTypeChoice.TimeseriesTrend:
    #                             for cls in self.log_history.get(f"{out}").get('class_metrics').keys():
    #                                 class_metric = 0.
    #                                 if out_task == LayerOutputTypeChoice.Classification or \
    #                                         out_task == LayerOutputTypeChoice.TimeseriesTrend:
    #                                     class_metric = self._get_metric_calculation(
    #                                         metric_name=metric_name,
    #                                         metric_obj=self.metrics_obj.get(f"{out}").get(metric_name),
    #                                         out=f"{out}",
    #                                         y_true=self.y_true.get('val').get(f"{out}")[
    #                                             self.class_idx.get('val').get(f"{out}").get(cls)],
    #                                         y_pred=self.y_pred.get(f"{out}")[
    #                                             self.class_idx.get('val').get(f"{out}").get(cls)],
    #                                         show_class=True
    #                                     )
    #                                 if out_task == LayerOutputTypeChoice.Segmentation or \
    #                                         out_task == LayerOutputTypeChoice.TextSegmentation:
    #                                     class_idx = classes_names.index(cls)
    #                                     class_metric = self._get_metric_calculation(
    #                                         metric_name=metric_name,
    #                                         metric_obj=self.metrics_obj.get(f"{out}").get(metric_name),
    #                                         out=f"{out}",
    #                                         y_true=self.y_true.get('val').get(f"{out}")[..., class_idx:class_idx + 1],
    #                                         y_pred=self.y_pred.get(f"{out}")[..., class_idx:class_idx + 1],
    #                                     )
    #                                 if data_idx or data_idx == 0:
    #                                     self.log_history[f"{out}"]['class_metrics'][cls][metric_name][data_idx] = \
    #                                         round_loss_metric(class_metric)
    #                                 else:
    #                                     self.log_history[f"{out}"]['class_metrics'][cls][metric_name].append(
    #                                         round_loss_metric(class_metric)
    #                                     )
    #
    #             if self.options.data.architecture in YOLO_ARCHITECTURE:
    #                 self.log_history['learning_rate'] = self.current_logs.get('learning_rate')
    #                 out = list(self.options.data.outputs.keys())[0]
    #                 classes_names = self.options.data.outputs.get(out).classes_names
    #                 for key in self.log_history['output']["loss"].keys():
    #                     for data_type in ['train', 'val']:
    #                         self.log_history['output']["loss"][key][data_type].append(
    #                             round_loss_metric(self.current_logs.get('output').get(
    #                                 data_type).get('loss').get(key))
    #                         )
    #                 for key in self.log_history['output']["metrics"].keys():
    #                     self.log_history['output']["metrics"][key].append(
    #                         round_loss_metric(self.current_logs.get('output').get(
    #                             'val').get('metrics').get(key))
    #                     )
    #                 for name in classes_names:
    #                     self.log_history['output']["class_loss"]['prob_loss'][name].append(
    #                         round_loss_metric(self.current_logs.get('output').get("val").get(
    #                             'class_loss').get("prob_loss").get(name))
    #                     )
    #                     self.log_history['output']["class_metrics"]['mAP50'][name].append(
    #                         round_loss_metric(self.current_logs.get('output').get("val").get(
    #                             'class_metrics').get("mAP50").get(name))
    #                     )
    #                     # self.log_history['output']["class_metrics"]['mAP95'][name].append(
    #                     #     self._round_loss_metric(self.current_logs.get('output').get("val").get(
    #                     #         'class_metrics').get("mAP95").get(name))
    #                     # )
    #                 for loss_name in self.log_history['output']["loss"].keys():
    #                     # fill loss progress state
    #                     if data_idx or data_idx == 0:
    #                         self.log_history['output']['progress_state']['loss'][loss_name]['mean_log_history'][
    #                             data_idx] = \
    #                             self._get_mean_log(self.log_history.get('output').get('loss').get(loss_name).get('val'))
    #                     else:
    #                         self.log_history['output']['progress_state']['loss'][loss_name]['mean_log_history'].append(
    #                             self._get_mean_log(self.log_history.get('output').get('loss').get(loss_name).get('val'))
    #                         )
    #                     # get progress state data
    #                     loss_underfitting = self._evaluate_underfitting(
    #                         loss_name,
    #                         self.log_history['output']['loss'][loss_name]['train'][-1],
    #                         self.log_history['output']['loss'][loss_name]['val'][-1],
    #                         metric_type='loss'
    #                     )
    #                     loss_overfitting = self._evaluate_overfitting(
    #                         loss_name,
    #                         self.log_history['output']['progress_state']['loss'][loss_name]['mean_log_history'],
    #                         metric_type='loss'
    #                     )
    #                     if loss_underfitting or loss_overfitting:
    #                         normal_state = False
    #                     else:
    #                         normal_state = True
    #                     if data_idx or data_idx == 0:
    #                         self.log_history['output']['progress_state']['loss'][loss_name][
    #                             'underfitting'][data_idx] = loss_underfitting
    #                         self.log_history['output']['progress_state']['loss'][loss_name][
    #                             'overfitting'][data_idx] = loss_overfitting
    #                         self.log_history['output']['progress_state']['loss'][loss_name][
    #                             'normal_state'][data_idx] = normal_state
    #                     else:
    #                         self.log_history['output']['progress_state']['loss'][loss_name]['underfitting'].append(
    #                             loss_underfitting)
    #                         self.log_history['output']['progress_state']['loss'][loss_name]['overfitting'].append(
    #                             loss_overfitting)
    #                         self.log_history['output']['progress_state']['loss'][loss_name]['normal_state'].append(
    #                             normal_state)
    #                 for metric_name in self.log_history.get('output').get('metrics').keys():
    #                     if data_idx or data_idx == 0:
    #                         self.log_history['output']['progress_state']['metrics'][metric_name]['mean_log_history'][
    #                             data_idx] = self._get_mean_log(self.log_history['output']['metrics'][metric_name])
    #                     else:
    #                         self.log_history['output']['progress_state']['metrics'][metric_name][
    #                             'mean_log_history'].append(
    #                             self._get_mean_log(self.log_history['output']['metrics'][metric_name])
    #                         )
    #                     metric_overfitting = self._evaluate_overfitting(
    #                         metric_name,
    #                         self.log_history['output']['progress_state']['metrics'][metric_name]['mean_log_history'],
    #                         metric_type='metric'
    #                     )
    #                     if metric_overfitting:
    #                         normal_state = False
    #                     else:
    #                         normal_state = True
    #                     if data_idx or data_idx == 0:
    #                         self.log_history['output']['progress_state']['metrics'][metric_name]['overfitting'][
    #                             data_idx] = metric_overfitting
    #                         self.log_history['output']['progress_state']['metrics'][metric_name]['normal_state'][
    #                             data_idx] = normal_state
    #                     else:
    #                         self.log_history['output']['progress_state']['metrics'][metric_name]['overfitting'].append(
    #                             metric_overfitting)
    #                         self.log_history['output']['progress_state']['metrics'][metric_name]['normal_state'].append(
    #                             normal_state)
    #     except Exception as e:
    #         print_error(InteractiveCallback().name, method_name, e)

    def _update_progress_table(self, epoch_time: float):
        method_name = '_update_progress_table'
        try:
            if self.options.data.architecture in BASIC_ARCHITECTURE:
                self.progress_table[self.current_epoch] = {
                    "time": epoch_time,
                    "data": {}
                }
                for out in list(self.log_history.keys())[1:]:
                    self.progress_table[self.current_epoch]["data"][f"Выходной слой «{out}»"] = {
                        'loss': {},
                        'metrics': {}
                    }
                    loss_name = list(self.log_history.get(out).get('loss').keys())[0]
                    self.progress_table[self.current_epoch]["data"][f"Выходной слой «{out}»"]["loss"] = {
                        'loss': f"{self.log_history.get(out).get('loss').get(loss_name).get('train')[-1]}",
                        'val_loss': f"{self.log_history.get(out).get('loss').get(loss_name).get('val')[-1]}"
                    }
                    for metric in self.log_history.get(out).get('metrics').keys():
                        self.progress_table[self.current_epoch]["data"][f"Выходной слой «{out}»"]["metrics"][metric] = \
                            f"{self.log_history.get(out).get('metrics').get(metric).get('train')[-1]}"
                        self.progress_table[self.current_epoch]["data"][f"Выходной слой «{out}»"]["metrics"][
                            f"val_{metric}"] = f"{self.log_history.get(out).get('metrics').get(metric).get('val')[-1]}"

            if self.options.data.architecture in YOLO_ARCHITECTURE:
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

    def _get_loss_graph_data_request(self) -> list:
        method_name = '_get_loss_graph_data_request'
        try:
            data_return = []
            if self.options.data.architecture in BASIC_ARCHITECTURE:
                if not self.interactive_config.loss_graphs or not self.log_history.get("epochs"):
                    return data_return
                for loss_graph_config in self.interactive_config.loss_graphs:
                    loss = self.losses.get(f"{loss_graph_config.output_idx}")
                    if self.options.data.architecture in YOLO_ARCHITECTURE:
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

            if self.options.data.architecture in YOLO_ARCHITECTURE:
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
            if self.options.data.architecture in BASIC_ARCHITECTURE:
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

            if self.options.data.architecture in YOLO_ARCHITECTURE:
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
