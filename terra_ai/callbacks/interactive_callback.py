import copy
import random
import string
import time

from pathlib import Path
from typing import Optional

from tensorflow.keras.utils import to_categorical
import numpy as np

from terra_ai import progress
from terra_ai.callbacks.classification_callbacks import ImageClassificationCallback, TextClassificationCallback, \
    AudioClassificationCallback, VideoClassificationCallback, DataframeClassificationCallback, TimeseriesTrendCallback
from terra_ai.callbacks.object_detection_callbacks import YoloV3Callback, YoloV4Callback
from terra_ai.callbacks.regression_callbacks import DataframeRegressionCallback
from terra_ai.callbacks.segmentation_callbacks import ImageSegmentationCallback, TextSegmentationCallback
from terra_ai.callbacks.time_series_callbacks import TimeseriesCallback
from terra_ai.callbacks.utils import loss_metric_config, fill_graph_plot_data, fill_graph_front_structure,\
    get_classes_colors, print_error, BASIC_ARCHITECTURE, CLASSIFICATION_ARCHITECTURE, YOLO_ARCHITECTURE,\
    class_metric_list, reformat_fit_array
from terra_ai.data.datasets.extra import LayerOutputTypeChoice, DatasetGroupChoice, LayerInputTypeChoice
from terra_ai.data.training.extra import LossGraphShowChoice, MetricGraphShowChoice, ArchitectureChoice
from terra_ai.data.training.train import TrainingDetailsData
from terra_ai.datasets.preparing import PrepareDataset

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
        self.training_details: Optional[TrainingDetailsData] = None
        self.last_training_details: Optional[TrainingDetailsData] = None

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
        self.addtrain_epochs = []
        self.progress_name = "training"
        self.preset_path = ""
        self.urgent_predict = False
        self.deploy_presets_data = None
        self.random_key = ''
        self.get_balance = True
        pass

    def set_attributes(self, dataset: PrepareDataset, params: TrainingDetailsData):
        self.options = dataset
        self._callback_router(dataset)
        self.class_graphics = self._class_metric_list()
        print('set_attributes', dataset.data.architecture)
        print('\ndataset_config', dataset.data)
        print('\nparams', params.native(), '\n')
        self.training_details = params
        self.last_training_details = copy.deepcopy(params)
        self.dataset_path = dataset.data.path
        self.class_colors = get_classes_colors(dataset)
        self.x_val, self.inverse_x_val = self.callback.get_x_array(dataset)
        self.random_key = ''.join(random.sample(string.ascii_letters + string.digits, 16))

    def clear_history(self):
        self.log_history = {}
        self.current_logs = {}
        self.progress_table = {}
        self.intermediate_result = {}
        self.statistic_result = {}
        self.addtrain_epochs = []
        self.deploy_presets_data = None

    def get_presets(self):
        return self.deploy_presets_data

    def update_state(self, arrays: dict = None, fit_logs=None, current_epoch_time=None,
                     on_epoch_end_flag=False, train_idx: list = None) -> dict:
        if self.log_history:
            if arrays:
                t = time.time()
                if self.options.data.architecture in BASIC_ARCHITECTURE:
                    t = time.time()
                    self.y_true = reformat_fit_array(
                        array={"train": arrays.get("train_true"), "val": arrays.get("val_true")},
                        options=self.options, train_idx=train_idx)
                    self.inverse_y_true = self.callback.get_inverse_array(self.y_true, self.options)
                    self.y_pred = reformat_fit_array(
                        array={"train": arrays.get("train_pred"), "val": arrays.get("val_pred")},
                        options=self.options, train_idx=train_idx)
                    self.inverse_y_pred = self.callback.get_inverse_array(self.y_pred, self.options)
                    print('\nInteractiveCallback y_true, y_pred:', round(time.time() - t, 3))
                    t = time.time()
                    if self.get_balance:
                        self.dataset_balance = self.callback.dataset_balance(
                            options=self.options, y_true=self.y_true,
                            preset_path=self.training_details.intermediate_path,
                            class_colors=self.class_colors
                        )
                        print('\n self.dataset_balance', self.dataset_balance)
                        if self.options.data.architecture in CLASSIFICATION_ARCHITECTURE:
                            self.class_idx = self.callback.prepare_class_idx(self.y_true, self.options)
                        self.seed_idx = self._prepare_seed()
                        self.get_balance = False
                    print('\nInteractiveCallback dataset_balance:', round(time.time() - t, 3))
                    t = time.time()
                    out = f"{self.training_details.interactive.intermediate_result.main_output}"
                    count = self.training_details.interactive.intermediate_result.num_examples
                    count = count if count > len(self.y_true.get('val').get(out)) \
                        else len(self.y_true.get('val').get(out))
                    self.example_idx = self.callback.prepare_example_idx_to_show(
                        array=self.y_pred.get("val").get(out),
                        true_array=self.y_true.get("val").get(out),
                        options=self.options,
                        output=int(out),
                        count=count,
                        choice_type=self.training_details.interactive.intermediate_result.example_choice_type,
                        seed_idx=self.seed_idx[:self.training_details.interactive.intermediate_result.num_examples]
                    )
                    print('\nInteractiveCallback prepare_example_idx_to_show', round(time.time() - t, 3))
                if self.options.data.architecture in YOLO_ARCHITECTURE:
                    t = time.time()
                    self.raw_y_pred = arrays.get("val_pred")
                    sensitivity = self.training_details.interactive.intermediate_result.sensitivity \
                        if self.training_details.interactive.intermediate_result.sensitivity else 0.3
                    threashold = self.training_details.interactive.intermediate_result.threashold \
                        if self.training_details.interactive.intermediate_result.threashold else 0.5
                    print('\nInteractiveCallback get_y_pred: start', sensitivity, threashold)
                    self.y_pred = self.callback.get_y_pred(
                        y_pred=arrays.get("val_pred"), options=self.options,
                        sensitivity=sensitivity,
                        threashold=threashold
                    )
                    if self.get_balance:
                        self.y_true, self.inverse_y_true = \
                            self.callback.get_y_true(self.options, self.options.data.path)
                        self.dataset_balance = self.callback.dataset_balance(
                            options=self.options, y_true=self.y_true,
                            preset_path=self.training_details.intermediate_path,
                            class_colors=self.class_colors
                        )
                        self.seed_idx = self._prepare_seed()
                        print('\nseed_idx', self.seed_idx[:10], '\n')
                        self.get_balance = False
                    print('\nInteractiveCallback get_y_pred', round(time.time() - t, 3), sensitivity, threashold)
                    count = self.training_details.interactive.intermediate_result.num_examples
                    count = count if count > len(self.options.dataframe.get('val')) \
                        else len(self.options.dataframe.get('val'))
                    self.raw_y_true = arrays.get("val_true")
                    print('\nInteractiveCallback prepare_example_idx_to_show: start')
                    t = time.time()
                    self.example_idx, _ = self.callback.prepare_example_idx_to_show(
                        array=self.y_pred,
                        true_array=self.y_true,
                        name_classes=self.options.data.outputs.get(
                            list(self.options.data.outputs.keys())[0]).classes_names,
                        box_channel=self.training_details.interactive.intermediate_result.box_channel,
                        count=count,
                        choice_type=self.training_details.interactive.intermediate_result.example_choice_type,
                        seed_idx=self.seed_idx,
                        sensitivity=self.training_details.interactive.intermediate_result.sensitivity,
                    )
                    print('\nInteractiveCallback example_idx', round(time.time() - t, 3))
                print('\nInteractiveCallback if self.options.data.architecture in BASIC_ARCHITECTURE:', round(time.time() - t, 3))
                t = time.time()
                if on_epoch_end_flag:
                    self.current_epoch = fit_logs.get('epochs')[-1]
                    self.log_history = fit_logs
                    self._update_progress_table(current_epoch_time)
                    if self.training_details.interactive.intermediate_result.autoupdate:
                        print('\nInteractiveCallback intermediate_result_request: start')
                        t = time.time()
                        self.intermediate_result = self.callback.intermediate_result_request(
                            options=self.options,
                            interactive_config=self.training_details.interactive,
                            example_idx=self.example_idx,
                            dataset_path=self.dataset_path,
                            preset_path=self.training_details.intermediate_path,
                            x_val=self.x_val,
                            inverse_x_val=self.inverse_x_val,
                            y_pred=self.y_pred,
                            inverse_y_pred=self.inverse_y_pred,
                            y_true=self.y_true,
                            inverse_y_true=self.inverse_y_true,
                            class_colors=self.class_colors,
                        )
                        print('\nInteractiveCallback intermediate_result_request', round(time.time() - t, 3))
                    if self.options.data.architecture in BASIC_ARCHITECTURE and \
                            self.training_details.interactive.statistic_data.output_id \
                            and self.training_details.interactive.statistic_data.autoupdate:
                        t = time.time()
                        self.statistic_result = self.callback.statistic_data_request(
                            interactive_config=self.training_details.interactive,
                            options=self.options,
                            y_true=self.y_true,
                            inverse_y_true=self.inverse_y_true,
                            y_pred=self.y_pred,
                            inverse_y_pred=self.inverse_y_pred,
                        )
                        print('\nInteractiveCallback statistic_data_request', round(time.time() - t, 3))
                    if self.options.data.architecture in YOLO_ARCHITECTURE and \
                            self.training_details.interactive.statistic_data.box_channel \
                            and self.training_details.interactive.statistic_data.autoupdate:
                        print('\nInteractiveCallback statistic_data_request: start')
                        t = time.time()
                        self.statistic_result = self.callback.statistic_data_request(
                            interactive_config=self.training_details.interactive,
                            options=self.options,
                            y_true=self.y_true,
                            y_pred=self.y_pred,
                            inverse_y_pred=self.inverse_y_pred,
                            inverse_y_true=self.inverse_y_true
                        )
                        print('\nInteractiveCallback statistic_data_request', round(time.time() - t, 3))
                else:
                    t = time.time()
                    self.intermediate_result = self.callback.intermediate_result_request(
                        options=self.options,
                        interactive_config=self.training_details.interactive,
                        example_idx=self.example_idx,
                        dataset_path=self.dataset_path,
                        preset_path=self.training_details.intermediate_path,
                        x_val=self.x_val,
                        inverse_x_val=self.inverse_x_val,
                        y_pred=self.y_pred,
                        inverse_y_pred=self.inverse_y_pred,
                        y_true=self.y_true,
                        inverse_y_true=self.inverse_y_true,
                        class_colors=self.class_colors,
                    )
                    print('\nInteractiveCallback intermediate_result_request', round(time.time() - t, 3))
                    if self.options.data.architecture in BASIC_ARCHITECTURE and \
                            self.training_details.interactive.statistic_data.output_id:
                        t = time.time()
                        self.statistic_result = self.callback.statistic_data_request(
                            interactive_config=self.training_details.interactive,
                            options=self.options,
                            y_true=self.y_true,
                            y_pred=self.y_pred,
                            inverse_y_pred=self.inverse_y_pred,
                            inverse_y_true=self.inverse_y_true,
                        )
                        print('\nInteractiveCallback statistic_data_request', round(time.time() - t, 3))
                    if self.options.data.architecture in YOLO_ARCHITECTURE and \
                            self.training_details.interactive.statistic_data.box_channel:
                        self.statistic_result = self.callback.statistic_data_request(
                            interactive_config=self.training_details.interactive,
                            options=self.options,
                            y_true=self.y_true,
                            y_pred=self.y_pred,
                            inverse_y_pred=self.inverse_y_pred,
                            inverse_y_true=self.inverse_y_true,
                        )
                self.urgent_predict = False
                self.random_key = ''.join(random.sample(string.ascii_letters + string.digits, 16))
                print('\nInteractiveCallback if on_epoch_end_flag:', round(time.time() - t, 3))
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
                    interactive_config=self.training_details.interactive
                ),
                'addtrain_epochs': self.addtrain_epochs,
            }
        else:
            return {}

    def get_train_results(self):
        print('InteractiveCallback.get_train_results')
        """Return dict with data for current interactive request"""
        if self.log_history and self.log_history.get("epochs", {}):
            if self.options.data.architecture in BASIC_ARCHITECTURE:
                if self.training_details.interactive.intermediate_result.show_results:
                    out = f"{self.training_details.interactive.intermediate_result.main_output}"
                    count = self.training_details.interactive.intermediate_result.num_examples
                    count = count if count > len(self.y_true.get('val').get(out)) \
                        else len(self.y_true.get('val').get(out))
                    self.example_idx = self.callback.prepare_example_idx_to_show(
                        array=self.y_true.get("val").get(out),
                        true_array=self.y_true.get("val").get(out),
                        options=self.options,
                        output=int(out),
                        count=count,
                        choice_type=self.training_details.interactive.intermediate_result.example_choice_type,
                        seed_idx=self.seed_idx[:self.training_details.interactive.intermediate_result.num_examples]
                    )
                if self.training_details.interactive.intermediate_result.show_results or \
                        self.training_details.interactive.statistic_data.output_id:
                    print('\nstatistic_data', self.training_details.interactive.statistic_data)
                    self.urgent_predict = True
                    self.intermediate_result = self.callback.intermediate_result_request(
                        options=self.options,
                        interactive_config=self.training_details.interactive,
                        example_idx=self.example_idx,
                        dataset_path=self.dataset_path,
                        preset_path=self.training_details.intermediate_path,
                        x_val=self.x_val,
                        inverse_x_val=self.inverse_x_val,
                        y_pred=self.y_pred,
                        inverse_y_pred=self.inverse_y_pred,
                        y_true=self.y_true,
                        inverse_y_true=self.inverse_y_true,
                        class_colors=self.class_colors,
                        raw_y_pred=self.raw_y_pred
                    )
                    self.statistic_result = self.callback.statistic_data_request(
                        interactive_config=self.training_details.interactive,
                        options=self.options,
                        y_true=self.y_true,
                        y_pred=self.y_pred,
                        inverse_y_pred=self.inverse_y_pred,
                        inverse_y_true=self.inverse_y_true,
                    )

            if self.options.data.architecture in YOLO_ARCHITECTURE:
                if self.training_details.interactive.intermediate_result.show_results:
                    self.y_pred = self.callback.get_y_pred(
                        y_pred=self.raw_y_pred, options=self.options,
                        sensitivity=self.training_details.interactive.intermediate_result.sensitivity,
                        threashold=self.training_details.interactive.intermediate_result.threashold
                    )
                    count = self.training_details.interactive.intermediate_result.num_examples
                    count = count if count > len(self.options.dataframe.get('val')) \
                        else len(self.options.dataframe.get('val'))
                    self.example_idx, _ = self.callback.prepare_example_idx_to_show(
                        array=self.y_pred,
                        true_array=self.y_true,
                        name_classes=self.options.data.outputs.get(
                            list(self.options.data.outputs.keys())[0]).classes_names,
                        box_channel=self.training_details.interactive.intermediate_result.box_channel,
                        count=count,
                        choice_type=self.training_details.interactive.intermediate_result.example_choice_type,
                        seed_idx=self.seed_idx,
                        sensitivity=self.training_details.interactive.intermediate_result.sensitivity,
                    )
                    self.intermediate_result = self.callback.intermediate_result_request(
                        options=self.options,
                        interactive_config=self.training_details.interactive,
                        example_idx=self.example_idx,
                        dataset_path=self.dataset_path,
                        preset_path=self.training_details.intermediate_path,
                        x_val=self.x_val,
                        inverse_x_val=self.inverse_x_val,
                        y_pred=self.y_pred,
                        inverse_y_pred=self.inverse_y_pred,
                        y_true=self.y_true,
                        inverse_y_true=self.inverse_y_true,
                        class_colors=self.class_colors,
                    )
                    self.statistic_result = self.callback.statistic_data_request(
                        interactive_config=self.training_details.interactive,
                        options=self.options,
                        y_true=self.y_true,
                        y_pred=self.y_pred,
                        inverse_y_pred=self.inverse_y_pred,
                        inverse_y_true=self.inverse_y_true,
                    )

            self.random_key = ''.join(random.sample(string.ascii_letters + string.digits, 16))
            result = {
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
                    interactive_config=self.training_details.interactive
                ),
                'addtrain_epochs': self.addtrain_epochs,
            }
            progress.pool(
                self.progress_name,
                finished=False,
            )
            self.training_details.result = {"train_data": result}

    def _callback_router(self, dataset: PrepareDataset):
        method_name = '_callback_router'
        try:
            print(method_name)
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
            print(method_name)
            return class_metric_list(self.options)
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _prepare_seed(self):
        method_name = '_prepare_seed'
        try:
            print(method_name)
            if self.options.data.architecture in YOLO_ARCHITECTURE:
                example_idx = np.arange(len(self.options.dataframe.get("val")))
                np.random.shuffle(example_idx)
            elif self.options.data.architecture in BASIC_ARCHITECTURE:
                output = self.training_details.interactive.intermediate_result.main_output
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
                    num_ex = 25 if len(y_true) > 25 else len(y_true)
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

    def _update_progress_table(self, epoch_time: float):
        method_name = '_update_progress_table'
        try:
            print(method_name)
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
                    # "learning_rate": self.current_logs.get("learning_rate"),
                    "data": {f"Прогресс обучения": {'loss': {}, 'metrics': {}}}
                }
                for loss in self.log_history['output']["loss"].keys():
                    self.progress_table[self.current_epoch]["data"]["Прогресс обучения"]["loss"][f'{loss}'] = \
                        f"{self.log_history.get('output').get('loss').get(loss).get('train')[-1]}"
                    self.progress_table[self.current_epoch]["data"]["Прогресс обучения"]["loss"][f'val_{loss}'] = \
                        f"{self.log_history.get('output').get('loss').get(loss).get('val')[-1]}"
                for metric in self.log_history['output']["metrics"].keys():
                    self.progress_table[self.current_epoch]["data"]["Прогресс обучения"]["metrics"][f"{metric}"] = \
                        f"{self.log_history.get('output').get('metrics').get(metric).get('val')[-1]}"
        except Exception as e:
            print_error(InteractiveCallback().name, method_name, e)

    def _get_loss_graph_data_request(self) -> list:
        method_name = '_get_loss_graph_data_request'
        try:
            data_return = []
            if self.options.data.architecture in BASIC_ARCHITECTURE:
                if not self.training_details.interactive.loss_graphs or not self.log_history.get("epochs"):
                    return data_return
                for loss_graph_config in self.training_details.interactive.loss_graphs:
                    loss_name = self.training_details.base.architecture.parameters.outputs[0].loss.name
                    if self.options.data.architecture in YOLO_ARCHITECTURE:
                        loss_graph_config.output_idx = 'output'
                    if loss_graph_config.show == LossGraphShowChoice.model:
                        if sum(self.log_history.get(f"{loss_graph_config.output_idx}").get("progress_state").get(
                                "loss").get(loss_name).get('overfitting')[-self.log_gap:]) >= self.progress_threashold:
                            progress_state = "overfitting"
                        elif sum(self.log_history.get(f"{loss_graph_config.output_idx}").get("progress_state").get(
                                "loss").get(loss_name).get('underfitting')[-self.log_gap:]) >= self.progress_threashold:
                            progress_state = "underfitting"
                        else:
                            progress_state = "normal"

                        train_list = self.log_history.get(f"{loss_graph_config.output_idx}").get('loss').get(
                            loss_name).get('train')
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
                            loss_name).get("val")
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
                            y=self.log_history.get(f"{loss_graph_config.output_idx}").get('loss').get(
                                loss_name).get("val"),
                            label="Проверочная выборка"
                        )

                        data_return.append(
                            fill_graph_front_structure(
                                _id=loss_graph_config.id,
                                _type='graphic',
                                graph_name=f"Выходной слой «{loss_graph_config.output_idx}» - График ошибки обучения",
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
                                           f"График ошибки обучения по классам",
                                short_name=f"{loss_graph_config.output_idx} - График ошибки обучения по классам",
                                x_label="Эпоха",
                                y_label="Значение",
                                plot_data=[
                                    fill_graph_plot_data(
                                        x=self.log_history.get("epochs"),
                                        y=self.log_history.get(f"{loss_graph_config.output_idx}").get('class_loss').get(
                                            class_name).get(loss_name).get("val"),
                                        label=f"Класс {class_name}"
                                    ) for class_name in
                                    self.options.data.outputs.get(int(loss_graph_config.output_idx)).classes_names
                                ],
                            )
                        )

            if self.options.data.architecture in YOLO_ARCHITECTURE:
                if not self.training_details.interactive.loss_graphs or not self.log_history.get("epochs"):
                    return data_return
                _id = 1
                for loss_graph_config in self.training_details.interactive.loss_graphs:
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
                                    graph_name=f"График ошибки обучения «{loss}»",
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
                                graph_name=f"График ошибки обучения «prob_loss» по классам - Проверочная выборка",
                                short_name=f"График ошибки обучения по классам - Проверочная",
                                x_label="Эпоха",
                                y_label="Значение",
                                plot_data=[
                                    fill_graph_plot_data(
                                        x=self.log_history.get("epochs"),
                                        y=self.log_history.get("output").get('class_loss').get('prob_loss').get(
                                            class_name).get("val"),
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
                if not self.training_details.interactive.metric_graphs or not self.log_history.get("epochs"):
                    return data_return
                for metric_graph_config in self.training_details.interactive.metric_graphs:
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
                                           f"{metric_graph_config.show_metric.name}",
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
                                           f"{metric_graph_config.show_metric.name} по классам",
                                short_name=f"{metric_graph_config.output_idx} - "
                                           f"{metric_graph_config.show_metric.name} по классам",
                                x_label="Эпоха",
                                y_label="Значение",
                                plot_data=[
                                    fill_graph_plot_data(
                                        x=self.log_history.get("epochs"),
                                        y=self.log_history.get(f"{metric_graph_config.output_idx}").get(
                                            'class_metrics').get(
                                            class_name).get(metric_graph_config.show_metric).get('val'),
                                        label=f"Класс {class_name}"
                                    ) for class_name in
                                    self.options.data.outputs.get(metric_graph_config.output_idx).classes_names
                                ],
                            )
                        )

            if self.options.data.architecture in YOLO_ARCHITECTURE:
                if not self.training_details.interactive.metric_graphs or not self.log_history.get("epochs"):
                    return data_return
                _id = 1
                for metric_graph_config in self.training_details.interactive.metric_graphs:
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
                            metric_graph_config.show_metric.name).get("val")
                        best_val_value = min(val_list) if min_max_mode == 'min' else max(val_list)
                        best_val = fill_graph_plot_data(
                            x=[self.log_history.get("epochs")[val_list.index(best_val_value)]],
                            y=[best_val_value],
                            label="Лучший результат на проверочной выборке"
                        )
                        val_plot = fill_graph_plot_data(
                            x=self.log_history.get("epochs"),
                            y=self.log_history.get("output").get('metrics').get(
                                metric_graph_config.show_metric.name).get("val"),
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
                                            metric_graph_config.show_metric.name).get(class_name).get("val"),
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
