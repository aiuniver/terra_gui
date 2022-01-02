import copy
import random
import string
# import time
from typing import Optional
import numpy as np
import tensorflow

from terra_ai import progress
from terra_ai.callbacks.classification_callbacks import ImageClassificationCallback, TextClassificationCallback, \
    AudioClassificationCallback, VideoClassificationCallback, DataframeClassificationCallback, TimeseriesTrendCallback
from terra_ai.callbacks.gan_callback import GANCallback, CGANCallback
from terra_ai.callbacks.object_detection_callbacks import YoloV3Callback, YoloV4Callback
from terra_ai.callbacks.regression_callbacks import DataframeRegressionCallback
from terra_ai.callbacks.segmentation_callbacks import ImageSegmentationCallback, TextSegmentationCallback
from terra_ai.callbacks.time_series_callbacks import TimeseriesCallback
from terra_ai.callbacks.utils import loss_metric_config, fill_graph_plot_data, fill_graph_front_structure, \
    get_classes_colors, BASIC_ARCHITECTURE, CLASSIFICATION_ARCHITECTURE, YOLO_ARCHITECTURE, \
    class_metric_list, reformat_fit_array, GAN_ARCHITECTURE
from terra_ai.data.datasets.extra import LayerOutputTypeChoice, DatasetGroupChoice, LayerInputTypeChoice
from terra_ai.data.training.extra import LossGraphShowChoice, MetricGraphShowChoice, ArchitectureChoice
from terra_ai.data.training.train import TrainingDetailsData
from terra_ai.datasets.preparing import PrepareDataset
import terra_ai.exceptions.callbacks as exception

__version__ = 0.086

from terra_ai.logging import logger


class InteractiveCallback:
    """Callback for interactive requests"""
    name = 'InteractiveCallback'

    def __init__(self):
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
        self.noise = 100
        pass

    def set_attributes(self, dataset: PrepareDataset, params: TrainingDetailsData):
        logger.debug(f"{InteractiveCallback.name}, {InteractiveCallback.set_attributes.__name__}")
        method_name = "set attributes"
        try:
            self.options = dataset
            self._callback_router(dataset)
            self.class_graphics = self._class_metric_list()
            logger.info(f"\ndataset architecture: {dataset.data.architecture}")
            logger.info(f"\ndataset config: \n{dataset.data}")
            logger.info(f"\ntraining parameters: \n{params.native()}\n")
            self.training_details = params
            self.last_training_details = copy.deepcopy(params)
            self.dataset_path = dataset.data.path
            self.class_colors = get_classes_colors(dataset)
            self.x_val, self.inverse_x_val = self.callback.get_x_array(dataset)
            self.random_key = ''.join(random.sample(string.ascii_letters + string.digits, 16))
        except Exception as error:
            raise exception.SetInteractiveAttributesException(
                self.__class__.__name__, method_name
            ).with_traceback(error.__traceback__)

    def clear_history(self):
        logger.debug(f"{InteractiveCallback.name}, {InteractiveCallback.clear_history.__name__}")
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
                logger.debug(f"{InteractiveCallback.name}, {InteractiveCallback.update_state.__name__}")
                data_type = self.training_details.interactive.intermediate_result.data_type.name
                if self.options.data.architecture in BASIC_ARCHITECTURE:
                    logger.debug(f"{InteractiveCallback.name}: обработка массивов...")
                    self.y_true = reformat_fit_array(
                        array={"train": arrays.get("train_true"), "val": arrays.get("val_true")}, train_idx=train_idx)
                    self.inverse_y_true = self.callback.get_inverse_array(self.y_true, self.options)
                    self.y_pred = reformat_fit_array(
                        array={"train": arrays.get("train_pred"), "val": arrays.get("val_pred")}, train_idx=train_idx)
                    self.inverse_y_pred = self.callback.get_inverse_array(self.y_pred, self.options)
                    if self.get_balance:
                        logger.debug(f"{InteractiveCallback.name}: расчет баланса датасета...")
                        self.dataset_balance = self.callback.dataset_balance(
                            options=self.options, y_true=self.y_true,
                            preset_path=self.training_details.intermediate_path,
                            class_colors=self.class_colors
                        )
                        if self.options.data.architecture in CLASSIFICATION_ARCHITECTURE:
                            self.class_idx = self.callback.prepare_class_idx(self.y_true, self.options)
                        logger.info(f"{InteractiveCallback.name}: получение индексов seed...",
                                    extra={"front_level": "info"})
                        self.seed_idx = self._prepare_seed()
                        self.get_balance = False
                    out = f"{self.training_details.interactive.intermediate_result.main_output}"
                    count = self.training_details.interactive.intermediate_result.num_examples
                    count = count if count > len(self.y_true.get(data_type).get(out)) \
                        else len(self.y_true.get(data_type).get(out))
                    logger.info(f"{InteractiveCallback.name}: получение индексов для промежуточных результатов...",
                                extra={"front_level": "info"})
                    self.example_idx = self.callback.prepare_example_idx_to_show(
                        array=self.y_pred.get(data_type).get(out),
                        true_array=self.y_true.get(data_type).get(out),
                        options=self.options,
                        output=int(out),
                        count=count,
                        choice_type=self.training_details.interactive.intermediate_result.example_choice_type,
                        seed_idx=self.seed_idx[data_type][
                                 :self.training_details.interactive.intermediate_result.num_examples]
                    )

                if self.options.data.architecture in YOLO_ARCHITECTURE:
                    logger.info(f"{InteractiveCallback.name}: обработка массивов...", extra={"front_level": "info"})
                    self.raw_y_pred = arrays.get("val_pred")
                    sensitivity = self.training_details.interactive.intermediate_result.sensitivity \
                        if self.training_details.interactive.intermediate_result.sensitivity else 0.3
                    threashold = self.training_details.interactive.intermediate_result.threashold \
                        if self.training_details.interactive.intermediate_result.threashold else 0.5
                    self.y_pred = self.callback.get_y_pred(
                        y_pred=arrays.get("val_pred"), options=self.options,
                        sensitivity=sensitivity,
                        threashold=threashold
                    )
                    if self.get_balance:
                        logger.info(f"{InteractiveCallback.name}: расчет баланса датасета...",
                                    extra={"front_level": "info"})
                        self.y_true, self.inverse_y_true = \
                            self.callback.get_y_true(self.options, self.options.data.path)
                        self.dataset_balance = self.callback.dataset_balance(
                            options=self.options, y_true=self.y_true,
                            preset_path=self.training_details.intermediate_path,
                            class_colors=self.class_colors
                        )
                        logger.info(f"{InteractiveCallback.name}: получение индексов seed...",
                                    extra={"front_level": "info"})
                        self.seed_idx = self._prepare_seed()
                        self.get_balance = False
                    count = self.training_details.interactive.intermediate_result.num_examples
                    count = count if count > len(self.options.dataframe.get(data_type)) \
                        else len(self.options.dataframe.get(data_type))
                    self.raw_y_true = arrays.get("val_true")
                    logger.info(f"{InteractiveCallback.name}: получение индексов для промежуточных результатов...",
                                extra={"front_level": "info"})
                    self.example_idx, _ = self.callback.prepare_example_idx_to_show(
                        array=self.y_pred,
                        true_array=self.y_true,
                        name_classes=self.options.data.outputs.get(
                            list(self.options.data.outputs.keys())[0]).classes_names,
                        box_channel=self.training_details.interactive.intermediate_result.box_channel,
                        count=count,
                        choice_type=self.training_details.interactive.intermediate_result.example_choice_type,
                        seed_idx=self.seed_idx['val'],
                        sensitivity=self.training_details.interactive.intermediate_result.sensitivity,
                    )

                if self.options.data.architecture in GAN_ARCHITECTURE:
                    logger.debug(f"{InteractiveCallback.name}: обработка массивов...")
                    self.y_pred = arrays
                    count = self.training_details.interactive.intermediate_result.num_examples
                    logger.debug(f"{InteractiveCallback.name}: получение индексов для промежуточных результатов...")
                    self.example_idx = self.callback.prepare_example_idx_to_show(
                        array=self.y_pred.get('train'),
                        seed_array=self.y_pred.get('seed'),
                        count=count,
                        choice_type=self.training_details.interactive.intermediate_result.example_choice_type
                    )

                if on_epoch_end_flag:
                    self.current_epoch = fit_logs.get('epochs')[-1]
                    logger.debug(f"{InteractiveCallback.name}: обновление логов и таблицы прогресса обучения...")
                    self.log_history = fit_logs
                    self._update_progress_table(current_epoch_time)
                    if self.training_details.interactive.intermediate_result.autoupdate:
                        logger.debug(f"{InteractiveCallback.name}: расчет промежуточных результатов...")
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
                    if self.options.data.architecture in BASIC_ARCHITECTURE and \
                            self.training_details.interactive.statistic_data.output_id \
                            and self.training_details.interactive.statistic_data.autoupdate:
                        logger.debug(f"{InteractiveCallback.name}: расчет статистических данных...")
                        self.statistic_result = self.callback.statistic_data_request(
                            interactive_config=self.training_details.interactive,
                            options=self.options,
                            y_true=self.y_true,
                            inverse_y_true=self.inverse_y_true,
                            y_pred=self.y_pred,
                            inverse_y_pred=self.inverse_y_pred,
                        )
                    if self.options.data.architecture in YOLO_ARCHITECTURE and \
                            self.training_details.interactive.statistic_data.box_channel \
                            and self.training_details.interactive.statistic_data.autoupdate:
                        logger.debug(f"{InteractiveCallback.name}: расчет статистических данных...")
                        self.statistic_result = self.callback.statistic_data_request(
                            interactive_config=self.training_details.interactive,
                            options=self.options,
                            y_true=self.y_true,
                            y_pred=self.y_pred,
                            inverse_y_pred=self.inverse_y_pred,
                            inverse_y_true=self.inverse_y_true
                        )
                else:
                    logger.debug(f"{InteractiveCallback.name}: расчет промежуточных результатов...")
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
                    if self.options.data.architecture in BASIC_ARCHITECTURE and \
                            self.training_details.interactive.statistic_data.output_id:
                        logger.debug(f"{InteractiveCallback.name}: расчет статистических данных...")
                        self.statistic_result = self.callback.statistic_data_request(
                            interactive_config=self.training_details.interactive,
                            options=self.options,
                            y_true=self.y_true,
                            y_pred=self.y_pred,
                            inverse_y_pred=self.inverse_y_pred,
                            inverse_y_true=self.inverse_y_true,
                        )
                    if self.options.data.architecture in YOLO_ARCHITECTURE and \
                            self.training_details.interactive.statistic_data.box_channel:
                        logger.debug(f"{InteractiveCallback.name}: расчет статистических данных...")
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
                logger.info(f"{InteractiveCallback.name}: передача данных на страницу обучения...",
                            extra={"front_level": "info"})
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
            self.get_balance = True
            return {}

    def get_train_results(self):
        """Return dict with data for current interactive request"""
        if self.log_history and self.log_history.get("epochs", {}):
            logger.debug(f"{InteractiveCallback.name}, {InteractiveCallback.get_train_results.__name__}")
            data_type = self.training_details.interactive.intermediate_result.data_type.name
            if self.options.data.architecture in BASIC_ARCHITECTURE:
                if self.training_details.interactive.intermediate_result.show_results:
                    out = f"{self.training_details.interactive.intermediate_result.main_output}"
                    count = self.training_details.interactive.intermediate_result.num_examples
                    count = count if count > len(self.y_true.get(data_type).get(out)) \
                        else len(self.y_true.get(data_type).get(out))
                    self.example_idx = self.callback.prepare_example_idx_to_show(
                        array=self.y_true.get(data_type).get(out),
                        true_array=self.y_true.get(data_type).get(out),
                        options=self.options,
                        output=int(out),
                        count=count,
                        choice_type=self.training_details.interactive.intermediate_result.example_choice_type,
                        seed_idx=self.seed_idx[data_type][
                                 :self.training_details.interactive.intermediate_result.num_examples]
                    )
                if self.training_details.interactive.intermediate_result.show_results or \
                        self.training_details.interactive.statistic_data.output_id:
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
                    count = count if count > len(self.options.dataframe.get(data_type)) \
                        else len(self.options.dataframe.get(data_type))
                    self.example_idx, _ = self.callback.prepare_example_idx_to_show(
                        array=self.y_pred,
                        true_array=self.y_true,
                        name_classes=self.options.data.outputs.get(
                            list(self.options.data.outputs.keys())[0]).classes_names,
                        box_channel=self.training_details.interactive.intermediate_result.box_channel,
                        count=count,
                        choice_type=self.training_details.interactive.intermediate_result.example_choice_type,
                        seed_idx=self.seed_idx['val'],
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
        logger.debug(f"{InteractiveCallback.name}, {InteractiveCallback._callback_router.__name__}")
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
            elif dataset.data.architecture == ArchitectureChoice.GAN:
                self.callback = GANCallback()
            elif dataset.data.architecture == ArchitectureChoice.CGAN:
                self.callback = CGANCallback()
            else:
                pass
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                InteractiveCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    def _class_metric_list(self):
        logger.debug(f"{InteractiveCallback.name}, {InteractiveCallback._class_metric_list.__name__}")
        method_name = '_class_metric_list'
        try:
            return class_metric_list(self.options)
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                InteractiveCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    def _prepare_seed(self):
        logger.debug(f"{InteractiveCallback.name}, {InteractiveCallback._prepare_seed.__name__}")
        method_name = '_prepare_seed'
        try:
            example_idx = {}
            if self.options.data.architecture in GAN_ARCHITECTURE:
                return example_idx
            for data_type in ['train', 'val']:
                if self.options.data.architecture in YOLO_ARCHITECTURE:
                    example_idx[data_type] = np.arange(len(self.options.dataframe.get(data_type)))
                    np.random.shuffle(example_idx[data_type])
                elif self.options.data.architecture in BASIC_ARCHITECTURE:
                    output = self.training_details.interactive.intermediate_result.main_output
                    example_idx[data_type] = []
                    if self.options.data.architecture in CLASSIFICATION_ARCHITECTURE:
                        y_true = np.argmax(self.y_true.get(data_type).get(f"{output}"), axis=-1)
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
                            example_idx[data_type].append(class_idx[key][0])
                            class_idx[key].pop(0)
                            num_ex -= 1
                    else:
                        if self.options.data.group == DatasetGroupChoice.keras or self.x_val:
                            example_idx[data_type] = np.arange(len(self.y_true.get(data_type).get(
                                list(self.y_true.get(data_type).keys())[0])))
                        else:
                            example_idx[data_type] = np.arange(len(self.options.dataframe.get(data_type)))
                        np.random.shuffle(example_idx[data_type])
                else:
                    example_idx[data_type] = np.arange(len(self.options.dataframe.get(data_type)))
                    np.random.shuffle(example_idx[data_type])
            return example_idx
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                InteractiveCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    def _update_progress_table(self, epoch_time: float):
        logger.debug(f"{InteractiveCallback.name}, {InteractiveCallback._update_progress_table.__name__}")
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

            if self.options.data.architecture in GAN_ARCHITECTURE:
                self.progress_table[self.current_epoch] = {
                    "time": epoch_time,
                    # "learning_rate": self.current_logs.get("learning_rate"),
                    "data": {f"Прогресс обучения": {'loss': {}, 'metrics': {}}}
                }
                for loss in self.log_history['output']["loss"].keys():
                    self.progress_table[self.current_epoch]["data"]["Прогресс обучения"]["loss"][f'{loss}'] = \
                        f"{self.log_history.get('output').get('loss').get(loss).get('train')[-1]}"
                    # self.progress_table[self.current_epoch]["data"]["Прогресс обучения"]["loss"][f'val_{loss}'] = \
                    #     f"{self.log_history.get('output').get('loss').get(loss).get('val')[-1]}"
                # for metric in self.log_history['output']["metrics"].keys():
                #     self.progress_table[self.current_epoch]["data"]["Прогресс обучения"]["metrics"][f"{metric}"] = \
                #         f"{self.log_history.get('output').get('metrics').get(metric).get('val')[-1]}"

        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                InteractiveCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    def _get_loss_graph_data_request(self) -> list:
        logger.debug(f"{InteractiveCallback.name}, {InteractiveCallback._get_loss_graph_data_request.__name__}")
        method_name = '_get_loss_graph_data_request'
        try:
            data_return = []
            if self.options.data.architecture in BASIC_ARCHITECTURE:
                if not self.training_details.interactive.loss_graphs or not self.log_history.get("epochs"):
                    return data_return
                for loss_graph_config in self.training_details.interactive.loss_graphs:
                    loss_name = self.training_details.base.architecture.parameters.outputs[0].loss.name
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
                                graph_name=f"Выход «{loss_graph_config.output_idx}» - График ошибки обучения",
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
                        loss_data_type = loss_graph_config.data_type.name
                        data_type_name = "Тренировочная" if loss_data_type == "train" else "Проверочная"
                        data_return.append(
                            fill_graph_front_structure(
                                _id=loss_graph_config.id,
                                _type='graphic',
                                graph_name=f"Выход «{loss_graph_config.output_idx}» - "
                                           f"График ошибки обучения по классам - {data_type_name} выборка",
                                short_name=f"{loss_graph_config.output_idx} - График ошибки обучения по классам",
                                x_label="Эпоха",
                                y_label="Значение",
                                plot_data=[
                                    fill_graph_plot_data(
                                        x=self.log_history.get("epochs"),
                                        y=self.log_history.get(f"{loss_graph_config.output_idx}").get('class_loss').get(
                                            class_name).get(loss_name).get(loss_data_type),
                                        label=f"Класс {class_name}"
                                    ) for class_name in
                                    self.options.data.outputs.get(int(loss_graph_config.output_idx)).classes_names
                                ],
                            )
                        )

            if self.options.data.architecture in YOLO_ARCHITECTURE:
                if not self.training_details.interactive.loss_graphs or not self.log_history.get("epochs"):
                    return data_return
                model_loss = 0
                for loss_graph_config in self.training_details.interactive.loss_graphs:
                    if loss_graph_config.show == LossGraphShowChoice.model:
                        idx = model_loss if model_loss <= 3 else model_loss % 4
                        loss = list(self.log_history.get('output').get('loss').keys())[idx]
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
                                _id=loss_graph_config.id,
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
                        model_loss += 1
                    if loss_graph_config.show == LossGraphShowChoice.classes:
                        loss_data_type = loss_graph_config.data_type.name
                        data_type_name = "Тренировочная" if loss_data_type == "train" else "Проверочная"
                        output_idx = list(self.options.data.outputs.keys())[0]
                        data_return.append(
                            fill_graph_front_structure(
                                _id=loss_graph_config.id,
                                _type='graphic',
                                graph_name=f"График ошибки обучения «prob_loss» по классам - {data_type_name} выборка",
                                short_name=f"График ошибки обучения по классам - {data_type_name}",
                                x_label="Эпоха",
                                y_label="Значение",
                                plot_data=[
                                    fill_graph_plot_data(
                                        x=self.log_history.get("epochs"),
                                        y=self.log_history.get("output").get('class_loss').get('prob_loss').get(
                                            class_name).get(loss_data_type),
                                        label=f"Класс {class_name}"
                                    ) for class_name in self.options.data.outputs.get(output_idx).classes_names
                                ],
                            )
                        )

            if self.options.data.architecture in GAN_ARCHITECTURE:
                if not self.training_details.interactive.loss_graphs or not self.log_history.get("epochs"):
                    return data_return
                model = 0
                for loss_graph_config in self.training_details.interactive.loss_graphs:
                    if model % 2 == 0:
                        progress_state = "normal"
                        gen_list = self.log_history.get(f"output").get('loss').get('gen_loss').get('train')
                        no_none_gen = []
                        for x in gen_list:
                            if x is not None:
                                no_none_gen.append(x)
                        best_gen_value = min(no_none_gen) if no_none_gen else None
                        best_gen = fill_graph_plot_data(
                            x=[self.log_history.get("epochs")[gen_list.index(best_gen_value)]
                               if best_gen_value is not None else None],
                            y=[best_gen_value],
                            label="Лучший результат генератора"
                        )
                        gen_plot = fill_graph_plot_data(
                            x=self.log_history.get("epochs"), y=gen_list, label="Генератор"
                        )
                        disc_list = self.log_history.get(f"output").get('loss').get('disc_loss').get("train")
                        no_none_disc = []
                        for x in disc_list:
                            if x is not None:
                                no_none_disc.append(x)
                        best_disc_value = min(no_none_disc) if no_none_disc else None
                        best_disc = fill_graph_plot_data(
                            x=[self.log_history.get("epochs")[disc_list.index(best_disc_value)]
                               if best_disc_value is not None else None],
                            y=[best_disc_value],
                            label="Лучший результат дискриминатора"
                        )
                        disc_plot = fill_graph_plot_data(
                            x=self.log_history.get("epochs"), y=disc_list, label="Дискриминатор"
                        )
                        data_return.append(
                            fill_graph_front_structure(
                                _id=loss_graph_config.id,
                                _type='graphic',
                                graph_name=f"Выход «{loss_graph_config.output_idx}» - График ошибки генератора и дискриминатора",
                                short_name=f"{loss_graph_config.output_idx} - График ошибки обучения",
                                x_label="Эпоха",
                                y_label="Значение",
                                plot_data=[gen_plot, disc_plot],
                                best=[best_gen, best_disc],
                                progress_state=progress_state
                            )
                        )
                        model += 1
                    else:
                        progress_state = "normal"
                        real_list = self.log_history.get(f"output").get('loss').get('disc_real_loss').get('train')
                        no_none_real = []
                        for x in real_list:
                            if x is not None:
                                no_none_real.append(x)
                        best_real_value = min(no_none_real) if no_none_real else None
                        best_real = fill_graph_plot_data(
                            x=[self.log_history.get("epochs")[real_list.index(best_real_value)]
                               if best_real_value is not None else None],
                            y=[best_real_value],
                            label="Лучший результат на реальных данных"
                        )
                        real_plot = fill_graph_plot_data(
                            x=self.log_history.get("epochs"), y=real_list, label="Реальные данные"
                        )
                        fake_list = self.log_history.get(f"output").get('loss').get('disc_fake_loss').get("train")
                        no_none_fake = []
                        for x in fake_list:
                            if x is not None:
                                no_none_fake.append(x)
                        best_fake_value = min(no_none_fake) if no_none_fake else None
                        best_fake = fill_graph_plot_data(
                            x=[self.log_history.get("epochs")[fake_list.index(best_fake_value)]
                               if best_fake_value is not None else None],
                            y=[best_fake_value],
                            label="Лучший результат на сгенерированных данных"
                        )
                        fake_plot = fill_graph_plot_data(
                            x=self.log_history.get("epochs"), y=fake_list, label="Сгенерированные данные"
                        )
                        data_return.append(
                            fill_graph_front_structure(
                                _id=loss_graph_config.id,
                                _type='graphic',
                                graph_name=f"Выход «{loss_graph_config.output_idx}» - "
                                           f"График ошибки дискриминатора на реальных и сгенерированных данных",
                                short_name=f"{loss_graph_config.output_idx} - Тип данных",
                                x_label="Эпоха",
                                y_label="Значение",
                                plot_data=[real_plot, fake_plot],
                                best=[best_real, best_fake],
                                progress_state=progress_state
                            )
                        )
                        model += 1

            return data_return
        except Exception as error:
            if self.first_error:
                self.first_error = False
                exc = exception.ErrorInClassInMethodException(
                    InteractiveCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
                # logger.error(exc)
                raise exc
            else:
                pass

    def _get_metric_graph_data_request(self) -> list:
        logger.debug(f"{InteractiveCallback.name}, {InteractiveCallback._get_metric_graph_data_request.__name__}")
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
                                graph_name=f"Выход «{metric_graph_config.output_idx}» - График метрики "
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
                        metric_data_type = metric_graph_config.data_type.name
                        data_type_name = "Тренировочная" if metric_data_type == "train" else "Проверочная"
                        data_return.append(
                            fill_graph_front_structure(
                                _id=metric_graph_config.id,
                                _type='graphic',
                                graph_name=f"Выход «{metric_graph_config.output_idx}» - График метрики "
                                           f"{metric_graph_config.show_metric.name} по классам - "
                                           f"{data_type_name} выборка",
                                short_name=f"{metric_graph_config.output_idx} - "
                                           f"{metric_graph_config.show_metric.name} по классам",
                                x_label="Эпоха",
                                y_label="Значение",
                                plot_data=[
                                    fill_graph_plot_data(
                                        x=self.log_history.get("epochs"),
                                        y=self.log_history.get(f"{metric_graph_config.output_idx}").get(
                                            'class_metrics').get(class_name).get(metric_graph_config.show_metric
                                                                                 ).get(metric_data_type),
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
                                graph_name=f"График метрики {metric_graph_config.show_metric.name}",
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
                                           f"Проверочная выборка",
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
        except Exception as error:
            if self.first_error:
                self.first_error = False
                exc = exception.ErrorInClassInMethodException(
                    InteractiveCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
                # logger.error(exc)
                raise exc
            else:
                pass
