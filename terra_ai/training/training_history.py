import copy
import importlib
import json
import math
import os

import numpy as np

from tensorflow.python.keras.utils.np_utils import to_categorical

from terra_ai.callbacks import interactive
from terra_ai.callbacks.classification_callbacks import BaseClassificationCallback
from terra_ai.callbacks.utils import BASIC_ARCHITECTURE, CLASS_ARCHITECTURE, YOLO_ARCHITECTURE, \
    CLASSIFICATION_ARCHITECTURE, loss_metric_config, round_loss_metric, class_metric_list, GAN_ARCHITECTURE
from terra_ai.data.datasets.extra import LayerEncodingChoice
from terra_ai.data.presets.training import Metric
from terra_ai.data.training.extra import StateStatusChoice
from terra_ai.data.training.train import TrainingDetailsData
from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.exceptions.training import NoHistoryLogsException
import terra_ai.exceptions.callbacks as exception
from terra_ai.logging import logger

OUTPUT_LOG_CONFIG = {
    "loss": {
        'giou_loss': {"train": [], "val": []},
        'conf_loss': {"train": [], "val": []},
        'prob_loss': {"train": [], "val": []},
        'total_loss': {"train": [], "val": []}
    },
    "class_loss": {'prob_loss': {}},
    "metrics": {'mAP50': {"train": [], "val": []}},
    "class_metrics": {'mAP50': {}},
    "progress_state": {
        "loss": {
            'giou_loss': {
                "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []},
            'conf_loss': {
                "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []},
            'prob_loss': {
                "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []},
            'total_loss': {
                "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []}
        },
        "metrics": {
            'mAP50': {"mean_log_history": [], "normal_state": [], "overfitting": []}
        }
    }
}


class History:
    name = "History"

    def __init__(self, dataset: PrepareDataset, training_details: TrainingDetailsData, deploy_type: str = ""):
        self.architecture_type = deploy_type
        self.current_logs = {}
        self.dataset = dataset
        self.training_detail = training_details
        self.training_detail.logs = None
        self.last_epoch = 0
        self.epochs = training_details.base.epochs
        self.sum_epoch = self.epochs
        self.log_history = self._load_logs(dataset=dataset, training_details=training_details)
        self.class_outputs = class_metric_list(dataset)
        if self.architecture_type in CLASSIFICATION_ARCHITECTURE:
            self.y_true, _ = BaseClassificationCallback().get_y_true(dataset)
            self.class_idx = BaseClassificationCallback().prepare_class_idx(self.y_true, dataset)

    def get_history(self):
        return self.log_history

    def save_logs(self):
        method_name = 'save_logs'
        try:
            logs = {
                "fit_log": self.log_history,
                "interactive_log": interactive.log_history,
                "progress_table": interactive.progress_table,
                "addtrain_epochs": interactive.addtrain_epochs,
                "sum_epoch": self.sum_epoch
            }
            self.training_detail.logs = logs
            with open(os.path.join(self.training_detail.model_path, "log.history"), "w", encoding="utf-8") as history:
                json.dump(logs, history)
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                History.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    def _load_logs(self, dataset: PrepareDataset, training_details: TrainingDetailsData):
        method_name = '_load_logs'
        try:
            if self.training_detail.state.status == StateStatusChoice.addtrain:
                if self.training_detail.logs:
                    logs = self.training_detail.logs
                else:
                    with open(os.path.join(self.training_detail.model_path, "log.history"), "r",
                              encoding="utf-8") as history:
                        logs = json.load(history)
                interactive.log_history = logs.get("interactive_log")
                interactive.progress_table = logs.get("progress_table")
                interactive.addtrain_epochs = logs.get("addtrain_epochs")
                fit_logs = logs.get("fit_log")

                self.last_epoch = max(fit_logs.get('epochs'))
                self.sum_epoch = logs.get("sum_epoch")
                if self.training_detail.state.status == "addtrain":
                    if logs.get("addtrain_epochs")[-1] >= self.sum_epoch:
                        self.sum_epoch += self.training_detail.base.epochs
                    if logs.get("addtrain_epochs")[-1] < self.sum_epoch:
                        self.epochs = self.sum_epoch - logs.get("addtrain_epochs")[-1]
                return fit_logs
            else:
                return self._prepare_log_history_template(options=dataset, params=training_details)
        except Exception as error:
            raise NoHistoryLogsException(
                self.__class__.__name__, method_name
            ).with_traceback(error.__traceback__)

    @staticmethod
    def _prepare_log_history_template(options: PrepareDataset, params: TrainingDetailsData):
        method_name = '_prepare_log_history_template'
        try:
            log_history = {"epochs": []}
            if options.data.architecture in BASIC_ARCHITECTURE:
                for output_layer in params.base.architecture.parameters.outputs:
                    out = f"{output_layer.id}"
                    log_history[out] = {
                        "loss": {}, "metrics": {},
                        "class_loss": {}, "class_metrics": {},
                        "progress_state": {"loss": {}, "metrics": {}}
                    }
                    log_history[out]["loss"][output_layer.loss.name] = {"train": [], "val": []}
                    log_history[out]["progress_state"]["loss"][output_layer.loss.name] = {
                        "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []
                    }
                    for metric in output_layer.metrics:
                        log_history[out]["metrics"][metric.name] = {"train": [], "val": []}
                        log_history[out]["progress_state"]["metrics"][metric.name] = {
                            "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []
                        }

                    if options.data.architecture in CLASS_ARCHITECTURE:
                        log_history[out]["class_loss"] = {}
                        log_history[out]["class_metrics"] = {}
                        for class_name in options.data.outputs.get(int(out)).classes_names:
                            log_history[out]["class_metrics"][class_name] = {}
                            log_history[out]["class_loss"][class_name] = \
                                {output_layer.loss.name: {"train": [], "val": []}}
                            for metric in output_layer.metrics:
                                log_history[out]["class_metrics"][class_name][metric.name] = {"train": [], "val": []}

            if options.data.architecture in YOLO_ARCHITECTURE:
                log_history['output'] = copy.deepcopy(OUTPUT_LOG_CONFIG)
                out = list(options.data.outputs.keys())[0]
                for class_name in options.data.outputs.get(out).classes_names:
                    log_history['output']["class_loss"]['prob_loss'][class_name] = {"train": [], "val": []}
                    log_history['output']["class_metrics"]['mAP50'][class_name] = {"train": [], "val": []}

            if options.data.architecture in GAN_ARCHITECTURE:
                log_history['output'] = {
                    "loss": {
                        'gen_loss': {"train": [], "val": []},
                        'disc_loss': {"train": [], "val": []},
                        'disc_real_loss': {"train": [], "val": []},
                        'disc_fake_loss': {"train": [], "val": []}
                    },
                    "metrics": {},
                    "progress_state": {
                        "loss": {
                            'gen_loss': {
                                "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []},
                            'disc_loss': {
                                "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []},
                            'disc_real_loss': {
                                "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []},
                            'disc_fake_loss': {
                                "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []}
                        }
                    }
                }
                # out = list(options.data.outputs.keys())[0]
                # for class_name in options.data.outputs.get(out).classes_names:
                #     log_history['output']["class_loss"]['prob_loss'][class_name] = {"train": [], "val": []}
                #     log_history['output']["class_metrics"]['mAP50'][class_name] = {"train": [], "val": []}
            return log_history
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                History.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def update_class_idx(dataset_class_idx, predict_idx):
        method_name = 'update_class_idx'
        try:
            update_idx = {'train': {}, "val": dataset_class_idx.get('val')}
            for out in dataset_class_idx['train'].keys():
                update_idx['train'][out] = {}
                for cls in dataset_class_idx['train'][out].keys():
                    shift = predict_idx[0]
                    update_idx['train'][out][cls] = list(np.array(dataset_class_idx['train'][out][cls]) - shift)
            return update_idx
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                History.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    def get_checkpoint_data(self, output: str, checkpoint_type: str, metric: str, indicator: str):
        return self.log_history.get(output, {}).get(checkpoint_type, {}).get(metric, {}).get(indicator, None)

    def current_basic_logs(self, epoch: int, arrays: dict, train_idx: list):
        method_name = 'current_basic_logs'
        try:
            self.current_logs = {"epochs": epoch}
            update_cls = {}
            if self.architecture_type in CLASSIFICATION_ARCHITECTURE:
                update_cls = self.update_class_idx(self.class_idx, train_idx)
            for output_layer in self.training_detail.base.architecture.parameters.outputs:
                out = f"{output_layer.id}"
                name_classes = self.dataset.data.outputs.get(output_layer.id).classes_names
                self.current_logs[out] = {"loss": {}, "metrics": {}, "class_loss": {}, "class_metrics": {}}

                # calculate loss
                loss_name = output_layer.loss.name
                loss_fn = getattr(
                    importlib.import_module(loss_metric_config.get("loss").get(loss_name, {}).get('module')), loss_name
                )
                train_loss = self._get_loss_calculation(
                    loss_obj=loss_fn, out=out, y_true=arrays.get("train_true").get(out),
                    y_pred=arrays.get("train_pred").get(out))
                val_loss = self._get_loss_calculation(
                    loss_obj=loss_fn, out=out, y_true=arrays.get("val_true").get(out),
                    y_pred=arrays.get("val_pred").get(out))
                self.current_logs[out]["loss"][output_layer.loss.name] = {"train": train_loss, "val": val_loss}
                if self.class_outputs.get(output_layer.id):
                    self.current_logs[out]["class_loss"][output_layer.loss.name] = {}
                    if self.dataset.data.architecture in CLASSIFICATION_ARCHITECTURE:
                        for i, cls in enumerate(name_classes):
                            train_class_loss = self._get_loss_calculation(
                                loss_obj=loss_fn, out=out,
                                y_true=arrays.get("train_true").get(out)[update_cls['train'][out][cls], ...],
                                y_pred=arrays.get("train_pred").get(out)[update_cls['train'][out][cls], ...])
                            val_class_loss = self._get_loss_calculation(
                                loss_obj=loss_fn, out=out,
                                y_true=arrays.get("val_true").get(out)[update_cls['val'][out][cls], ...],
                                y_pred=arrays.get("val_pred").get(out)[update_cls['val'][out][cls], ...])
                            self.current_logs[out]["class_loss"][output_layer.loss.name][cls] = \
                                {"train": train_class_loss, "val": val_class_loss}
                    else:
                        for i, cls in enumerate(name_classes):
                            train_class_loss = self._get_loss_calculation(
                                loss_obj=loss_fn, out=out, class_idx=i, show_class=True,
                                y_true=arrays.get("train_true").get(out), y_pred=arrays.get("train_pred").get(out))
                            val_class_loss = self._get_loss_calculation(
                                loss_obj=loss_fn, out=out, class_idx=i, show_class=True,
                                y_true=arrays.get("val_true").get(out), y_pred=arrays.get("val_pred").get(out))
                            self.current_logs[out]["class_loss"][output_layer.loss.name][cls] = \
                                {"train": train_class_loss, "val": val_class_loss}

                # calculate metrics
                for metric_name in output_layer.metrics:
                    metric_name = metric_name.name
                    metric_fn = getattr(
                        importlib.import_module(loss_metric_config.get("metric").get(metric_name, {}).get('module')),
                        metric_name
                    )
                    train_metric = self._get_metric_calculation(
                        metric_name, metric_fn, out,
                        arrays.get("train_true").get(out), arrays.get("train_pred").get(out))
                    val_metric = self._get_metric_calculation(
                        metric_name, metric_fn, out, arrays.get("val_true").get(out), arrays.get("val_pred").get(out))
                    self.current_logs[out]["metrics"][metric_name] = {"train": train_metric, "val": val_metric}
                    if self.class_outputs.get(output_layer.id):
                        self.current_logs[out]["class_metrics"][metric_name] = {}
                        if self.dataset.data.architecture in CLASSIFICATION_ARCHITECTURE and \
                                metric_name not in [Metric.BalancedRecall, Metric.BalancedPrecision,
                                                    Metric.BalancedFScore, Metric.FScore]:
                            for i, cls in enumerate(name_classes):
                                train_class_metric = self._get_metric_calculation(
                                    metric_name=metric_name, metric_obj=metric_fn, out=out,
                                    y_true=arrays.get("train_true").get(out)[update_cls['train'][out][cls], ...],
                                    y_pred=arrays.get("train_pred").get(out)[update_cls['train'][out][cls], ...])
                                val_class_metric = self._get_metric_calculation(
                                    metric_name=metric_name, metric_obj=metric_fn, out=out,
                                    y_true=arrays.get("val_true").get(out)[update_cls['val'][out][cls], ...],
                                    y_pred=arrays.get("val_pred").get(out)[update_cls['val'][out][cls], ...])
                                self.current_logs[out]["class_metrics"][metric_name][cls] = \
                                    {"train": train_class_metric, "val": val_class_metric}
                        else:
                            for i, cls in enumerate(name_classes):
                                train_class_metric = self._get_metric_calculation(
                                    metric_name=metric_name, metric_obj=metric_fn, out=out, show_class=True,
                                    y_true=arrays.get("train_true").get(out),
                                    y_pred=arrays.get("train_pred").get(out), class_idx=i)
                                val_class_metric = self._get_metric_calculation(
                                    metric_name=metric_name, metric_obj=metric_fn, out=out, show_class=True,
                                    y_true=arrays.get("val_true").get(out),
                                    y_pred=arrays.get("val_pred").get(out), class_idx=i)
                                self.current_logs[out]["class_metrics"][metric_name][cls] = \
                                    {"train": train_class_metric, "val": val_class_metric}
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                History.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    def _get_loss_calculation(self, loss_obj, out: str, y_true, y_pred, show_class=False, class_idx=0):
        method_name = '_get_loss_calculation'
        try:
            encoding = self.dataset.data.outputs.get(int(out)).encoding
            num_classes = self.dataset.data.outputs.get(int(out)).num_classes
            if show_class and (encoding == LayerEncodingChoice.ohe or encoding == LayerEncodingChoice.multi) and \
                    loss_obj.__name__ in ['CategoricalHinge', 'CategoricalCrossentropy', 'BinaryCrossentropy']:
                array_shape = list(y_true.shape[:-1])
                array_shape.append(2)
                true_array = np.zeros(array_shape).astype('float32')
                true_array[..., 0] = y_true[..., class_idx]
                true_array[..., 1] = 1 - y_true[..., class_idx]
                pred_array = np.zeros(array_shape).astype('float32')
                pred_array[..., 0] = y_pred[..., class_idx]
                pred_array[..., 1] = 1 - y_pred[..., class_idx]
            elif show_class and (encoding == LayerEncodingChoice.ohe or encoding == LayerEncodingChoice.multi):
                true_array = y_true[..., class_idx:class_idx + 1]
                pred_array = y_pred[..., class_idx:class_idx + 1]
            elif show_class:
                true_array = to_categorical(y_true, num_classes)[..., class_idx:class_idx + 1]
                pred_array = y_pred[..., class_idx:class_idx + 1]
            else:
                true_array = y_true
                pred_array = y_pred
            loss_value = float(loss_obj()(true_array, pred_array).numpy())
            return loss_value if not math.isnan(loss_value) else None
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                History.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    def _get_metric_calculation(self, metric_name, metric_obj, out: str, y_true, y_pred, show_class=False, class_idx=0):
        method_name = '_get_metric_calculation'
        try:
            encoding = self.dataset.data.outputs.get(int(out)).encoding
            num_classes = self.dataset.data.outputs.get(int(out)).num_classes
            if metric_name == Metric.MeanIoU:
                m = metric_obj(num_classes)
            elif metric_name == Metric.BalancedDiceCoef:
                m = metric_obj(encoding=encoding.name)
            else:
                m = metric_obj()
            if show_class and (encoding == LayerEncodingChoice.ohe or encoding == LayerEncodingChoice.multi):
                if metric_name == Metric.Accuracy:
                    true_array = to_categorical(np.argmax(y_true, axis=-1), num_classes)[..., class_idx]
                    pred_array = to_categorical(np.argmax(y_pred, axis=-1), num_classes)[..., class_idx]
                    m.update_state(true_array, pred_array)
                elif metric_name in [Metric.BalancedRecall, Metric.BalancedPrecision, Metric.BalancedFScore,
                                     Metric.FScore, Metric.BalancedDiceCoef]:
                    m.update_state(y_true, y_pred, show_class=show_class, class_idx=class_idx)
                else:
                    m.update_state(y_true[..., class_idx:class_idx + 1], y_pred[..., class_idx:class_idx + 1])
            elif show_class:
                if metric_name == Metric.Accuracy:
                    true_array = y_true[..., class_idx]
                    pred_array = to_categorical(np.argmax(y_pred, axis=-1), num_classes)[..., class_idx]
                    m.update_state(true_array, pred_array)
                else:
                    true_array = to_categorical(y_true, num_classes)[..., class_idx:class_idx + 1]
                    pred_array = y_pred[..., class_idx:class_idx + 1]
                    m.update_state(true_array, pred_array)
            else:
                if metric_name in [Metric.UnscaledMAE, Metric.PercentMAE]:
                    m.update_state(y_true, y_pred, output=int(out), preprocess=self.dataset.preprocessing)
                else:
                    m.update_state(y_true, y_pred)
            metric_value = float(m.result().numpy())
            return metric_value if not math.isnan(metric_value) else None
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                History.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    def update_log_history(self):
        method_name = 'update_log_history'
        try:
            if self.current_logs['epochs'] in self.log_history['epochs']:
                logger.warning(f"Текущая эпоха {self.current_logs['epochs']} уже записана ранее в логи")
            self.log_history['epochs'].append(self.current_logs['epochs'])
            if self.dataset.data.architecture in BASIC_ARCHITECTURE:
                for output_layer in self.training_detail.base.architecture.parameters.outputs:
                    out = f"{output_layer.id}"
                    classes_names = self.dataset.data.outputs.get(output_layer.id).classes_names
                    loss_name = output_layer.loss.name
                    for data_type in ['train', 'val']:
                        self.log_history[out]['loss'][loss_name][data_type].append(
                            round_loss_metric(self.current_logs.get(out).get('loss').get(loss_name).get(data_type))
                        )
                    self.log_history[out]['progress_state']['loss'][loss_name]['mean_log_history'].append(
                        round_loss_metric(
                            self._get_mean_log(self.log_history.get(out).get('loss').get(loss_name).get('val')))
                    )
                    loss_underfitting = self._evaluate_underfitting(
                        metric_name=loss_name, train_log=self.log_history[out]['loss'][loss_name]['train'][-1],
                        val_log=self.log_history[out]['loss'][loss_name]['val'][-1], metric_type='loss'
                    )
                    loss_overfitting = self._evaluate_overfitting(
                        metric_name=loss_name, metric_type='loss',
                        mean_log=self.log_history[out]['progress_state']['loss'][loss_name]['mean_log_history']
                    )
                    if loss_underfitting or loss_overfitting:
                        normal_state = False
                    else:
                        normal_state = True

                    self.log_history[out]['progress_state']['loss'][loss_name][
                        'underfitting'].append(loss_underfitting)
                    self.log_history[out]['progress_state']['loss'][loss_name][
                        'overfitting'].append(loss_overfitting)
                    self.log_history[out]['progress_state']['loss'][loss_name][
                        'normal_state'].append(normal_state)

                    if self.current_logs.get(out).get("class_loss"):
                        for cls in classes_names:
                            self.log_history[out]['class_loss'][cls][loss_name]["train"].append(
                                round_loss_metric(self.current_logs[out]['class_loss'][loss_name][cls]["train"])
                            )
                            self.log_history[out]['class_loss'][cls][loss_name]["val"].append(
                                round_loss_metric(self.current_logs[out]['class_loss'][loss_name][cls]["val"])
                            )

                    for metric_name in output_layer.metrics:
                        metric_name = metric_name.name
                        for data_type in ['train', 'val']:
                            self.log_history[out]['metrics'][metric_name][data_type].append(
                                round_loss_metric(
                                    self.current_logs.get(out).get('metrics').get(metric_name).get(data_type)))
                        self.log_history[out]['progress_state']['metrics'][metric_name]['mean_log_history'].append(
                            round_loss_metric(self._get_mean_log(self.log_history[out]['metrics'][metric_name]['val']))
                        )
                        metric_underfittng = self._evaluate_underfitting(
                            metric_name=metric_name, metric_type='metric',
                            train_log=self.log_history[f"{out}"]['metrics'][metric_name]['train'][-1],
                            val_log=self.log_history[f"{out}"]['metrics'][metric_name]['val'][-1]
                        )
                        metric_overfitting = self._evaluate_overfitting(
                            metric_name=metric_name, metric_type='metric',
                            mean_log=self.log_history[out]['progress_state']['metrics']
                            [metric_name]['mean_log_history']
                        )
                        if metric_underfittng or metric_overfitting:
                            normal_state = False
                        else:
                            normal_state = True
                        self.log_history[out]['progress_state']['metrics'][metric_name][
                            'underfitting'].append(metric_underfittng)
                        self.log_history[out]['progress_state']['metrics'][metric_name][
                            'overfitting'].append(metric_overfitting)
                        self.log_history[out]['progress_state']['metrics'][metric_name][
                            'normal_state'].append(normal_state)
                        if self.current_logs.get(out).get("class_metrics"):
                            for cls in classes_names:
                                self.log_history[out]['class_metrics'][cls][metric_name]["train"].append(
                                    round_loss_metric(
                                        self.current_logs[out]['class_metrics'][metric_name][cls]["train"]))
                                self.log_history[out]['class_metrics'][cls][metric_name]["val"].append(
                                    round_loss_metric(self.current_logs[out]['class_metrics'][metric_name][cls]["val"]))

            if self.dataset.data.architecture in YOLO_ARCHITECTURE:
                out = list(self.dataset.data.outputs.keys())[0]
                classes_names = self.dataset.data.outputs.get(out).classes_names
                for key in self.log_history['output']["loss"].keys():
                    for data_type in ['train', 'val']:
                        self.log_history['output']["loss"][key][data_type].append(
                            round_loss_metric(self.current_logs.get('loss').get(key).get(data_type)))
                for key in self.log_history['output']["metrics"].keys():
                    self.log_history['output']["metrics"][key]["val"].append(
                        round_loss_metric(self.current_logs.get('metrics').get(key).get('val'))
                    )
                for name in classes_names:
                    self.log_history['output']["class_loss"]['prob_loss'][name]["val"].append(
                        round_loss_metric(self.current_logs.get('class_loss').get("prob_loss").get(name).get("val"))
                    )
                    self.log_history['output']["class_loss"]['prob_loss'][name]["train"].append(
                        round_loss_metric(self.current_logs.get('class_loss').get("prob_loss").get(name).get("train"))
                    )
                    self.log_history['output']["class_metrics"]['mAP50'][name]["val"].append(
                        round_loss_metric(self.current_logs.get('class_metrics').get("mAP50").get(name).get("val"))
                    )
                for loss_name in self.log_history['output']["loss"].keys():
                    # fill loss progress state
                    self.log_history['output']['progress_state']['loss'][loss_name]['mean_log_history'].append(
                        round_loss_metric(
                            self._get_mean_log(self.log_history.get('output').get('loss').get(loss_name).get('val')))
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
                    self.log_history['output']['progress_state']['loss'][loss_name]['underfitting'].append(
                        loss_underfitting)
                    self.log_history['output']['progress_state']['loss'][loss_name]['overfitting'].append(
                        loss_overfitting)
                    self.log_history['output']['progress_state']['loss'][loss_name]['normal_state'].append(
                        normal_state)
                for metric_name in self.log_history.get('output').get('metrics').keys():
                    self.log_history['output']['progress_state']['metrics'][metric_name]['mean_log_history'].append(
                        round_loss_metric(self._get_mean_log(self.log_history['output']['metrics'][metric_name]["val"]))
                    )
                    metric_overfitting = self._evaluate_overfitting(
                        metric_name=metric_name,
                        mean_log=self.log_history['output']['progress_state']['metrics'][
                            metric_name]['mean_log_history'],
                        metric_type='metric'
                    )
                    # logger.debug(f"mean_log: {self.log_history['output']['progress_state']['metrics'][metric_name]['mean_log_history']}\n"
                    #              f"metric_overfitting: {metric_overfitting}")
                    if metric_overfitting:
                        normal_state = False
                    else:
                        normal_state = True
                    self.log_history['output']['progress_state']['metrics'][metric_name]['overfitting'].append(
                        metric_overfitting)
                    self.log_history['output']['progress_state']['metrics'][metric_name]['normal_state'].append(
                        normal_state)

            if self.dataset.data.architecture in GAN_ARCHITECTURE:
                for key in self.log_history['output']["loss"].keys():
                    self.log_history['output']["loss"][key]['train'].append(
                            round_loss_metric(self.current_logs.get('loss').get(key).get('train')))
                for loss_name in self.log_history['output']["loss"].keys():
                    self.log_history['output']['progress_state']['loss'][loss_name]['underfitting'].append(False)
                    self.log_history['output']['progress_state']['loss'][loss_name]['overfitting'].append(False)
                    self.log_history['output']['progress_state']['loss'][loss_name]['normal_state'].append(True)

        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                History.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def _get_mean_log(logs):
        method_name = '_get_mean_log'
        try:
            copy_logs = copy.deepcopy(logs)
            while None in copy_logs:
                copy_logs.pop(copy_logs.index(None))
            if len(copy_logs) < 5:
                return float(np.mean(copy_logs))
            else:
                return float(np.mean(copy_logs[-5:]))
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                History.name, method_name, str(error)).with_traceback(error.__traceback__)
            logger.error(exc)
            return 0.

    @staticmethod
    def _evaluate_overfitting(metric_name: str, mean_log: list, metric_type: str):
        method_name = '_evaluate_overfitting'
        try:
            mode = loss_metric_config.get(metric_type).get(metric_name).get("mode")
            overfitting = False
            if mean_log[-1] == 0:
                if mode == 'min' and min(mean_log) != 0:
                    overfitting = True
                if mode == 'max' and max(mean_log) != 0:
                    overfitting = True
            elif mean_log[-1] is None or min(mean_log) is None or max(mean_log) is None:
                if mode == 'min' and not min(mean_log):
                    overfitting = False
                if mode == 'max' and not max(mean_log) != 0:
                    overfitting = False
            elif mode == 'min':
                if mean_log[-1] > min(mean_log) and \
                        (mean_log[-1] - min(mean_log)) * 100 / min(mean_log) > 2:
                    overfitting = True
            elif mode == 'max':
                if mean_log[-1] < max(mean_log) and \
                        (max(mean_log) - mean_log[-1]) * 100 / max(mean_log) > 2:
                    overfitting = True
            return overfitting
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                History.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def _evaluate_underfitting(metric_name: str, train_log: float, val_log: float, metric_type: str):
        method_name = '_evaluate_underfitting'
        try:
            mode = loss_metric_config.get(metric_type).get(metric_name).get("mode")
            underfitting = False
            if not train_log or not val_log:
                underfitting = True
            elif mode == 'min':
                if val_log < 1 and train_log < 1 and (val_log - train_log) > 0.05:
                    underfitting = True
                if (val_log >= 1 or train_log >= 1) and (val_log - train_log) / train_log * 100 > 5:
                    underfitting = True
            else:
                if (train_log - val_log) / train_log * 100 > 3:
                    underfitting = True
            return underfitting
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                History.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc
