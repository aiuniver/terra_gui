import copy
import importlib
import os
import re
from tempfile import NamedTemporaryFile
from typing import Union

import tensorflow
from PIL import Image, ImageDraw, ImageFont  # Модули работы с изображениями

from tensorflow.keras.utils import to_categorical
from pydub import AudioSegment
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import moviepy.editor as moviepy_editor
from tensorflow.python.keras.api.keras.preprocessing import image

from terra_ai import progress
from terra_ai.data.datasets.dataset import DatasetOutputsData
from terra_ai.data.datasets.extra import LayerInputTypeChoice, LayerOutputTypeChoice, DatasetGroupChoice
from terra_ai.data.presets.training import Metric, Loss
from terra_ai.data.training.extra import TaskChoice
from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.utils import camelize, decamelize

__version__ = 0.056


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
        'CohenKappa': {
            "log_name": "cohen_kappa",
            "mode": "max",
            "module": "tensorflow_addons.metrics"
        },
        "CosineSimilarity": {
            "log_name": "cosine_similarity",
            "mode": "max",
            "module": "tensorflow.keras.metrics"
        },  # min if loss, max if metric
        "DiceCoefficient": {
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
        "DiceCoef": {
            "log_name": "dice_coef",
            "mode": "max"
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
        self.dataset_config = None
        # self.dataset = None
        self.x_val = None
        self.y_true = {}
        self.y_pred = {}
        self.current_epoch = None

        # overfitting params
        self.log_gap = 5
        self.progress_threashold = 5

        self.current_logs = {}
        self.log_history = {}
        self.progress_table = {}
        self.dataset_balance = None
        self.class_idx = None

        self.show_examples = 10
        self.ex_type_choice = 'seed'
        self.seed_idx = None
        self.example_idx = []
        self.intermediate_result = {}
        self.statistic_result = {}
        self.train_progress = {}
        self.progress_name = "training"

        self.urgent_predict = False

        self.train_states = {
            "status": "no_train",  # training, trained, stopped, retrain
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

        self.interactive_config = {
            # 'loss_graphs': [
            #     # {
            #     #     'id': 1,
            #     #     'output_idx': 2,
            #     #     'show': ['model'],
            #     # },
            #     # {
            #     #     'id': 2,
            #     #     'output_idx': 2,
            #     #     'show': ['classes'],
            #     # },
            # ],
            # 'metric_graphs': [
            #     # {
            #     #     'id': 1,
            #     #     'output_idx': 2,
            #     #     'show': ['model'],
            #     #     'show_metric': 'CategoricalAccuracy'
            #     # },
            #     # {
            #     #     'id': 2,
            #     #     'output_idx': 2,
            #     #     'show': ['classes'],
            #     #     'show_metric': 'CategoricalAccuracy'
            #     # }
            # ],
            # 'intermediate_result': {
            #     'show_results': False,
            #     'example_choice_type': 'seed',
            #     'main_output': 2,
            #     'num_examples': 10,
            #     'show_statistic': False,
            #     'autoupdate': False
            # },
            # 'progress_table': [
            #     {
            #         'output_idx': 2,
            #         'show_loss': True,
            #         'show_metrics': True,
            #     }
            # ],
            # 'statistic_data': {
            #     'output_id': [2],
            #     'autoupdate': False
            # },
            # 'data_balance': {
            #     'show_train': True,
            #     'show_val': True,
            #     'sorted': 'by_name'  # 'descending', 'ascending'
            # }
        }
        pass

    def set_attributes(self, dataset: PrepareDataset,
                       metrics: dict,
                       losses: dict,
                       dataset_path: str,
                       initial_config: dict):
        self.losses = losses
        self.metrics = self._reformat_metrics(metrics)
        self.loss_obj = self._prepare_loss_obj(losses)
        self.metrics_obj = self._prepare_metric_obj(metrics)
        self.interactive_config = initial_config
        # self.dataset = dataset
        self._prepare_dataset_config(dataset, dataset_path)
        self.x_val = dataset.X.get("val") if dataset.data.group == DatasetGroupChoice.keras else None
        self._prepare_y_true(dataset)
        # self._prepare_interactive_config()

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
            self.train_states["buttons"]["train"]["title"] = "Обучить"
            self.train_states["buttons"]["train"]["visible"] = True
            self.train_states["buttons"]["stop"]["visible"] = False
            self.train_states["buttons"]["clear"]["visible"] = False
            self.train_states["buttons"]["save"]["visible"] = False

    def get_states(self):
        return self.train_states

    def update_train_progress(self, data: dict):
        self.train_progress = data

    def update_state(self, y_pred, fit_logs=None, current_epoch_time=None, on_epoch_end_flag=False) -> dict:
        if on_epoch_end_flag:
            self.current_epoch = fit_logs.get('epoch')
            self.current_logs = self._reformat_fit_logs(fit_logs)
            self._reformat_y_pred(y_pred)
            self.example_idx = self._prepare_example_idx_to_show()
            self._update_log_history()
            self._update_progress_table(current_epoch_time)
            if self.interactive_config.get('intermediate_result').get('show_results') and \
                    self.interactive_config.get('intermediate_result').get('autoupdate'):
                self.intermediate_result = self._get_intermediate_result_request()
            if self.interactive_config.get('statistic_data').get('output_id') \
                    and self.interactive_config.get('statistic_data').get('autoupdate'):
                self.statistic_result = self._get_statistic_data_request()
        else:
            self._reformat_y_pred(y_pred)
            self.example_idx = self._prepare_example_idx_to_show()
            if self.interactive_config.get('intermediate_result').get('show_results'):
                self.intermediate_result = self._get_intermediate_result_request()
            if self.interactive_config.get('statistic_data').get('output_id'):
                self.statistic_result = self._get_statistic_data_request()
        self.urgent_predict = False
        return {
            'loss_graphs': self._get_loss_graph_data_request(),
            'metric_graphs': self._get_metric_graph_data_request(),
            'intermediate_result': self.intermediate_result,
            'progress_table': self.progress_table,
            'statistic_data': self.statistic_result,
            'data_balance': self._get_balance_data_request(),
        }

    def get_train_results(self, config: dict = None) -> Union[dict, None]:
        """Return dict with data for current interactive request"""
        self.interactive_config = config if config else self.interactive_config
        self.example_idx = self._prepare_example_idx_to_show()
        if config.get('intermediate_result').get('show_results') or config.get('statistic_data').get('output_id'):
            self.urgent_predict = True
            return

        self.train_progress['train_data'] = {
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

    # Методы для set_attributes()
    @staticmethod
    def _reformat_metrics(metrics: dict) -> dict:
        output = {}
        for out, out_metrics in metrics.items():
            output[f"{out}"] = []
            for metric in out_metrics:
                metric_name = metric.name
                if re.search(r'_\d+$', metric_name):
                    end = len(f"_{metric_name.split('_')[-1]}")
                    metric_name = metric_name[:-end]
                output[f"{out}"].append(camelize(metric_name))
        return output

    @staticmethod
    def _prepare_loss_obj(losses: dict) -> dict:
        loss_obj = {}
        for out in losses.keys():
            loss_obj[f"{out}"] = getattr(
                importlib.import_module(loss_metric_config.get("loss").get(losses.get(out)).get("module")),
                losses.get(out)
            )
        return loss_obj

    @staticmethod
    def _prepare_metric_obj(metrics: dict) -> dict:
        metrics_obj = {}
        for out in metrics.keys():
            metrics_obj[f"{out}"] = {}
            for metric in metrics.get(out):
                metric_name = metric.name
                if re.search(r'_\d+$', metric_name):
                    end = len(f"_{metric_name.split('_')[-1]}")
                    metric_name = metric_name[:-end]
                metrics_obj[f"{out}"][camelize(metric_name)] = metric
        return metrics_obj

    def _prepare_dataset_config(self, dataset: PrepareDataset, dataset_path: str):
        self.dataset_config = {
            "dataset_path": dataset_path,
            "group": dataset.data.group.name,
            "dataframe": dataset.dataframe,
            "use_generator": dataset.data.use_generator,
            "inputs": {},
            "outputs": {}
        }
        for inp in dataset.data.inputs.keys():
            self.dataset_config["inputs"][f"{inp}"] = {
                'encoding': dataset.data.inputs.get(inp).encoding.name,
                'task': dataset.data.inputs.get(inp).task.name,
                "input_shape": dataset.data.inputs.get(inp).shape
            }
        for out in dataset.data.outputs.keys():
            self.dataset_config["outputs"][f"{out}"] = {
                'classes_colors': self._get_classes_colors(dataset.data.outputs.get(out)),
                'classes_names': dataset.data.outputs.get(out).classes_names,
                'encoding': dataset.data.outputs.get(out).encoding.name,
                'num_classes': dataset.data.outputs.get(out).num_classes,
                'task': dataset.data.outputs.get(out).task.name
            }

    @staticmethod
    def _get_classes_colors(dataset_output: DatasetOutputsData):
        colors = {}
        if dataset_output.task == LayerOutputTypeChoice.TextSegmentation and \
                dataset_output.classes_colors:
            for i, tag in enumerate(dataset_output.classes_names):
                colors[tag] = dataset_output.classes_colors[i].as_rgb_tuple()
            return colors
        elif dataset_output.task == LayerOutputTypeChoice.TextSegmentation and \
                not dataset_output.classes_colors:
            for tag in dataset_output.classes_names:
                colors[tag] = tuple(np.random.randint(256, size=3))
            return colors
        elif dataset_output.task == LayerOutputTypeChoice.Segmentation:
            return [classes_color.as_rgb_tuple() for classes_color in dataset_output.classes_colors]
        else:
            return dataset_output.classes_colors

    def _prepare_y_true(self, dataset: PrepareDataset):
        self.y_true = {
            "train": {},
            "val": {}
        }
        for data_type in self.y_true.keys():
            for out in self.dataset_config.get("outputs").keys():
                if (
                        self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Classification
                        and self.dataset_config.get("outputs").get(out).get("use_generator")
                        and self.dataset_config.get("dataframe")
                ):
                    self.y_true.get(data_type)[out] = []
                    for column_name in self.dataset_config.get("dataframe").get(data_type).columns:
                        if column_name.split('_')[0] == out:
                            for lbl in list(self.dataset_config.get("dataframe").get(data_type)[column_name]):
                                self.y_true[data_type][out].append(
                                    to_categorical(
                                        self.dataset_config.get("outputs").get(out).get("classes_names").index(lbl),
                                        num_classes=self.dataset_config.get("outputs").get(out).get("num_classes")
                                    )
                                    if self.dataset_config.get("outputs").get(out).get("encoding") == 'ohe'
                                    else self.dataset_config.get("outputs").get(out).get("classes_names").index(lbl))
                            self.y_true[data_type][f'{out}'] = np.array(self.y_true[data_type][f'{out}'])
                            break
                elif (
                        self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Classification
                        and not self.dataset_config.get("outputs").get(out).get("use_generator")
                ):
                    self.y_true[data_type][out] = dataset.Y.get(data_type).get(f"{out}")
                elif (
                        self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Segmentation
                        and self.dataset_config.get("outputs").get(out).get("use_generator")
                ):
                    # TODO: загрузка из генераторов занимает уйму времени, нужны другие варианты
                    self.y_true[data_type][out] = []
                    for _, y_val in dataset.dataset[data_type].batch(1):
                        self.y_true[data_type][out].extend(y_val.get(f'{out}').numpy())
                    self.y_true[data_type][out] = np.array(self.y_true[data_type][out])
                elif (
                        self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Segmentation or
                        self.dataset_config.get("outputs").get(out).get(
                            "task") == LayerOutputTypeChoice.TextSegmentation
                        and not self.dataset_config.get("outputs").get(out).get("use_generator")
                ):
                    self.y_true[data_type][out] = dataset.Y.get(data_type).get(f"{out}")
                else:
                    pass

    def _prepare_null_log_history_template(self):
        """
        self.log_history_example = {
            'epochs': [],
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
                    'class_name': {
                        'loss_name': []
                    },
                },
                'class_metrics': {
                    'class_name': {
                        'metric_name': [],
                    },
                }
            }
        }
        """
        self.log_history["epochs"] = []
        for out in self.dataset_config.get("outputs").keys():
            # out: str
            self.log_history[out] = {
                "loss": {},
                "metrics": {},
                "progress_state": {
                    "loss": {},
                    "metrics": {}
                }
            }
            # if self.losses.get(f'{out}') and isinstance(self.losses.get(f'{out}'), str):
            #     self.losses[f'{out}'] = self.losses.get(f'{out}')
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
            if self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Classification or \
                    self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Segmentation or \
                    self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.TextSegmentation:
                self.log_history[out]["class_loss"] = {}
                self.log_history[out]["class_metrics"] = {}
                for class_name in self.dataset_config.get("outputs").get(out).get("classes_names"):
                    self.log_history[out]["class_metrics"][f"{class_name}"] = {}
                    self.log_history[out]["class_loss"][f"{class_name}"] = {}
                    self.log_history[out]["class_loss"][f"{class_name}"][self.losses.get(out)] = []
                    for metric in self.metrics.get(out):
                        self.log_history[out]["class_metrics"][f"{class_name}"][f"{metric}"] = []

    def _prepare_dataset_balance(self) -> dict:
        """
        return = {
            "output_name": {
                'data_type': {
                    'class_name': int,
                },
            }
        }
        """
        dataset_balance = {}
        for out in self.dataset_config.get("outputs").keys():
            dataset_balance[out] = {}
            if self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Classification:
                for data_type in self.y_true.keys():
                    dataset_balance[out][data_type] = class_counter(
                        self.y_true.get(data_type).get(out),
                        self.dataset_config.get("outputs").get(out).get("classes_names"),
                        self.dataset_config.get("outputs").get(out).get("encoding") == 'ohe'
                    )
            if (
                    self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Segmentation and
                    self.dataset_config.get("outputs").get(out).get("encoding") == 'ohe'
            ):
                for data_type in self.y_true.keys():
                    dataset_balance[out][data_type] = {
                        "presence_balance": {},
                        "percent_balance": {}
                    }
                    classes = np.arange(self.dataset_config.get("outputs").get(out).get("num_classes"))
                    class_percent = {}
                    class_count = {}
                    for cl in classes:
                        class_percent[self.dataset_config.get("outputs").get(out).get("classes_names")[cl]] = np.round(
                            np.sum(self.y_true.get(data_type).get(out)[:, :, :, cl]) * 100
                            / np.prod(self.y_true.get(data_type).get(out)[:, :, :, 0].shape))
                        class_count[self.dataset_config.get("outputs").get(out).get("classes_names")[cl]] = 0

                    for img_array in np.argmax(self.y_true.get(data_type).get(out), axis=-1):
                        for cl in classes:
                            if cl in img_array:
                                class_count[
                                    self.dataset_config.get("outputs").get(out).get("classes_names")[cl]
                                ] += 1
                    dataset_balance[out][data_type]["presence_balance"] = class_count
                    dataset_balance[out][data_type]["square_balance"] = class_percent
            if (
                    self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.TextSegmentation
                    and self.dataset_config.get("outputs").get(out).get("encoding") == 'ohe'
                    or self.dataset_config.get("outputs").get(out).get("encoding") == 'multi'
            ):
                for data_type in self.y_true.keys():
                    dataset_balance[out][data_type] = {
                        "presence_balance": {},
                        "percent_balance": {}
                    }
                    classes = np.arange(self.dataset_config.get("outputs").get(out).get("num_classes"))
                    class_count = {}
                    class_percent = {}
                    for cl in classes:
                        class_count[self.dataset_config.get("outputs").get(out).get("classes_names")[cl]] = \
                            np.sum(self.y_true.get(data_type).get(out)[:, :, cl])
                        class_percent[self.dataset_config.get("outputs").get(out).get("classes_names")[cl]] = np.round(
                            np.sum(self.y_true.get(data_type).get(out)[:, :, cl]) * 100
                            / np.prod(self.y_true.get(data_type).get(out)[:, :, cl].shape))
                    dataset_balance[out][data_type]["presence_balance"] = class_count
                    dataset_balance[out][data_type]["percent_balance"] = class_percent
            pass
        return dataset_balance

    def _prepare_class_idx(self) -> dict:
        """
        class_idx_dict -> train -> output_idx -> class_name
        """
        class_idx = {}
        for data_type in self.y_true.keys():
            class_idx[data_type] = {}
            for out in self.y_true.get(data_type).keys():
                class_idx[data_type][out] = {}
                if self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Classification:
                    for name in self.dataset_config.get("outputs").get(out).get("classes_names"):
                        class_idx[data_type][out][name] = []
                    y_true = np.argmax(self.y_true.get(data_type).get(out), axis=-1) \
                        if self.dataset_config.get("outputs").get(out).get("encoding") == 'ohe' \
                        else self.y_true.get(data_type).get(out)
                    for idx in range(len(y_true)):
                        class_idx[data_type][out][
                            self.dataset_config.get("outputs").get(out).get("classes_names")[y_true[idx]]].append(idx)
        return class_idx

    def _prepare_seed(self):
        if self.dataset_config.get('group') == 'keras':
            data_lenth = np.arange(len(self.y_true.get("val").get(list(self.y_true.get("val").keys())[0])))
        else:
            data_lenth = np.arange(len(self.dataset_config.get("dataframe").get("val")))
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

            if len(self.losses.keys()) == 1:
                interactive_log[out]['loss'] = {
                    self.losses.get(out): {
                        'train': update_logs.get('loss'),
                        'val': update_logs.get('val_loss')
                    }
                }
            else:
                interactive_log[out]['loss'] = {}
                interactive_log[out]['loss'][self.losses.get(out)] = {
                    'train': update_logs.get(f'{out}_loss'),
                    'val': update_logs.get(f'val_{out}_loss')
                }

            interactive_log[out]['metrics'] = {}
            if len(self.metrics.keys()) == 1:
                for metric_name in self.metrics.get(out):
                    interactive_log[out]['metrics'][metric_name] = {}
                    interactive_log[out]['metrics'][metric_name] = {
                        'train': update_logs.get(loss_metric_config.get('metric').get(metric_name).get('log_name')),
                        'val': update_logs.get(
                            f"val_{loss_metric_config.get('metric').get(metric_name).get('log_name')}")
                    }
            else:
                for metric_name in self.metrics.get(out):
                    interactive_log[out]['metrics'][metric_name] = {}
                    interactive_log[out]['metrics'][metric_name] = {
                        'train': update_logs.get(
                            f"{out}_{loss_metric_config.get('metric').get(metric_name).get('log_name')}"),
                        'val': update_logs.get(
                            f"val_{out}_{loss_metric_config.get('metric').get(metric_name).get('log_name')}")
                    }
        return interactive_log

    def _reformat_y_pred(self, y_pred):
        """
        y_pred: {
            'output_id': predict_array
        }
        """
        self.y_pred = {}
        for idx, out in enumerate(self.y_true.get('val').keys()):
            if len(self.y_true.get('val').keys()) == 1:
                self.y_pred[out] = y_pred
            else:
                self.y_pred[out] = y_pred[idx]

    def _prepare_example_idx_to_show(self) -> dict:
        """
        example_idx = {
            output_id: []
        }
        """
        example_idx = {}
        out = f"{self.interactive_config.get('intermediate_result').get('main_output')}"
        ohe = self.dataset_config.get("outputs").get(out).get("encoding") == 'ohe'
        count = self.interactive_config.get('intermediate_result').get('num_examples')
        choice_type = self.interactive_config.get("intermediate_result").get("example_choice_type")
        if choice_type == "best" or choice_type == "worst":
            if self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Classification:
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
                if choice_type == "best":
                    example_idx = sorted_args[::-1][:count]
                if choice_type == "worst":
                    example_idx = sorted_args[:count]

            elif self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Segmentation or \
                    self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.TextSegmentation:
                y_true = self.y_true.get("val").get(out)
                y_pred = to_categorical(
                    np.argmax(self.y_pred.get(out), axis=-1),
                    num_classes=self.dataset_config.get("outputs").get(out).get("num_classes")
                )
                dice_val = self._dice_coef(y_true, y_pred, batch_mode=True)
                dice_dict = dict(zip(np.arange(0, len(dice_val)), dice_val))
                if choice_type == "best":
                    example_idx, _ = sort_dict(dice_dict, mode="descending")
                    example_idx = example_idx[:count]
                if choice_type == "worst":
                    example_idx, _ = sort_dict(dice_dict, mode="ascending")
                    example_idx = example_idx[:count]
            else:
                pass
        elif choice_type == "seed":
            example_idx = self.seed_idx[:self.interactive_config.get("intermediate_result").get("num_examples")]
        elif choice_type == "random":
            example_idx = np.random.randint(
                0,
                len(self.y_true.get("val").get(list(self.y_true.get('val').keys())[0])),
                self.interactive_config.get("intermediate_result").get("num_examples")
            )
        else:
            pass
        return example_idx

    def _update_log_history(self):
        data_idx = None
        if self.current_epoch in self.log_history['epochs']:
            data_idx = self.log_history['epochs'].index(self.current_epoch)
        else:
            self.log_history['epochs'].append(self.current_epoch)
        for out in self.dataset_config.get("outputs").keys():
            for loss_name in self.log_history.get(out).get('loss').keys():
                for data_type in ['train', 'val']:
                    # fill losses
                    if data_idx or data_idx == 0:
                        self.log_history[out]['loss'][loss_name][data_type][data_idx] = \
                            self.current_logs.get(out).get('loss').get(loss_name).get(data_type) \
                                if self.current_logs.get(out).get('loss').get(loss_name).get(data_type) else 0.
                    else:
                        self.log_history[out]['loss'][loss_name][data_type].append(
                            self.current_logs.get(out).get('loss').get(loss_name).get(data_type)
                            if self.current_logs.get(out).get('loss').get(loss_name).get(data_type) else 0.
                        )
                # fill loss progress state
                if data_idx or data_idx == 0:
                    self.log_history[out]['progress_state']['loss'][loss_name]['mean_log_history'][data_idx] = \
                        self._get_mean_log(self.log_history.get(out).get('loss').get(loss_name).get('val'))
                else:
                    self.log_history[out]['progress_state']['loss'][loss_name]['mean_log_history'].append(
                        self._get_mean_log(self.log_history.get(out).get('loss').get(loss_name).get('val'))
                    )
                # get progress state data
                loss_underfitting = self._evaluate_underfitting(
                    loss_name,
                    self.log_history[out]['loss'][loss_name]['train'][-1],
                    self.log_history[out]['loss'][loss_name]['val'][-1],
                    metric_type='loss'
                )
                loss_overfitting = self._evaluate_overfitting(
                    loss_name,
                    self.log_history[out]['progress_state']['loss'][loss_name]['mean_log_history'],
                    metric_type='loss'
                )
                if loss_underfitting or loss_overfitting:
                    normal_state = False
                else:
                    normal_state = True

                if data_idx or data_idx == 0:
                    self.log_history[out]['progress_state']['loss'][loss_name]['underfitting'][data_idx] = \
                        loss_underfitting
                    self.log_history[out]['progress_state']['loss'][loss_name]['overfitting'][data_idx] = \
                        loss_overfitting
                    self.log_history[out]['progress_state']['loss'][loss_name]['normal_state'][data_idx] = \
                        normal_state
                else:
                    self.log_history[out]['progress_state']['loss'][loss_name]['underfitting'].append(
                        loss_underfitting)
                    self.log_history[out]['progress_state']['loss'][loss_name]['overfitting'].append(
                        loss_overfitting)
                    self.log_history[out]['progress_state']['loss'][loss_name]['normal_state'].append(
                        normal_state)

                for cls in self.log_history.get(out).get('class_loss').keys():
                    class_loss = 0.

                    # get Classification loss
                    if self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Classification:
                        class_loss = self._get_loss_calculation(
                            loss_name=self.losses.get(out),
                            loss_obj=self.loss_obj.get(out),
                            out=out,
                            y_true=self.y_true.get('val').get(out)[
                                self.class_idx.get('val').get(out).get(cls)],
                            y_pred=self.y_pred.get(out)[self.class_idx.get('val').get(out).get(cls)],
                        )
                    # get Segmentation loss
                    if self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Segmentation:
                        class_loss = self._get_loss_calculation(
                            loss_name=self.losses.get(out),
                            loss_obj=self.loss_obj.get(out),
                            out=out,
                            y_true=self.y_true.get('val').get(out)[
                                   :, :, :,
                                   self.dataset_config.get("outputs").get(out).get("classes_names").index(cls)],
                            y_pred=self.y_pred.get(out)[
                                   :, :, :,
                                   self.dataset_config.get("outputs").get(out).get("classes_names").index(cls)],
                        )
                    # get TextSegmentation loss
                    if self.dataset_config.get("outputs").get(out).get(
                            "task") == LayerOutputTypeChoice.TextSegmentation:
                        class_loss = self._get_loss_calculation(
                            loss_name=self.losses.get(out),
                            loss_obj=self.loss_obj.get(out),
                            out=out,
                            y_true=self.y_true.get('val').get(out)[
                                   :, :, self.dataset_config.get("outputs").get(out).get("classes_names").index(cls)],
                            y_pred=self.y_pred.get(out)[
                                   :, :, self.dataset_config.get("outputs").get(out).get("classes_names").index(cls)],
                        )

                    if data_idx or data_idx == 0:
                        self.log_history[out]['class_loss'][cls][loss_name][data_idx] = \
                            class_loss if class_loss else 0.
                    else:
                        self.log_history[out]['class_loss'][cls][loss_name].append(
                            class_loss if class_loss else 0.
                        )

            for metric_name in self.log_history.get(out).get('metrics').keys():
                for data_type in ['train', 'val']:
                    # fill metrics
                    if data_idx or data_idx == 0:
                        # print(data_idx,  self.log_history[f'{out}']['metrics'][metric_name])
                        self.log_history[out]['metrics'][metric_name][data_type][data_idx] = \
                            self.current_logs.get(out).get('metrics').get(metric_name).get(data_type) \
                                if self.current_logs.get(out).get('metrics').get(metric_name).get(
                                data_type) else 0.
                    else:
                        self.log_history[out]['metrics'][metric_name][data_type].append(
                            self.current_logs.get(out).get('metrics').get(metric_name).get(data_type)
                            if self.current_logs.get(out).get('metrics').get(metric_name).get(data_type) else 0.
                        )

                # fill metric progress state
                if data_idx or data_idx == 0:
                    self.log_history[out]['progress_state']['metrics'][metric_name]['mean_log_history'][data_idx] = \
                        self._get_mean_log(self.log_history[out]['metrics'][metric_name]['val'])
                else:
                    self.log_history[out]['progress_state']['metrics'][metric_name]['mean_log_history'].append(
                        self._get_mean_log(self.log_history[out]['metrics'][metric_name]['val'])
                    )
                loss_underfittng = self._evaluate_underfitting(
                    metric_name,
                    self.log_history[out]['metrics'][metric_name]['train'][-1],
                    self.log_history[out]['metrics'][metric_name]['val'][-1],
                    metric_type='metric'
                )
                loss_overfittng = self._evaluate_overfitting(
                    metric_name,
                    self.log_history[out]['progress_state']['metrics'][metric_name]['mean_log_history'],
                    metric_type='metric'
                )
                if loss_underfittng or loss_overfittng:
                    normal_state = False
                else:
                    normal_state = True

                if data_idx or data_idx == 0:
                    self.log_history[out]['progress_state']['metrics'][metric_name]['underfitting'][data_idx] = \
                        loss_underfittng
                    self.log_history[out]['progress_state']['metrics'][metric_name]['overfitting'][data_idx] = \
                        loss_overfittng
                    self.log_history[out]['progress_state']['metrics'][metric_name]['normal_state'][data_idx] = \
                        normal_state
                else:
                    self.log_history[out]['progress_state']['metrics'][metric_name]['underfitting'].append(
                        loss_underfittng)
                    self.log_history[out]['progress_state']['metrics'][metric_name]['overfitting'].append(
                        loss_overfittng)
                    self.log_history[out]['progress_state']['metrics'][metric_name]['normal_state'].append(
                        normal_state)

                # fill class losses
                for cls in self.log_history.get(out).get('class_metrics').keys():
                    class_metric = 0.
                    if self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Classification:
                        class_metric = self._get_metric_calculation(
                            metric_name=metric_name,
                            metric_obj=self.metrics_obj.get(out).get(metric_name),
                            out=out,
                            y_true=self.y_true.get('val').get(out)[
                                self.class_idx.get('val').get(out).get(cls)],
                            y_pred=self.y_pred.get(out)[self.class_idx.get('val').get(out).get(cls)],
                        )
                    if self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Segmentation:
                        class_metric = self._get_metric_calculation(
                            metric_name=metric_name,
                            metric_obj=self.metrics_obj.get(out).get(metric_name),
                            out=out,
                            y_true=self.y_true.get('val').get(out)[
                                   :, :, :,
                                   self.dataset_config.get("outputs").get(out).get("classes_names").index(cls)
                                   ],
                            y_pred=self.y_pred.get(out)[
                                   :, :, :,
                                   self.dataset_config.get("outputs").get(out).get("classes_names").index(cls)
                                   ],
                        )
                    if self.dataset_config.get("outputs").get(out).get(
                            "task") == LayerOutputTypeChoice.TextSegmentation:
                        class_metric = self._get_metric_calculation(
                            metric_name=metric_name,
                            metric_obj=self.metrics_obj.get(out).get(metric_name),
                            out=out,
                            y_true=self.y_true.get('val').get(out)[
                                   :, :, self.dataset_config.get("outputs").get(out).get("classes_names").index(cls)
                                   ],
                            y_pred=self.y_pred.get(out)[
                                   :, :, self.dataset_config.get("outputs").get(out).get("classes_names").index(cls)
                                   ],
                        )
                    if data_idx or data_idx == 0:
                        self.log_history[out]['class_metrics'][cls][metric_name][data_idx] = \
                            class_metric if class_metric else 0.
                    else:
                        self.log_history[out]['class_metrics'][cls][metric_name].append(
                            class_metric if class_metric else 0.
                        )

    def _update_progress_table(self, epoch_time: float):
        """
        "epoch": {
            "time": 7.233344078063965,
            "data": {
                  "Output_2": {
                        "loss": {
                              "loss": 0.9713165163993835,
                              "val_loss": 0.942412257194519
                        },
                        "metrics": {
                              "CategoricalAccuracy": 0.5084558725357056,
                              "val_CategoricalAccuracy": 0.568359375,
                              "AUC": 0.7065025568008423,
                              "val_AUC": 0.7676296234130859
                        }
                  }
            }
        },
        """
        self.progress_table[self.current_epoch] = {
            "time": epoch_time,
            "data": {}
        }
        for out in self.dataset_config.get("outputs").keys():
            self.progress_table[self.current_epoch]["data"][f"Выходной слой {out}"] = {
                'loss': {},
                'metrics': {}
            }
            self.progress_table[self.current_epoch]["data"][f"Выходной слой {out}"]["loss"] = {
                'loss': self.log_history.get(out).get('loss').get(self.losses.get(out)).get('train')[-1],
                'val_loss': self.log_history.get(out).get('loss').get(self.losses.get(out)).get('val')[-1]
            }
            for metric in self.metrics.get(out):
                self.progress_table[self.current_epoch]["data"][f"Выходной слой {out}"]["metrics"][metric] = \
                    self.log_history.get(out).get('metrics').get(metric).get('train')[-1]
                self.progress_table[self.current_epoch]["data"][f"Выходной слой {out}"]["metrics"][f"val_{metric}"] = \
                    self.log_history.get(out).get('metrics').get(metric).get('val')[-1]

    def _get_loss_calculation(self, loss_name, loss_obj, out: str, y_true, y_pred):
        if self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Classification:
            if loss_name == Loss.SparseCategoricalCrossentropy:
                return float(loss_obj()(
                    np.argmax(y_true, axis=-1) if self.dataset_config.get("outputs").get(out).get("encoding") == "ohe"
                    else y_true, y_pred
                ).numpy())
            else:
                return float(loss_obj()(
                    y_true if self.dataset_config.get("outputs").get(out).get("encoding") == "ohe"
                    else to_categorical(y_true, self.dataset_config.get("outputs").get(out).get("num_classes")), y_pred
                ).numpy())

        elif self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Segmentation or \
                (self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.TextSegmentation and
                 self.dataset_config.get("outputs").get(out).get("encoding") == "ohe"):
            if loss_name == Loss.SparseCategoricalCrossentropy:
                return float(loss_obj()(
                    np.expand_dims(np.argmax(y_true, axis=-1), axis=-1)
                    if self.dataset_config.get("outputs").get(out).get("encoding") == "ohe" else y_true, y_pred
                ).numpy())
            else:
                return float(loss_obj()(
                    y_true if self.dataset_config.get("outputs").get(out).get("encoding") == "ohe"
                    else to_categorical(y_true, self.dataset_config.get("outputs").get(out).get("num_classes")), y_pred
                ).numpy())
        elif self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.TextSegmentation and \
                self.dataset_config.get("outputs").get(out).get("encoding") == "multi":
            if loss_name == Loss.SparseCategoricalCrossentropy:
                return 0.
            else:
                return float(loss_obj()(y_true, y_pred).numpy())
        else:
            return 0.

    def _get_metric_calculation(self, metric_name, metric_obj, out: str, y_true, y_pred):
        if self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Classification:
            if metric_name == Metric.Accuracy:
                metric_obj.update_state(
                    np.argmax(y_true, axis=-1) if self.dataset_config.get("outputs").get(out).get("encoding") == "ohe"
                    else y_true, np.argmax(y_pred, axis=-1)
                )
            elif metric_name == Metric.SparseCategoricalAccuracy or \
                    metric_name == Metric.SparseTopKCategoricalAccuracy or \
                    metric_name == Metric.SparseCategoricalCrossentropy:
                metric_obj.update_state(
                    np.argmax(y_true, axis=-1) if self.dataset_config.get("outputs").get(out).get("encoding") == "ohe"
                    else y_true, y_pred
                )
            else:
                metric_obj.update_state(
                    y_true if self.dataset_config.get("outputs").get(out).get("encoding") == "ohe"
                    else to_categorical(y_true, self.dataset_config.get("outputs").get(out).get("num_classes")), y_pred
                )
            return float(metric_obj.result().numpy())

        elif self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Segmentation or \
                (self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.TextSegmentation and
                 self.dataset_config.get("outputs").get(out).get("encoding") == "ohe"):
            if metric_name == Metric.SparseCategoricalAccuracy or \
                    metric_name == Metric.SparseTopKCategoricalAccuracy or \
                    metric_name == Metric.SparseCategoricalCrossentropy:
                metric_obj.update_state(
                    np.expand_dims(np.argmax(y_true, axis=-1), axis=-1)
                    if self.dataset_config.get("outputs").get(out).get("encoding") == "ohe" else y_true, y_pred
                )
            else:
                metric_obj.update_state(
                    y_true if self.dataset_config.get("outputs").get(out).get("encoding") == "ohe"
                    else to_categorical(y_true, self.dataset_config.get("outputs").get(out).get("num_classes")), y_pred
                )
            return float(metric_obj.result().numpy())
        elif self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.TextSegmentation and \
                self.dataset_config.get("outputs").get(out).get("encoding") == "multi":
            if metric_name == Metric.SparseCategoricalAccuracy or \
                    metric_name == Metric.SparseTopKCategoricalAccuracy or \
                    metric_name == Metric.SparseCategoricalCrossentropy:
                return 0.
            else:
                metric_obj.update_state(y_true, y_pred)
                return float(metric_obj.result().numpy())
        else:
            return 0.

    def _get_mean_log(self, logs):
        if len(logs) < self.log_gap:
            return float(np.mean(logs))
        else:
            return float(np.mean(logs[-self.log_gap:]))

    @staticmethod
    def _evaluate_overfitting(metric_name: str, mean_log: list, metric_type: str):
        if min(mean_log) or max(mean_log):
            if loss_metric_config.get(metric_type).get(metric_name).get("mode") == 'min' and \
                    mean_log[-1] > min(mean_log) and \
                    (mean_log[-1] - min(mean_log)) * 100 / min(mean_log) > 10:
                return True
            elif loss_metric_config.get(metric_type).get(metric_name).get("mode") == 'max' and \
                    mean_log[-1] < max(mean_log) and \
                    (max(mean_log) - mean_log[-1]) * 100 / max(mean_log) > 10:
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def _evaluate_underfitting(metric_name: str, train_log: list, val_log: list, metric_type: str):
        if train_log:
            if loss_metric_config.get(metric_type).get(metric_name).get("mode") == 'min' and \
                    (val_log - train_log) / train_log * 100 > 10:
                return True
            elif loss_metric_config.get(metric_type).get(metric_name).get("mode") == 'max' and \
                    (train_log - val_log) / train_log * 100 > 10:
                return True
            else:
                return False
        else:
            return False

    # Методы для конечных данных для вывода
    def _get_loss_graph_data_request(self) -> list:
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
                        'x': []:
                        'y': []
                    },
                    {
                        'label': 'Проверочная выборка',
                        'x': []:
                        'y': []
                    },
                ],
                "progress_state": "normal",
            },

            # Пример для классов
            {
                'id': 2,
                'graph_name': f'Output_{output_idx} - График ошибки обучения по классам - Эпоха №{epoch_num}',
                'x_label': 'Эпоха',
                'y_label': 'Значение',
                'plot_data': [
                    {
                        'class_label': f'Класс {class_name}',
                        'x': [],
                        'y': []
                    },
                ],
            }
        ]
        """
        data_return = []
        if not self.interactive_config.get('loss_graphs'):
            return data_return

        for loss_graph_config in self.interactive_config.get('loss_graphs'):
            if loss_graph_config.get('show') == 'model':
                if sum(self.log_history.get(f"{loss_graph_config.get('output_idx')}").get("progress_state").get(
                        "loss").get(self.losses.get(f"{loss_graph_config.get('output_idx')}")).get(
                    'overfitting')[-self.log_gap:]) >= self.progress_threashold:
                    progress_state = "overfitting"
                elif sum(self.log_history.get(f"{loss_graph_config.get('output_idx')}").get("progress_state").get(
                        "loss").get(self.losses.get(f"{loss_graph_config.get('output_idx')}")).get(
                    'underfitting')[-self.log_gap:]) >= self.progress_threashold:
                    progress_state = "underfitting"
                else:
                    progress_state = "normal"
                data_return.append(
                    {
                        "id": loss_graph_config.get('id'),
                        "graph_name": f"Output_{loss_graph_config.get('output_idx')} - "
                                      f"График ошибки обучения - Эпоха №{self.log_history.get('epochs')[-1]}",
                        "x_label": "Эпоха",
                        "y_label": "Значение",
                        "plot_data": [
                            {
                                "label": "Тренировочная выборка",
                                "x": self.log_history.get("epochs"),
                                "y": self.log_history.get(
                                    f"{loss_graph_config.get('output_idx')}").get('loss').get(
                                    self.losses.get(f"{loss_graph_config.get('output_idx')}")).get('train')
                            },
                            {
                                "label": "Проверочная выборка",
                                "x": self.log_history.get("epochs"),
                                "y": self.log_history.get(
                                    f"{loss_graph_config.get('output_idx')}").get('loss').get(
                                    self.losses.get(f"{loss_graph_config.get('output_idx')}")).get("val")
                            }
                        ],
                        "progress_state": progress_state
                    }
                )
            elif loss_graph_config.get('show') == 'classes':
                data_return.append(
                    {
                        "id": loss_graph_config.get('id'),
                        "graph_name": f"Output_{loss_graph_config.get('output_idx')} - "
                                      f"График ошибки обучения по классам - "
                                      f"Эпоха №{self.log_history.get('epochs')[-1]}",
                        "x_label": "Эпоха",
                        "y_label": "Значение",
                        "plot_data": [
                            {
                                'class_label': f'Класс {class_name}',
                                'x': self.log_history.get("epochs"),
                                'y': self.log_history.get(
                                    f"{loss_graph_config.get('output_idx')}").get('class_loss').get(class_name).get(
                                    self.losses.get(f"{loss_graph_config.get('output_idx')}"))
                            } for class_name in self.dataset_config.get("outputs").get(
                                f"{loss_graph_config.get('output_idx')}").get("classes_names")
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
        return data_return

    def _get_metric_graph_data_request(self) -> list:
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
                        'x': []:
                        'y': []
                    },
                    {
                        'label': 'Проверочная выборка',
                        'x': []:
                        'y': []
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
                        'x': [],
                        'y': []
                    },
                ],
            }
        ]
        """
        data_return = []
        if not self.interactive_config.get('metric_graphs'):
            return data_return

        for metric_graph_config in self.interactive_config.get('metric_graphs'):
            if metric_graph_config.get('show') == 'model':
                if sum(self.log_history.get(f"{metric_graph_config.get('output_idx')}").get("progress_state").get(
                        "metrics").get(metric_graph_config.get('show_metric')).get(
                    'overfitting')[-self.log_gap:]) >= self.progress_threashold:
                    progress_state = 'overfitting'
                elif sum(self.log_history.get(f"{metric_graph_config.get('output_idx')}").get("progress_state").get(
                        "metrics").get(metric_graph_config.get('show_metric')).get(
                    'underfitting')[-self.log_gap:]) >= self.progress_threashold:
                    progress_state = 'underfitting'
                else:
                    progress_state = 'normal'
                data_return.append(
                    {
                        "id": metric_graph_config.get('id'),
                        "graph_name": f"Output_{metric_graph_config.get('output_idx')} - "
                                      f"График метрики {metric_graph_config.get('show_metric')} - "
                                      f"Эпоха №{self.log_history.get('epochs')[-1]}",
                        "x_label": "Эпоха",
                        "y_label": "Значение",
                        "plot_data": [
                            {
                                "label": "Тренировочная выборка",
                                "x": self.log_history.get("epochs"),
                                "y": self.log_history.get(
                                    f"{metric_graph_config.get('output_idx')}").get('metrics').get(
                                    metric_graph_config.get('show_metric')).get("train")
                            },
                            {
                                "label": "Проверочная выборка",
                                "x": self.log_history.get("epochs"),
                                "y": self.log_history.get(
                                    f"{metric_graph_config.get('output_idx')}").get('metrics').get(
                                    metric_graph_config.get('show_metric')).get("val")
                            }
                        ],
                        "progress_state": progress_state
                    }
                )
            elif metric_graph_config.get('show') == 'classes':
                data_return.append(
                    {
                        "id": metric_graph_config.get('id'),
                        "graph_name": f"Output_{metric_graph_config.get('output_idx')} - "
                                      f"График метрики {metric_graph_config.get('show_metric')} по классам - "
                                      f"Эпоха №{self.log_history.get('epochs')[-1]}",
                        "x_label": "Эпоха",
                        "y_label": "Значение",
                        "plot_data": [
                            {
                                'class_label': f'Класс {class_name}',
                                'x': self.log_history.get("epochs"),
                                'y': self.log_history.get(
                                    f"{metric_graph_config.get('output_idx')}").get('class_metrics').get(
                                    class_name).get(metric_graph_config.get('show_metric'))
                            } for class_name in self.dataset_config.get("outputs").get(
                                f"{metric_graph_config.get('output_idx')}").get("classes_names")
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
        return data_return

    def _get_intermediate_result_request(self) -> dict:
        """
        'intermediate_result': {
            1: {
                'initial_value': {
                    'Вход 1': {
                        {
                            'type': 'image',
                            'data': '/content/file.webm'
                        },
                        # {
                        #     'type': 'video',
                        #     'data': '/content/file.webm'
                        # },
                        # {
                        #     'type': 'text',
                        #     'data': text,
                        #     }
                        # },
                        # {
                        #     'type': 'audio',
                        #     'data': '/content/file.webp'
                        # }
                    }
                },
                'true_value': {
                    'Выход 2':
                        {
                            'type': 'str',
                            'data': text,
                            'color_mark': None
                            'tags_color': {
                                s1: #fffff,
                                s2: #tyuhg
                            }
                        }
                },
                'predict_value': {
                    'Выход 2': {
                        'type': str,
                        'data': text,
                        'color_mark': 'wrong', None, 'success'
                        'tags_color': {
                                s1: #fffff,
                                s2: #tyuhg
                            }
                    }
                },
                'statistic_values': {
                    Выход 2: {
                        'class name': {
                            "value": str,
                            'color_mark': 'wrong', None, 'success'
                        },
                    }
                }
            },
        }
        """
        return_data = {}
        if self.interactive_config.get('intermediate_result').get('show_results'):
            for idx in range(self.interactive_config.get('intermediate_result').get('num_examples')):
                return_data[idx + 1] = {
                    'initial_data': {},
                    'true_value': {},
                    'predict_value': {},
                    'statistic_values': {}
                }
                if len(self.dataset_config.get("outputs").keys()) == 1 and self.dataset_config.get("outputs").get(
                        list(self.dataset_config.get("outputs").keys())[0]).get("task") == \
                        LayerOutputTypeChoice.TextSegmentation:
                    return_data[idx + 1]['initial_data'] = {}
                else:
                    for inp in self.dataset_config.get("inputs").keys():
                        path, type_choice = self._postprocess_initial_data(
                            input_id=inp,
                            save_id=idx + 1,
                            example_idx=self.example_idx[idx],
                        )
                        return_data[idx + 1]['initial_data'][f"Входной слой {inp}"] = {
                            'data': path,
                            'type': type_choice,
                        }
                for out in self.dataset_config.get("outputs").keys():
                    true_value, predict_value, color_mark, stat, out_type = self._postprocess_result_data(
                        output_id=out,
                        data_type='val',
                        save_id=idx + 1,
                        example_idx=self.example_idx[idx],
                        show_stat=self.interactive_config.get('intermediate_result').get('show_statistic'),
                    )
                    return_data[idx + 1]['true_value'][f"Выходной слой {out}"] = {
                        "type": out_type,
                        "data": true_value,
                        "color_mark": None,
                    }
                    return_data[idx + 1]['predict_value'][f"Выходной слой {out}"] = {
                        "type": out_type,
                        "data": predict_value,
                        "color_mark": color_mark,
                    },
                    return_data[idx + 1]['tags_color'] = \
                        self.dataset_config.get("outputs").get(out).get('classes_colors') if \
                            self.dataset_config.get("outputs").get(
                                list(self.dataset_config.get("outputs").keys())[0]).get(
                                "task") == LayerOutputTypeChoice.TextSegmentation else None
                    if stat:
                        return_data[idx + 1]['statistic_values'][f"Выходной слой {out}"] = stat
        return return_data

    def _get_statistic_data_request(self) -> dict:
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
            if self.dataset_config.get("outputs").get(f"{out}").get("task") == LayerOutputTypeChoice.Classification:
                cm, cm_percent = self._get_confusion_matrix(
                    np.argmax(self.y_true.get("val").get(f'{out}'), axis=-1)
                    if self.dataset_config.get("outputs").get(f"{out}").get("encoding") == "ohe"
                    else self.y_true.get("val").get(f'{out}'),
                    np.argmax(self.y_pred.get(f'{out}'), axis=-1),
                    get_percent=True
                )
                return_data[f'{out}'] = dict(
                    id=_id,
                    task_type=LayerOutputTypeChoice.Classification.name,
                    graph_name=f"Output_{out} - Confusion matrix",
                    x_label="Предсказание",
                    y_label="Истинное значение",
                    labels=self.dataset_config.get("outputs").get(f"{out}").get("classes_names"),
                    data_array=cm,
                    data_percent_array=cm_percent)
                _id += 1

            elif self.dataset_config.get("outputs").get(f"{out}").get("task") == LayerOutputTypeChoice.Segmentation or \
                    (
                            self.dataset_config.get("outputs").get(f"{out}").get("task") ==
                            LayerOutputTypeChoice.TextSegmentation and self.dataset_config.get(
                        "outputs").get(f"{out}").get('encoding') == 'ohe'
                    ):
                cm, cm_percent = self._get_confusion_matrix(
                    np.argmax(self.y_true.get("val").get(f"{out}"), axis=-1).reshape(
                        np.prod(np.argmax(self.y_true.get("val").get(f"{out}"), axis=-1).shape)).astype('int'),
                    np.argmax(self.y_pred.get(f'{out}'), axis=-1).reshape(
                        np.prod(np.argmax(self.y_pred.get(f'{out}'), axis=-1).shape)).astype('int'),
                    get_percent=True
                )
                return_data[f"{out}"] = dict(
                    id=_id,
                    task_type=LayerOutputTypeChoice.Segmentation.name,
                    graph_name=f"Выходной слой {out} - Confusion matrix",
                    x_label="Предсказание",
                    y_label="Истинное значение",
                    labels=self.dataset_config.get("outputs").get(f"{out}").get("classes_names"),
                    data_array=cm,
                    data_percent_array=cm_percent
                )
                _id += 1

            elif self.dataset_config.get("outputs").get(f"{out}").get("task") == LayerOutputTypeChoice.TextSegmentation \
                    and self.dataset_config.get("outputs").get(f"{out}").get("encoding") == "multi":
                report = self._get_classification_report(
                    self.y_true.get("val").get(f"{out}").reshape((np.prod(
                        self.y_true.get("val").get(f"{out}").shape[:-1]),
                                                                  self.y_true.get("val").get(f"{out}").shape[-1])
                    ),
                    np.where(self.y_pred.get(f"{out}") >= 0.9, 1, 0).reshape(
                        (np.prod(self.y_pred.get(f"{out}").shape[:-1]), self.y_pred.get(f"{out}").shape[-1])
                    ),
                    self.dataset_config.get("outputs").get(f"{out}").get("classes_names")
                )
                return_data[f"{out}"] = dict(
                    id=_id,
                    task_type=LayerOutputTypeChoice.TextSegmentation.name,
                    graph_name=f"Выходной слой {out} - Отчет по классам",
                    type="Table",
                    table_data=report,
                )
                _id += 1

            elif self.dataset_config.get("outputs").get(f"{out}").get("task") == LayerOutputTypeChoice.Regression:
                # Scatter
                pass

            elif self.dataset_config.get("outputs").get(f"{out}").get("task") == LayerOutputTypeChoice.Timeseries:
                # autocorrelation
                pass

            elif self.dataset_config.get("outputs").get(f"{out}").get("task") == LayerOutputTypeChoice.ObjectDetection:
                # accuracy for classes? smth else?
                pass

            else:
                return_data[f"{out}"] = {}
        return return_data

    def _get_balance_data_request(self) -> dict:
        """
        'data_balance': {
            'output_id': [
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
        for out in self.dataset_config.get("outputs").keys():
            if self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Classification:
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
                        'graph_name': 'Проверчная выборка',
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

            elif self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Segmentation:
                presence_train_names, presence_train_count = sort_dict(
                    self.dataset_balance.get(out).get('train').get('presence_balance'),
                    mode=self.interactive_config.get('data_balance').get('sorted')
                )
                presence_val_names, presence_val_count = sort_dict(
                    self.dataset_balance.get(out).get('val').get('presence_balance'),
                    mode=self.interactive_config.get('data_balance').get('sorted')
                )
                square_train_names, square_train_count = sort_dict(
                    self.dataset_balance.get(out).get('train').get('square_balance'),
                    mode=self.interactive_config.get('data_balance').get('sorted')
                )
                square_val_names, square_val_count = sort_dict(
                    self.dataset_balance.get(out).get('val').get('square_balance'),
                    mode=self.interactive_config.get('data_balance').get('sorted')
                )
                return_data[out] = [
                    {
                        'id': _id,
                        'graph_name': 'Тренировочная выборка - баланс присутсвия',
                        'x_label': 'Название класса',
                        'y_label': 'Значение',
                        'plot_data': [
                            {
                                'labels': presence_train_names,
                                'values': presence_train_count
                            },
                        ]
                    },
                    {
                        'id': _id + 1,
                        'graph_name': 'Проверочная выборка - баланс присутсвия',
                        'x_label': 'Название класса',
                        'y_label': 'Значение',
                        'plot_data': [
                            {
                                'labels': presence_val_names,
                                'values': presence_val_count
                            },
                        ]
                    },
                    {
                        'id': _id + 2,
                        'graph_name': 'Тренировочная выборка - процент пространства',
                        'x_label': 'Название класса',
                        'y_label': 'Значение',
                        'plot_data': [
                            {
                                'labels': square_train_names,
                                'values': square_train_count
                            },
                        ]
                    },
                    {
                        'id': _id + 3,
                        'graph_name': 'Проверочная выборка - процент пространства',
                        'x_label': 'Название класса',
                        'y_label': 'Значение',
                        'plot_data': [
                            {
                                'labels': square_val_names,
                                'values': square_val_count
                            },
                        ]
                    }
                ]
                _id += 4

            elif self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.TextSegmentation:
                presence_train_names, presence_train_count = sort_dict(
                    self.dataset_balance.get(out).get('train').get('presence_balance'),
                    mode=self.interactive_config.get('data_balance').get('sorted')
                )
                presence_val_names, presence_val_count = sort_dict(
                    self.dataset_balance.get(out).get('val').get('presence_balance'),
                    mode=self.interactive_config.get('data_balance').get('sorted')
                )
                return_data[out] = [
                    {
                        'id': _id,
                        'graph_name': 'Тренировочная выборка - баланс присутсвия',
                        'x_label': 'Название класса',
                        'y_label': 'Значение',
                        'plot_data': [
                            {
                                'labels': presence_train_names,
                                'values': presence_train_count
                            },
                        ]
                    },
                    {
                        'id': _id + 1,
                        'graph_name': 'Проверочная выборка - баланс присутсвия',
                        'x_label': 'Название класса',
                        'y_label': 'Значение',
                        'plot_data': [
                            {
                                'labels': presence_val_names,
                                'values': presence_val_count
                            },
                        ]
                    },
                ]
                _id += 2

            if self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Regression:
                # histograms for result and any chosen categorizing column
                pass

            if self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.Timeseries:
                # just graphic in case dataframe
                pass

            if self.dataset_config.get("outputs").get(out).get("task") == LayerOutputTypeChoice.ObjectDetection:
                # frequency of classes, like with segmentation
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
        return cm.astype('float'), cm_percent.astype('float')

    @staticmethod
    def _get_classification_report(y_true, y_pred, labels):
        cr = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        return_stat = {}
        for lbl in labels:
            return_stat[lbl] = {
                "Точность": round(cr.get(lbl).get('precision') * 100, 2),
                "Чувствительность": round(cr.get(lbl).get('recall') * 100, 2),
                "F1-мера": round(cr.get(lbl).get('f1-score') * 100, 2),
                "Количество": cr.get(lbl).get('support')
            }
        for i in ['macro avg', 'micro avg', 'samples avg', 'weighted avg']:
            return_stat[i] = {
                "Точность": round(cr.get(i).get('precision') * 100, 2),
                "Чувствительность": round(cr.get(i).get('recall') * 100, 2),
                "F1-мера": round(cr.get(i).get('f1-score') * 100, 2),
                "Количество": cr.get(i).get('support')
            }
        return return_stat

    @staticmethod
    def _dice_coef(y_true, y_pred, batch_mode=True, smooth=1.0):
        axis = tuple(np.arange(1, len(y_true.shape))) if batch_mode else None
        intersection = np.sum(y_true * y_pred, axis=axis)
        union = np.sum(y_true, axis=axis) + np.sum(y_pred, axis=axis)
        return (2.0 * intersection + smooth) / (union + smooth)

    def _postprocess_initial_data(self, input_id: str, save_id: int, example_idx: int):
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
        # if not self.dataset.g
        column_idx = 0
        if self.dataset_config.get("group") != 'keras':
            for column_name in self.dataset_config.get("dataframe").get('val').columns:
                if column_name.split('_')[0] == input_id:
                    column_idx = self.dataset_config.get("dataframe").get('val').columns.tolist().index(column_name)
            initial_file_path = os.path.join(
                self.dataset_config.get("dataset_path"),
                self.dataset_config.get("dataframe").get('val').iat[example_idx, column_idx]
            ) if self.dataset_config.get("inputs").get(input_id).get("task") != LayerInputTypeChoice.Text else ""
        else:
            initial_file_path = ""
        if self.dataset_config.get("inputs").get(input_id).get("task") == LayerInputTypeChoice.Image:
            if self.dataset_config.get("group") != 'keras':
                img = Image.open(initial_file_path)
                img = img.resize(
                    self.dataset_config.get("inputs").get(input_id).get("input_shape")[0:2][::-1],
                    Image.ANTIALIAS
                )
            else:
                img = image.array_to_img(self.x_val.get(input_id)[example_idx])
            img = img.convert('RGB')
            # filepath = NamedTemporaryFile()
            # save_path = f"/tmp/initial_data_image_{save_id}_input_{input_id}.webp"
            save_path = f"initial_data_image_{save_id}_input_{input_id}.webp"
            img.save(save_path, 'webp')
            return save_path, LayerInputTypeChoice.Image.name

        elif self.dataset_config.get("inputs").get(input_id).get("task") == LayerInputTypeChoice.Text:
            text_str = self.dataset_config.get("dataframe").get('val').iat[example_idx, column_idx]
            return text_str, LayerInputTypeChoice.Text.name
        elif self.dataset_config.get("inputs").get(input_id).get("task") == LayerInputTypeChoice.Video:
            clip = moviepy_editor.VideoFileClip(initial_file_path)
            # filepath = NamedTemporaryFile()
            # save_path = f"/tmp/initial_data_video_{save_id}_input_{input_id}.webm"
            save_path = f"initial_data_video_{save_id}_input_{input_id}.webm"
            clip.write_videofile(save_path)
            return save_path, LayerInputTypeChoice.Video.name
        elif self.dataset_config.get("inputs").get(input_id).get("task") == LayerInputTypeChoice.Audio:
            # filepath = NamedTemporaryFile()
            # save_path = f"/tmp/initial_data_audio_{save_id}_input_{input_id}.webp"
            save_path = f"initial_data_audio_{save_id}_input_{input_id}.webp"
            AudioSegment.from_file(initial_file_path).export(save_path, format="webm")
            return save_path, LayerInputTypeChoice.Audio.name
        elif self.dataset_config.get("inputs").get(input_id).get("task") == LayerInputTypeChoice.Dataframe:
            # TODO: обсудить как пересылать датафреймы на фронт
            return initial_file_path, LayerInputTypeChoice.Dataframe.name
        else:
            return initial_file_path, None

    def _postprocess_result_data(self, output_id: str, data_type: str, save_id: int, example_idx: int, show_stat=True):

        def add_tags_to_word(word: str, tag: str):
            """
            Tag  = <s1>
            """
            if tag:
                return f"<{tag[1:-1]}>{word}</{tag[1:-1]}>"
            else:
                return word

        def color_mixer(colors: list):
            if colors:
                result = np.zeros((3,))
                for color in colors:
                    result += np.array(color)
                return tuple((result / len(colors)).astype('int'))

        def tag_mixer(tags: list, colors: dict):
            tags = list(set(sorted(tags, reverse=True)))
            mix_tag = f"<{tags[0][1:-1]}"
            for tag in tags[1:]:
                mix_tag += f"+{tag[1:-1]}"
            mix_tag = f"{mix_tag}>"
            if mix_tag not in colors.keys():
                colors[mix_tag] = color_mixer([colors[tag] for tag in tags])
            return mix_tag

        def reformat_tags(y_array, classes_names: list, colors: dict, sensitivity: float = 0.9):
            norm_array = np.where(y_array >= sensitivity, 1, 0).astype('int')
            reformat_tags = []
            for word_tag in norm_array:
                if np.sum(word_tag) == 0:
                    reformat_tags.append(None)
                elif np.sum(word_tag) == 1:
                    reformat_tags.append(classes_names[np.argmax(word_tag, axis=-1)])
                else:
                    mix_tag = []
                    for i, tag in enumerate(word_tag):
                        if tag == 1:
                            mix_tag.append(classes_names[i])
                    reformat_tags.append(tag_mixer(mix_tag, colors))
            return reformat_tags

        def text_colorization(text: str, labels: list, classes_names: list, colors: dict):
            text = text.split(" ")
            labels = reformat_tags(labels, classes_names, colors)
            # if isinstance(labels, str):
            #     labels = ast.literal_eval(labels)
            colored_text = []
            for i, word in enumerate(text):
                colored_text.append(add_tags_to_word(word, labels[i]))
            return ' '.join(colored_text)

        if self.dataset_config.get("outputs").get(output_id).get("task") == LayerOutputTypeChoice.Classification:
            labels = self.dataset_config.get("outputs").get(output_id).get("classes_names")
            ohe = True if self.dataset_config.get("outputs").get(output_id).get("encoding") == 'ohe' else False

            y_true = np.argmax(self.y_true.get(data_type).get(output_id)[example_idx]) if ohe \
                else self.y_true.get(data_type).get(output_id)[example_idx]

            predict = self.y_pred.get(output_id)[example_idx]
            if y_true == np.argmax(predict):
                color_mark = 'success'
            else:
                color_mark = 'wrong'

            class_stat = {}
            if show_stat:
                for i, val in enumerate(predict):
                    if val == max(predict) and i == y_true:
                        class_color_mark = "success"
                    elif val == max(predict) and i != y_true:
                        class_color_mark = "wrong"
                    else:
                        class_color_mark = None
                    class_stat[labels[i]] = {
                        "value": f"{round(val * 100, 1)}%",
                        "color_mark": class_color_mark
                    }
            return labels[y_true], labels[np.argmax(predict)], color_mark, class_stat, "str"

        elif self.dataset_config.get("outputs").get(output_id).get("task") == LayerOutputTypeChoice.Segmentation:
            labels = self.dataset_config.get("outputs").get(output_id).get("classes_names")

            # prepare y_true image
            y_true = np.expand_dims(np.argmax(self.y_true.get(data_type).get(output_id)[example_idx], axis=-1), axis=-1)
            for color_idx in range(len(self.dataset_config.get("outputs").get(output_id).get("classes_colors"))):
                y_true = np.where(
                    y_true == [color_idx],
                    np.array(
                        self.dataset_config.get("outputs").get(output_id).get("classes_colors")[color_idx]),
                    y_true
                )
            y_true = tensorflow.keras.utils.array_to_img(y_true)
            y_true = y_true.convert('RGB')
            # filepath_true = NamedTemporaryFile()
            # y_true_save_path = f"/tmp/true_segmentation_data_image_{save_id}_output_{output_id}.webp"
            y_true_save_path = f"true_segmentation_data_image_{save_id}_output_{output_id}.webp"
            y_true.save(y_true_save_path, 'webp')

            # prepare y_pred image
            y_pred = np.expand_dims(np.argmax(self.y_pred.get(output_id)[example_idx], axis=-1), axis=-1)
            for color_idx in range(len(self.dataset_config.get("outputs").get(output_id).get("classes_colors"))):
                y_pred = np.where(
                    y_pred == [color_idx],
                    np.array(self.dataset_config.get("outputs").get(output_id).get("classes_colors")[color_idx]),
                    y_pred
                )
            y_pred = tensorflow.keras.utils.array_to_img(y_pred)
            y_pred = y_pred.convert('RGB')
            # filepath_pred = NamedTemporaryFile()
            # y_pred_save_path = f"/tmp/predict_segmentation_data_image_{save_id}_output_{output_id}.webp"
            y_pred_save_path = f"predict_segmentation_data_image_{save_id}_output_{output_id}.webp"
            y_pred.save(y_pred_save_path, 'webp')

            class_stat = {}
            if show_stat:
                y_true = np.array(self.y_true.get(data_type).get(output_id)[example_idx]).astype('int')
                y_pred = to_categorical(np.argmax(self.y_pred.get(output_id)[example_idx], axis=-1),
                                        self.dataset_config.get("outputs").get(output_id).get("num_classes"))
                for idx, cls in enumerate(labels):
                    dice_val = np.round(self._dice_coef(y_true[:, :, idx], y_pred[:, :, idx], batch_mode=False) * 100,
                                        1)
                    class_stat[cls] = {
                        "value": f"{dice_val}%",
                        "color_mark": 'success' if dice_val >= 90 else 'wrong'
                    }
            return y_true_save_path, y_pred_save_path, None, class_stat, "Image"

        elif self.dataset_config.get("outputs").get(output_id).get("task") == LayerOutputTypeChoice.TextSegmentation:
            # TODO: пока исходим что для сегментации текста есть только один вход с текстом, если будут сложные модели
            #  на сегментацию текста на несколько входов то придется искать решения

            classes_names = self.dataset_config.get("outputs").get(output_id).get("classes_names")
            text_for_preparation = self.dataset_config.get('dataframe').get('val').iat[example_idx, 0]
            true_text_segmentation = text_colorization(
                text_for_preparation,
                self.y_true.get(data_type).get(output_id)[example_idx],
                classes_names,
                self.dataset_config.get("outputs").get(output_id).get('classes_colors')
            )
            pred_text_segmentation = text_colorization(
                text_for_preparation,
                self.y_pred.get(output_id)[example_idx],
                classes_names,
                self.dataset_config.get("outputs").get(output_id).get('classes_colors')
            )
            class_stat = {}
            if show_stat:
                y_true = np.array(self.y_true.get(data_type).get(output_id)[example_idx]).astype('int')
                y_pred = np.where(self.y_pred.get(output_id)[example_idx] >= 0.9, 1., 0.)
                for idx, cls in enumerate(classes_names):
                    if np.sum(y_true[:, idx]) == 0 and np.sum(y_pred[:, idx]) == 0:
                        class_stat[cls] = {
                            "value": "-",
                            "color_mark": None
                        }
                    else:
                        dice_val = np.round(self._dice_coef(y_true[:, idx], y_pred[:, idx], batch_mode=False) * 100, 1)
                        class_stat[cls] = {
                            "value": f"{dice_val}%",
                            "color_mark": 'success' if dice_val >= 90 else 'wrong'
                        }
            return true_text_segmentation, pred_text_segmentation, None, class_stat, "Text"

        elif self.dataset_config.get("outputs").get(output_id).get("task") == LayerOutputTypeChoice.Regression:
            # values
            # stat - deviation
            pass
        elif self.dataset_config.get("outputs").get(output_id).get("task") == LayerOutputTypeChoice.Timeseries:
            # values
            # stat - deviation
            pass
        elif self.dataset_config.get("outputs").get(output_id).get("task") == LayerOutputTypeChoice.ObjectDetection:
            # image with bb
            # accuracy, correlation bb for classes
            pass
