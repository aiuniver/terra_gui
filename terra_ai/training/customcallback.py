import copy
import os
import re

from PIL import Image, ImageDraw, ImageFont  # Модули работы с изображениями

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from pydub import AudioSegment
from sklearn.metrics import confusion_matrix
import numpy as np
import moviepy.editor as moviepy_editor

from terra_ai import progress
from terra_ai.data.datasets.extra import LayerInputTypeChoice, LayerOutputTypeChoice
from terra_ai.data.presets.training import Metric, Loss
from terra_ai.data.training.extra import TaskChoice
from terra_ai.datasets.preparing import PrepareDTS
from terra_ai.utils import camelize, decamelize


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
            "mode": "min"
        },
        "CategoricalCrossentropy": {
            "log_name": "categorical_crossentropy",
            "mode": "min"
        },
        "CategoricalHinge": {
            "log_name": "categorical_hinge",
            "mode": "min"
        },
        "CosineSimilarity": {
            "log_name": "cosine_similarity",
            "mode": "min"
        },  # min if loss, max if metric
        "Hinge": {
            "log_name": "hinge",
            "mode": "min"
        },
        "Huber": {
            "log_name": "huber",
            "mode": "min"
        },
        "KLDivergence": {
            "log_name": "kullback_leibler_divergence",
            "mode": "min"
        },
        "LogCosh": {
            "log_name": "logcosh",
            "mode": "min"
        },
        "MeanAbsoluteError": {
            "log_name": "mean_absolute_error",
            "mode": "min"
        },
        "MeanAbsolutePercentageError": {
            "log_name": "mean_absolute_percentage_error",
            "mode": "min"
        },
        "MeanSquaredError": {
            "log_name": "mean_squared_error",
            "mode": "min"
        },
        "MeanSquaredLogarithmicError": {
            "log_name": "mean_squared_logarithmic_error",
            "mode": "min"
        },
        "Poisson": {
            "log_name": "poisson",
            "mode": "min"
        },
        "SparseCategoricalCrossentropy": {
            "log_name": "sparse_categorical_crossentropy",
            "mode": "min"
        },
        "SquaredHinge": {
            "log_name": "squared_hinge",
            "mode": "min"
        },
    },
    "metric": {
        "AUC": {
            "log_name": "auc",
            "mode": "max"
        },
        "Accuracy": {
            "log_name": "accuracy",
            "mode": "max"
        },
        "BinaryAccuracy": {
            "log_name": "binary_accuracy",
            "mode": "max"
        },
        "BinaryCrossentropy": {
            "log_name": "binary_crossentropy",
            "mode": "min"
        },
        "CategoricalAccuracy": {
            "log_name": "categorical_accuracy",
            "mode": "max"
        },
        "CategoricalCrossentropy": {
            "log_name": "categorical_crossentropy",
            "mode": "min"
        },
        "CategoricalHinge": {
            "log_name": "categorical_hinge",
            "mode": "min"
        },
        "CosineSimilarity": {
            "log_name": "cosine_similarity",
            "mode": "max"
        },  # min if loss, max if metric
        "FalseNegatives": {
            "log_name": "false_negatives",
            "mode": "min"
        },
        "FalsePositives": {
            "log_name": "false_positives",
            "mode": "min"
        },
        "Hinge": {
            "log_name": "hinge",
            "mode": "min"
        },
        "KLDivergence": {
            "log_name": "kullback_leibler_divergence",
            "mode": "min"
        },
        "LogCoshError": {
            "log_name": "logcosh",
            "mode": "min"
        },
        "MeanAbsoluteError": {
            "log_name": "mean_absolute_error",
            "mode": "min"
        },
        "MeanAbsolutePercentageError": {
            "log_name": "mean_absolute_percentage_error",
            "mode": "min"
        },
        "MeanIoU": {
            "log_name": "mean_io_u",
            "mode": "max"
        },
        "MeanSquaredError": {
            "log_name": "mean_squared_error",
            "mode": "min"
        },
        "MeanSquaredLogarithmicError": {
            "log_name": "mean_squared_logarithmic_error",
            "mode": "min"
        },
        "Poisson": {
            "log_name": "poisson",
            "mode": "min"
        },
        "Precision": {
            "log_name": "precision",
            "mode": "max"
        },
        "Recall": {
            "log_name": "recall",
            "mode": "max"
        },
        "RootMeanSquaredError": {
            "log_name": "root_mean_squared_error",
            "mode": "min"
        },
        "SparseCategoricalAccuracy": {
            "log_name": "sparse_categorical_accuracy",
            "mode": "max"
        },
        "SparseCategoricalCrossentropy": {
            "log_name": "sparse_categorical_crossentropy",
            "mode": "min"
        },
        "SparseTopKCategoricalAccuracy": {
            "log_name": "sparse_top_k_categorical_accuracy",
            "mode": "max"
        },
        "SquaredHinge": {
            "log_name": "squared_hinge",
            "mode": "min"
        },
        "TopKCategoricalAccuracy": {
            "log_name": "top_k_categorical_accuracy",
            "mode": "max"
        },
        "TrueNegatives": {
            "log_name": "true_negatives",
            "mode": "max"
        },
        "TruePositives": {
            "log_name": "true_positives",
            "mode": "max"
        },
    }
}


class InteractiveCallback:
    """Callback for interactive requests"""

    def __init__(self):
        """
        log_history:    epoch_num -> metrics/loss -> output_idx - > metric/loss -> train ->  total/classes
        """
        self.losses = None
        self.metrics = None
        self.dataset = None
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

        self.interactive_config = {
            'loss_graphs': [
                # {
                #     'id': 1,
                #     'output_idx': 2,
                #     'show_for_model': True,
                #     'show_for_classes': False
                # },
            ],
            'metric_graphs': [
                # {
                #     'id': 1,
                #     'output_idx': 2,
                #     'show_for_model': True,
                #     'show_for_classes': False,
                #     'show_metric': 'CategoricalAccuracy'
                # }
            ],
            'intermediate_result': {
                'show_results': False,
                # 'data_for_calculation': 'val',
                'example_choice_type': 'seed',
                'main_output': 2,
                'num_examples': 10,
                'show_statistic': False,
                'autoupdate': False
            },
            'progress_table': [
                # {
                #     'output_idx': 2,
                #     'show_loss': True,
                #     'show_metrics': True,
                # }
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

    def set_attributes(self, dataset: PrepareDTS, metrics: dict, losses: dict):
        self.losses = losses
        self.metrics = self._reformat_metrics(metrics)
        self._prepare_interactive_config()
        self.dataset = dataset
        self._prepare_y_true()
        self._prepare_null_log_history_template()
        self.dataset_balance = self._prepare_dataset_balance()
        self.class_idx = self._prepare_class_idx()
        self.seed_idx = self._prepare_seed()
        self.example_idx = self._prepare_example_idx_to_show()

    def _prepare_interactive_config(self):
        # fill loss_graphs config
        _id = 1
        for out in self.losses.keys():
            self.interactive_config.get('loss_graphs').append(
                {
                    'id': _id,
                    'output_idx': out,
                    'show_for_model': True,
                    'show_for_classes': False
                }
            )
            _id += 1

        # fill metric_graphs config
        _id = 1
        for out in self.metrics.keys():
            for metric_name in self.metrics.get(out):
                self.interactive_config.get('metric_graphs').append(
                    {
                        'id': _id,
                        'output_idx': out,
                        'show_for_model': True,
                        'show_for_classes': False,
                        'show_metric': metric_name
                    }
                )
                _id += 1

        # fill progress_table_config
        for out in self.metrics.keys():
            self.interactive_config.get('progress_table').append(
                {
                    'output_idx': out,
                    'show_loss': True,
                    'show_metrics': True,
                }
            )

    @staticmethod
    def _reformat_metrics(metrics: dict) -> dict:
        output = {}
        for out, out_metrics in metrics.items():
            output[out] = []
            for metric in out_metrics:
                if camelize(metric.name) == 'KullbackLeiblerDivergence':
                    output[out].append('KLDivergence')
                else:
                    output[out].append(camelize(metric.name))
        return output

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
        self.log_history['epochs'] = []
        for out in self.dataset.data.outputs.keys():
            # out: int
            self.log_history[f'{out}'] = {
                'loss': {},
                'metrics': {},
                'progress_state': {
                    'loss': {},
                    'metrics': {}
                }
            }
            # if self.losses.get(f'{out}') and isinstance(self.losses.get(f'{out}'), str):
            #     self.losses[f'{out}'] = self.losses.get(f'{out}')
            if self.metrics.get(f'{out}') and isinstance(self.metrics.get(f'{out}'), str):
                self.metrics[f'{out}'] = [self.metrics.get(f'{out}')]

            self.log_history[f'{out}']['loss'][self.losses.get(f'{out}')] = {'train': [], 'val': []}
            self.log_history[f'{out}']['progress_state']['loss'][self.losses.get(f'{out}')] = {
                'mean_log_history': [], 'normal_state': [], 'underfitting': [], 'overfitting': []
            }
            for metric in self.metrics.get(f'{out}'):
                self.log_history[f'{out}']['metrics'][f"{metric}"] = {'train': [], 'val': []}
                self.log_history[f'{out}']['progress_state']['metrics'][f"{metric}"] = {
                    'mean_log_history': [], 'normal_state': [], 'underfitting': [], 'overfitting': []
                }
            if self.dataset.data.task_type.get(out) == TaskChoice.Classification:
                self.log_history[f'{out}']['class_loss'] = {}
                self.log_history[f'{out}']['class_metrics'] = {}
                for class_name in self.dataset.data.classes_names.get(out):
                    self.log_history[f'{out}']['class_metrics'][f"{class_name}"] = {}
                    self.log_history[f'{out}']['class_loss'][f"{class_name}"] = {}
                    self.log_history[f'{out}']['class_loss'][f"{class_name}"][self.losses.get(f'{out}')] = []
                    for metric in self.metrics.get(f'{out}'):
                        self.log_history[f'{out}']['class_metrics'][f"{class_name}"][f"{metric}"] = []

    def _prepare_y_true(self):
        self.y_true = {
            'train': {},
            'val': {}
        }
        for task_type in self.y_true.keys():
            for out in self.dataset.data.outputs.keys():
                if self.dataset.data.outputs.get(
                        out).task == TaskChoice.Classification and self.dataset.data.use_generator:
                    self.y_true.get(task_type)[f'{out}'] = []
                    for column_name in self.dataset.dataframe.get(task_type).columns:
                        if column_name.split('_')[0] == f'{out}':
                            for lbl in list(self.dataset.dataframe.get(task_type)[column_name]):
                                self.y_true[task_type][f'{out}'].append(
                                    to_categorical(self.dataset.data.classes_names.get(out).index(lbl),
                                                   num_classes=self.dataset.data.num_classes.get(out))
                                    if self.dataset.data.encoding.get(out) == 'ohe'
                                    else self.dataset.data.classes_names.get(out).index(lbl))
                            self.y_true[task_type][f'{out}'] = np.array(self.y_true[task_type][f'{out}'])
                            break
                elif self.dataset.data.outputs.get(out).task == TaskChoice.Classification and \
                        not self.dataset.data.use_generator:
                    self.y_true[task_type][f'{out}'] = self.dataset.Y.get(task_type).get(f"{out}")
                else:
                    pass

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
        for out in self.dataset.data.outputs.keys():
            dataset_balance[f'{out}'] = {}
            if self.dataset.data.task_type.get(out) == TaskChoice.Classification:
                for data_type in self.y_true.keys():
                    dataset_balance[f'{out}'][data_type] = class_counter(
                        self.y_true.get(data_type).get(f'{out}'),
                        self.dataset.data.classes_names.get(out),
                        self.dataset.data.encoding.get(out) == 'ohe'
                    )
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
                for name in self.dataset.data.classes_names.get(int(out)):
                    class_idx[data_type][out][name] = []
                y_true = np.argmax(self.y_true.get(data_type).get(out), axis=-1) \
                    if self.dataset.data.encoding.get(int(out)) == 'ohe' else self.y_true.get(data_type).get(out)
                for idx in range(len(y_true)):
                    class_idx[data_type][out][self.dataset.data.classes_names.get(int(out))[y_true[idx]]].append(idx)
        return class_idx

    def _prepare_seed(self):
        data_lenth = np.arange(len(self.dataset.dataframe.get('val')))
        np.random.shuffle(data_lenth)
        return data_lenth

    def update_epoch_state(self, fit_logs, y_pred, current_epoch_time, on_epoch_end_flag=True) -> dict:
        if on_epoch_end_flag:
            self.current_epoch = fit_logs.get('epoch')
            self.current_logs = self._reformat_fit_logs(fit_logs)
            self._reformat_y_pred(y_pred)
            self._update_log_history()
            self._update_progress_table(current_epoch_time)
            if self.interactive_config.get('intermediate_result').get('show_results') and \
                    self.interactive_config.get('intermediate_result').get('autoupdate'):
                self.intermediate_result = self._get_intermediate_result_request()
            if self.interactive_config.get('statistic_data').get('show_results') \
                    and self.interactive_config.get('statistic_data').get('autoupdate'):
                self.statistic_result = self._get_statistic_data_request()
        else:
            self._reformat_y_pred(y_pred)
            if self.interactive_config.get('intermediate_result').get('show_results'):
                self.intermediate_result = self._get_intermediate_result_request()
            if self.interactive_config.get('statistic_data').get('show_results'):
                self.statistic_result = self._get_statistic_data_request()
        return {
            'loss_graphs': self._get_loss_graph_data_request(),
            'metric_graphs': self._get_metric_graph_data_request(),
            'intermediate_result': self.intermediate_result,
            'progress_table': self.progress_table,
            'statistic_data': self.statistic_result,
            'data_balance': self._get_balance_data_request(),
        }

    def _reformat_fit_logs(self, logs) -> dict:
        interactive_log = {}
        update_logs = {}
        for log, val in logs.items():
            if re.search(r'_\d+$', log):
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

    def _update_log_history(self):
        self.log_history['epochs'].append(self.current_epoch)
        for out in self.dataset.data.outputs.keys():
            if self.dataset.data.encoding.get(out) == 'ohe':
                ohe = True
            else:
                ohe = False
            num_classes = self.dataset.data.num_classes.get(out)

            for loss_name in self.log_history.get(f'{out}').get('loss').keys():
                for data_type in ['train', 'val']:
                    # fill losses
                    self.log_history[f'{out}']['loss'][loss_name][data_type].append(
                        self.current_logs.get(f'{out}').get('loss').get(loss_name).get(data_type)
                        if self.current_logs.get(f'{out}').get('loss').get(loss_name).get(data_type) else 0.
                    )

                # fill loss progress state
                self.log_history[f'{out}']['progress_state']['loss'][loss_name]['mean_log_history'].append(
                    self._get_mean_log(self.log_history.get(f'{out}').get('loss').get(loss_name).get('val'))
                )

                # get progress state data
                loss_underfitting = self._evaluate_underfitting(
                    loss_name,
                    self.log_history[f'{out}']['loss'][loss_name]['train'][-1],
                    self.log_history[f'{out}']['loss'][loss_name]['val'][-1],
                    metric_type='loss'
                )
                loss_overfittng = self._evaluate_overfitting(
                    loss_name,
                    self.log_history[f'{out}']['progress_state']['loss'][loss_name]['mean_log_history'],
                    metric_type='loss'
                )
                if loss_underfitting or loss_overfittng:
                    normal_state = False
                else:
                    normal_state = True

                self.log_history[f'{out}']['progress_state']['loss'][loss_name]['underfitting'].append(loss_underfittng)
                self.log_history[f'{out}']['progress_state']['loss'][loss_name]['overfitting'].append(loss_overfittng)
                self.log_history[f'{out}']['progress_state']['loss'][loss_name]['normal_state'].append(normal_state)

                # get Classification loss logs
                if self.dataset.data.task_type.get(out) == TaskChoice.Classification:
                    # fill class losses
                    for cls in self.log_history.get(f'{out}').get('class_loss').keys():
                        class_loss = self._get_loss_calculation(
                            loss_name,
                            self.y_true.get('val').get(f'{out}')[self.class_idx.get('val').get(f'{out}').get(cls)],
                            self.y_pred.get(f'{out}')[self.class_idx.get('val').get(f'{out}').get(cls)],
                            ohe,
                            num_classes
                        )
                        self.log_history[f'{out}']['class_loss'][cls][loss_name].append(
                            class_loss if class_loss else 0.
                        )

            for metric_name in self.log_history.get(f'{out}').get('metrics').keys():
                for data_type in ['train', 'val']:
                    # fill metrics
                    self.log_history[f'{out}']['metrics'][metric_name][data_type].append(
                        self.current_logs.get(f'{out}').get('metrics').get(metric_name).get(data_type)
                        if self.current_logs.get(f'{out}').get('metrics').get(metric_name).get(data_type) else 0.
                    )

                # fill metric progress state
                self.log_history[f'{out}']['progress_state']['metrics'][metric_name]['mean_log_history'].append(
                    self._get_mean_log(self.log_history[f'{out}']['metrics'][metric_name]['val'])
                )
                loss_underfittng = self._evaluate_underfitting(
                    metric_name,
                    self.log_history[f'{out}']['metrics'][metric_name]['train'][-1],
                    self.log_history[f'{out}']['metrics'][metric_name]['val'][-1],
                    metric_type='metric'
                )
                loss_overfittng = self._evaluate_overfitting(
                    metric_name,
                    self.log_history[f'{out}']['progress_state']['metrics'][metric_name]['mean_log_history'],
                    metric_type='metric'
                )
                if loss_underfittng or loss_overfittng:
                    normal_state = False
                else:
                    normal_state = True
                self.log_history[f'{out}']['progress_state']['metrics'][metric_name]['underfitting'].append(
                    loss_underfittng)
                self.log_history[f'{out}']['progress_state']['metrics'][metric_name]['overfitting'].append(
                    loss_overfittng)
                self.log_history[f'{out}']['progress_state']['metrics'][metric_name]['normal_state'].append(
                    normal_state)

                if self.dataset.data.task_type.get(out) == TaskChoice.Classification:
                    # fill class losses
                    for cls in self.log_history.get(f'{out}').get('class_metrics').keys():
                        class_metric = self._get_metric_calculation(
                            metric_name,
                            self.y_true.get('val').get(f'{out}')[self.class_idx.get('val').get(f'{out}').get(cls)],
                            self.y_pred.get(f'{out}')[self.class_idx.get('val').get(f'{out}').get(cls)],
                            ohe,
                            num_classes
                        )
                        self.log_history[f'{out}']['class_metrics'][cls][metric_name].append(
                            class_metric if class_metric else 0.
                        )

    def _update_progress_table(self, epoch_time: float):
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
        self.progress_table[self.current_epoch] = {
            "time": epoch_time,
        }
        for out in self.dataset.data.outputs.keys():
            self.progress_table[self.current_epoch][f"Output_{out}"] = {
                'loss': {},
                'metrics': {}
            }
            self.progress_table[self.current_epoch][f"Output_{out}"]["loss"] = {
                'loss': self.log_history.get(f'{out}').get('loss').get(self.losses.get(f'{out}')).get('train')[-1],
                'val_loss': self.log_history.get(f'{out}').get('loss').get(self.losses.get(f'{out}')).get('val')[-1]
            }
            for metric in self.metrics.get(f'{out}'):
                self.progress_table[self.current_epoch][f"Output_{out}"]["metrics"][metric] = \
                    self.log_history.get(f'{out}').get('metrics').get(metric).get('train')[-1]
                self.progress_table[self.current_epoch][f"Output_{out}"]["metrics"][f"val_{metric}"] = \
                    self.log_history.get(f'{out}').get('metrics').get(metric).get('val')[-1]

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

    def return_interactive_results(self, config: dict) -> dict:
        """Return dict with data for current interactive request"""
        self.interactive_config = config if config else self.interactive_config
        self.example_idx = self._prepare_example_idx_to_show()
        return {
            'loss_graphs': self._get_loss_graph_data_request(),
            'metric_graphs': self._get_metric_graph_data_request(),
            'intermediate_result': self.intermediate_result,
            'progress_table': self.progress_table,
            'statistic_data': self.statistic_result,
            'data_balance': self._get_balance_data_request(),
        }

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
        data_return = []
        if self.interactive_config.get('loss_graphs'):
            for loss_graph_config in self.interactive_config.get('loss_graphs'):
                if loss_graph_config.get('show_for_model'):
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
                                        self.losses.get(f"{loss_graph_config.get('output_idx')}")).get('train')
                                },
                                {
                                    "label": "Проверочная выборка",
                                    "epochs": self.log_history.get("epochs"),
                                    "values": self.log_history.get(
                                        f"{loss_graph_config.get('output_idx')}").get('loss').get(
                                        self.losses.get(f"{loss_graph_config.get('output_idx')}")).get("val")
                                }
                            ],
                            "progress_state": progress_state
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
                                        self.losses.get(f"{loss_graph_config.get('output_idx')}"))
                                } for class_name in self.dataset.data.classes_names.get(
                                    loss_graph_config.get('output_idx'))
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
            pass
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
        data_return = []
        if self.interactive_config.get('metric_graphs'):
            for metric_graph_config in self.interactive_config.get('metric_graphs'):
                if metric_graph_config.get('show_for_model'):
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
                            "progress_state": progress_state
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
                                        class_name).get(metric_graph_config.get('show_metric'))
                                } for class_name in self.dataset.data.classes_names.get(
                                    metric_graph_config.get('output_idx'))
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
            pass
        return data_return

    def _get_intermediate_result_request(self) -> dict:

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
        return_data = {}
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
                for out in self.y_true.get('train').keys():
                    true_lbl, predict_lbl, color_mark, stat = self._postprocess_result_data(
                        output_id=out,
                        data_type='val',
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
            if self.dataset.data.task_type.get(out) == TaskChoice.Classification:
                cm = self._get_confusion_matrix(
                    np.argmax(self.y_true.get("val").get(f'{out}'), axis=-1)
                    if self.dataset.data.encoding.get(out) == "ohe" else self.y_true.get("val").get(f'{out}'),
                    np.argmax(self.y_pred.get(f'{out}'), axis=-1)
                )
                return_data[f'{out}'] = {
                    "id": _id,
                    "task_type": TaskChoice.Classification.name,
                    "graph_name": f"Output_{out} - Confusion matrix",
                    "x_label": "Предсказание",
                    "y_label": "Истинное значение",
                    "labels": self.dataset.data.classes_names.get(out),
                    "data_array": cm[0],
                    "data_percent_array": cm[1]
                }
                _id += 1
            else:
                return_data[f'{out}'] = {}
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
        for out in self.y_true.get('train').keys():
            if self.dataset.data.task_type.get(int(out)) == TaskChoice.Classification:
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

    @staticmethod
    def _get_confusion_matrix(y_true, y_pred) -> tuple:
        cm = confusion_matrix(y_true, y_pred)

        cm_percent = np.zeros_like(cm).astype('float32')
        for i in range(len(cm)):
            total = np.sum(cm[i])
            for j in range(len(cm[i])):
                cm_percent[i][j] = round(cm[i][j] * 100 / total, 1)
        return (cm, cm_percent)

    def _prepare_example_idx_to_show(self) -> dict:
        """
        example_idx = {
            output_id: []
        }
        """
        example_idx = {}
        count = self.interactive_config.get('intermediate_result').get('num_examples')
        choice_type = self.interactive_config.get('intermediate_result').get('example_choice_type')
        if choice_type == 'best' or choice_type == 'worst':
            y_true = self.dataset.Y.get('val').get(
                f"{self.interactive_config.get('intermediate_result').get('main_output')}")
            y_pred = self.y_pred.get(f"{self.interactive_config.get('intermediate_result').get('main_output')}")
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
                example_idx = sorted_args[::-1][:count]
            elif choice_type == 'worst':
                example_idx = sorted_args[:count]
            else:
                example_idx = np.random.choice(len(probs), count, replace=False)

        elif choice_type == 'seed':
            example_idx = self.seed_idx[:self.interactive_config.get('intermediate_result').get('num_examples')]
        elif choice_type == 'random':
            example_idx = np.random.randint(
                0,
                len(self.dataset.X.get('val')),
                self.interactive_config.get('intermediate_result').get('num_examples')
            )
        else:
            pass
        return example_idx

    def _postprocess_initial_data(self, input_id: str, example_idx: int):
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
            'val').iat[example_idx, 0])
        # TODO: посмотреть как реализовать '.iat[example_idx, 0]' для нескольких входов
        if self.dataset.data.inputs.get(int(input_id)).task == LayerInputTypeChoice.Image:
            img = Image.open(initial_file_path)
            img = img.convert('RGB')
            save_path = f"/tmp/initial_data_image{example_idx}_input{input_id}.webp"
            img.save(save_path, 'webp')
            return save_path
        elif self.dataset.data.inputs.get(int(input_id)).task == LayerInputTypeChoice.Text:
            text_str = open(initial_file_path, 'r')
            return text_str
        elif self.dataset.data.inputs.get(int(input_id)).task == LayerInputTypeChoice.Video:
            clip = moviepy_editor.VideoFileClip(initial_file_path)
            save_path = f"/tmp/initial_data_video{example_idx}_input{input_id}.webp"
            clip.write_videofile(save_path)
            return save_path
        elif self.dataset.data.inputs.get(int(input_id)).task == LayerInputTypeChoice.Audio:
            save_path = f"/tmp/initial_data_audio{example_idx}_input{input_id}.webp"
            AudioSegment.from_file(initial_file_path).export(save_path, format="webm")
            return save_path
        elif self.dataset.data.inputs.get(int(input_id)).task == LayerInputTypeChoice.Dataframe:
            # TODO: обсудить как пересылать датафреймы на фронт
            return initial_file_path
        else:
            return initial_file_path

    def _postprocess_result_data(self, output_id: str, data_type: str, example_idx: int, show_stat=True):
        if self.dataset.data.outputs.get(int(output_id)).task == LayerOutputTypeChoice.Classification:
            labels = self.dataset.data.classes_names.get(int(output_id))
            ohe = True if self.dataset.data.encoding.get(int(output_id)) == 'ohe' else False

            y_true = np.argmax(self.y_true.get(data_type).get(output_id)[example_idx]) if ohe \
                else self.y_true.get(data_type).get(output_id)[example_idx]

            predict = self.y_pred.get(output_id)[example_idx]
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

        elif self.dataset.data.outputs.get(int(output_id)).task == LayerOutputTypeChoice.Segmentation:
            pass
        elif self.dataset.data.outputs.get(int(output_id)).task == LayerOutputTypeChoice.TextSegmentation:
            pass
        elif self.dataset.data.outputs.get(int(output_id)).task == LayerOutputTypeChoice.Regression:
            pass
        elif self.dataset.data.outputs.get(int(output_id)).task == LayerOutputTypeChoice.Timeseries:
            pass
        elif self.dataset.data.outputs.get(int(output_id)).task == LayerOutputTypeChoice.ObjectDetection:
            pass
