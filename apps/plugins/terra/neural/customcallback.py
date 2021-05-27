import base64
import copy
import os
import tempfile

import tensorflow as tf

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
import numpy as np
import types
import time
from terra_ai.guiexchange import Exchange
from terra_ai.trds import DTS

__version__ = 0.04


class CustomCallback(keras.callbacks.Callback):
    """CustomCallback for all task type"""

    def __init__(
            self,
            params: dict = None,
            step=1,
            show_final=True,
            dataset=DTS(),
            exchange=Exchange(),
            samples_x: dict = None,
            samples_y: dict = None,
            batch_size: int = None,
            epochs: int = None,
            save_model_path: str = "./",
            model_name: str = "noname"):

        """
        Init for Custom callback
        Args:
            params (list):              список используемых параметров
            step int():                 шаг вывода хода обучения, по умолчанию step = 1
            show_final (bool):          выводить ли в конце обучения график, по умолчанию True
            dataset (DTS):              экземпляр класса DTS
            exchange (Exchange)         экземпляр класса Exchange
        Returns:
            None
        """
        super().__init__()
        if params is None:
            params = {}
        if samples_y is None:
            samples_y = {}
        if samples_x is None:
            samples_x = {}
        self.step = step
        self.clbck_params = params
        self.show_final = show_final
        self.Exch = exchange
        self.DTS = dataset
        self.save_model_path = save_model_path
        self.nn_name = model_name
        self.metrics = []
        self.loss = []
        self.callbacks = []
        self.callbacks_name = []
        self.task_name = []
        self.num_classes = []
        self.y_Scaler = []
        self.tokenizer = []
        self.one_hot_encoding = []
        self.data_tag = []
        self.x_Val = samples_x
        self.y_true = samples_y
        self.batch_size = batch_size
        self.epochs = epochs
        self.batch = 0
        self.num_batches = self.DTS.X['input_1']['data'][0].shape[0] // self.batch_size
        self.msg_epoch = ""
        self.y_pred = []
        self.epoch = 0
        self.history = {}
        self._start_time = time.time()
        self._now_time = time.time()
        self._time_first_step = time.time()

        self.task_type_defaults_dict = {
            "classification": {
                "optimizer_name": "Adam",
                "loss": "categorical_crossentropy",
                "metrics": ["accuracy"],
                "batch_size": 32,
                "epochs": 20,
                "shuffle": True,
                "clbck_object": ClassificationCallback,
                "callback_kwargs": {
                    "metrics": ["loss", "accuracy"],
                    "step": 1,
                    "class_metrics": [],
                    "num_classes": 2,
                    "data_tag": "images",
                    "show_best": True,
                    "show_worst": False,
                    "show_final": True,
                    "dataset": self.DTS,
                    "exchange": self.Exch,
                },
            },
            "segmentation": {
                "optimizer_name": "Adam",
                "loss": "categorical_crossentropy",
                "metrics": ["dice_coef"],
                "batch_size": 16,
                "epochs": 20,
                "shuffle": False,
                "clbck_object": SegmentationCallback,
                "callback_kwargs": {
                    "metrics": ["dice_coef"],
                    "step": 1,
                    "class_metrics": [],
                    "num_classes": 2,
                    "data_tag": "images",
                    "show_worst": False,
                    "show_final": True,
                    "dataset": self.DTS,
                    "exchange": self.Exch,
                },
            },
            "regression": {
                "optimizer_name": "Adam",
                "loss": "mse",
                "metrics": ["mae"],
                "batch_size": 32,
                "epochs": 20,
                "shuffle": True,
                "clbck_object": RegressionCallback,
                "callback_kwargs": {
                    "metrics": ["loss", "mse"],
                    "step": 1,
                    "plot_scatter": True,
                    "show_final": True,
                    "dataset": self.DTS,
                    "exchange": self.Exch,
                },
            },
            "timeseries": {
                "optimizer_name": "Adam",
                "loss": "mse",
                "metrics": ["mae"],
                "batch_size": 32,
                "epochs": 20,
                "shuffle": False,
                "clbck_object": TimeseriesCallback,
                "callback_kwargs": {
                    "metrics": ["loss", "mse"],
                    "step": 1,
                    "corr_step": 10,
                    "plot_pred_and_true": True,
                    "show_final": True,
                    "dataset": self.DTS,
                    "exchange": self.Exch,
                },
            },
        }
        self.callback_kwargs = []
        self.clbck_object = []
        self.prepare_params()
        self.Exch.show_text_data(
            f"Добавлены колбэки: {self.callbacks_name}"
        )

    def save_lastmodel(self) -> None:
        """
        Saving last model on each epoch end

        Returns:
            None
        """
        model_name = f"model_{self.nn_name}_on_epoch_end_last"
        file_path_model: str = os.path.join(
            self.save_model_path, f"{model_name}.h5"
        )
        self.model.save(file_path_model)
        self.Exch.print_2status_bar(
            ("Инфо", f"Последняя модель сохранена как {file_path_model}")
        )
        pass

    def prepare_callbacks(
            self, task_type: str = "", metrics: list = None, num_classes: int = None,
            clbck_options: dict = {}, tags: dict = {}) -> None:
        """
        if terra in raw mode  - setting callback if its set
        if terra with django - checking switches and set callback options from switches

        Returns:
            None
        """

        __task_type_defaults_kwargs = self.task_type_defaults_dict.get(task_type)
        callback_kwargs = __task_type_defaults_kwargs["callback_kwargs"]
        if metrics:
            callback_kwargs["metrics"] = copy.copy(metrics)
        if task_type == "classification" or task_type == "segmentation":
            callback_kwargs["num_classes"] = num_classes
            if tags["input_1"]:
                callback_kwargs["data_tag"] = tags["input_1"]

        for option_name, option_value in clbck_options.items():

            if option_name == "show_every_epoch":
                if option_value:
                    callback_kwargs["step"] = 1
                else:
                    callback_kwargs["step"] = 0
            elif option_name == "plot_loss_metric":
                if option_value:
                    if not ("loss" in callback_kwargs["metrics"]):
                        callback_kwargs["metrics"].append("loss")
                else:
                    if "loss" in callback_kwargs["metrics"]:
                        callback_kwargs["metrics"].remove("loss")
            elif option_name == "plot_metric":
                if option_value:
                    if not (metrics[0] in callback_kwargs["metrics"]):
                        callback_kwargs["metrics"].append(metrics[0])
                else:
                    if metrics[0] in callback_kwargs["metrics"]:
                        callback_kwargs["metrics"].remove(metrics[0])
            elif option_name == "plot_final":
                if option_value:
                    callback_kwargs["show_final"] = True
                else:
                    callback_kwargs["show_final"] = False

        if (task_type == "classification") or (task_type == "segmentation"):
            for option_name, option_value in clbck_options.items():
                if option_name == "plot_loss_for_classes":
                    if option_value:
                        if not ("loss" in callback_kwargs["class_metrics"]):
                            callback_kwargs["class_metrics"].append("loss")
                    else:
                        if "loss" in callback_kwargs["class_metrics"]:
                            callback_kwargs["class_metrics"].remove("loss")
                elif option_name == "plot_metric_for_classes":
                    if option_value:
                        if not (metrics[0] in callback_kwargs["class_metrics"]):
                            callback_kwargs["class_metrics"].append(metrics[0])
                    else:
                        if metrics[0] in callback_kwargs["class_metrics"]:
                            callback_kwargs["class_metrics"].remove(metrics[0])
                elif option_name == "show_worst_images" and option_value:
                    if option_value:
                        callback_kwargs["show_worst"] = True
                    else:
                        callback_kwargs["show_worst"] = False
                elif option_name == 'show_best_images':
                    if option_value:
                        callback_kwargs['show_best'] = True
                    else:
                        callback_kwargs['show_best'] = False

        if task_type == 'regression':
            for option_name, option_value in clbck_options.items():
                if option_name == 'plot_scatter':
                    if option_value:
                        callback_kwargs['plot_scatter'] = True
                    else:
                        callback_kwargs['plot_scatter'] = False

        if task_type == 'timeseries':
            for option_name, option_value in clbck_options.items():
                if option_name == 'plot_autocorrelation':
                    if option_value:
                        callback_kwargs['corr_step'] = 10
                    else:
                        callback_kwargs['corr_step'] = 0
                elif option_name == 'plot_pred_and_true':
                    if option_value:
                        callback_kwargs['plot_pred_and_true'] = True
                    else:
                        callback_kwargs['plot_pred_and_true'] = False

        self.callback_kwargs.append(callback_kwargs)
        clbck_object = __task_type_defaults_kwargs['clbck_object']
        self.clbck_object.append(clbck_object)
        initialized_callback = clbck_object(**callback_kwargs)
        self.callbacks.append(initialized_callback)
        self.callbacks_name.append(initialized_callback.__name__)

        pass

    def prepare_params(self):

        for _key in self.clbck_params.keys():
            self.metrics.append(self.clbck_params[_key]["metrics"])
            self.loss.append(self.clbck_params[_key]["loss"])
            self.task_name.append(self.clbck_params[_key]["task"])
            self.num_classes.append(self.clbck_params.setdefault(_key)["num_classes"])
            self.y_Scaler.append(self.DTS.y_Scaler.setdefault(_key))
            self.tokenizer.append(self.DTS.tokenizer.setdefault(_key))
            self.one_hot_encoding.append(self.DTS.one_hot_encoding.setdefault(_key))
            self.prepare_callbacks(
                task_type=self.clbck_params[_key]["task"].value,
                metrics=self.clbck_params[_key]["metrics"],
                num_classes=self.clbck_params.setdefault(_key)["num_classes"],
                clbck_options=self.clbck_params[_key]["callbacks"],
                tags=self.DTS.tags,
            )

    def _estimate_step(self, current, start, now):
        if current:
            _time_per_unit = (now - start) / current
        else:
            _time_per_unit = (now - start)
        return _time_per_unit

    def update_progress(self, target, current, start_time, finalize=False):
        """
        Updates the progress bar.
        """
        if finalize:
            _now_time = time.time()
            eta = _now_time - start_time
        else:
            _now_time = time.time()

            time_per_unit = self._estimate_step(current, start_time, _now_time)

            eta = time_per_unit * (target - current)

        if eta > 3600:
            eta_format = '%d ч %02d мин %02d сек' % (eta // 3600,
                                           (eta % 3600) // 60, eta % 60)
        elif eta > 60:
            eta_format = '%d мин %02d сек' % (eta // 60, eta % 60)
        else:
            eta_format = '%d сек' % eta

        info = ' %s' % eta_format
        return info

    def on_train_begin(self, logs=None):
        self._start_time = time.time()
        self.Exch.show_current_epoch(self.epoch)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self._time_first_step = time.time()

    def on_batch_begin(self, batch, logs=None):
        stop = self.Exch.get_stop_training_flag()
        if stop:
            self.model.stop_training = True
            msg = f'эпоха: {self.epoch + 1}, модель сохранена'
            self.Exch.print_2status_bar(('Обучение остановлено пользователем', msg))

    def on_train_batch_end(self, batch, logs=None):
        msg_batch = f'Батч {batch}/{self.num_batches}'
        msg_epoch = f'Эпоха {self.epoch + 1}/{self.epochs}:' \
                    f'{self.update_progress(self.num_batches, batch, self._time_first_step)}, '
        msg_progress_end = f'Расчетное время окончания:' \
                       f'{self.update_progress(self.num_batches * self.epochs+1, self.batch, self._start_time)}, '
        msg_progress_start = f'Время выполнения:' \
                       f'{self.update_progress(self.num_batches * self.epochs+1, self.batch, self._start_time, finalize=True)}, '
        self.batch += 1
        self.Exch.print_2status_bar(('Прогресс обучения', msg_progress_start +
                                     msg_progress_end + msg_epoch + msg_batch))

    def on_epoch_end(self, epoch, logs=None):
        """
        Returns:
            {}:
        """
        self.msg_epoch = self.update_progress(self.num_batches, self.batch, self._time_first_step, finalize=True)
        if self.x_Val["input_1"] is not None:
            self.y_pred = self.model.predict(self.x_Val)
        else:
            self.y_pred = self.y_true
        if isinstance(self.y_pred, list):
            for i, output_key in enumerate(self.clbck_params.keys()):
                self.callbacks[i].epoch_end(
                    epoch,
                    logs=logs,
                    output_key=output_key,
                    y_pred=self.y_pred[i],
                    y_true=self.y_true[output_key],
                    loss=self.loss[i],
                    msg_epoch=self.msg_epoch
                )
        else:
            for i, output_key in enumerate(self.clbck_params.keys()):
                self.callbacks[i].epoch_end(
                    epoch,
                    logs=logs,
                    output_key=output_key,
                    y_pred=self.y_pred,
                    y_true=self.y_true[output_key],
                    loss=self.loss[i],
                    msg_epoch=self.msg_epoch
                )
        self.Exch.show_current_epoch(epoch)
        self.save_lastmodel()


    def on_train_end(self, logs=None):
        for i, output_key in enumerate(self.clbck_params.keys()):
            self.callbacks[i].train_end(output_key=output_key, x_val=self.x_Val)
        self.save_lastmodel()
        self.Exch.show_text_data(
            f'Затрачено времени на обучение: '
            f'{self.update_progress(self.num_batches * self.epochs + 1, self.batch, self._start_time, finalize=True)}')


class ClassificationCallback:
    """Callback for classification"""

    def __init__(
            self,
            metrics=[],
            step=1,
            class_metrics=[],
            data_tag=None,
            num_classes=2,
            show_worst=False,
            show_best=False,
            show_final=True,
            dataset=DTS(),
            exchange=Exchange(),
    ):
        """
        Init for classification callback
        Args:
            metrics (list):             список используемых метрик: по умолчанию [], что соответсвует 'loss'
            class_metrics:              вывод графиков метрик по каждому сегменту: по умолчанию []
            step int():                 шаг вывода хода обучения, по умолчанию step = 1
            show_worst (bool):          выводить ли справа отдельно экземпляры, худшие по метрикам, по умолчанию False
            show_best (bool):           выводить ли справа отдельно экземпляры, лучшие по метрикам, по умолчанию False
            show_final (bool):          выводить ли в конце обучения график, по умолчанию True
            dataset (DTS):              экземпляр класса DTS
        Returns:
            None
        """
        self.__name__ = "Callback for classification"
        self.step = step
        self.clbck_metrics = metrics
        self.class_metrics = class_metrics
        self.show_worst = show_worst
        self.show_best = show_best
        self.show_final = show_final
        self.dataset = dataset
        self.Exch = exchange
        self.data_tag = data_tag
        self.epoch = 0
        self.history = {}
        self.accuracy_metric = [[] for i in range(len(self.clbck_metrics))]
        self.accuracy_val_metric = [[] for i in range(len(self.clbck_metrics))]
        self.num_classes = num_classes
        self.predict_cls = {}
        self.max_accuracy_value = 0
        self.idx = 0

        self.acls_lst = [
            [[] for i in range(self.num_classes + 1)]
            for i in range(len(self.clbck_metrics))
        ]
        self.predict_cls = (
            {}
        )  # словарь для сбора истории предикта по классам и метрикам
        self.batch_count = 0
        self.x_Val = {}
        self.y_true = []
        self.y_pred = []
        self.loss = ""
        pass

    def plot_result(self, output_key: str = None):
        """
        Returns:
            None:
        """
        plot_data = {}
        msg_epoch = f"Эпоха №{self.epoch + 1:03d}"
        if len(self.clbck_metrics) >= 1:
            for metric_name in self.clbck_metrics:
                if not isinstance(metric_name, str):
                    metric_name = metric_name.__name__
                if len(self.dataset.Y) > 1:
                    # определяем, что демонстрируем во 2м и 3м окне
                    metric_name = f"{output_key}_{metric_name}"
                    val_metric_name = f"val_{metric_name}"
                else:
                    val_metric_name = f"val_{metric_name}"

                metric_title = f"{metric_name} и {val_metric_name} {msg_epoch}"
                xlabel = "эпоха"
                ylabel = f"{metric_name}"
                labels = (metric_title, xlabel, ylabel)
                plot_data[labels] = [
                    [
                        list(range(len(self.history[metric_name]))),
                        self.history[metric_name],
                        f"{metric_name}",
                    ],
                    [
                        list(range(len(self.history[val_metric_name]))),
                        self.history[val_metric_name],
                        f"{val_metric_name}",
                    ],
                ]

            if self.class_metrics:
                for metric_name in self.class_metrics:
                    if not isinstance(metric_name, str):
                        metric_name = metric_name.__name__
                    if len(self.dataset.Y) > 1:
                        metric_name = f'{output_key}_{metric_name}'
                        val_metric_name = f"val_{metric_name}"
                    else:
                        val_metric_name = f"val_{metric_name}"
                    classes_title = f"{val_metric_name} из {self.num_classes} классов. {msg_epoch}"
                    xlabel = "эпоха"
                    ylabel = val_metric_name
                    labels = (classes_title, xlabel, ylabel)
                    plot_data[labels] = [[list(range(len(self.predict_cls[val_metric_name][j]))),
                                          self.predict_cls[val_metric_name][j],
                                          f"{val_metric_name} класс {j}", ] for j in range(self.num_classes)]
            self.Exch.show_plot_data(plot_data)
        pass

    def image_indices(self, count=6) -> np.ndarray:
        """
        Computes indices of images based on instance mode ('worst', 'best')
        Returns: array of best or worst predictions indices
        """
        if "categorical_crossentropy" in self.loss:
            classes = np.argmax(self.y_true, axis=-1)
        elif "sparse_categorical_crossentropy" in self.loss:
            classes = np.reshape(self.y_true, (self.y_true.shape[0]))
        elif "binary_crossentropy" in self.loss:
            classes = np.reshape(self.y_true, (self.y_true.shape[0]))
        else:
            classes = np.reshape(self.y_true, (self.y_true.shape[0]))
        probs = np.array([pred[classes[i]]
                          for i, pred in enumerate(self.y_pred)])
        sorted_args = np.argsort(probs)
        if self.show_best:
            indices = sorted_args[-count:]
        else:
            indices = sorted_args[:count]
        return indices

    def plot_images(self, output_key: str = None):
        """
        Plot images based on indices in dataset
        Returns: None
        """
        img_indices = self.image_indices()

        classes_labels = np.arange(self.num_classes)
        data = []
        if "categorical_crossentropy" in self.loss:
            y_pred = np.argmax(self.y_pred, axis=-1)
            y_true = np.argmax(self.y_true, axis=-1)
        elif "sparse_categorical_crossentropy" in self.loss:
            y_pred = np.argmax(self.y_pred, axis=-1)
            y_true = np.reshape(self.y_true, (self.y_true.shape[0]))
        elif "binary_crossentropy" in self.loss:
            y_pred = np.reshape(self.y_pred, (self.y_pred.shape[0]))
            y_true = np.reshape(self.y_true, (self.y_true.shape[0]))
        else:
            y_pred = np.reshape(self.y_pred, (self.y_pred.shape[0]))
            y_true = np.reshape(self.y_true, (self.y_true.shape[0]))

        for idx in img_indices:
            # TODO нужно как то определять тип входа по тэгу (images)
            image = self.x_Val['input_1'][idx]
            true_idx = y_true[idx]
            pred_idx = y_pred[idx]
            title = f"Выход: {output_key} \n Предикт: {classes_labels[pred_idx]} \n" \
                    f" Истина: {classes_labels[true_idx]}"
            data.append((image, title))
        out_data = {'images': image_to_base64(data)}
        self.Exch.show_image_data(out_data)

    # # Распознаём тестовую выборку и выводим результаты
    # def recognize_classes(self):
    #     y_pred_classes = np.argmax(self.y_pred, axis=-1)
    #     y_true_classes = np.argmax(self.y_true, axis=-1)
    #     classes_accuracy = []
    #     for j in range(self.num_classes + 1):
    #         accuracy_value = 0
    #         y_true_count_sum = 0
    #         y_pred_count_sum = 0
    #         for i in range(self.y_true.shape[0]):
    #             y_true_diff = y_true_classes[i] - j
    #             if not y_true_diff:
    #                 y_pred_count_sum += 1
    #             y_pred_diff = y_pred_classes[i] - j
    #             if not (y_true_diff and y_pred_diff):
    #                 y_true_count_sum += 1
    #             if not y_pred_count_sum:
    #                 accuracy_value = 0
    #             else:
    #                 accuracy_value = y_true_count_sum / y_pred_count_sum
    #         classes_accuracy.append(accuracy_value)
    #     return classes_accuracy

    # Распознаём тестовую выборку и выводим результаты
    def evaluate_accuracy(self):
        y_true = self.y_true
        y_pred = self.y_pred
        metric_classes = []
        if "categorical_crossentropy" in self.loss:
            pred_classes = np.argmax(y_pred, axis=-1)
            true_classes = np.argmax(y_true, axis=-1)
        elif "sparse_categorical_crossentropy" in self.loss:
            pred_classes = np.argmax(y_pred, axis=-1)
            true_classes = np.reshape(y_true, (y_true.shape[0]))
        elif "binary_crossentropy" in self.loss:
            pred_classes = np.reshape(y_pred, (y_pred.shape[0]))
            true_classes = np.reshape(y_true, (y_true.shape[0]))
        else:
            pred_classes = np.reshape(y_pred, (y_pred.shape[0]))
            true_classes = np.reshape(y_true, (y_true.shape[0]))
        for j in range(self.num_classes):
            y_true_count_sum = 0
            y_pred_count_sum = 0
            for i in range(y_true.shape[0]):
                y_true_diff = true_classes[i] - j
                if y_true_diff == 0:
                    y_pred_count_sum += 1
                y_pred_diff = pred_classes[i] - j
                if y_pred_diff == 0 and y_true_diff == 0:
                    y_true_count_sum += 1
            if y_pred_count_sum == 0:
                acc_val = 0
            else:
                acc_val = y_true_count_sum / y_pred_count_sum
            metric_classes.append(acc_val)
        return metric_classes

    def evaluate_f1(self):
        y_true = self.y_true
        y_pred = self.y_pred
        metric_classes = []
        if "categorical_crossentropy" in self.loss:
            pred_classes = np.argmax(y_pred, axis=-1)
            true_classes = np.argmax(y_true, axis=-1)
        elif "sparse_categorical_crossentropy" in self.loss:
            pred_classes = np.argmax(y_pred, axis=-1)
            true_classes = np.reshape(y_true, (y_true.shape[0]))
        elif "binary_crossentropy" in self.loss:
            pred_classes = np.reshape(y_pred, (y_pred.shape[0]))
            true_classes = np.reshape(y_true, (y_true.shape[0]))
        else:
            pred_classes = np.reshape(y_pred, (y_pred.shape[0]))
            true_classes = np.reshape(y_true, (y_true.shape[0]))
        for j in range(self.num_classes):
            tp = 0
            fp = 0
            fn = 0
            for i in range(y_true.shape[0]):
                cross = pred_classes[i][pred_classes[i] == true_classes[i]]
                tp += cross[cross == j].size

                pred_uncross = pred_classes[i][pred_classes[i] != true_classes[i]]
                fp += pred_uncross[pred_uncross == j].size

                true_uncross = true_classes[i][true_classes[i] != pred_classes[i]]
                fn += true_uncross[true_uncross == j].size

            recall = (tp + 1) / (tp + fp + 1)
            precision = (tp + 1) / (tp + fn + 1)
            f1 = 2 * precision * recall / (precision + recall)
            metric_classes.append(f1)
        return metric_classes

    # TODO как разбирать лосс по классам при "sparse_categorical_crossentropy" и "binary_crossentropy"
    def evaluate_loss(self):
        y_true = self.y_true
        y_pred = self.y_pred
        metric_classes = []
        if "categorical_crossentropy" in self.loss:
            cross_entropy = CategoricalCrossentropy()
            for i in range(self.num_classes):
                loss = cross_entropy(y_true[..., i], y_pred[..., i]).numpy()
                metric_classes.append(loss)
        elif "sparse_categorical_crossentropy" in self.loss:
            y_true = tf.keras.utils.to_categorical(self.y_true, num_classes=self.num_classes)
            cross_entropy = CategoricalCrossentropy()
            for i in range(self.num_classes):
                loss = cross_entropy(y_true[..., i], y_pred[..., i]).numpy()
                metric_classes.append(loss)
        elif "binary_crossentropy" in self.loss:
            y_true = tf.keras.utils.to_categorical(self.y_true, num_classes=self.num_classes)
            y_pred = tf.keras.utils.to_categorical(self.y_pred, num_classes=self.num_classes)
            cross_entropy = CategoricalCrossentropy()
            for i in range(self.num_classes):
                loss = cross_entropy(y_true[..., i], y_pred[..., i]).numpy()
                metric_classes.append(loss)
        # else:

        return metric_classes

    def epoch_end(
            self,
            epoch,
            logs: dict = None,
            output_key: str = None,
            y_pred: list = None,
            y_true: dict = None,
            loss: str = None,
            msg_epoch: str = None,
    ):
        """
        Returns:
            {}:
        """
        self.idx = 0
        self.epoch = epoch
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss = loss
        epoch_metric_data = ""
        epoch_val_metric_data = ""
        for metric_idx in range(len(self.clbck_metrics)):
            # # проверяем есть ли метрика заданная функцией
            if not isinstance(self.clbck_metrics[metric_idx], str):
                metric_name = self.clbck_metrics[metric_idx].__name__
                self.clbck_metrics[metric_idx] = metric_name

            if len(self.dataset.Y) > 1:  # or (len(self.clbck_metrics) > 1 and
                # 'loss' not in self.clbck_metrics):
                metric_name = f'{output_key}_{self.clbck_metrics[metric_idx]}'
                val_metric_name = f"val_{metric_name}"
            else:
                metric_name = f"{self.clbck_metrics[metric_idx]}"
                val_metric_name = f"val_{metric_name}"

            # определяем лучшую метрику для вывода данных при class_metrics='best'
            if logs[val_metric_name] > self.max_accuracy_value:
                self.max_accuracy_value = logs[val_metric_name]
            self.idx = metric_idx
            # собираем в словарь по метрикам
            self.accuracy_metric[metric_idx].append(logs[metric_name])
            self.accuracy_val_metric[metric_idx].append(logs[val_metric_name])
            dm = {str(metric_name): self.accuracy_metric[metric_idx]}
            self.history.update(dm)
            dv = {str(val_metric_name): self.accuracy_val_metric[metric_idx]}
            self.history.update(dv)

            epoch_metric_data += (
                f" - {metric_name}: {self.history[metric_name][-1]: .4f}"
            )
            epoch_val_metric_data += (
                f" - {val_metric_name}: {self.history[val_metric_name][-1]: .4f}"
            )

            if self.y_pred is not None:
                # try:
                # # распознаем и выводим результат по классам
                # classes_accuracy = self.recognize_classes(self.y_pred[output_num], self.y_true[output_key])
                # распознаем и выводим результат по классам
                # TODO считаем каждую метрику на каждом выходе
                if metric_name.endswith("accuracy"):
                    metric_classes = self.evaluate_accuracy()
                elif metric_name.endswith('loss'):
                    metric_classes = self.evaluate_loss()
                else:
                    metric_classes = self.evaluate_f1()

                # собираем в словарь по метрикам и классам
                dclsup = {}
                for j in range(self.num_classes):
                    self.acls_lst[metric_idx][j].append(metric_classes[j])
                dcls = {str(val_metric_name): self.acls_lst[metric_idx]}
                dclsup.update(dcls)
                self.predict_cls.update(dclsup)
                # except Exception:
                #     pass

        if self.step:
            if (self.epoch % self.step == 0) and (self.step >= 1):
                self.plot_result(output_key)

        self.Exch.show_text_data(
            f"Эпоха {epoch + 1:03d}, затраченное время: {msg_epoch}, выход: {output_key}, {epoch_metric_data}{epoch_val_metric_data}"
        )
        # return

    def train_end(self, output_key: str = None, x_val: dict = None):
        self.x_Val = x_val
        if self.show_final:
            self.plot_result(output_key)
            if self.data_tag == 'images':
                if self.show_best or self.show_worst:
                    self.plot_images(output_key=output_key)


class SegmentationCallback:
    """Callback for segmentation"""

    def __init__(
            self,
            metrics=[],
            step=1,
            num_classes=2,
            class_metrics=[],
            data_tag=None,
            show_worst=False,
            show_best=False,
            show_final=True,
            dataset=DTS(),
            exchange=Exchange(),
    ):
        """
        Init for classification callback
        Args:
            metrics (list):             список используемых метрик
            class_metrics:              вывод графиков метрик по каждому сегменту
            step int(list):             шаг вывода хода обучения, по умолчанию step = 1
            show_worst bool():          выводить ли справа отдельно, плохие метрики, по умолчанию False
            show_final bool ():         выводить ли в конце обучения график, по умолчанию True
            dataset (trds.DTS):         instance of DTS class
            exchange:                   экземпляр Exchange (для вывода текстовой и графической инф-ии)
        Returns:
            None
        """
        self.__name__ = "Callback for segmentation"
        self.step = step
        self.clbck_metrics = metrics
        self.class_metrics = class_metrics
        self.show_worst = show_worst
        self.show_best = show_best
        self.show_final = show_final
        self.dataset = dataset
        self.Exch = exchange
        self.data_tag = data_tag
        self.epoch = 0
        self.history = {}
        self.accuracy_metric = [[] for i in range(len(self.clbck_metrics))]
        self.accuracy_val_metric = [[] for i in range(len(self.clbck_metrics))]
        self.max_accuracy_value = 0
        self.idx = 0
        self.num_classes = num_classes  # количество классов
        self.acls_lst = [
            [[] for i in range(self.num_classes + 1)]
            for i in range(len(self.clbck_metrics))
        ]
        self.predict_cls = (
            {}
        )  # словарь для сбора истории предикта по классам и метрикам
        self.batch_count = 0
        self.x_Val = {}
        self.y_true = []
        self.y_pred = []
        self.loss = ""
        self.metric_classes = []
        pass

    def plot_result(self, output_key: str = None) -> None:
        """
        Returns:
            None:
        """
        plot_data = {}
        msg_epoch = f"Эпоха №{self.epoch + 1:03d}"
        if len(self.clbck_metrics) >= 1:
            for metric_name in self.clbck_metrics:
                if not isinstance(metric_name, str):
                    metric_name = metric_name.__name__

                if len(self.dataset.Y) > 1:
                    # определяем, что демонстрируем во 2м и 3м окне
                    metric_name = f"{output_key}_{metric_name}"
                    val_metric_name = f"val_{metric_name}"
                else:
                    val_metric_name = f"val_{metric_name}"
                # определяем, что демонстрируем во 2м и 3м окне
                metric_title = f"{metric_name} и {val_metric_name} {msg_epoch}"
                xlabel = "эпох"
                ylabel = f"{metric_name}"
                labels = (metric_title, xlabel, ylabel)
                plot_data[labels] = [
                    [
                        list(range(len(self.history[metric_name]))),
                        self.history[metric_name],
                        f"{metric_name}",
                    ],
                    [
                        list(range(len(self.history[val_metric_name]))),
                        self.history[val_metric_name],
                        f"{val_metric_name}",
                    ],
                ]

            # if self.class_metrics:
            #     for metric_name in self.class_metrics:
            #         if not isinstance(metric_name, str):
            #             metric_name = metric_name.__name__
            #         if len(self.dataset.Y) > 1:
            #             metric_name = f'{output_key}_{metric_name}'
            #             val_metric_name = f"val_{metric_name}"
            #         else:
            #             val_metric_name = f"val_{metric_name}"
            #         classes_title = f"{val_metric_name} of {self.num_classes} classes. {msg_epoch}"
            #         xlabel = "epoch"
            #         ylabel = val_metric_name
            #         labels = (classes_title, xlabel, ylabel)
            #         plot_data[labels] = [[list(range(len(self.predict_cls[val_metric_name][j]))),
            #                               self.predict_cls[val_metric_name][j],
            #                               f"{val_metric_name} class {j}", ] for j in range(self.num_classes)]
            self.Exch.show_plot_data(plot_data)
        pass

    def _get_colored_mask(self, mask, input_key: str = None):
        """
        Transforms prediction mask to colored mask

        Parameters:
        mask : numpy array                 segmentation mask

        Returns:
        colored_mask : numpy array         mask with colors by classes
        """

        def index2color(pix, num_classes, classes_colors):
            index = np.argmax(pix)
            color = []
            for i in range(num_classes):
                if index == i:
                    color = classes_colors[i]
            return color

        colored_mask = []
        mask = mask.reshape(-1, self.num_classes)
        for pix in range(len(mask)):
            colored_mask.append(
                index2color(mask[pix], self.num_classes, self.dataset.classes_colors)
            )
        colored_mask = np.array(colored_mask).astype(np.uint8)
        self.colored_mask = colored_mask.reshape(self.dataset.input_shape[input_key])

    def _dice_coef(self, smooth=1.0):
        """
        Compute dice coefficient for each mask

        Parameters:
        smooth : float     to avoid division by zero

        Returns:
        -------
        None
        """

        intersection = np.sum(self.y_true * self.y_pred, axis=(1, 2, 3))
        union = np.sum(self.y_true, axis=(1, 2, 3)) + np.sum(
            self.y_pred, axis=(1, 2, 3)
        )
        self.dice = (2.0 * intersection + smooth) / (union + smooth)

    def plot_images(self, input_key: str = None):
        """
        Returns:
            None:
        """

        image_data = []
        true_mask_data = []
        pred_mask_data = []

        self._dice_coef()

        # выбираем 5 лучших либо 5 худших результатов сегментации
        if self.show_best:
            indexes = np.argsort(self.dice)[-5:]
        elif self.show_worst:
            indexes = np.argsort(self.dice)[:5]

        for idx in indexes:
            # исходное изобаржение
            image = np.squeeze(
                self.x_Val[input_key][idx].reshape(self.dataset.input_shape[input_key])
            )
            title = "Изображение"
            image_data.append((image, title))

            # истинная маска
            self._get_colored_mask(mask=self.y_true[idx], input_key=input_key)
            image = np.squeeze(self.colored_mask)
            title = "Истинная маска"
            true_mask_data.append((image, title))

            # предсказанная маска
            self._get_colored_mask(mask=self.y_pred[idx], input_key=input_key)
            image = np.squeeze(self.colored_mask)
            title = "Маска предикта"
            pred_mask_data.append((image, title))

        data = {
            'images': image_to_base64(image_data),
            'ground_truth_masks': image_to_base64(true_mask_data),
            'predicted_mask': image_to_base64(pred_mask_data)
        }
        self.Exch.show_image_data(data)

    # Распознаём тестовую выборку и выводим результаты
    def evaluate_accuracy(self, smooth=1.0):
        """
        Compute accuracy for classes

        Parameters:
        smooth : float     to avoid division by zero

        Returns:
        -------
        None
        """
        predsegments = np.argmax(self.y_pred, axis=-1)
        truesegments = np.argmax(self.y_true, axis=-1)
        self.metric_classes = []
        for j in range(self.num_classes):
            summ_val = 0
            for i in range(self.y_true.shape[0]):
                # делаем сегметн класса для сверки
                testsegment = np.ones_like(predsegments[0]) * j
                truezero = np.abs(truesegments[i] - testsegment)
                predzero = np.abs(predsegments[i] - testsegment)
                summ_val += (
                                    testsegment.size - np.count_nonzero(truezero + predzero)
                            ) / (testsegment.size - np.count_nonzero(predzero) + smooth)
            acc_val = summ_val / self.y_true.shape[0]
            self.metric_classes.append(acc_val)

    def evaluate_dice_coef(self, input_key: str = "input_1", smooth=1.0):
        """
        Compute dice coefficient for classes

        Parameters:
        smooth : float     to avoid division by zero

        Returns:
        -------
        None
        """
        # TODO сделать для нескольких входов
        if self.dataset.tags[input_key] == "images":
            axis = (1, 2)
        elif self.dataset.tags[input_key] == "text":
            axis = 1
        intersection = np.sum(self.y_true * self.y_pred, axis=axis)
        union = np.sum(self.y_true, axis=axis) + np.sum(self.y_pred, axis=axis)
        dice = np.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)
        self.metric_classes = dice

    def evaluate_loss(self):
        """
        Compute loss for classes

        Returns:
        -------
        None
        """
        self.metric_classes = []
        bce = BinaryCrossentropy()
        for i in range(self.num_classes):
            loss = bce(self.y_true[..., i], self.y_pred[..., i]).numpy()
            self.metric_classes.append(loss)

    def epoch_end(
            self,
            epoch: int = None,
            logs: dict = None,
            output_key: str = None,
            y_pred: list = None,
            y_true: dict = None,
            loss: str = None,
            msg_epoch: str = None,
    ):
        """
        Returns:
            {}:
        """
        self.epoch = epoch
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss = loss
        epoch_metric_data = ""
        epoch_val_metric_data = ""
        self.idx = 0
        for metric_idx in range(len(self.clbck_metrics)):
            # проверяем есть ли метрика заданная функцией
            if not isinstance(self.clbck_metrics[metric_idx], str):
                metric_name = self.clbck_metrics[metric_idx].__name__
                self.clbck_metrics[metric_idx] = metric_name
            if len(self.dataset.Y) > 1:  # or (len(self.clbck_metrics) > 1 and 'loss' not in self.clbck_metrics):
                metric_name = f'{output_key}_{self.clbck_metrics[metric_idx]}'
                val_metric_name = f"val_{metric_name}"
            else:
                metric_name = f"{self.clbck_metrics[metric_idx]}"
                val_metric_name = f"val_{metric_name}"

            if logs[val_metric_name] > self.max_accuracy_value:
                self.max_accuracy_value = logs[val_metric_name]
                self.idx = metric_idx
            # собираем в словарь по метрикам
            self.accuracy_metric[metric_idx].append(logs[metric_name])
            self.accuracy_val_metric[metric_idx].append(logs[val_metric_name])
            dm = {str(metric_name): self.accuracy_metric[metric_idx]}
            self.history.update(dm)
            dv = {str(val_metric_name): self.accuracy_val_metric[metric_idx]}
            self.history.update(dv)

            epoch_metric_data += (
                f" - {metric_name}: {self.history[metric_name][-1]: .4f}"
            )
            epoch_val_metric_data += (
                f" - {val_metric_name}: {self.history[val_metric_name][-1]: .4f}"
            )

            if self.y_pred is not None:
                # TODO Добавить другие варианты по используемым метрикам
                # вычисляем результат по классам
                if metric_name.endswith("accuracy"):
                    self.evaluate_accuracy()
                elif metric_name.endswith("dice_coef"):
                    self.evaluate_dice_coef(input_key="input_1")
                elif metric_name.endswith("loss"):
                    self.evaluate_loss()
                else:
                    print(f"Выбранная метрика {metric_name} не поддерживается для вычислений")
                # собираем в словарь по метрикам и классам
                dclsup = {}
                for j in range(self.num_classes):
                    self.acls_lst[metric_idx][j].append(self.metric_classes[j])
                dcls = {val_metric_name: self.acls_lst}
                dclsup.update(dcls)
                self.predict_cls.update(dclsup)

        if self.step > 0:
            if self.epoch % self.step == 0:
                self.plot_result(output_key=output_key)

        self.Exch.show_text_data(
            f"Эпоха {epoch + 1:03d}, затраченное время: {msg_epoch}, выход: {output_key}, {epoch_metric_data}{epoch_val_metric_data}"
        )

    def train_end(self, output_key: str = None, x_val: dict = None):
        self.x_Val = x_val
        if self.show_final:
            self.plot_result(output_key=output_key)
            if self.data_tag == 'images':
                if self.show_best or self.show_worst:
                    self.plot_images(input_key="input_1")
        pass


class TimeseriesCallback:
    def __init__(
            self,
            metrics=None,
            step=1,
            corr_step=50,
            show_final=True,
            plot_pred_and_true=True,
            dataset=DTS(),
            exchange=Exchange(),
    ):
        """
        Init for timeseries callback
        Args:
            metrics (list):             список используемых метрик (по умолчанию clbck_metrics = list()), что соответсвует 'loss'
            step int():                 шаг вывода хода обучения, по умолчанию step = 1
            show_final (bool):          выводить ли в конце обучения график, по умолчанию True
            plot_pred_and_true (bool):  выводить ли графики реальных и предсказанных рядов
            dataset (DTS):              экземпляр класса DTS
            corr_step (int):            количество шагов для отображения корреляции (при <= 0 не отображается)
        Returns:
            None
        """
        self.__name__ = "Callback for timeseries"
        if metrics is None:
            metrics = ["loss"]
        self.metrics = metrics
        self.step = step
        self.show_final = show_final
        self.plot_pred_and_true = plot_pred_and_true
        self.dataset = dataset
        self.Exch = exchange
        self.corr_step = corr_step
        self.epoch = 0
        self.x_Val = {}
        self.y_true = []
        self.y_pred = []
        self.loss = ""

        self.losses = (
            self.metrics if "loss" in self.metrics else self.metrics + ["loss"]
        )
        self.met = [[] for _ in range(len(self.losses))]
        self.valmet = [[] for _ in range(len(self.losses))]
        self.history = {}
        self.predicts = {}

    def plot_result(self, output_key=None):
        for i in range(len(self.losses)):
            # проверяем есть ли метрика заданная функцией
            if type(self.losses[i]) == types.FunctionType:
                metric_name = self.losses[i].__name__
                self.losses[i] = metric_name
            if len(self.dataset.Y) > 1:  # or (len(self.losses) > 1 and 'loss' not in self.losses):
                showmet = f'{output_key}_{self.losses[i]}'
                vshowmet = f"val_{showmet}"
            else:
                showmet = f'{self.losses[i]}'
                vshowmet = f"val_{showmet}"
            # showmet = self.losses[self.idx]
            # vshowmet = f"val_{showmet}"
            epochcomment = f" epoch {self.epoch + 1}"
            loss_len = len(self.history[showmet])
            data = {}

            # loss_title = (f"loss and val_loss {epochcomment}", "epochs", f"{showmet}")
            # data.update(
            #     {
            #         loss_title: [
            #             [range(loss_len), self.history["loss"], "loss"],
            #             [range(loss_len), self.history["val_loss"], "val_loss"],
            #         ]
            #     }
            # )

            metric_title = (
                f"{showmet} metric = {showmet} and {vshowmet}{epochcomment}",
                "epochs",
                f"{showmet}",
            )
            data.update(
                {
                    metric_title: [
                        [list(range(loss_len)), self.history[showmet], showmet],
                        [list(range(loss_len)), self.history[vshowmet], vshowmet],
                    ]
                }
            )

            if self.plot_pred_and_true:
                y_true, y_pred = self.predicts[vshowmet]
                pred_title = ("Predictions", "steps", f"{showmet}")
                data.update(
                    {
                        pred_title: [
                            [list(range(len(y_true))), y_true, "Actual"],
                            [list(range(len(y_pred))), y_pred, "Prediction"],
                        ]
                    }
                )
            self.Exch.show_plot_data(data)

    @staticmethod
    def autocorr(a, b):
        ma = a.mean()
        mb = b.mean()
        mab = (a * b).mean()
        sa = a.std()
        sb = b.std()
        corr = 1
        if (sa > 0) & (sb > 0):
            corr = (mab - ma * mb) / (sa * sb)
        return corr

    @staticmethod
    def collect_correlation_data(y_pred, y_true, channel, corr_steps=10):
        corr_list = []
        autocorr_list = []
        yLen = y_true.shape[0]
        for i in range(corr_steps):
            corr_list.append(
                TimeseriesCallback.autocorr(
                    y_true[: yLen - i, channel], y_pred[i:, channel]
                )
            )
            autocorr_list.append(
                TimeseriesCallback.autocorr(
                    y_true[: yLen - i, channel], y_true[i:, channel]
                )
            )
        corr_label = f"Предсказание на {corr_steps + 1} шаг"
        autocorr_label = "Эталон"
        title = ("Автокорреляция", '', '')
        correlation_data = {
            title: [
                [list(range(corr_steps)), corr_list, corr_label],
                [list(range(corr_steps)), autocorr_list, autocorr_label],
            ]
        }
        return correlation_data

    def epoch_end(
            self,
            epoch: int = None,
            logs: dict = None,
            output_key: str = None,
            y_pred: list = None,
            y_true: dict = None,
            loss: str = None,
            msg_epoch: str = None,
    ):
        self.epoch = epoch
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss = loss
        epoch_metric_data = ""
        epoch_val_metric_data = ""
        for i in range(len(self.losses)):
            # проверяем есть ли метрика заданная функцией
            if type(self.losses[i]) == types.FunctionType:
                metric_name = self.losses[i].__name__
                self.losses[i] = metric_name
            if len(self.dataset.Y) > 1:  # or (len(self.losses) > 1 and 'loss' not in self.losses):
                metric_name = f'{output_key}_{self.losses[i]}'
                val_metric_name = f"val_{metric_name}"
            else:
                metric_name = f'{self.losses[i]}'
                val_metric_name = f"val_{metric_name}"

            # собираем в словарь по метрикам
            self.met[i].append(logs[metric_name])
            self.valmet[i].append(logs[val_metric_name])
            self.history[metric_name] = self.met[i]
            self.history[val_metric_name] = self.valmet[i]

            if self.y_pred is not None:
                self.predicts[val_metric_name] = (y_true, y_pred)
                self.vmet_name = val_metric_name

            epoch_metric_data += (
                f" - {metric_name}: {self.history[metric_name][-1]: .4f}"
            )
            epoch_val_metric_data += (
                f" - {val_metric_name}: {self.history[val_metric_name][-1]: .4f}"
            )

        if self.step:
            if (self.epoch % self.step == 0) and (self.step >= 1):
                self.comment = f" epoch {epoch + 1}"
                self.idx = 0
                self.plot_result(output_key=output_key)
        self.Exch.show_text_data(
            f"Эпоха {epoch + 1:03d}, затраченное время: {msg_epoch}, выход: {output_key}, {epoch_metric_data}{epoch_val_metric_data}"
        )

    def train_end(self, output_key: str = None, x_val: dict = None):
        self.x_Val = x_val
        if self.show_final:
            self.comment = f"on {self.epoch + 1} epochs"
            self.idx = 0
            self.plot_result(output_key=output_key)
        if self.corr_step > 0:
            y_true, y_pred = self.predicts[self.vmet_name]
            corr_data = TimeseriesCallback.collect_correlation_data(
                y_pred, y_true, 0, self.corr_step
            )
            # Plot correlation and autocorrelation graphics
            self.Exch.show_plot_data(corr_data)


class RegressionCallback:
    def __init__(
            self,
            metrics,
            step=1,
            show_final=True,
            plot_scatter=False,
            dataset=DTS(),
            exchange=Exchange(),
    ):
        """
        Init for regression callback
        Args:
            metrics (list):         список используемых метрик
            step int():             шаг вывода хода обучения, по умолчанию step = 1
            plot_scatter (bool):    вывод граыика скатера по умолчанию True
            show_final (bool):      выводить ли в конце обучения график, по умолчанию True
            exchange:               экземпляр Exchange (для вывода текстовой и графической инф-ии)
        Returns:
            None
        """
        self.__name__ = "Callback for regression"
        if metrics is None:
            metrics = ["loss"]
        self.step = step
        self.metrics = metrics
        self.show_final = show_final
        self.plot_scatter = plot_scatter
        self.dataset = dataset
        self.exchange = exchange
        self.epoch = 0
        self.x_Val = {}
        self.y_true = []
        self.y_pred = []
        self.loss = ''
        self.max_accuracy_value = 0

        self.losses = (
            self.metrics if "loss" in self.metrics else self.metrics + ["loss"]
        )
        self.met = [[] for _ in range(len(self.losses))]
        self.valmet = [[] for _ in range(len(self.losses))]
        self.history = {}
        self.predicts = {}

    def plot_result(self, output_key=None):
        for i in range(len(self.losses)):
            # проверяем есть ли метрика заданная функцией
            if type(self.losses[i]) == types.FunctionType:
                metric_name = self.losses[i].__name__
                self.losses[i] = metric_name
            if len(self.dataset.Y) > 1:
                showmet = f'{output_key}_{self.losses[i]}'
                vshowmet = f"val_{showmet}"
            else:
                showmet = f'{self.losses[i]}'
                vshowmet = f"val_{showmet}"
            # showmet = self.losses[self.idx]
            # vshowmet = f"val_{showmet}"
            epochcomment = f" epoch {self.epoch + 1}"
            loss_len = len(self.history[showmet])
            data = {}

            # loss_title = f"loss and val_loss{epochcomment}"
            # xlabel = "epoch"
            # ylabel = f"{showmet}"
            # key = (loss_title, xlabel, ylabel)
            # value = [
            #     [range(loss_len), self.history[showmet], showmet],
            #     [range(loss_len), self.history[vshowmet], vshowmet],
            # ]
            # data.update({key: value})

            metric_title = f"{showmet} metric = {showmet} and {vshowmet}{epochcomment}"
            xlabel = "epoch"
            ylabel = f"{showmet}"
            key = (metric_title, xlabel, ylabel)
            value = [
                (list(range(loss_len)), self.history[showmet], showmet),
                (list(range(loss_len)), self.history[vshowmet], vshowmet),
            ]
            data.update({key: value})
            self.exchange.show_plot_data(data)

        if self.plot_scatter:
            data = {}
            scatter_title = "Scatter"
            xlabel = "True values"
            ylabel = "Predictions"
            y_true, y_pred = self.predicts[vshowmet]
            key = (scatter_title, xlabel, ylabel)
            value = [(y_true.reshape(-1), y_pred.reshape(-1), "Regression")]
            data.update({key: value})
            self.exchange.show_scatter_data(data)

        pass

    def epoch_end(
            self,
            epoch: int = None,
            logs: dict = None,
            output_key: str = None,
            y_pred: list = None,
            y_true: dict = None,
            loss: str = None,
            msg_epoch: str = None,
    ):

        self.epoch = epoch
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss = loss
        epoch_metric_data = ""
        epoch_val_metric_data = ""
        for i in range(len(self.losses)):
            # проверяем есть ли метрика заданная функцией
            if type(self.losses[i]) == types.FunctionType:
                metric_name = self.losses[i].__name__
                self.losses[i] = metric_name
            if len(self.dataset.Y) > 1:  # or (len(self.losses) > 1 and 'loss' not in self.losses):
                metric_name = f'{output_key}_{self.losses[i]}'
                val_metric_name = f"val_{metric_name}"
            else:
                metric_name = f'{self.losses[i]}'
                val_metric_name = f"val_{metric_name}"
            # собираем в словарь по метрикам
            self.met[i].append(logs[metric_name])
            self.valmet[i].append(logs[val_metric_name])
            self.history[metric_name] = self.met[i]
            self.history[val_metric_name] = self.valmet[i]

            if self.y_pred is not None:
                self.predicts[val_metric_name] = (y_true, y_pred)

            epoch_metric_data += (
                f" - {metric_name}: {self.history[metric_name][-1]: .4f}"
            )
            epoch_val_metric_data += (
                f" - {val_metric_name}: {self.history[val_metric_name][-1]: .4f}"
            )

        if self.step > 0:
            if self.epoch % self.step == 0:
                self.comment = f" epoch {epoch + 1}"
                self.idx = 0
                self.plot_result(output_key=output_key)

        self.exchange.show_text_data(
            f"Эпоха {epoch + 1:03d}, затраченное время: {msg_epoch}, выход: {output_key}, {epoch_metric_data}{epoch_val_metric_data}"
        )
        pass

    def train_end(self, output_key: str = None, x_val: dict = None):
        self.x_Val = x_val
        if self.show_final:
            self.comment = f"on {self.epoch + 1} epochs"
            self.idx = 0
            self.plot_result(output_key=output_key)
        pass


def image_to_base64(image_as_array):
    output = []

    for i, (image, title) in enumerate(image_as_array):
        if image.dtype == 'int32':
            image = image.astype(np.uint8)
        temp_image = tempfile.NamedTemporaryFile(prefix='image_', suffix='tmp.png', delete=False)
        try:
            plt.imsave(temp_image.name, image, cmap='Greys')
        except Exception as e:
            print(e.__str__())
            plt.imsave(temp_image.name, image.reshape(image.shape[:-1]), cmap='gray')
        with open(temp_image.name, 'rb') as img:
            output_image = base64.b64encode(img.read())
        output.append({'image': output_image, 'title': title})
        temp_image.close()
        os.remove(temp_image.name)
    return output
