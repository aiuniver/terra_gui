import base64
import copy
import os
import tempfile
import colorsys
import random
from PIL import Image, ImageDraw, ImageFont  # Модули работы с изображениями

import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
import numpy as np
import types
import time
from terra_ai.guiexchange import Exchange

from terra_ai.training.data import task_type_defaults_dict

__version__ = 0.13


class BaseCallback():
    """Callback for callbacks"""

    def __init__(
            self,
            metrics=None,
            step=1,
            class_metrics=None,
            data_tag=None,
            num_classes=2,
            show_worst=False,
            show_best=True,
            show_final=True,
            dataset=DTS(),
            exchange=Exchange(),
    ):
        """
        Init for base callback
        Args:
            metrics (list):             список используемых метрик: по умолчанию [], что соответсвует 'loss'
            class_metrics:              вывод графиков метрик по каждому сегменту: по умолчанию []
            step int():                 шаг вывода хода обучения, по умолчанию step = 1
            show_worst (bool):          выводить ли экземпляры, худшие по метрикам, по умолчанию False
            show_best (bool):           выводить ли экземпляры, лучшие по метрикам, по умолчанию False
            show_final (bool):          выводить ли в конце обучения график, по умолчанию True
            dataset (DTS):              экземпляр класса DTS
            exchange (Exchange)         экземпляр класса Exchange  (для вывода текстовой и графической инф-ии)
        Returns:
            None
        """
        self.step = step
        if metrics is None:
            metrics = []
        if class_metrics is None:
            class_metrics = []
        if data_tag is None:
            data_tag = []
        self.clbck_metrics = metrics
        self.class_metrics = class_metrics
        self.exchange = exchange
        self.dataset = dataset
        self.show_final = show_final
        self.show_best = show_best
        self.show_worst = show_worst
        self.num_classes = num_classes
        self.data_tag = data_tag
        self.epoch = 0
        self.history = {}
        self.predict_cls = {}
        self.max_accuracy_value = 0
        self.predict_cls = (
            {}
        )  # словарь для сбора истории предикта по классам и метрикам
        self.batch_count = 0
        self.x_Val = {}
        self.y_true = []
        self.y_pred = []
        self.loss = ""
        self.dice = []

    def plot_result(self, output_key: str = None):
        """
        Returns: plot_data
        """
        plot_data = {}
        try:

            msg_epoch = f"Эпоха №{self.epoch + 1:03d}"
            if len(self.clbck_metrics) >= 1:
                for metric_name in self.clbck_metrics:
                    if not isinstance(metric_name, str):
                        metric_name = metric_name.name
                    if len(self.dataset.Y) > 1:
                        # определяем, что демонстрируем во 2м и 3м окне
                        metric_name = f"{output_key}_{metric_name}"
                        val_metric_name = f"val_{metric_name}"
                    else:
                        val_metric_name = f"val_{metric_name}"
                    metric_title = f"{metric_name} и {val_metric_name} {msg_epoch}"
                    xlabel = "Эпоха"
                    ylabel = "Значение"
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

                if len(self.class_metrics):
                    for metric_name in self.class_metrics:
                        if not isinstance(metric_name, str):
                            metric_name = metric_name.name
                        if len(self.dataset.Y) > 1:
                            metric_name = f'{output_key}_{metric_name}'
                            val_metric_name = f"val_{metric_name}"
                        else:
                            val_metric_name = f"val_{metric_name}"
                        xlabel = "Эпоха"
                        if metric_name.endswith("loss"):
                            classes_title = f"Ошибка {output_key} для {self.num_classes} классов. {msg_epoch}"
                            ylabel = "Значение"
                        else:
                            classes_title = f"Точность {output_key} для {self.num_classes} классов. {msg_epoch}"
                            ylabel = "Значение"
                        labels = (classes_title, xlabel, ylabel)
                        plot_data[labels] = [
                            [
                                list(range(len(self.predict_cls[val_metric_name][j]))),
                                self.predict_cls[val_metric_name][j],
                                f" класс {l}", ] for j, l in
                            enumerate(self.dataset.classes_names[output_key])]
        except Exception as e:
            print("Exception", e.__str__())
        return plot_data

    def image_indices(self, count=10, output_key: str = None) -> np.ndarray:
        """
        Computes indices of images based on instance mode ('worst', 'best', "random")
        Returns: array of best or worst predictions indices
        """
        if (self.y_pred.shape[-1] == self.y_true.shape[-1]) \
                and (self.dataset.one_hot_encoding[output_key]) \
                and (self.y_true.shape[-1] > 1):
            classes = np.argmax(self.y_true, axis=-1)
        elif (len(self.y_true.shape) == 1) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_pred.shape[-1] > 1):
            classes = copy.copy(self.y_true)
        elif (len(self.y_true.shape) == 1) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_pred.shape[-1] == 1):
            classes = copy.deepcopy(self.y_true)
        else:
            classes = copy.deepcopy(self.y_true)

        probs = np.array([pred[classes[i]]
                          for i, pred in enumerate(self.y_pred)])
        sorted_args = np.argsort(probs)
        if self.show_best:
            indices = sorted_args[-count:]
        elif self.show_worst:
            indices = sorted_args[:count]
        else:
            indices = np.random.choice(len(probs), count, replace=False)
        return indices, probs

    def plot_images(self, input_key: str = 'input_1', output_key: str = None):
        """
        Plot images based on indices in dataset
        Returns:
            images
        """
        images = {"images": []}
        if self.show_best:
            img_title = "лучшее по метрике: "
        elif self.show_worst:
            img_title = "худшее по метрике: "
        else:
            img_title = "случайное: "
        img_indices, probs = self.image_indices(output_key=output_key)
        classes_labels = self.dataset.classes_names[output_key]
        if (self.y_pred.shape[-1] == self.y_true.shape[-1]) \
                and (self.dataset.one_hot_encoding[output_key]) \
                and (self.y_true.shape[-1] > 1):
            y_pred = np.argmax(self.y_pred, axis=-1)
            y_true = np.argmax(self.y_true, axis=-1)
        elif (len(self.y_true.shape) == 1) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_pred.shape[-1] > 1):
            y_pred = np.argmax(self.y_pred, axis=-1)
            y_true = copy.deepcopy(self.y_true)
        elif (len(self.y_true.shape) == 1) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_pred.shape[-1] == 1):
            y_pred = np.reshape(self.y_pred, (self.y_pred.shape[0]))
            y_true = copy.deepcopy(self.y_true)
        else:
            y_pred = np.reshape(self.y_pred, (self.y_pred.shape[0]))
            y_true = copy.deepcopy(self.y_true)

        for idx in img_indices:
            # TODO нужно как то определять тип входа по тэгу (images)
            image = self.x_Val[input_key][idx]
            true_idx = y_true[idx]
            pred_idx = y_pred[idx]

            # исходное изобаржение
            image = self.inverse_scaler(image, input_key)
            image_data = {
                "image": self.image_to_base64(image),
                "title": f"{img_title + str(round(probs[idx], 4))}",
                "info": [
                    {
                        "label": "Выход",
                        "value": output_key,
                    },
                    {
                        "label": "Распознано",
                        "value": classes_labels[pred_idx],
                    },
                    {
                        "label": "Верный ответ",
                        "value": classes_labels[true_idx],
                    }
                ]
            }
            images["images"].append(image_data)

        return images

    def evaluate_accuracy(self, output_key: str = None):
        """
        Compute accuracy for classes

        Returns:
            metric_classes
        """
        metric_classes = []
        if (self.y_pred.shape[-1] == self.y_true.shape[-1]) \
                and (self.dataset.one_hot_encoding[output_key]) \
                and (self.y_true.shape[-1] > 1):
            pred_classes = np.argmax(self.y_pred, axis=-1)
            true_classes = np.argmax(self.y_true, axis=-1)
        elif (len(self.y_true.shape) == 1) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_pred.shape[-1] > 1):
            pred_classes = np.argmax(self.y_pred, axis=-1)
            true_classes = copy.deepcopy(self.y_true)
        elif (len(self.y_true.shape) == 1) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_pred.shape[-1] == 1):
            pred_classes = np.reshape(self.y_pred, (self.y_pred.shape[0]))
            true_classes = copy.deepcopy(self.y_true)
        else:
            pred_classes = np.reshape(self.y_pred, (self.y_pred.shape[0]))
            true_classes = copy.deepcopy(self.y_true)
        for j in range(self.num_classes):
            y_true_count_sum = 0
            y_pred_count_sum = 0
            for i in range(self.y_true.shape[0]):
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

    def evaluate_f1(self, output_key: str = None):
        metric_classes = []

        if (self.y_pred.shape[-1] == self.y_true.shape[-1]) \
                and (self.dataset.one_hot_encoding[output_key]) \
                and (self.y_true.shape[-1] > 1):
            pred_classes = np.argmax(self.y_pred, axis=-1)
            true_classes = np.argmax(self.y_true, axis=-1)
        elif (len(self.y_true.shape) == 1) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_pred.shape[-1] > 1):
            pred_classes = np.argmax(self.y_pred, axis=-1)
            true_classes = copy.deepcopy(self.y_true)
        elif (len(self.y_true.shape) == 1) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_pred.shape[-1] == 1):
            pred_classes = np.reshape(self.y_pred, (self.y_pred.shape[0]))
            true_classes = copy.deepcopy(self.y_true)
        else:
            pred_classes = np.reshape(self.y_pred, (self.y_pred.shape[0]))
            true_classes = copy.deepcopy(self.y_true)
        for j in range(self.num_classes):
            tp = 0
            fp = 0
            fn = 0
            for i in range(self.y_true.shape[0]):
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

    def evaluate_loss(self, output_key: str = None):
        """
        Compute loss for classes

        Returns:
            metric_classes
        """
        metric_classes = []

        if (self.y_pred.shape[-1] == self.y_true.shape[-1]) \
                and (self.dataset.one_hot_encoding[output_key]) \
                and (self.y_true.shape[-1] > 1):
            cross_entropy = CategoricalCrossentropy()
            for i in range(self.num_classes):
                loss = cross_entropy(self.y_true[..., i], self.y_pred[..., i]).numpy()
                metric_classes.append(loss)
        elif (len(self.y_true.shape) == 1) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_pred.shape[-1] > 1):
            y_true = tf.keras.utils.to_categorical(self.y_true, num_classes=self.num_classes)
            cross_entropy = CategoricalCrossentropy()
            for i in range(self.num_classes):
                loss = cross_entropy(y_true[..., i], self.y_pred[..., i]).numpy()
                metric_classes.append(loss)
        elif (len(self.y_true.shape) == 1) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_pred.shape[-1] == 1):
            y_true = tf.keras.utils.to_categorical(self.y_true, num_classes=self.num_classes)
            y_pred = tf.keras.utils.to_categorical(np.reshape(self.y_pred, (self.y_pred.shape[0])),
                                                   num_classes=self.num_classes)
            cross_entropy = CategoricalCrossentropy()
            for i in range(self.num_classes):
                loss = cross_entropy(y_true[..., i], y_pred[..., i]).numpy()
                metric_classes.append(loss)
        else:
            bce = BinaryCrossentropy()
            for i in range(self.num_classes):
                loss = bce(self.y_true[..., i], self.y_pred[..., i]).numpy()
                metric_classes.append(loss)

        return metric_classes

    def plot_text(self, input_key: str = 'input_1', output_key: str = None):
        """
        Plot sample text based on indices in dataset
        Returns:
            images
        """
        text = {"text": []}

        indices, probs = self.image_indices(output_key=output_key)
        if self.show_best:
            text_title = "лучшее по метрике: "
        elif self.show_worst:
            text_title = "худшее по метрике: "
        else:
            text_title = "случайное: "
        classes_labels = self.dataset.classes_names.get(output_key)
        if self.dataset.task_type.get(output_key) != "segmentation":
            if (self.y_pred.shape[-1] == self.y_true.shape[-1]) \
                    and (self.dataset.one_hot_encoding[output_key]) \
                    and (self.y_true.shape[-1] > 1):
                y_pred = np.argmax(self.y_pred, axis=-1)
                y_true = np.argmax(self.y_true, axis=-1)
            elif (len(self.y_true.shape) == 1) \
                    and (not self.dataset.one_hot_encoding[output_key]) \
                    and (self.y_pred.shape[-1] > 1):
                y_pred = np.argmax(self.y_pred, axis=-1)
                y_true = copy.deepcopy(self.y_true)
            elif (len(self.y_true.shape) == 1) \
                    and (not self.dataset.one_hot_encoding[output_key]) \
                    and (self.y_pred.shape[-1] == 1):
                y_pred = np.reshape(self.y_pred, (self.y_pred.shape[0]))
                y_true = copy.deepcopy(self.y_true)
            else:
                y_pred = np.reshape(self.y_pred, (self.y_pred.shape[0]))
                y_true = copy.deepcopy(self.y_true)

            for idx in indices:
                # TODO нужно как то определять тип входа по тэгу (images)
                sample = self.x_Val[input_key][idx]
                true_idx = y_true[idx]
                pred_idx = y_pred[idx]

                # исходный формат примера
                sample = self.inverse_scaler(sample, input_key)
                text_data = {
                    "text": self.dataset.inverse_data(input_key, sample),
                    "title": f"{text_title + str(round(probs[idx], 4))}",
                    "info": [
                        {
                            "label": "Выход",
                            "value": output_key,
                        },
                        {
                            "label": "Распознано",
                            "value": classes_labels[pred_idx],
                        },
                        {
                            "label": "Верный ответ",
                            "value": classes_labels[true_idx],
                        }
                    ]
                }
                text["text"].append(text_data)

        else:
            y_pred = copy.deepcopy(self.y_pred)
            y_true = copy.deepcopy(self.y_true)

            #######################
            # Функция, выводящая точность распознавания каждой категории отдельно
            #######################
            def recognizeSet(tagI, pred, tags, length, value, num_classes):
                total = 0

                for j in range(num_classes):  # общее количество тегов
                    correct = 0
                    for i in range(len(tagI)):  # проходимся по каждому списку списка тегов
                        for k in range(length):  # проходимся по каждому тегу
                            if tagI[i][k][j] == (pred[i][k][j] > value).astype(
                                    int):  # если соответствующие индексы совпадают, значит сеть распознала верно
                                correct += 1
                    print("Сеть распознала категорию '{}' на {}%".format(tags[j], 100 * correct / (len(tagI) * length)))
                    total += 100 * correct / (len(tagI) * length)
                print("средняя точность {}%".format(total / num_classes))

            recognizeSet(y_true, y_pred, classes_labels, y_true.shape[1], 0.999,
                         self.dataset.num_classes.get(output_key))
        return text

    # def image_to_gif(image_directory: pathlib.Path, frames_per_second: float, **kwargs):
    #     """
    #     import imageio
    #     import pathlib
    #     from datetime import datetime
    #
    #     Makes a .gif which shows many images at a given frame rate.
    #     All images should be in order (don't know how this works) in the image directory
    #
    #     Only tested with .png images but may work with others.
    #
    #     :param image_directory:
    #     :type image_directory: pathlib.Path
    #     :param frames_per_second:
    #     :type frames_per_second: float
    #     :param kwargs: image_type='png' or other
    #     :return: nothing
    #     Example:
    #         fps = 5
    #         png_dir = pathlib.Path('/content/valid')
    #         image_to_gif(png_dir, fps)
    #     """
    #     assert isinstance(image_directory, pathlib.Path), "input must be a pathlib object"
    #     image_type = kwargs.get('type', 'jpg')
    #
    #     timestampStr = datetime.now().strftime("%y%m%d_%H%M%S")
    #     gif_dir = image_directory.joinpath(timestampStr + "_GIF.gif")
    #
    #     print('Started making GIF')
    #     print('Please wait... ')
    #
    #     images = []
    #     for file_name in image_directory.glob('*.' + image_type):
    #         images.append(imageio.imread(image_directory.joinpath(file_name)))
    #     imageio.mimsave(gif_dir.as_posix(), images, fps=frames_per_second)
    #
    #     print('Finished making GIF!')
    #     print('GIF can be found at: ' + gif_dir.as_posix())

    # # build gif
    # with imageio.get_writer('mygif.gif', mode='I') as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)

    def image_to_base64(self, image_as_array):
        if image_as_array.dtype != 'uint8':
            image_as_array = image_as_array.astype(np.uint8)
        temp_image = tempfile.NamedTemporaryFile(prefix='image_', suffix='tmp.png', delete=False)
        try:
            plt.imsave(temp_image.name, image_as_array, cmap='Greys')
        except Exception as e:
            plt.imsave(temp_image.name, image_as_array.reshape(image_as_array.shape[:-1]), cmap='gray')
        with open(temp_image.name, 'rb') as img:
            output_image = base64.b64encode(img.read()).decode('utf-8')
        temp_image.close()
        os.remove(temp_image.name)
        return output_image

    def inverse_scaler(self, x, key):
        x_shape = x.shape
        if self.dataset.scaler.get(key) is not None:
            if len(x_shape) > 2:
                x_inverse = self.dataset.scaler.get(key).inverse_transform(x.reshape((-1, 1)))
                x_inverse = x_inverse.reshape(x_shape)
            else:
                x_inverse = self.dataset.scaler.get(key).inverse_transform(x)
            return x_inverse
        else:
            return x


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
            model_name: str = "noname",
    ):

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
        self.last_epoch = 0
        self.batch = 0
        self.num_batches = 0
        self.msg_epoch = ""
        self.y_pred = []
        self.epoch = 0
        self.history = {}
        self._start_time = time.time()
        self._time_batch_step = time.time()
        self._time_first_step = time.time()
        self._sum_time = 0
        self.out_table_data = {}
        self.stop_training = False
        self.retrain_flag = False
        self.stop_flag = False
        self.retrain_epochs = 0

        self.callback_kwargs = []
        self.clbck_object = []
        self.prepare_params()

    def save_lastmodel(self) -> None:
        """
        Saving last model on each epoch end

        Returns:
            None
        """
        model_name = f"model_{self.nn_name}_on_epoch_end.last.h5"
        file_path_model: str = os.path.join(
            self.save_model_path, f"{model_name}"
        )
        self.model.save(file_path_model)
        self.Exch.print_2status_bar(
            ("Инфо", f"Последняя модель сохранена как {file_path_model}")
        )
        pass

    def prepare_callbacks(
            self, task_type: str = "", metrics: list = None, num_classes: int = None,
            clbck_options: dict = {}, tags: dict = {}):
        """
        if terra in raw mode  - setting callback if its set
        if terra with django - checking switches and set callback options from switches

        Returns:
            initialized_callback
        """
        _task_type_defaults_kwargs = self.task_type_defaults_dict.get(task_type)
        callback_kwargs = _task_type_defaults_kwargs["callback_kwargs"]
        if metrics:
            callback_kwargs["metrics"] = copy.deepcopy(metrics)
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

        if (task_type == "classification") or (task_type == "segmentation") or (task_type == "object_detection"):
            callback_kwargs["num_classes"] = copy.deepcopy(num_classes)
            if tags["input_1"]:
                callback_kwargs["data_tag"] = tags["input_1"]
            callback_kwargs["class_metrics"] = []
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
                elif option_name == "show_worst_images":
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
        print("callback_kwargs", callback_kwargs)
        clbck_object = _task_type_defaults_kwargs['clbck_object']
        initialized_callback = clbck_object(**callback_kwargs)
        return initialized_callback

    def prepare_params(self):
        print("self.clbck_params", self.clbck_params)
        print("self.DTS.tags", self.DTS.tags)
        for _key in self.clbck_params.keys():
            if (_key == 'output_1' and self.clbck_params[_key]["task"] == 'object_detection') \
                    or (self.clbck_params[_key]["task"] != 'object_detection'):
                self.metrics.append(self.clbck_params[_key]["metrics"])
                self.loss.append(self.clbck_params[_key]["loss"])
                self.task_name.append(self.clbck_params[_key]["task"])
                self.num_classes.append(self.clbck_params.setdefault(_key)["num_classes"])
                self.y_Scaler.append(self.DTS.scaler.setdefault(_key))
                self.tokenizer.append(self.DTS.tokenizer.setdefault(_key))
                self.one_hot_encoding.append(self.DTS.one_hot_encoding.setdefault(_key))
                initialized_callback = self.prepare_callbacks(
                    task_type=self.clbck_params[_key]["task"].value,
                    metrics=self.clbck_params[_key]["metrics"],
                    num_classes=self.clbck_params.setdefault(_key)["num_classes"],
                    clbck_options=self.clbck_params[_key]["callbacks"],
                    tags=self.DTS.tags,
                )
                self.callbacks.append(initialized_callback)

    def _estimate_step(self, current, start, now):
        if current:
            _time_per_unit = (now - start) / current
        else:
            _time_per_unit = (now - start)
        return _time_per_unit

    def eta_format(self, eta):
        if eta > 3600:
            eta_format = '%d ч %02d мин %02d сек' % (eta // 3600,
                                                     (eta % 3600) // 60, eta % 60)
        elif eta > 60:
            eta_format = '%d мин %02d сек' % (eta // 60, eta % 60)
        else:
            eta_format = '%d сек' % eta

        info = ' %s' % eta_format
        return info

    def update_progress(self, target, current, start_time, finalize=False, stop_current=0, stop_flag=False):
        """
        Updates the progress bar.
        """
        if finalize:
            _now_time = time.time()
            eta = _now_time - start_time
        else:
            _now_time = time.time()
            if stop_flag:
                time_per_unit = self._estimate_step(stop_current, start_time, _now_time)
            else:
                time_per_unit = self._estimate_step(current, start_time, _now_time)

            eta = time_per_unit * (target - current)
        info = self.eta_format(eta)

        return [info, int(eta)]

    def predict_yolo(self, x):
        out1 = self.model.get_layer(name='output_1').output
        out2 = self.model.get_layer(name='output_2').output
        out3 = self.model.get_layer(name='output_3').output
        model = keras.models.Model(self.model.input, [out1, out2, out3])
        return model.predict(x, batch_size=self.batch_size)

    def on_train_begin(self, logs=None):
        # self.model.stop_training = False
        self.stop_training = False
        self._start_time = time.time()
        if not self.stop_flag:
            self.batch = 0
        self.num_batches = self.DTS.X['input_1']['data'][0].shape[0] // self.batch_size

        self.Exch.show_current_epoch(self.last_epoch)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self._time_first_step = time.time()

    def on_train_batch_end(self, batch, logs=None):
        stop = self.Exch.get_stop_training_flag()
        if stop:
            self.model.stop_training = True
            self.stop_training = True
            self.stop_flag = True
            msg = f'ожидайте остановку...'
            self.batch += 1
            self.Exch.print_2status_bar(('Обучение остановлено пользователем', msg))
        else:
            msg_batch = f'Батч {batch}/{self.num_batches}'
            msg_epoch = f'Эпоха {self.last_epoch + 1}/{self.epochs}:' \
                        f'{self.update_progress(self.num_batches, batch, self._time_first_step)[0]}, '
            time_start = \
                self.update_progress(self.num_batches * self.epochs + 1, self.batch, self._start_time, finalize=True)[1]
            if self.retrain_flag:
                msg_progress_end = f'Расчетное время окончания:' \
                                   f'{self.update_progress(self.num_batches * self.retrain_epochs + 1, self.batch, self._start_time)[0]}, '
                msg_progress_start = f'Время выполнения дообучения:' \
                                     f'{self.eta_format(time_start)}, '
            elif self.stop_flag:
                msg_progress_end = f'Расчетное время окончания после остановки:' \
                                   f'{self.update_progress(self.num_batches * self.epochs + 1, self.batch, self._start_time, stop_current=batch, stop_flag=True)[0]}'
                msg_progress_start = f'Время выполнения:' \
                                     f'{self.eta_format(self._sum_time + time_start)}, '
            else:
                msg_progress_end = f'Расчетное время окончания:' \
                                   f'{self.update_progress(self.num_batches * self.epochs + 1, self.batch, self._start_time)[0]}, '
                msg_progress_start = f'Время выполнения:' \
                                     f'{self.eta_format(time_start)}, '
            self.batch += 1
            self.Exch.print_2status_bar(('Прогресс обучения', msg_progress_start +
                                         msg_progress_end + msg_epoch + msg_batch))

    def on_epoch_end(self, epoch, logs=None):
        """
        Returns:
            {}:
        """
        out_table_data = {
            "epoch": {
                "number": self.last_epoch + 1,
                "time": self.update_progress(self.num_batches, self.batch, self._time_first_step, finalize=True)[1],
                "data": {},
            },
            "summary": "",
        }
        out_plots_data = {}
        if self.x_Val.get("input_1") is not None:
            if self.clbck_params['output_1']["task"] == 'object_detection':
                self.y_pred = self.predict_yolo(self.x_Val)
            else:
                self.y_pred = self.model.predict(self.x_Val, batch_size=self.batch_size)
        else:
            self.y_pred = copy.copy(self.y_true)
        if isinstance(self.y_pred, list):
            for i, output_key in enumerate(self.clbck_params.keys()):
                if output_key == 'output_1' and self.clbck_params[output_key]["task"] == 'object_detection':
                    callback_out_data = self.callbacks[i].epoch_end(
                        self.last_epoch,
                        logs=logs,
                        output_key=output_key,
                        y_pred=self.y_pred,
                        y_true=self.y_true,
                        loss=self.loss[i],
                        msg_epoch=self.msg_epoch
                    )
                elif self.clbck_params[output_key]["task"] != 'object_detection':
                    callback_out_data = self.callbacks[i].epoch_end(
                        self.last_epoch,
                        logs=logs,
                        output_key=output_key,
                        y_pred=self.y_pred[i],
                        y_true=self.y_true.get(output_key),
                        loss=self.loss[i],
                        msg_epoch=self.msg_epoch
                    )
                else:
                    callback_out_data = {}
                if len(callback_out_data) != 0:
                    for key in callback_out_data.keys():
                        if key == "table":
                            out_table_data["epoch"]["data"].update({output_key: callback_out_data[key][output_key]})
                        elif key == "plots":
                            out_plots_data.update(callback_out_data[key])

        else:
            for i, output_key in enumerate(self.clbck_params.keys()):
                callback_out_data = self.callbacks[i].epoch_end(
                    self.last_epoch,
                    logs=logs,
                    output_key=output_key,
                    y_pred=self.y_pred,
                    y_true=self.y_true.get(output_key),
                    loss=self.loss[i],
                    msg_epoch=self.msg_epoch
                )
                if len(callback_out_data) != 0:
                    for key in callback_out_data.keys():
                        if key == "table":
                            out_table_data["epoch"]["data"].update({output_key: callback_out_data[key][output_key]})
                        elif key == "plots":
                            out_plots_data.update(callback_out_data[key])

        self.Exch.show_current_epoch(self.last_epoch)
        self.last_epoch += 1
        if len(out_table_data) != 0:
            self.Exch.show_text_data(out_table_data)
        if len(out_plots_data) != 0:
            self.Exch.show_plot_data(out_plots_data)
        self.save_lastmodel()

    def on_train_end(self, logs=None):
        out_table_data = {
            "epoch": {},
            "summary": "",
        }
        out_plots_data = {}
        out_images_data = {"images": {
            "title": "Исходное изображение",
            "values": []
        },
            "ground_truth_masks": {
                "title": "Маска сегментации",
                "values": []
            },
            "predicted_mask": {
                "title": "Результат работы модели",
                "values": []
            },
            "ground_truth_bbox": {
                "title": "Правильный bbox",
                "values": []
            },
            "predicted_bbox": {
                "title": "Результат работы модели",
                "values": []
            },
        }
        for i, output_key in enumerate(self.clbck_params.keys()):
            if (output_key == 'output_1' and self.clbck_params[output_key]["task"] == 'object_detection') \
                    or (self.clbck_params[output_key]["task"] != 'object_detection'):
                callback_out_data = self.callbacks[i].train_end(output_key=output_key, x_val=self.x_Val)
                if len(callback_out_data) != 0:
                    for key in callback_out_data.keys():
                        if key == "plots":
                            out_plots_data.update(callback_out_data[key])
                        elif key == "images":
                            for im_key in callback_out_data[key].keys():
                                if im_key == "images":
                                    out_images_data["images"]["values"].extend(callback_out_data[key][im_key])
                                elif im_key == "ground_truth_masks":
                                    out_images_data["ground_truth_masks"]["values"].extend(
                                        callback_out_data[key][im_key])
                                elif im_key == "predicted_mask":
                                    out_images_data["predicted_mask"]["values"].extend(
                                        callback_out_data[key][im_key])
                                elif im_key == "ground_truth_bbox":
                                    out_images_data["ground_truth_bbox"]["values"].extend(
                                        callback_out_data[key][im_key])
                                elif im_key == "predicted_bbox":
                                    out_images_data["predicted_bbox"]["values"].extend(
                                        callback_out_data[key][im_key])
        self.save_lastmodel()
        time_end = self.update_progress(self.num_batches * self.epochs + 1,
                                        self.batch, self._start_time, finalize=True)[1]
        self._sum_time += time_end
        if self.model.stop_training:
            out_table_data["summary"] = f'Затрачено времени на обучение: ' \
                                        f'{self.eta_format(self._sum_time)} '
            self.Exch.show_text_data(self.out_table_data)
            msg = f'Модель сохранена.'
            self.Exch.print_2status_bar(('Обучение остановлено пользователем!', msg))
            self.Exch.out_data['stop_flag'] = True
        else:
            if self.retrain_flag:
                out_table_data["summary"] = f'Затрачено времени на обучение: ' \
                                            f'{self.eta_format(time_end)} '
            else:
                out_table_data["summary"] = f'Затрачено времени на обучение: ' \
                                            f'{self.eta_format(self._sum_time)} '
            self.Exch.show_text_data(out_table_data)
        if (len(out_images_data["images"]["values"]) or len(out_images_data["ground_truth_bbox"]["values"])) != 0:
            self.Exch.show_image_data(out_images_data)
        if len(out_plots_data) != 0:
            self.Exch.show_plot_data(out_plots_data)


class ClassificationCallback(BaseCallback):
    """Callback for classification"""

    def __init__(
            self,
            metrics=None,
            step=1,
            class_metrics=None,
            data_tag=None,
            num_classes=2,
            show_worst=False,
            show_best=True,
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
        super().__init__()
        if class_metrics is None:
            class_metrics = []
        if metrics is None:
            metrics = []
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

        self.accuracy_metric = [[] for i in range(len(self.clbck_metrics))]
        self.accuracy_val_metric = [[] for i in range(len(self.clbck_metrics))]
        self.num_classes = num_classes
        self.acls_lst = [
            [[] for i in range(self.num_classes + 1)]
            for i in range(len(self.clbck_metrics))
        ]

        pass

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
        self.epoch = epoch
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss = loss
        epoch_table_data = {
            output_key: {}
        }
        out_data = {}
        for metric_idx in range(len(self.clbck_metrics)):
            # # проверяем есть ли метрика заданная функцией
            if not isinstance(self.clbck_metrics[metric_idx], str):
                metric_name = self.clbck_metrics[metric_idx].name
                self.clbck_metrics[metric_idx] = metric_name

            if len(self.dataset.Y) > 1:
                metric_name = f'{output_key}_{self.clbck_metrics[metric_idx]}'
                val_metric_name = f"val_{metric_name}"
            else:
                metric_name = f"{self.clbck_metrics[metric_idx]}"
                val_metric_name = f"val_{metric_name}"
            # определяем лучшую метрику для вывода данных при class_metrics='best'
            if logs[val_metric_name] > self.max_accuracy_value:
                self.max_accuracy_value = logs[val_metric_name]
            # собираем в словарь по метрикам
            self.accuracy_metric[metric_idx].append(logs[metric_name])
            self.accuracy_val_metric[metric_idx].append(logs[val_metric_name])
            dm = {metric_name: self.accuracy_metric[metric_idx]}
            self.history.update(dm)
            dv = {val_metric_name: self.accuracy_val_metric[metric_idx]}
            self.history.update(dv)

            epoch_table_data[output_key].update({metric_name: self.history[metric_name][-1]})
            epoch_table_data[output_key].update({val_metric_name: self.history[val_metric_name][-1]})
            out_data.update({"table": epoch_table_data})

            if self.y_pred is not None:
                # распознаем и выводим результат по классам
                # TODO считаем каждую метрику на каждом выходе
                if metric_name.endswith("accuracy"):
                    metric_classes = self.evaluate_accuracy(output_key=output_key)
                elif metric_name.endswith('loss'):
                    metric_classes = self.evaluate_loss(output_key=output_key)
                else:
                    metric_classes = self.evaluate_f1(output_key=output_key)

                # собираем в словарь по метрикам и классам
                if len(metric_classes):
                    dclsup = {}
                    for j in range(self.num_classes):
                        self.acls_lst[metric_idx][j].append(metric_classes[j])
                    dcls = {val_metric_name: self.acls_lst[metric_idx]}
                    dclsup.update(dcls)
                    self.predict_cls.update(dclsup)
        if self.step:
            if (self.epoch % self.step == 0) and (self.step >= 1):
                plot_data = self.plot_result(output_key)
                out_data.update({"plots": plot_data})

        return out_data

    def train_end(self, output_key: str = None, x_val: dict = None):
        self.x_Val = x_val
        out_data = {}
        if self.show_final:
            plot_data = self.plot_result(output_key)
            out_data.update({"plots": plot_data})
            if self.data_tag == 'images':
                if self.show_best or self.show_worst:
                    images = self.plot_images(output_key=output_key)
                    out_data.update({"images": images})
            elif self.data_tag == 'text':
                text = self.plot_text(output_key=output_key)
                print("text", text)

        return out_data


class SegmentationCallback(BaseCallback):
    """Callback for segmentation"""

    def __init__(
            self,
            metrics=None,
            step=1,
            num_classes=2,
            class_metrics=None,
            data_tag=None,
            show_worst=False,
            show_best=True,
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
        super().__init__()
        if class_metrics is None:
            class_metrics = []
        if metrics is None:
            metrics = []
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
        self.accuracy_metric = [[] for i in range(len(self.clbck_metrics))]
        self.accuracy_val_metric = [[] for i in range(len(self.clbck_metrics))]
        self.num_classes = num_classes  # количество классов
        self.acls_lst = [
            [[] for i in range(self.num_classes + 1)]
            for i in range(len(self.clbck_metrics))
        ]

        pass

    def _get_colored_mask(self, mask, input_key: str = None, output_key: str = None):
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
                index2color(mask[pix], self.num_classes, self.dataset.classes_colors[output_key])
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

    def image_indices(self, count=5, output_key: str = None) -> np.ndarray:
        """
        Computes indices of mask based on instance mode ('worst', 'best', 'random')
        Returns: array of best or worst predictions indices
        """
        self._dice_coef()
        # выбираем count лучших либо худших или случайных результатов сегментации
        if self.show_best:
            indices = np.argsort(self.dice)[-count:]
        elif self.show_worst:
            indices = np.argsort(self.dice)[:count]
        else:
            indices = np.random.choice(len(self.dice), count, replace=False)

        return indices

    def plot_images(self, input_key: str = None, output_key: str = None):
        """
        Returns:
            images
        """
        images = {"images": [],
                  "ground_truth_masks": [],
                  "predicted_mask": [],
                  }
        if self.show_best:
            img_title = "лучшее по метрике: "
        elif self.show_worst:
            img_title = "худшее по метрике: "
        else:
            img_title = "случайное: "
        indexes = self.image_indices(output_key=output_key)

        for idx in indexes:
            # исходное изобаржение
            image_data = {
                "image": None,
                "title": None,
                "info": [
                    {
                        "label": "Выход",
                        "value": output_key,
                    }
                ]
            }

            image = np.squeeze(
                self.x_Val[input_key][idx].reshape(self.dataset.input_shape[input_key])
            )
            image = self.inverse_scaler(image, input_key)
            image_data["image"] = self.image_to_base64(image)
            images["images"].append(image_data)

            # истинная маска
            truth_masks_data = {
                "image": None,
                "title": None,
                "info": [
                    {
                        "label": "Выход",
                        "value": output_key,
                    }
                ]
            }
            self._get_colored_mask(mask=self.y_true[idx], input_key=input_key, output_key=output_key)
            image = np.squeeze(self.colored_mask)

            truth_masks_data["image"] = self.image_to_base64(image)
            images["ground_truth_masks"].append(truth_masks_data)

            # предсказанная маска
            predicted_mask_data = {
                "image": None,
                "title": f'{img_title + str(round(self.dice[idx], 4))}',
                "info": [
                    {
                        "label": "Выход",
                        "value": output_key,
                    }
                ]
            }
            self._get_colored_mask(mask=self.y_pred[idx], input_key=input_key, output_key=output_key)
            image = np.squeeze(self.colored_mask)
            predicted_mask_data["image"] = self.image_to_base64(image)
            images["predicted_mask"].append(predicted_mask_data)

        return images

    def evaluate_accuracy(self, smooth=1.0, output_key: str = None):
        """
        Compute accuracy for classes

        Parameters:
        smooth : float     to avoid division by zero

        Returns:
            metric_classes
        """
        metric_classes = []
        if (self.y_pred.shape[-1] == self.y_true.shape[-1]) \
                and (self.dataset.one_hot_encoding[output_key]) \
                and (self.y_true.shape[-1] > 1):
            predsegments = np.argmax(self.y_pred, axis=-1)
            truesegments = np.argmax(self.y_true, axis=-1)
        elif (self.y_pred.shape[-1] > self.y_true.shape[-1]) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_true.shape[-1] == 1):
            predsegments = np.argmax(self.y_pred, axis=-1)
            truesegments = np.reshape(self.y_true, (self.y_true.shape[0]))
        elif (self.y_pred.shape[-1] == self.y_true.shape[-1]) \
                and (not self.dataset.one_hot_encoding[output_key]) \
                and (self.y_true.shape[-1] == 1):
            predsegments = np.reshape(self.y_pred, (self.y_pred.shape[0]))
            truesegments = np.reshape(self.y_true, (self.y_true.shape[0]))
        else:
            predsegments = np.reshape(self.y_pred, (self.y_pred.shape[0]))
            truesegments = np.reshape(self.y_true, (self.y_true.shape[0]))

        for j in range(self.num_classes):
            summ_val = 0
            for i in range(self.y_true.shape[0]):
                # делаем сегмент класса для сверки
                testsegment = np.ones_like(predsegments[0]) * j
                truezero = np.abs(truesegments[i] - testsegment)
                predzero = np.abs(predsegments[i] - testsegment)
                summ_val += (
                                    testsegment.size - np.count_nonzero(truezero + predzero)
                            ) / (testsegment.size - np.count_nonzero(predzero) + smooth)
            acc_val = summ_val / self.y_true.shape[0]
            metric_classes.append(acc_val)

        return metric_classes

    def evaluate_dice_coef(self, input_key: str = "input_1", smooth=1.0):
        """
        Compute dice coefficient for classes

        Parameters:
        smooth : float     to avoid division by zero

        Returns:
            dice
        """
        # TODO сделать для нескольких входов
        if self.dataset.tags[input_key] == "images":
            axis = (1, 2)
        elif self.dataset.tags[input_key] == "text":
            axis = 1
        intersection = np.sum(self.y_true * self.y_pred, axis=axis)
        union = np.sum(self.y_true, axis=axis) + np.sum(self.y_pred, axis=axis)
        dice = np.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)

        return dice

        # def evaluate_mean_io_u(self, input_key: str = "input_1", smooth=1.0):
        #     """
        #     Compute dice coefficient for classes
        #
        #     Parameters:
        #     smooth : float     to avoid division by zero
        #
        #     Returns:
        #     -------
        #     None
        #     """
        #     # TODO сделать для нескольких входов
        #     if self.dataset.tags[input_key] == "images":
        #         axis = (1, 2)
        #     elif self.dataset.tags[input_key] == "text":
        #         axis = 1
        #     intersection = np.sum(self.y_true * self.y_pred, axis=axis)
        #     union = np.sum(self.y_true, axis=axis) + np.sum(self.y_pred, axis=axis)
        #     dice = np.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)
        #     self.metric_classes = dice

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
        epoch_table_data = {
            output_key: {}
        }
        out_data = {}
        try:
            for metric_idx in range(len(self.clbck_metrics)):
                # проверяем есть ли метрика заданная функцией
                if not isinstance(self.clbck_metrics[metric_idx], str):
                    metric_name = self.clbck_metrics[metric_idx].name
                    self.clbck_metrics[metric_idx] = metric_name
                if len(self.dataset.Y) > 1:
                    metric_name = f'{output_key}_{self.clbck_metrics[metric_idx]}'
                    val_metric_name = f"val_{metric_name}"
                else:
                    metric_name = f"{self.clbck_metrics[metric_idx]}"
                    val_metric_name = f"val_{metric_name}"

                if logs[val_metric_name] > self.max_accuracy_value:
                    self.max_accuracy_value = logs[val_metric_name]

                # собираем в словарь по метрикам
                self.accuracy_metric[metric_idx].append(logs[metric_name])
                self.accuracy_val_metric[metric_idx].append(logs[val_metric_name])
                dm = {str(metric_name): self.accuracy_metric[metric_idx]}
                self.history.update(dm)
                dv = {str(val_metric_name): self.accuracy_val_metric[metric_idx]}
                self.history.update(dv)

                epoch_table_data[output_key].update({metric_name: self.history[metric_name][-1]})
                epoch_table_data[output_key].update({val_metric_name: self.history[val_metric_name][-1]})
                out_data.update({"table": epoch_table_data})

                if self.y_pred is not None:
                    # TODO Добавить другие варианты по используемым метрикам
                    # вычисляем результат по классам
                    if metric_name.endswith("accuracy"):
                        metric_classes = self.evaluate_accuracy(output_key=output_key)
                    elif metric_name.endswith("dice_coef"):
                        metric_classes = self.evaluate_dice_coef(input_key="input_1")
                    elif metric_name.endswith("loss"):
                        metric_classes = self.evaluate_loss(output_key=output_key)
                    else:
                        metric_classes = self.evaluate_f1(output_key=output_key)
                    # собираем в словарь по метрикам и классам
                    if len(metric_classes):
                        dclsup = {}
                        for j in range(self.num_classes):
                            self.acls_lst[metric_idx][j].append(metric_classes[j])
                        dcls = {val_metric_name: self.acls_lst[metric_idx]}
                        dclsup.update(dcls)
                        self.predict_cls.update(dclsup)
        except Exception as e:
            print("Exception epoch_end", e.__str__())

        if self.step > 0:
            if self.epoch % self.step == 0:
                plot_data = self.plot_result(output_key=output_key)
                out_data.update({"plots": plot_data})

        return out_data

    def train_end(self, output_key: str = None, x_val: dict = None, input_key="input_1"):
        self.x_Val = x_val
        out_data = {}
        if self.show_final:

            plot_data = self.plot_result(output_key=output_key)
            out_data.update({"plots": plot_data})

            if self.data_tag == 'images':
                if self.show_best or self.show_worst:
                    images = self.plot_images(input_key=input_key, output_key=output_key)
                    out_data.update({"images": images})

            elif self.data_tag == 'text':

                try:
                    text = self.plot_text(output_key=output_key)
                    print("text", text)
                except Exception as e:
                    print("Exception train_end", e.__str__())

        return out_data


class TimeseriesCallback(BaseCallback):
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
        super().__init__()
        self.__name__ = "Callback for timeseries"
        if metrics is None:
            metrics = []
        self.metrics = metrics
        self.step = step
        self.show_final = show_final
        self.plot_pred_and_true = plot_pred_and_true
        self.dataset = dataset
        self.Exch = exchange
        self.corr_step = corr_step

        self.losses = (
            self.metrics if "loss" in self.metrics else self.metrics + ["loss"]
        )
        self.met = [[] for _ in range(len(self.losses))]
        self.valmet = [[] for _ in range(len(self.losses))]

    def plot_result(self, input_key: str = "input_1", output_key: str = None):
        """
        Returns: plot_data
        """
        plot_data = {}
        for i in range(len(self.losses)):
            # проверяем есть ли метрика заданная функцией
            if not isinstance(self.losses[i], str):
                metric_name = self.losses[i].name
                self.losses[i] = metric_name
            if len(self.dataset.Y) > 1:
                showmet = f'{output_key}_{self.losses[i]}'
                vshowmet = f"val_{showmet}"
            else:
                showmet = f'{self.losses[i]}'
                vshowmet = f"val_{showmet}"
            epochcomment = f" эпоха {self.epoch + 1}"
            loss_len = len(self.history[showmet])

            metric_title = (
                f"{showmet} и {vshowmet}{epochcomment}",
                "эпоха",
                f"{showmet}",
            )
            plot_data.update(
                {
                    metric_title: [
                        [list(range(loss_len)), self.history[showmet], showmet],
                        [list(range(loss_len)), self.history[vshowmet], vshowmet],
                    ]
                }
            )

        if self.plot_pred_and_true and self.y_true is not None:
            x_val = self.inverse_scaler(self.dataset.X.get(input_key)['data'][1], input_key)
            print('x_val[0]', x_val[0].shape)
            y_true = self.inverse_scaler(self.y_true, output_key)
            y_pred = self.inverse_scaler(self.y_pred, output_key)
            if len(y_true.shape) == 2:
                for ch in range(y_true.shape[1]):
                    pred_title = (
                        f"Предикт канала {self.dataset.classes_names.get(output_key)[ch]}", "Время", "Значение")
                    plot_data.update(
                        {
                            pred_title: [
                                [list(range(len(x_val[0]))), x_val[0][:, 0], "Исходный"],
                                [list(range(len(y_true))), y_true[:60, ch], "Истина"],
                                [list(range(len(y_pred))), y_pred[60:, ch], "Предикт"],
                            ]
                        }
                    )

        return plot_data

    @staticmethod
    def autocorr(a, b):
        ma = a.mean()
        mb = b.mean()
        mab = (a * b).mean()
        sa = a.std()
        sb = b.std()
        corr = 0
        if (sa > 0) & (sb > 0):
            corr = (mab - ma * mb) / (sa * sb)
        return corr

    @staticmethod
    def collect_correlation_data(y_pred, y_true, channel, ch_label, corr_steps=10):
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
        corr_label = f"Предсказание на {corr_steps} шаг"
        autocorr_label = "Эталон"
        title = (f"Автокорреляция канала {ch_label[channel]}", '', '')
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
        epoch_table_data = {
            output_key: {}
        }
        out_data = {}
        for i in range(len(self.losses)):
            # проверяем есть ли метрика заданная функцией
            if type(self.losses[i]) == types.FunctionType:
                metric_name = self.losses[i].name
                self.losses[i] = metric_name
            if len(self.dataset.Y) > 1:
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

            epoch_table_data[output_key].update({metric_name: self.history[metric_name][-1]})
            epoch_table_data[output_key].update({val_metric_name: self.history[val_metric_name][-1]})
            out_data.update({"table": epoch_table_data})

        if self.step:
            if (self.epoch % self.step == 0) and (self.step >= 1):
                plot_data = self.plot_result(output_key=output_key)
                out_data.update({"plots": plot_data})

        return out_data

    def train_end(self, output_key: str = None, x_val: dict = None):
        self.x_Val = x_val
        out_data = {}
        if self.show_final and self.y_true is not None:
            plot_data = self.plot_result(output_key=output_key)
            out_data.update({"plots": plot_data})
            # Plot correlation and autocorrelation graphics
            if self.corr_step > 0:
                if len(self.y_true.shape) == 2:
                    for ch in range(self.y_true.shape[1]):
                        corr_data = TimeseriesCallback.collect_correlation_data(
                            self.y_pred, self.y_true, ch, self.dataset.classes_names.get(output_key), self.corr_step
                        )
                        out_data["plots"].update(corr_data)

        return out_data


class RegressionCallback(BaseCallback):
    def __init__(
            self,
            metrics=None,
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
        super().__init__()
        self.__name__ = "Callback for regression"
        if metrics is None:
            metrics = []
        self.step = step
        self.metrics = metrics
        self.show_final = show_final
        self.plot_scatter = plot_scatter
        self.dataset = dataset
        self.Exch = exchange
        self.losses = (
            self.metrics if "loss" in self.metrics else self.metrics + ["loss"]
        )
        self.met = [[] for _ in range(len(self.losses))]
        self.valmet = [[] for _ in range(len(self.losses))]

    def plot_result(self, output_key=None):
        """
        Returns: plot_data
        """
        plot_data = {}
        for i in range(len(self.losses)):
            # проверяем есть ли метрика заданная функцией
            if not isinstance(self.losses[i], str):
                metric_name = self.losses[i].name
                self.losses[i] = metric_name
            if len(self.dataset.Y) > 1:
                showmet = f'{output_key}_{self.losses[i]}'
                vshowmet = f"val_{showmet}"
            else:
                showmet = f'{self.losses[i]}'
                vshowmet = f"val_{showmet}"
            epochcomment = f" эпоха {self.epoch + 1}"
            loss_len = len(self.history[showmet])

            metric_title = f"метрика: {showmet} и {vshowmet}{epochcomment}"
            xlabel = "эпоха"
            ylabel = f"{showmet}"
            labels = (metric_title, xlabel, ylabel)
            plot_data[labels] = [
                (list(range(loss_len)), self.history[showmet], showmet),
                (list(range(loss_len)), self.history[vshowmet], vshowmet),
            ]

        if self.plot_scatter and self.y_true is not None:
            y_true = self.inverse_scaler(self.y_true, output_key)
            y_pred = self.inverse_scaler(self.y_pred, output_key)
            data = {}
            scatter_title = "Scatter"
            xlabel = "Истинные значения"
            ylabel = "Предсказанные значения"
            key = (scatter_title, xlabel, ylabel)
            value = [(y_true.reshape(-1), y_pred.reshape(-1), "Регрессия")]
            data.update({key: value})
            self.Exch.show_scatter_data(data)

        return plot_data

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
        epoch_table_data = {
            output_key: {}
        }
        out_data = {}
        for i in range(len(self.losses)):
            # проверяем есть ли метрика заданная функцией
            if type(self.losses[i]) == types.FunctionType:
                metric_name = self.losses[i].name
                self.losses[i] = metric_name
            if len(self.dataset.Y) > 1:
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

            epoch_table_data[output_key].update({metric_name: self.history[metric_name][-1]})
            epoch_table_data[output_key].update({val_metric_name: self.history[val_metric_name][-1]})
            out_data.update({"table": epoch_table_data})

        if self.step > 0:
            if self.epoch % self.step == 0:
                plot_data = self.plot_result(output_key=output_key)
                out_data.update({"plots": plot_data})

        return out_data

    def train_end(self, output_key: str = None, x_val: dict = None):
        self.x_Val = x_val
        out_data = {}
        if self.show_final:
            plot_data = self.plot_result(output_key=output_key)
            out_data.update({"plots": plot_data})

        return out_data


class ObjectdetectionCallback(BaseCallback):
    """Callback for object_detection"""

    def __init__(
            self,
            metrics=None,
            step=1,
            class_metrics=None,
            data_tag=None,
            num_classes=2,
            show_worst=False,
            show_best=True,
            show_final=True,
            dataset=DTS(),
            exchange=Exchange(),
    ):
        """
        Init for ObjectdetectionCallback callback
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
        super().__init__()
        if class_metrics is None:
            class_metrics = []
        if metrics is None:
            metrics = []
        self.__name__ = "Callback for object_detection"
        self.step = step
        self.clbck_metrics = metrics
        self.class_metrics = class_metrics
        self.show_worst = show_worst
        self.show_best = show_best
        self.show_final = show_final
        self.dataset = dataset
        self.Exch = exchange
        self.data_tag = data_tag

        self.accuracy_metric = [[] for i in range(len(self.clbck_metrics))]
        self.accuracy_val_metric = [[] for i in range(len(self.clbck_metrics))]
        self.num_classes = num_classes
        self.acls_lst = [
            [[] for i in range(self.num_classes + 1)]
            for i in range(len(self.clbck_metrics))
        ]

        pass

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
        self.epoch = epoch
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss = loss
        epoch_table_data = {
            output_key: {}
        }
        out_data = {}
        for metric_idx in range(len(self.clbck_metrics)):
            # # проверяем есть ли метрика заданная функцией
            if not isinstance(self.clbck_metrics[metric_idx], str):
                metric_name = self.clbck_metrics[metric_idx].name
                self.clbck_metrics[metric_idx] = metric_name

            # if len(self.dataset.Y) > 1:
            #     metric_name = f'{output_key}_{self.clbck_metrics[metric_idx]}'
            #     val_metric_name = f"val_{metric_name}"
            # else:
            metric_name = f"{self.clbck_metrics[metric_idx]}"
            val_metric_name = f"val_{metric_name}"

            # определяем лучшую метрику для вывода данных при class_metrics='best'
            if logs[val_metric_name] > self.max_accuracy_value:
                self.max_accuracy_value = logs[val_metric_name]
            # собираем в словарь по метрикам
            self.accuracy_metric[metric_idx].append(logs[metric_name])
            self.accuracy_val_metric[metric_idx].append(logs[val_metric_name])
            dm = {metric_name: self.accuracy_metric[metric_idx]}
            self.history.update(dm)
            dv = {val_metric_name: self.accuracy_val_metric[metric_idx]}
            self.history.update(dv)

            epoch_table_data[output_key].update({metric_name: self.history[metric_name][-1]})
            epoch_table_data[output_key].update({val_metric_name: self.history[val_metric_name][-1]})
            out_data.update({"table": epoch_table_data})

        if self.step:
            if (self.epoch % self.step == 0) and (self.step >= 1):
                plot_data = self.plot_result(output_key)
                out_data.update({"plots": plot_data})

        return out_data

    def train_end(self, output_key: str = None, x_val: dict = None):
        self.x_Val = x_val
        out_data = {}
        if self.show_final:
            plot_data = self.plot_result(output_key)
            out_data.update({"plots": plot_data})
            print("self.data_tag", self.data_tag)
            if self.data_tag == 'images':
                print("self.show_best or self.show_worst", self.show_best, self.show_worst)
                if self.show_best or self.show_worst:
                    images = self.plot_images(input_key="input_1", output_key=output_key)
                    out_data.update({"images": images})

        return out_data

    def plot_result(self, output_key: str = None):
        """
        Returns: plot_data
        """
        plot_data = {}
        msg_epoch = f"Эпоха №{self.epoch + 1:03d}"
        if len(self.clbck_metrics) >= 1:
            for metric_name in self.clbck_metrics:
                if not isinstance(metric_name, str):
                    metric_name = metric_name.name
                # if len(self.dataset.Y) > 1:
                #     # определяем, что демонстрируем во 2м и 3м окне
                #     metric_name = f"{output_key}_{metric_name}"
                #     val_metric_name = f"val_{metric_name}"
                # else:
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

        return plot_data

    def plot_images(self, input_key: str = None, output_key: str = None):
        """
        Returns:
            images
        """
        images = {"images": [],
                  "ground_truth_bbox": [],
                  "predicted_bbox": [],
                  }
        # self._dice_coef()
        # выбираем 5 лучших либо 5 худших результатов сегментации
        # if self.show_best:
        #     indexes = np.argsort(self.dice)[-5:]
        # elif self.show_worst:
        #     indexes = np.argsort(self.dice)[:5]
        print("self.x_Val.get(input_key).shape", self.x_Val.get(input_key).shape)
        indexes = np.random.choice((self.x_Val.get(input_key).shape[0]) - 1, 5, replace=False)
        print("indexes", indexes)
        for idx in indexes:
            # исходное изобаржение
            original_image = self.x_Val.get(input_key)[idx]
            image_shape = original_image.shape
            if self.dataset.scaler.get(input_key) is not None:
                original_image = self.dataset.scaler.get(input_key).inverse_transform(original_image.reshape((-1, 1)))
                original_image = original_image.reshape(image_shape)
                print("original_image scaler", original_image)
            image_data = {
                "image": None,
                "title": None,
                "info": [
                    {
                        "label": 'Выход',
                        "value": output_key,
                    }
                ]
            }

            # image = np.squeeze(
            #     self.x_Val.get(input_key)[idx])  # .reshape(self.dataset.input_shape.get(input_key)
            image_data["image"] = self.image_to_base64(original_image)
            images["images"].append(image_data)

            # истинная маска
            truth_bbox_data = {
                "image": None,
                "title": None,
                "info": [
                    {
                        "label": None,
                        "value": None,
                    }
                ]
            }
            # self._get_colored_mask(mask=self.y_true[idx], input_key=input_key, output_key=output_key)
            # image = np.squeeze(self.colored_mask)
            #
            # print("self.y_true", self.y_true[0].shape, self.y_true[1].shape, self.y_true[2].shape)
            image = self.get_frame(original_image,
                                   self.y_true, input_key=input_key,
                                   output_key=output_key)
            print("Posle images truth")
            truth_bbox_data["image"] = self.image_to_base64(image)
            images["ground_truth_bbox"].append(truth_bbox_data)

            # предсказанная маска
            predicted_bbox_data = {
                "image": None,
                "title": None,
                "info": [
                    {
                        "label": "Выход",
                        "value": output_key,
                    }
                ]
            }
            print("do self.get_frame")
            image = self.get_frame(original_image,
                                   self.y_pred, input_key=input_key,
                                   output_key=output_key)  # .reshape(self.dataset.input_shape.get(input_key)
            # self._get_colored_mask(mask=self.y_pred[idx], input_key=input_key, output_key=output_key)
            # image = np.squeeze(self.colored_mask)
            predicted_bbox_data["image"] = self.image_to_base64(image)
            # print("predicted_bbox_dat", predicted_bbox_data)
            images["predicted_bbox"].append(predicted_bbox_data)

        return images

    def get_frame(self, original_image, predict, input_key: str = None, output_key: str = None):
        ANCHORS = [[[10, 13], [16, 30], [33, 23]],
                   [[30, 61], [62, 45], [59, 119]],
                   [[116, 90], [156, 198], [373, 326]]]
        input_size = 416
        score_threshold = 0.3
        iou_threshold = 0.45
        pred_bbox = []
        for i in range(len(ANCHORS)):
            pred_bbox.append(self.decode(predict[i], i))
        print("pred_bbox", len(pred_bbox))
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        print("reshape predbox")
        pred_bbox = tf.concat(pred_bbox, axis=0)
        print("concat predbox")
        bboxes = self.postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
        print("bboxes postproces")
        bboxes = self.nms(bboxes, iou_threshold, method='nms')
        print("bboxes nms", bboxes)
        print("self.dataset.classes_names", self.dataset.classes_names)

        image = self.draw_bbox(original_image, bboxes, CLASS=self.dataset.classes_names[output_key])
        # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))
        print("image.shape", image.shape)
        print(image)
        return image

    def video_detection(self, yolo, video_path, output_path, classes, score_threshold=0.3, input_size=416,
                        iou_threshold=0.45, rectangle_colors='', input_key: str = None, output_key: str = None):

        times, times_2 = [], []

        vid = cv2.VideoCapture(video_path)  # detect on video

        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, codec, fps, (width, height))  # output_path must be .mp4

        while True:
            _, frame = vid.read()

            try:
                original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            except:
                break

            # TODO здесь должен быть метод датасета препроцессинга картинки.
            image_data = 0  #
            # image_data = image_preprocess(np.copy(original_frame), [input_size, input_size])
            # image_data = image_data[np.newaxis, ...].astype(np.float32)

            t1 = time.time()

            pred_bbox = yolo.predict(image_data)

            # t1 = time.time()
            # pred_bbox = Yolo.predict(image_data)
            t2 = time.time()

            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)

            bboxes = self.postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
            bboxes = self.nms(bboxes, iou_threshold, method='nms')

            # extract bboxes to boxes (x, y, width, height), scores and names
            boxes, scores, names = [], [], []
            for bbox in bboxes:
                boxes.append(
                    [bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int) - bbox[0].astype(int),
                     bbox[3].astype(int) - bbox[1].astype(int)])
                scores.append(bbox[4])
                names.append(classes[int(bbox[5])])

            t3 = time.time()
            times.append(t2 - t1)
            times_2.append(t3 - t1)

            times = times[-20:]
            times_2 = times_2[-20:]

            ms = sum(times) / len(times) * 1000
            fps = 1000 / ms
            fps2 = 1000 / (sum(times_2) / len(times_2) * 1000)

            image = cv2.putText(image, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (0, 0, 255), 2)

            # draw original yolo detection
            image = self.draw_bbox(image, bboxes, CLASS=classes, show_label=True, rectangle_colors=rectangle_colors)

            print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
            if output_path != '':
                out.write(image)

    @staticmethod
    def decode(conv_output, i=0):
        ANCHORS = [[[10, 13], [16, 30], [33, 23]],
                   [[30, 61], [62, 45], [59, 119]],
                   [[116, 90], [156, 198], [373, 326]]]
        STRIDES = [32, 16, 8]
        # where i = 0, 1 or 2 to correspond to the three grid scales
        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]

        # conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]  # offset of center position
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]  # Prediction box length and width offset
        conv_raw_conf = conv_output[:, :, :, :, 4:5]  # confidence of the prediction box
        conv_raw_prob = conv_output[:, :, :, :, 5:]  # category probability of the prediction box

        # next need Draw the grid. Where output_size is equal to 13, 26 or 52
        y = tf.range(output_size, dtype=tf.int32)
        y = tf.expand_dims(y, -1)
        y = tf.tile(y, [1, output_size])
        x = tf.range(output_size, dtype=tf.int32)
        x = tf.expand_dims(x, 0)
        x = tf.tile(x, [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        # Calculate the center position of the prediction box:
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
        # Calculate the length and width of the prediction box:
        pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]

        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        pred_conf = tf.sigmoid(conv_raw_conf)  # object box calculates the predicted confidence
        pred_prob = tf.sigmoid(conv_raw_prob)  # calculating the predicted probability category box object

        # calculating the predicted probability category box object
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    @staticmethod
    def bbox_iou(boxes1, boxes2):
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return 1.0 * inter_area / union_area

    @staticmethod
    def draw_bbox(image, bboxes, CLASS, show_label=True, show_confidence=True,
                  Text_colors=(255, 255, 0), rectangle_colors=''):

        num_classes = len(CLASS)
        print("CLASS", CLASS, num_classes)
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        # print("hsv_tuples", hsv_tuples)
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for i, bbox in enumerate(bboxes):
            coor = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            print("bbox[5]", bbox[5])
            class_ind = int(bbox[5])
            print("class_ind", class_ind)
            bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 1000)
            if bbox_thick < 1: bbox_thick = 1
            fontScale = 0.75 * bbox_thick
            (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

            # put object rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick * 2)

            if show_label:
                # get text label
                score_str = " {:.2f}".format(score) if show_confidence else ""

                try:
                    label = "{}".format(CLASS[class_ind]) + score_str
                except KeyError:
                    print("You received KeyError, this might be that you are trying to use yolo original weights")
                    print(
                        "while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")

                # get text size
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                      fontScale, thickness=bbox_thick)
                # put filled text rectangle
                cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color,
                              thickness=cv2.FILLED)

                # put text above rectangle
                cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

        return image

    @staticmethod
    def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
        """
        :param bboxes: (xmin, ymin, xmax, ymax, score, class)
        Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
              https://github.com/bharatsingh430/soft-nms
        """
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        def bboxes_iou(boxes1, boxes2):
            boxes1 = np.array(boxes1)
            boxes2 = np.array(boxes2)

            boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
            boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

            left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
            right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

            inter_section = np.maximum(right_down - left_up, 0.0)
            inter_area = inter_section[..., 0] * inter_section[..., 1]
            union_area = boxes1_area + boxes2_area - inter_area
            ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

            return ious

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]
            # Process 1: Determine whether the number of bounding boxes is greater than 0
            while len(cls_bboxes) > 0:
                # Process 2: Select the bounding box with the highest score according to socre order A
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                # Process 3: Calculate this bounding box A and
                # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold
                iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                assert method in ['nms', 'soft-nms']

                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0

                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes

    @staticmethod
    def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
        valid_scale = [0, np.inf]
        pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                    pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
        # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        org_h, org_w = original_image.shape[:2]
        resize_ratio = min(input_size / org_w, input_size / org_h)

        dw = (input_size - resize_ratio * org_w) / 2
        dh = (input_size - resize_ratio * org_h) / 2

        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # 3. clip some boxes those are out of range
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # 4. discard some invalid boxes
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # 5. discard boxes with low scores
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > score_threshold
        mask = np.logical_and(scale_mask, score_mask)
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

        return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
