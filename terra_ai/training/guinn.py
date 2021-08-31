# from threading import Thread
from typing import Tuple
import numpy as np
import os
import gc
# import copy
import tensorflow as tf
import sys
from tensorflow import keras
from tensorflow.keras.models import load_model

from terra_ai import progress
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.datasets.extra import LayerOutputTypeChoice
from terra_ai.data.modeling.model import ModelData, ModelDetailsData
from terra_ai.data.training.train import TrainData
from terra_ai.datasets.preparing import PrepareDTS
from terra_ai.modeling.validator import ModelValidator
from terra_ai.training.customcallback import FitCallback
from terra_ai.training.customlosses import DiceCoefficient, yolo_loss

from terra_ai.data.training.extra import CheckpointIndicatorChoice, CheckpointTypeChoice, MetricChoice

from terra_ai.training.data import custom_losses_dict

__version__ = 0.01


class GUINN:
    """
    GUINN: class, for train model
    """

    def __init__(self) -> None:
        """
        GUINN init method

        Args:
            exch_obj:   exchange object for terra
        """
        self.callbacks = []
        self.chp_monitor = 'loss'

        """
        For model settings
        """
        self.nn_name: str = ''
        self.model = None
        self.training_path: str = ""
        self.optimizer = None
        self.loss: dict = {}
        self.metrics: dict = {}

        self.batch_size = 128
        self.epochs = 5
        self.sum_epoch = 0
        self.stop_training = False
        self.retrain_flag = False
        self.shuffle: bool = True
        self.model_is_trained: bool = False

        """
        Logs
        """
        self.history: dict = {}

    @staticmethod
    def _check_metrics(metrics: list, num_classes: int = 2) -> list:
        output = []
        for metric in metrics:
            if metric == MetricChoice.MeanIoU.value:
                output.append(getattr(sys.modules.get("tensorflow.keras.metrics"), metric)(num_classes))
            elif metric == MetricChoice.DiceCoef:
                output.append(DiceCoefficient())
            else:
                output.append(getattr(sys.modules.get("tensorflow.keras.metrics"), metric)())
        return output

    def _set_training_params(self, dataset: DatasetData, params: TrainData, training_path: str) -> None:
        self.dataset = self._prepare_dataset(dataset)
        self.training_path = training_path
        self.epochs = params.epochs
        self.batch_size = params.batch
        self.set_optimizer(params)
        self.set_chp_monitor(params)
        for output_layer in params.architecture.outputs_dict:
            self.metrics.update({
                str(output_layer["id"]):
                    self._check_metrics(metrics=output_layer.get("metrics", []),
                                        num_classes=output_layer.get("classes_quantity"))
            })
            self.loss.update({str(output_layer["id"]): output_layer["loss"]})

    def _set_callbacks(self, dataset: object, batch_size: int, epochs: int, checkpoint: dict) -> None:
        print(('Добавление колбэков', '...'))
        clsclbk = FitCallback(dataset=dataset, exchange=None, batch_size=batch_size, epochs=epochs)
        self.callbacks = [clsclbk]
        checkpoint.update([('filepath', 'F:\\tmp\\models\\test_model.h5')])
        self.callbacks.append(keras.callbacks.ModelCheckpoint(**checkpoint))
        print(('Добавление колбэков', 'выполнено'))

    @staticmethod
    def _prepare_dataset(dataset: DatasetData):
        prepared_dataset = PrepareDTS(dataset)
        prepared_dataset.prepare_dataset()
        return prepared_dataset

    @staticmethod
    def _set_model(model: dict) -> object:
        validator = ModelValidator(ModelDetailsData(**model))
        train_model = validator.get_keras_model()
        return train_model

    def set_optimizer(self, params) -> None:
        """
        Set optimizer method for using terra w/o gui
        """

        optimizer_object = getattr(keras.optimizers, params.optimizer.type.value)
        self.optimizer = optimizer_object(**params.optimizer.parameters_dict)
        print(params.optimizer.parameters_dict)
        print(self.optimizer)

    def set_custom_metrics(self, params=None) -> None:
        for i_key in self.metrics.keys():
            for idx, metric in enumerate(self.metrics[i_key]):
                if metric in custom_losses_dict.keys():
                    if metric == "mean_io_u":
                        self.metrics[i_key][idx] = custom_losses_dict[metric](
                            num_classes=self.output_params[i_key]['num_classes'], name=metric)
                    else:
                        self.metrics[i_key][idx] = custom_losses_dict[metric](name=metric)

    def set_chp_monitor(self, params) -> None:
        layer_id = params.architecture.parameters.checkpoint.layer
        output = params.architecture.parameters.outputs.get(layer_id)
        if len(self.dataset.data.inputs) > 1:
            if params.architecture.parameters.checkpoint.indicator == CheckpointIndicatorChoice.train:
                if params.architecture.parameters.checkpoint.type == CheckpointTypeChoice.Metrics:
                    for output in params.architecture.parameters.outputs:
                        if str(output.id) == str(params.architecture.parameters.checkpoint.layer):
                            self.chp_monitor = f'{params.architecture.parameters.checkpoint.layer}_{output.metrics[0].value}'
                else:
                    for output in params.architecture.parameters.outputs:
                        if str(output.id) == str(params.architecture.parameters.checkpoint.layer):
                            self.chp_monitor = f'{params.architecture.parameters.checkpoint.layer}_{output.loss.value}'
            else:
                if params.architecture.parameters.checkpoint.type == CheckpointTypeChoice.Metrics:
                    for output in params.architecture.parameters.outputs:
                        if str(output.id) == str(params.architecture.parameters.checkpoint.layer):
                            self.chp_monitor = f'val_{params.architecture.parameters.checkpoint.layer}_{output.metrics[0].value}'
                else:
                    for output in params.architecture.parameters.outputs:
                        if str(output.id) == str(params.architecture.parameters.checkpoint.layer):
                            self.chp_monitor = f'val_{params.architecture.parameters.checkpoint.layer}_{output.loss.value}'
        else:
            if params.architecture.parameters.checkpoint.indicator == CheckpointIndicatorChoice.Train:
                if params.architecture.parameters.checkpoint.type == CheckpointTypeChoice.Metrics:
                    for output in params.architecture.parameters.outputs:
                        if str(output.id) == str(params.architecture.parameters.checkpoint.layer):
                            self.chp_monitor = f'{output.metrics[0].value}'
                else:
                    self.chp_monitor = 'loss'
            else:
                if params.architecture.parameters.checkpoint.type == CheckpointTypeChoice.Metrics:
                    for output in params.architecture.parameters.outputs:
                        if str(output.id) == str(params.architecture.parameters.checkpoint.layer):
                            self.chp_monitor = f'val_{output.metrics[0].value}'
                else:
                    self.chp_monitor = 'val_loss'

    def show_training_params(self) -> None:
        """
        output the parameters of the neural network: batch_size, epochs, shuffle, callbacks, loss, metrics,
        x_train_shape, num_classes
        """

        # print("self.DTS.X", self.DTS.X)
        # print("self.DTS.Y", self.DTS.Y)
        print("\nself.DTS.classes_names", self.DTS.classes_names)
        x_shape = []
        v_shape = []
        t_shape = []
        for i_key in self.DTS.X.keys():
            x_shape.append([i_key, self.DTS.X[i_key]['data'][0].shape])
            v_shape.append([i_key, self.DTS.X[i_key]['data'][1].shape])
            t_shape.append([i_key, self.DTS.X[i_key]['data'][2].shape])

        msg = f'num_classes = {self.DTS.num_classes}, x_Train_shape = {x_shape}, x_Val_shape = {v_shape}, \n' \
              f'x_Test_shape = {t_shape}, epochs = {self.epochs}, learning_rate={self.learning_rate}, \n' \
              f'callbacks = {self.callbacks}, batch_size = {self.batch_size},shuffle = {self.shuffle}, \n' \
              f'loss = {self.loss}, metrics = {self.metrics} \n'

        print(msg)
        pass

    def terra_fit(self, dataset: DatasetData, gui_model: dict, training_params: TrainData = None,
                  training_path: str = "", verbose: int = 0) -> None:
        """
        This method created for using wth externally compiled models

        Args:
            nnmodel (obj): keras model for fit
            training_params:
            verbose:    verbose arg from tensorflow.keras.model.fit

        Return:
            None
        """
        self.nn_cleaner(retrain=True if self.model_is_trained else False)
        self._set_training_params(dataset=dataset, params=training_params, training_path=training_path)
        nnmodel = self._set_model(model=gui_model)

        if self.model_is_trained:
            try:
                list_files = os.listdir(self.training_path)
                model_name = [x for x in list_files if x.endswith("last.h5")]
                custom_objects = {}
                for output_key in self.metrics.keys():
                    for metric_name in self.metrics[output_key]:
                        if not isinstance(metric_name, str):
                            metric_name = metric_name.name
                        if metric_name == "dice_coef":
                            custom_objects.update({"DiceCoefficient": DiceCoefficient})
                if not custom_objects:
                    custom_objects = None
                self.model = load_model(os.path.join(self.training_path, model_name[0]), compile=False,
                                        custom_objects=custom_objects)

                self.nn_name = f"{self.model.name}"
                self.Exch.print_2status_bar(('Загружена модель', model_name[0]))
            except Exception:
                self.Exch.print_2status_bar(('Ошибка загрузки модели', "!!!"))
                print("Ошибка загрузки модели!!!")

            if self.stop_training and (self.callbacks[0].last_epoch != self.sum_epoch):
                if self.retrain_flag:
                    self.epochs = self.sum_epoch - self.callbacks[0].last_epoch
                else:
                    self.epochs = self.epochs - self.callbacks[0].last_epoch
            else:
                self.retrain_flag = True
                self.callbacks[0].stop_flag = False
                self.sum_epoch += self.epochs
                self.callbacks[0].batch_size = self.batch_size
                self.callbacks[0].retrain_flag = True
                self.callbacks[0].retrain_epochs = self.epochs
                self.callbacks[0].epochs = self.epochs + self.callbacks[0].last_epoch

            self.model.stop_training = False
            self.stop_training = False
            self.model_is_trained = False
            if list(self.dataset.data.outputs.values())[0].task == LayerOutputTypeChoice.ObjectDetection:
                self.yolomodel_fit(params=training_params, dataset=self.dataset, verbose=1, retrain=True)
            else:
                self.basemodel_fit(params=training_params, dataset=self.dataset, verbose=0, retrain=True)

        else:
            self.model = nnmodel
            self.nn_name = f"{self.model.name}"
            if list(self.dataset.data.outputs.values())[0].task == LayerOutputTypeChoice.ObjectDetection:
                self.yolomodel_fit(params=training_params, dataset=self.dataset, verbose=1, retrain=False)
            else:
                self.basemodel_fit(params=training_params, dataset=self.dataset, verbose=0, retrain=False)

            self.sum_epoch += self.epochs
        self.model_is_trained = True
        # self.stop_training = self.callbacks[0].stop_training

    def nn_cleaner(self, retrain=False) -> None:
        keras.backend.clear_session()
        self.DTS = None
        self.model = None
        if not retrain:
            self.stop_training = False
            self.model_is_trained = False
            self.retrain_flag = False
            self.sum_epoch = 0
            self.chp_monitor = ""
            self.optimizer = None
            self.loss = {}
            self.metrics = {}
            self.callbacks = []
            self.history = {}
        gc.collect()

    def get_nn(self):
        self.nn_cleaner(retrain=True)

        return self

    @progress.threading
    def basemodel_fit(self, params, dataset, verbose=0, retrain=False) -> None:
        print(('Компиляция модели', '...'))
        self.set_custom_metrics()
        print(self.loss)
        print(self.metrics)
        self.model.compile(loss=self.loss,
                           optimizer='adam',  # self.optimizer,
                           metrics=self.metrics
                           )
        print(('Компиляция модели', 'выполнена'))
        print(('Начало обучения', '...'))
        if not retrain:
            self._set_callbacks(dataset=dataset, batch_size=params.batch,
                                epochs=params.epochs, checkpoint=params.architecture.parameters.checkpoint.native())

        print(('Начало обучения', '...'))

        self.history = self.model.fit(
            self.dataset.dataset.get('train').batch(self.batch_size, drop_remainder=True).take(-1),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            validation_data=self.dataset.dataset.get('val').batch(self.batch_size, drop_remainder=True).take(-1),
            epochs=self.epochs,
            verbose=verbose,
            callbacks=self.callbacks
        )

    def yolomodel_fit(self, params, dataset, verbose=0, retrain=False) -> None:
        # Массив используемых анкоров (в пикселях). Используетя по 3 анкора на каждый из 3 уровней сеток
        # данные значения коррелируются с размерностью входного изображения input_shape
        anchors = np.array(
            [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
        num_anchors = len(anchors)  # Сохраняем количество анкоров

        @tf.autograph.experimental.do_not_convert
        def create_model(
                input_shape,
                num_anchor,
                model,
                num_classes,
        ):
            """
                Функция создания полной модели
                    Входные параметры:
                      input_shape - размерность входного изображения для модели YOLO
                      num_anchors - общее количество анкоров
                      model - спроектированная модель
                      num_classes - количесво классов
            """
            w, h, ch = input_shape  # Получаем ширину и высоту и глубину входного изображения
            # inputs = keras.layers.Input(shape=(w, h, ch))  # Создаем входной слой модели, добавляя размерность для
            # глубины цвета

            # Создаем три входных слоя y_true с размерностями ((None, 13, 13, 3, 6), (None, 26, 26, 3, 6) и (None,
            # 52, 52, 3, 6)) 2 и 3 параметры (13, 13) указывают размерность сетки, на которую условно будет разбито
            # исходное изображение каждый уровень сетки отвечает за обнаружение объектов различных размеров (13 -
            # крупных, 26 - средних, 52 - мелких) 4 параметр - количество анкоров на каждый уровень сетки 5 параметр
            # - 4 параметра описывающие параметры анкора (координаты центра, ширина и высота) + вероятность
            # обнаружения объекта + OHE номер класса
            y_true = [
                keras.layers.Input(shape=(w // 32, h // 32, num_anchor // 3, num_classes + 5), name="input_2"),
                keras.layers.Input(shape=(w // 16, h // 16, num_anchor // 3, num_classes + 5), name="input_3"),
                keras.layers.Input(shape=(w // 8, h // 8, num_anchor // 3, num_classes + 5), name="input_4")
            ]

            model_yolo = model  # create_YOLOv3(inputs, num_anchors // 3)  # Создаем модель YOLOv3
            print('Создана модель YOLO. Количество классов: {}.'.format(
                num_classes))  # Выводим сообщение о создании модели
            print('model_yolo.summary()', model_yolo.summary())
            # Создаем выходной слой Lambda (выходом которого будет значение ошибки модели)
            # На вход слоя подается:
            #   - model_yolo.output (выход модели model_yolo (то есть то, что посчитала сеть))
            #   - y_true (оригинальные данные из обучающей выборки)
            outputs = keras.layers.Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                                          arguments={'num_anchors': num_anchor})([*model_yolo.output, *y_true])

            return keras.models.Model([model_yolo.input, *y_true], outputs)  # Возвращаем модель

        # Создаем модель
        model_YOLO = create_model(input_shape=(416, 416, 3), num_anchor=num_anchors, model=self.model,
                                  num_classes=list(self.dataset.data.num_classes.values())[0])
        print(model_YOLO.summary())

        # Компилируем модель
        print(('Компиляция модели', '...'))
        # self.set_custom_metrics()
        model_YOLO.compile(optimizer=self.optimizer,
                           loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        print(('Компиляция модели', 'выполнена'))
        print(('Начало обучения', '...'))

        if not retrain:
            self._set_callbacks(dataset=dataset, batch_size=params.batch,
                                epochs=params.epochs, checkpoint=params.architecture.parameters.checkpoint.native())

        print(('Начало обучения', '...'))
        self.history = model_YOLO.fit(
            self.dataset.dataset.get('train'),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            validation_data=self.dataset.dataset.get('val'),
            epochs=self.epochs,
            verbose=verbose,
            callbacks=self.callbacks
        )
