# from threading import Thread
from typing import Tuple
import numpy as np
import os
import gc
# import copy
import tensorflow as tf
import operator
from tensorflow import keras
from tensorflow.keras.models import load_model
from apps.plugins.terra.neural.customcallback import CustomCallback
from apps.plugins.terra.neural.customlosses import DiceCoefficient, yolo_loss

__version__ = 0.06


class GUINN:
    """
    GUINN: class, for train model
    """

    def __init__(self, exch_obj) -> None:
        """
        GUINN init method

        Args:
            exch_obj:   exchange object for terra
        """

        self.Exch = exch_obj
        self.DTS = None
        self.callbacks = []
        self.output_params = {}
        self.chp_indicator = 'val'
        self.chp_monitor = 'loss'
        self.chp_monitors = {'output': 'output_1', 'out_type': 'loss', 'out_monitor': 'mse'}
        self.chp_mode = 'min'
        self.chp_save_best = True
        self.chp_save_weights = True

        """
        For samples from dataset
        """
        self.x_Train: dict = {}
        self.x_Val: dict = {}
        self.y_Train: dict = {}
        self.y_Val: dict = {}
        self.x_Test: dict = {}
        self.y_Test: dict = {}
        self.y_Val_bbox: list = []
        """
        For model settings
        """
        self.nn_name: str = ''
        self.model = keras.Model
        self.modelling_path: str = ""
        self.training_path: str = ""
        # self.external_model: bool = False
        self.learning_rate = 1e-3
        self.optimizer_name: str = 'Adam'
        self.optimizer_object = keras.optimizers.Adam
        self.optimizer_kwargs = {}
        self.optimizer = keras.optimizers.Adam()
        self.loss: dict = {}
        self.metrics: dict = {}
        self.custom_losses_dict: dict = {"dice_coef": DiceCoefficient, "mean_io_u": keras.metrics.MeanIoU}
        self.batch_size = 32
        self.epochs = 20
        self.sum_epoch = 0
        self.stop_training = False
        self.retrain_flag = False
        self.shuffle: bool = True

        """
        Logs
        """
        self.best_epoch: dict = {}
        self.best_epoch_num: int = 0
        self.stop_epoch: int = 0
        self.model_is_trained: bool = False
        self.history: dict = {}
        self.best_metric_result = "0000"
        self.monitor: str = 'accuracy'
        self.monitor2: str = "loss"

    def set_optimizer(self) -> None:
        """
        Set optimizer method for using terra w/o gui

        """
        self.optimizer_object = getattr(keras.optimizers, self.optimizer_name)
        self.optimizer = self.optimizer_object(**self.optimizer_kwargs)

        pass

    def set_custom_metrics(self):
        for i_key in self.metrics.keys():
            for idx, metric in enumerate(self.metrics[i_key]):
                if metric in self.custom_losses_dict.keys():
                    if metric == "mean_io_u":
                        self.metrics[i_key][idx] = self.custom_losses_dict[metric](
                            num_classes=self.output_params[i_key]['num_classes'], name=metric)
                    else:
                        self.metrics[i_key][idx] = self.custom_losses_dict[metric](name=metric)
                pass

    def set_chp_monitor(self) -> None:
        if len(self.x_Train) > 1 and self.DTS.task_type.get('output_1') != 'object_detection':
            if self.chp_indicator == 'train':
                self.chp_monitor = f'{self.chp_monitors["output"]}_{self.chp_monitors["out_monitor"]}'
            else:
                self.chp_monitor = f'val_{self.chp_monitors["output"]}_{self.chp_monitors["out_monitor"]}'
        else:
            if self.chp_indicator == 'train':
                if self.chp_monitors["out_type"] == 'loss':
                    self.chp_monitor = 'loss'
                else:
                    self.chp_monitor = f'{self.chp_monitors["out_monitor"]}'
            else:
                if self.chp_monitors["out_type"] == 'loss':
                    self.chp_monitor = 'val_loss'
                else:
                    self.chp_monitor = f'val_{self.chp_monitors["out_monitor"]}'

    def set_main_params(self, output_params: dict = None, clbck_chp: dict = None,
                        shuffle: bool = True, epochs: int = 10, batch_size: int = 32,
                        optimizer_params: dict = None) -> None:
        self.output_params = output_params
        self.chp_indicator = clbck_chp['indicator'].value  # 'train' или 'val'
        self.chp_monitors = clbck_chp[
            'monitor']  # это словарь {'output': 'output_1', 'out_type': 'loss', 'out_monitor': 'mse'}
        self.chp_mode = clbck_chp['mode'].value  # 'min' или 'max'
        self.chp_save_best = clbck_chp['save_best']  # bool
        self.chp_save_weights = clbck_chp['save_weights']  # bool
        self.shuffle = shuffle
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_name = optimizer_params['op_name'].value
        self.optimizer_kwargs = optimizer_params['op_kwargs']
        self.set_optimizer()
        self.set_chp_monitor()
        for output_key in self.output_params.keys():
            self.metrics.update({output_key: self.output_params[output_key]['metrics']})
            self.loss.update({output_key: self.output_params[output_key]['loss']})

        pass

    def set_dataset(self, dts_obj: object) -> None:
        """
        Setting task nn_name

        Args:
            dts_obj (object): setting dataset
        """
        if not self.model_is_trained:
            self.nn_cleaner()
        self.DTS = dts_obj
        self.prepare_dataset()
        pass

    def show_training_params(self) -> None:
        """
        output the parameters of the neural network: batch_size, epochs, shuffle, callbacks, loss, metrics,
        x_train_shape, num_classes
        """
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

        # TODO: change to print_2status_bar then remove debug_mode
        # self.Exch.show_text_data(msg)
        print(msg)
        pass

    def save_nnmodel(self) -> None:
        """
        Saving model if the model is trained

        Returns:
            None
        """
        if self.model_is_trained:
            model_name = f"model_{self.nn_name}_ep_{self.best_epoch_num:002d}_m_{self.best_metric_result:.4f}.last"
            file_path_model: str = os.path.join(
                self.training_path, f"{model_name}.h5"
            )
            self.model.save(file_path_model)
            self.Exch.print_2status_bar(
                ("Info", f"Модель сохранена как: {file_path_model}")
            )
        else:
            self.Exch.print_error(("Ошибка", "Сохранение не возможно. Модель не обучена."))

        pass

    def save_model_weights(self) -> None:
        """
        Saving model weights if the model is trained

        Returns:
            None
        """

        if self.model_is_trained:
            model_weights_name = \
                f'weights_{self.nn_name}_ep_{self.best_epoch_num:002d}_m_{self.best_metric_result:.4f}.last'
            file_path_weights: str = os.path.join(self.training_path, f'{model_weights_name}.h5')
            self.model.save_weights(filepath=file_path_weights)
            self.Exch.print_2status_bar(('info', f'Веса сохранены как {file_path_weights}'))
        else:
            self.Exch.print_error(("Ошибка", "Сохранение не возможно. Модель не обучена."))

        pass

    def prepare_dataset(self) -> None:
        """
        reformat samples of dataset

        Returns:
            None
        """
        self.Exch.print_2status_bar(('Поготовка датасета', '...'))
        if self.DTS.task_type.get('output_1') == 'object_detection': #Заглушка
            for input_key in self.DTS.X.keys():

                self.x_Train.update({input_key: self.DTS.X[input_key]['data'][0]})
                if self.DTS.X[input_key]['data'][1] is not None:
                    self.x_Val.update({input_key: self.DTS.X[input_key]['data'][1]})
                if self.DTS.X[input_key]['data'][2] is not None:
                    self.x_Test.update({input_key: self.DTS.X[input_key]['data'][2]})

            for output_key in self.DTS.Y.keys():
                if output_key == 'output_1':
                    self.x_Train.update({'input_2': self.DTS.Y[output_key]['data'][0]})
                    if self.DTS.Y[output_key]['data'][1] is not None:
                        self.x_Val.update({'input_2': self.DTS.Y[output_key]['data'][1]})
                        self.y_Val_bbox.append(np.array(self.DTS.Y[output_key]['data'][1]))
                    if self.DTS.Y[output_key]['data'][2] is not None:
                        self.x_Test.update({'input_2': self.DTS.Y[output_key]['data'][2]})

                    self.y_Train.update({'yolo_loss': np.zeros(self.DTS.Y[output_key]['data'][0].shape)})
                    if self.DTS.Y[output_key]['data'][1] is not None:
                        self.y_Val.update({'yolo_loss': np.zeros(self.DTS.Y[output_key]['data'][1].shape)})
                    if self.DTS.Y[output_key]['data'][2] is not None:
                        self.y_Test.update({'yolo_loss': np.zeros(self.DTS.Y[output_key]['data'][2].shape)})
                elif output_key == 'output_2':
                    self.x_Train.update({'input_3': self.DTS.Y[output_key]['data'][0]})
                    if self.DTS.Y[output_key]['data'][1] is not None:
                        self.x_Val.update({'input_3': self.DTS.Y[output_key]['data'][1]})
                        self.y_Val_bbox.append(np.array(self.DTS.Y[output_key]['data'][1]))
                    if self.DTS.Y[output_key]['data'][2] is not None:
                        self.x_Test.update({'input_3': self.DTS.Y[output_key]['data'][2]})
                elif output_key == 'output_3':
                    self.x_Train.update({'input_4': self.DTS.Y[output_key]['data'][0]})
                    if self.DTS.Y[output_key]['data'][1] is not None:
                        self.x_Val.update({'input_4': self.DTS.Y[output_key]['data'][1]})
                        self.y_Val_bbox.append(np.array(self.DTS.Y[output_key]['data'][1]))
                    if self.DTS.Y[output_key]['data'][2] is not None:
                        self.x_Test.update({'input_4': self.DTS.Y[output_key]['data'][2]})

        else:
            for input_key in self.DTS.X.keys():

                self.x_Train.update({input_key: self.DTS.X[input_key]['data'][0]})
                if self.DTS.X[input_key]['data'][1] is not None:
                    self.x_Val.update({input_key: self.DTS.X[input_key]['data'][1]})
                if self.DTS.X[input_key]['data'][2] is not None:
                    self.x_Test.update({input_key: self.DTS.X[input_key]['data'][2]})

            for output_key in self.DTS.Y.keys():

                self.y_Train.update({output_key: self.DTS.Y[output_key]['data'][0]})
                if self.DTS.Y[output_key]['data'][1] is not None:
                    self.y_Val.update({output_key: self.DTS.Y[output_key]['data'][1]})
                if self.DTS.Y[output_key]['data'][2] is not None:
                    self.y_Test.update({output_key: self.DTS.Y[output_key]['data'][2]})
        self.Exch.print_2status_bar(('Поготовка датасета', 'выполнена'))
        pass

    def terra_fit(self, nnmodel: object = keras.Model, verbose: int = 0) -> None:
        """
        This method created for using wth externally compiled models

        Args:
            nnmodel (obj): keras model for fit
            verbose:    verbose arg from tensorflow.keras.model.fit

        Return:
            None
        """
        self.show_training_params()
        # print("self.DTS.X", self.DTS.X)
        # print("self.DTS.Y", self.DTS.Y)
        print("\nself.DTS.classes_names", self.DTS.classes_names)
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
            if self.DTS.task_type.get('output_1') == 'object_detection':
                self.yolomodel_fit(verbose=1, retrain=False)
            else:
                self.basemodel_fit(verbose=0, retrain=True)

        else:
            self.model = nnmodel
            self.nn_name = f"{self.model.name}"
            if self.DTS.task_type.get('output_1') == 'object_detection':
                self.yolomodel_fit(verbose=1, retrain=False)
            else:
                self.basemodel_fit(verbose=0, retrain=False)

            self.sum_epoch += self.epochs
        self.model_is_trained = True
        self.stop_training = self.callbacks[0].stop_training

        # self.monitor = self.chp_monitor
        # self.best_epoch, self.best_epoch_num, self.stop_epoch = self._search_best_epoch_data(
        #     history=self.history, monitor=self.monitor, monitor2=self.monitor2)
        # self.best_metric_result = self.best_epoch[self.monitor]
        #
        # try:
        #     self.save_nnmodel()
        # except RuntimeError:
        #     self.Exch.print_2status_bar(('Внимание!', 'Ошибка сохранения модели.'))
        # self.save_model_weights()

    def nn_cleaner(self, retrain=False) -> None:
        keras.backend.clear_session()
        # del self.DTS
        # del self.model
        # del self.x_Train
        # del self.x_Val
        # del self.y_Train
        # del self.y_Val
        # del self.x_Test
        # del self.y_Test
        self.DTS = None
        self.model = None
        self.x_Train = {}
        self.x_Val = {}
        self.y_Train = {}
        self.y_Val = {}
        self.x_Test = {}
        self.y_Test = {}
        if not retrain:
            self.stop_training = False
            self.model_is_trained = False
            self.retrain_flag = False
            self.sum_epoch = 0
            self.optimizer = keras.optimizers.Adam()
            self.loss = {}
            self.metrics = {}
            self.callbacks = []
            self.history = {}
        gc.collect()

    def get_nn(self):
        self.nn_cleaner(retrain=True)

        return self

    @staticmethod
    def _search_best_epoch_data(
            history, monitor="accuracy", monitor2="loss"
    ) -> Tuple[dict, int, int]:
        """
        Searching in history for best epoch with metrics from 'monitor' kwargs

        Args:
            history (Any):    history from training
            monitor (str):    1st metric (main)
            monitor2 (str):   2nd metric

        Returns:
            best_epoch (dict):          dictionary with all data for best epoch
            best_epoch_num + 1 (int):   best epoch number
            stop_epoch (int):           stop epoch
        """
        max_monitors = ["accuracy",
                        "dice_coef",
                        "mean_io_u",
                        "accuracy",
                        "binary_accuracy",
                        "categorical_accuracy",
                        "sparse_categorical_accuracy",
                        "top_k_categorical_accuracy",
                        "sparse_top_k_categorical_accuracy",
                        ]
        min_monitors = ["loss", "mae", "mape", "mse", "msle"]

        if not isinstance(monitor, str):
            monitor = str(monitor)

        if not isinstance(monitor2, str):
            monitor2 = str(monitor2)

        if monitor.split('_')[-1] in max_monitors:
            funct = np.argmax
            check = operator.gt

        elif ("error" in monitor) or monitor.split('_')[-1] in min_monitors:
            funct = np.argmin
            check = operator.lt

        else:
            funct = np.argmin
            check = operator.lt

        if monitor2.split('_')[-1] in max_monitors:
            check2 = operator.gt
        elif ("error" in monitor2) or monitor2.split('_')[-1] in min_monitors:
            check2 = operator.lt
        else:
            check2 = operator.gt

        best_epoch = dict()
        best_epoch_num = funct(history.history[monitor])

        if np.isnan(history.history[monitor][best_epoch_num]):
            n_range = best_epoch_num - 1
            best_epoch_num = funct(history.history[monitor][: best_epoch_num - 1])
        else:
            n_range = len(history.history[monitor])

        for i in range(n_range):
            if (
                    (
                            check(
                                history.history[monitor][i],
                                history.history[monitor][best_epoch_num],
                            )
                    )
                    & (
                    check2(
                        history.history[monitor2][i],
                        history.history[monitor2][best_epoch_num],
                    )
            )
                    & (not np.isnan(history.history[monitor][i]))
            ):
                best_epoch_num = i
            elif (
                    (
                            history.history[monitor][i]
                            == history.history[monitor][best_epoch_num]
                    )
                    & (
                            history.history[monitor2][i]
                            == history.history[monitor2][best_epoch_num]
                    )
                    & (not np.isnan(history.history[monitor][i]))
            ):
                best_epoch_num = i

        early_stop_epoch = len(history.history[monitor])
        for key, val in history.history.items():
            best_epoch.update({key: history.history[key][best_epoch_num]})
        return best_epoch, best_epoch_num + 1, early_stop_epoch

    def basemodel_fit(self, verbose=0, retrain=False) -> None:

        self.Exch.print_2status_bar(('Компиляция модели', '...'))
        self.set_custom_metrics()
        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=self.metrics
                           )
        self.Exch.print_2status_bar(('Компиляция модели', 'выполнена'))
        self.Exch.print_2status_bar(('Начало обучения', '...'))
        if not retrain:
            self.Exch.print_2status_bar(('Добавление колбэков', '...'))
            clsclbk = CustomCallback(params=self.output_params, step=1, show_final=True, dataset=self.DTS,
                                     exchange=self.Exch, samples_x=self.x_Val, samples_y=self.y_Val,
                                     batch_size=self.batch_size, epochs=self.epochs, save_model_path=self.training_path,
                                     model_name=self.nn_name)
            self.callbacks = [clsclbk]
            self.callbacks.append(keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.training_path, f'model_{self.nn_name}.best.h5'),
                verbose=1, save_best_only=self.chp_save_best, save_weights_only=self.chp_save_weights,
                monitor=self.chp_monitor, mode=self.chp_mode))
            self.Exch.print_2status_bar(('Добавление колбэков', 'выполнено'))

        self.Exch.print_2status_bar(('Начало обучения', '...'))
        if self.x_Val['input_1'] is not None:
            self.history = self.model.fit(
                self.x_Train,
                self.y_Train,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                validation_data=(self.x_Val, self.y_Val),
                epochs=self.epochs,
                verbose=verbose,
                callbacks=self.callbacks
            )
        else:
            self.history = self.model.fit(
                self.x_Train,
                self.y_Train,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                validation_split=0.2,
                epochs=self.epochs,
                verbose=verbose,
                callbacks=self.callbacks
            )

    def yolomodel_fit(self, verbose=0, retrain=False) -> None:
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
                                  num_classes=self.DTS.num_classes['output_1'])
        print(model_YOLO.summary())

        # Компилируем модель
        self.Exch.print_2status_bar(('Компиляция модели', '...'))
        # self.set_custom_metrics()
        model_YOLO.compile(optimizer=self.optimizer,
                           loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        self.Exch.print_2status_bar(('Компиляция модели', 'выполнена'))
        self.Exch.print_2status_bar(('Начало обучения', '...'))

        if not retrain:
            self.Exch.print_2status_bar(('Добавление колбэков', '...'))
            clsclbk = CustomCallback(params=self.output_params, step=1, show_final=True, dataset=self.DTS,
                                     exchange=self.Exch, samples_x=self.x_Val, samples_y=self.y_Val_bbox,
                                     batch_size=self.batch_size, epochs=self.epochs, save_model_path=self.training_path,
                                     model_name=self.nn_name)
            self.callbacks = [clsclbk]
            self.callbacks.append(keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.training_path, f'model_{self.nn_name}.best.h5'),
                verbose=1, save_best_only=self.chp_save_best, save_weights_only=self.chp_save_weights,
                monitor=self.chp_monitor, mode=self.chp_mode))
            self.Exch.print_2status_bar(('Добавление колбэков', 'выполнено'))

        self.Exch.print_2status_bar(('Начало обучения', '...'))
        self.history = model_YOLO.fit(
            self.x_Train,
            self.y_Train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            validation_data=(self.x_Val, self.y_Val),
            epochs=self.epochs,
            verbose=verbose,
            callbacks=self.callbacks
        )
