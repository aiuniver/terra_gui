# from threading import Thread
from typing import Tuple
import numpy as np
import os
import gc
# import copy
import operator
from tensorflow import keras
from tensorflow.keras.models import load_model
from apps.plugins.terra.neural.customcallback import CustomCallback
from apps.plugins.terra.neural.customlosses import DiceCoefficient
# import apps.plugins.terra.data as Data
import terra_ai.data.training.extra as Extra
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

    def set_optimizer(self, params) -> None:
        """
        Set optimizer method for using terra w/o gui

        """
        optimizer_object = getattr(keras.optimizers, params.optimizer.type.value)
        self.optimizer = optimizer_object(**params.optimizer.parameters_dict)

        pass

    def set_custom_metrics(self, params):
        for i_key in self.metrics.keys():
            for idx, metric in enumerate(self.metrics[i_key]):
                if metric in self.custom_losses_dict.keys():
                    if metric == "mean_io_u":
                        self.metrics[i_key][idx] = self.custom_losses_dict[metric](
                            num_classes=self.output_params[i_key]['num_classes'], name=metric)
                    else:
                        self.metrics[i_key][idx] = self.custom_losses_dict[metric](name=metric)
                pass

    def set_chp_monitor(self, params) -> None:
        if len(self.x_Train) > 1:
            if params.architecture.parameters.checkpoint.indicator == Extra.CheckpointIndicatorChoice.train:
                if params.architecture.parameters.checkpoint.type == Extra.CheckpointTypeChoice.Metrics:
                    for output in params.architecture.parameters.outputs:
                        if output.alias == params.architecture.parameters.checkpoint.layer:
                            self.chp_monitor = f'{params.architecture.parameters.checkpoint.layer}_{output.metrics[0].value}'
                else:
                    for output in params.architecture.parameters.outputs:
                        if output.alias == params.architecture.parameters.checkpoint.layer:
                            self.chp_monitor = f'{params.architecture.parameters.checkpoint.layer}_{output.loss.value}'
            else:
                if params.architecture.parameters.checkpoint.type == Extra.CheckpointTypeChoice.Metrics:
                    for output in params.architecture.parameters.outputs:
                        if output.alias == params.architecture.parameters.checkpoint.layer:
                            self.chp_monitor = f'val_{params.architecture.parameters.checkpoint.layer}_{output.metrics[0].value}'
                else:
                    for output in params.architecture.parameters.outputs:
                        if output.alias == params.architecture.parameters.checkpoint.layer:
                            self.chp_monitor = f'val_{params.architecture.parameters.checkpoint.layer}_{output.loss.value}'
        else:
            if params.architecture.parameters.checkpoint.indicator == Extra.CheckpointIndicatorChoice.train:
                if params.architecture.parameters.checkpoint.type == Extra.CheckpointTypeChoice.Metrics:
                    for output in params.architecture.parameters.outputs:
                        if output.alias == params.architecture.parameters.checkpoint.layer:
                            self.chp_monitor = f'{output.metrics[0].value}'
                else:
                    self.chp_monitor = 'loss'
            else:
                if params.architecture.parameters.checkpoint.type == Extra.CheckpointTypeChoice.Metrics:
                    for output in params.architecture.parameters.outputs:
                        if output.alias == params.architecture.parameters.checkpoint.layer:
                            self.chp_monitor = f'val_{output.metrics[0].value}'
                else:
                    self.chp_monitor = 'val_loss'

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
            dts_obj (object): setting task_name
        """


        if not self.model_is_trained:
            self.nn_cleaner()
            self.DTS = dts_obj
            self.prepare_dataset()
        else:
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
        self.Exch.show_text_data(msg)
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
            self.Exch.print_2status_bar(('info', f'Weights are saved as {file_path_weights}'))
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

    def terra_fit(self, nnmodel: object = keras.Model, training_params=None, verbose: int = 0) -> None:
        """
        This method created for using wth externally compiled models

        Args:
            nnmodel (obj): keras model for fit
            training_params:
            verbose:    verbose arg from tensorflow.keras.model.fit

        Return:
            None
        """

        self.epochs = training_params.epochs
        self.batch_size = training_params.batch
        self.optimizer_name = training_params.optimizer.type
        self.optimizer_kwargs = training_params.optimizer.parameters
        self.set_optimizer(training_params)
        self.set_chp_monitor(training_params)
        for output_key in self.output_params.keys():
            self.metrics.update({output_key: self.output_params[output_key]['metrics']})
            self.loss.update({output_key: self.output_params[output_key]['loss']})


        if self.model_is_trained:
            try:
                list_files = os.listdir(self.training_path)
                model_name = [x for x in list_files if x.endswith("last.h5")]
                # if len(model_name) > 1:
                #     self.Exch.print_error(("Ошибка", "в папке обучения находится больше одной сохраненной модели"))
                custom_objects = {}
                for output_key in self.metrics.keys():
                    for metric_name in self.metrics[output_key]:
                        if not isinstance(metric_name, str):
                            metric_name = metric_name.name
                        if metric_name == "dice_coef":
                            custom_objects.update({"DiceCoefficient": DiceCoefficient})
                if custom_objects:
                    self.model = load_model(os.path.join(self.training_path, model_name[0]), compile=False,
                                            custom_objects=custom_objects)
                else:
                    self.model = load_model(os.path.join(self.training_path, model_name[0]), compile=False)

                self.nn_name = f"{self.model.name}"
                self.Exch.print_2status_bar(('Загружена модель', model_name[0]))
            except Exception:
                self.Exch.print_2status_bar(('Ошибка загрузки модели', "!!!"))  # self.Exch.print_error
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
            self.Exch.print_2status_bar(('Компиляция модели', '...'))
            self.set_custom_metrics()
            self.model.compile(loss=self.loss,
                               optimizer=self.optimizer,
                               metrics=self.metrics
                               )
            self.Exch.print_2status_bar(('Компиляция модели', 'выполнена'))
            self.Exch.print_2status_bar(('Начало обучения', '...'))
            if self.x_Val['input_1'] is not None:
                # training = Thread(target=self.tr_thread)
                # training.start()
                # training.join()
                # del training
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
        else:
            self.model = nnmodel
            self.nn_name = f"{self.model.name}"
            self.set_custom_metrics()
            self.Exch.print_2status_bar(('Компиляция модели', '...'))
            self.model.compile(loss=self.loss,
                               optimizer=self.optimizer,
                               metrics=self.metrics
                               )
            self.Exch.print_2status_bar(('Компиляция модели', 'выполнена'))
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
            # print("self.x_Train", self.x_Train)
            # print("self.x_Train shape", self.x_Train["input_1"].shape)
            # print("self.y_Train", self.y_Train)
            # print("self.y_Train shape", self.y_Train["output_1"].shape)
            # print("self.x_Val", self.x_Val)
            # print("self.x_Val shape", self.x_Val["input_1"].shape)
            # print("self.y_Val", self.y_Val)
            # print("self.y_Val shape", self.y_Val["output_1"].shape)
            # print(self.DTS.one_hot_encoding)
            # print(self.DTS.scaler)
            # print(self.DTS.classes_names)
            # print(self.DTS.classes_colors)
            # print("self.DTS.num_classes", self.DTS.num_classes)
            if self.x_Val['input_1'] is not None:
                # training = Thread(target=self.tr_thread)
                # training.start()
                # training.join()
                # del training
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
            self.sum_epoch += self.epochs
        self.model_is_trained = True
        self.stop_training = self.callbacks[0].stop_training

        #     msg = f'Модель сохранена на последней эпохе.'
        #     self.Exch.print_2status_bar(('Обучение завершено пользователем!', msg))
        #     self.Exch.out_data['stop_flag'] = True

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

    def tr_thread(self, verbose: int = 0):
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
                    & (
                    not np.isnan(history.history[monitor][i]))
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
