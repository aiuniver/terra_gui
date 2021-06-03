from threading import Thread
from typing import Tuple
import numpy as np
import os
import gc
import operator
from tensorflow import keras
from apps.plugins.terra.neural.customcallback import CustomCallback
from apps.plugins.terra.neural.customlosses import DiceCoefficient

__version__ = 0.05


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

        Args:
            optimizer_name (str):   name of keras optimizer
            kwargs (dict):          kwargs for optimizer
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
        if len(self.x_Train) > 1:
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
            dts_obj (object): setting task_name
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
        self.Exch.show_text_data(msg)
        pass

    def save_nnmodel(self) -> None:
        """
        Saving model if the model is trained

        Returns:
            None
        """
        if self.model_is_trained:
            model_name = f"model_{self.nn_name}_ep_{self.best_epoch_num:002d}_m_{self.best_metric_result:.4f}_last"
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
                f'weights_{self.nn_name}_ep_{self.best_epoch_num:002d}_m_{self.best_metric_result:.4f}_last'
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

    def terra_fit(self, nnmodel: object = keras.Model, verbose: int = 0) -> None:
        """
        This method created for using wth externally compiled models

        Args:
            nnmodel (obj): keras model for fit
            verbose:    verbose arg from tensorflow.keras.model.fit

        Return:
            None
        """
        if self.model_is_trained:
            if self.model.stop_training:
                self.epochs = self.epochs - self.callbacks[0].last_epoch
            else:
                self.callbacks[0].batch_size = self.batch_size
                self.callbacks[0].epochs = self.epochs
                self.model.stop_training = False
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
                filepath=os.path.join(self.training_path, f'model_{self.nn_name}_best.h5'),
                verbose=1, save_best_only=self.chp_save_best, save_weights_only=self.chp_save_weights,
                monitor=self.chp_monitor, mode=self.chp_mode))
            self.Exch.print_2status_bar(('Добавление колбэков', 'выполнено'))
            self.Exch.print_2status_bar(('Начало обучения', '...'))
            # self.show_training_params()
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
        self.model_is_trained = True

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

        pass

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

    def nn_cleaner(self) -> None:
        keras.backend.clear_session()
        del self.model
        del self.DTS
        del self.x_Train
        del self.x_Val
        del self.y_Train
        del self.y_Val
        del self.x_Test
        del self.y_Test
        gc.collect()
        self.model_is_trained = False
        self.DTS = None
        self.model = keras.Model
        self.optimizer = keras.optimizers.Adam()
        self.loss = {}
        self.metrics = {}
        self.callbacks = []
        self.history = {}
        self.x_Train = {}
        self.x_Val = {}
        self.y_Train = {}
        self.y_Val = {}
        self.x_Test = {}
        self.y_Test = {}
        pass

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
