from typing import List, Tuple
# from dataclasses import dataclass
import numpy as np
import sys
import uuid
import os.path
import os
import operator
# import copy
# import gc
# import time
# from tensorflow import keras
import tensorflow.keras.optimizers
import tensorflow
import trds
from apps.plugins.terra.guiexchange import Exchange
from callbacks import ClassificationCallback, SegmentationCallback, TimeseriesCallback, RegressionCallback

__version__ = 0.1

tr2dj_obj = Exchange()


class GUINN:

    """
    GUINN class, for using
    """
    # instance class counter
    instance = 0

    def __init__(self, exch_obj=tr2dj_obj) -> None:
        """
        GUI_NN init method

        Args:
            exch_obj:   default exchange object for terra
        """
        super().__init__(exch_obj)
        self.__class__.instance += 1

        self.Exch = exch_obj

        self.debug_verbose = 3
        self.django_flag = False
        if self.Exch.property_of != 'TERRA':
            self.django_flag = True

        self.nn_name = ''
        self.model = tensorflow.keras.Model()
        self.external_model = False

        self.DTS = trds.DTS()
        self.input_datatype = 'Dim'

        '''
        Setting experiment_UUID and experiment_name 
        '''
        self.experiment_name = ''
        self.experiment_UUID = ''
        self.experiment_path = ''
        self.set_experiment_UUID()
        # self.set_experiment_name(str(self.experiment_UUID))

        '''
        Setting location for task_name in current project for _current_ user 
        if task_type is currently = ''
        it's setting to None   
        '''
        self.task_name = ''
        self.task_type = ''
        self.task_path = ''
        self.set_task_type(self.task_type)

        self.clbck_object = None
        self.callbacks = []
        self.callback_kwargs = None
        self.initialized_callback: object = ClassificationCallback

        self.best_epoch = dict()
        self.best_epoch_num: int = 0
        self.stop_epoch: int = 0
        self.model_is_trained = False
        self.history = dict()
        self.best_metric_result = '0000'

        self.task_type_defaults_dict = {'classification': {'optimizer_name': 'Adam',
                                                           'loss': 'categorical_crossentropy',
                                                           'metrics': ['accuracy'],
                                                           'batch_size': 32,
                                                           'epochs': 20,
                                                           'shuffle': True,
                                                           'clbck_object': ClassificationCallback,
                                                           'callback_kwargs': {'metrics': ['loss', 'accuracy'],
                                                                               'step': 1,
                                                                               'class_metrics': [],
                                                                               'data_tag': 'images',
                                                                               'show_worst': False,
                                                                               'show_final': True,
                                                                               'dataset': trds.DTS,
                                                                               'exchange': self.Exch
                                                                               }
                                                           },
                                        'segmentation': {'optimizer_name': 'Adam',
                                                         'loss': 'categorical_crossentropy',
                                                         'metrics': ['dice_coef'],
                                                         'batch_size': 16,
                                                         'epochs': 20,
                                                         'shuffle': False,
                                                         'clbck_object': SegmentationCallback,
                                                         'callback_kwargs': {'metrics': ['loss', 'dice_coef'],
                                                                             'step': 1,
                                                                             'class_metrics': [],
                                                                             'data_tag': 'images',
                                                                             'show_worst': False,
                                                                             'show_final': True,
                                                                             'dataset': trds.DTS,
                                                                             'exchange': self.Exch
                                                                             }
                                                         },
                                        'regression': {'optimizer_name': 'Adam',
                                                       'loss': 'mse',
                                                       'metrics': ['mae'],
                                                       'batch_size': 32,
                                                       'epochs': 20,
                                                       'shuffle': True,
                                                       'clbck_object': RegressionCallback,
                                                       'callback_kwargs': {'metrics': ['loss', 'mse'],
                                                                           'step': 1,
                                                                           'plot_scatter': True,
                                                                           'show_final': True,
                                                                           'dataset': trds.DTS,
                                                                           'exchange': self.Exch
                                                                           }
                                                       },
                                        'timeseries': {'optimizer_name': 'Adam',
                                                       'loss': 'mse',
                                                       'metrics': ['mae'],
                                                       'batch_size': 32,
                                                       'epochs': 20,
                                                       'shuffle': False,
                                                       'clbck_object': TimeseriesCallback,
                                                       'callback_kwargs': {'metrics': ['loss', 'mse'],
                                                                           'step': 1,
                                                                           'corr_step': 10,
                                                                           'plot_pred_and_true': True,
                                                                           'show_final': True,
                                                                           'dataset': trds.DTS,
                                                                           'exchange': self.Exch
                                                                           }
                                                       },
                                        }
        self.learning_rate = 1e-3
        self.optimizer_name = 'Adam'
        self.loss = 'categorical_crossentropy'
        self.metrics: List[str] = ['accuracy']
        self.batch_size = 32
        self.epochs = 20
        self.shuffle = True

        if not isinstance(self.metrics[0], str):
            self.monitor = str(self.metrics[0])
        else:
            self.monitor = self.metrics[0]
        self.monitor2 = 'loss'

    @staticmethod
    def _search_best_epoch_data(history, monitor="val_accuracy", monitor2="val_loss") -> Tuple[dict, int, int]:
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
        max_monitors = ['accuracy', 'dice_coef']
        min_monitors = ['loss', 'mae', 'mape', 'mse', 'msle']

        if not isinstance(monitor, str):
            monitor = str(monitor)

        if not isinstance(monitor2, str):
            monitor2 = str(monitor2)

        if monitor in max_monitors:
            funct = np.argmax
            check = operator.gt

        elif ('error' in monitor) or monitor in min_monitors:
            funct = np.argmin
            check = operator.lt

        else:
            funct = np.argmin
            check = operator.lt

        if monitor2 in max_monitors:
            check2 = operator.gt
        elif ('error' in monitor2) or monitor2 in min_monitors:
            check2 = operator.lt
        else:
            check2 = operator.gt

        best_epoch = dict()
        best_epoch_num = funct(history.history[monitor])

        if np.isnan(history.history[monitor][best_epoch_num]):
            n_range = best_epoch_num - 1
            best_epoch_num = funct(history.history[monitor][:best_epoch_num - 1])
        else:
            n_range = len(history.history[monitor])

        for i in range(n_range):
            if (check(history.history[monitor][i], history.history[monitor][best_epoch_num])) & (
                    check2(history.history[monitor2][i], history.history[monitor2][best_epoch_num])) & (
                    not np.isnan(history.history[monitor][i])):
                best_epoch_num = i
            elif (history.history[monitor][i] == history.history[monitor][best_epoch_num]) & (
                    history.history[monitor2][i] == history.history[monitor2][best_epoch_num]) & (
                    not np.isnan(history.history[monitor][i])):
                best_epoch_num = i

        early_stop_epoch = len(history.history[monitor])
        for key, val in history.history.items():
            best_epoch.update({key: history.history[key][best_epoch_num]})
        return best_epoch, best_epoch_num + 1, early_stop_epoch

    def _task_type_defaults(self, task_type):
        __task_type_defaults_kwargs = self.task_type_defaults_dict.get(task_type)
        for __var_name, __var_value in __task_type_defaults_kwargs.items():
            setattr(self, __var_name, __var_value)
        pass

    def set_task_type(self, task_type: str) -> None:
        """
        Setting task_type to crete logistic 'pipe' from start to end
        also set the task_name to same string value if it's not changed from default

        Args:
            task_type (str):  task type ('classification', 'segmentation', 'regression' or etc) for training NN

        """
        if task_type == '':
            self.task_type = None
        else:
            if self.task_type == self.task_name or self.task_name == '':
                self.task_type = task_type
                self.task_name = self.task_type
            if self.django_flag:
                self.Exch.set_task_name(self.task_name)
            else:
                self.task_type = task_type
            self._task_type_defaults(task_type)
            # self.task_path = os.path.join(self.project_path, self.task_type)
        pass

    def set_experiment_UUID(self) -> None:
        """
        Setting experiment UUID
        """
        self.experiment_UUID = uuid.uuid4()
        if self.django_flag:
            self.Exch.set_experiment_UUID(self.experiment_UUID)
        self.experiment_path = os.path.join(self.task_path, str(self.experiment_UUID))
        pass

    def load_dataset(self, dataset_obj: trds.DTS, task_type: str) -> None:
        """
        Load dataset object

        Args:
            dataset_obj (object):   trds.DTS dataset object
            task_type (str):        task type for NN

        Returns:
            None
        """
        self.DTS = dataset_obj
        self.input_datatype = self.DTS.input_datatype
        self._reinit(task_type)
        pass

    def _reinit(self, task_type) -> None:
        self.set_task_type(task_type)
        self.set_experiment_UUID()
        if self.django_flag:
            task_type_defaults_kwargs = self.task_type_defaults_dict.get(task_type)
            self.Exch.set_task_type_defaults(task_type_defaults_kwargs)
        self._reset()
        pass

    def _reset(self):
        self.callbacks = []
        pass

    def prepare_callbacks(self) -> None:
        """
        if terra in raw mode  - setting callback if its set
        if terra with django - checking switches and set callback options from switches

        Returns:
            None
        """
        if self.django_flag:
            clbck_options = self.Exch.get_callback_show_options_from_django(self.task_type)

            for option_name, option_value in clbck_options.items():
                if option_name == 'show_every_epoch':
                    if option_value:
                        self.callback_kwargs['step'] = 1
                    else:
                        self.callback_kwargs['step'] = 0
                elif option_name == 'plot_loss_metric':
                    if option_value:
                        if not ('loss' in self.callback_kwargs['metrics']):
                            self.callback_kwargs['metrics'].append('loss')
                    else:
                        if 'loss' in self.callback_kwargs['metrics']:
                            self.callback_kwargs['metrics'].remove('loss')
                elif option_name == 'plot_metric':
                    if option_value:
                        if not (self.metrics[0] in self.callback_kwargs['metrics']):
                            self.callback_kwargs['metrics'].append(self.metrics[0])
                    else:
                        if self.metrics[0] in self.callback_kwargs['metrics']:
                            self.callback_kwargs['metrics'].remove(self.metrics[0])
                elif option_name == 'plot_final':
                    if option_value:
                        self.callback_kwargs['show_final'] = True
                    else:
                        self.callback_kwargs['show_final'] = False

            if (self.task_type == 'classification') or (self.task_type == 'segmentation'):
                for option_name, option_value in clbck_options.items():
                    if option_name == 'plot_loss_for_classes':
                        if option_value:
                            if not ('loss' in self.callback_kwargs['class_metrics']):
                                self.callback_kwargs['class_metrics'].append('loss')
                        else:
                            if 'loss' in self.callback_kwargs['class_metrics']:
                                self.callback_kwargs['class_metrics'].remove('loss')
                    elif option_name == 'plot_metric_for_classes':
                        if option_value:
                            if not (self.metrics[0] in self.callback_kwargs['class_metrics']):
                                self.callback_kwargs['class_metrics'].append(self.metrics[0])
                        else:
                            if self.metrics[0] in self.callback_kwargs['class_metrics']:
                                self.callback_kwargs['class_metrics'].remove(self.metrics[0])
                    elif option_name == 'show_worst_images' and option_value:
                        if option_value:
                            self.callback_kwargs['show_worst'] = True
                        else:
                            self.callback_kwargs['show_worst'] = False
                    # elif option_name == 'show_best_images':
                    #     if option_value:
                    #         self.callback_kwargs['show_best'] = True
                    #     else:
                    #         self.callback_kwargs['show_best'] = False

            if self.task_type == 'regression':
                for option_name, option_value in clbck_options.items():
                    if option_name == 'plot_scatter':
                        if option_value:
                            self.callback_kwargs['plot_scatter'] = True
                        else:
                            self.callback_kwargs['plot_scatter'] = False

            if self.task_type == 'timeseries':
                for option_name, option_value in clbck_options.items():
                    if option_name == 'plot_autocorrelation':
                        if option_value:
                            self.callback_kwargs['corr_step'] = 10
                        else:
                            self.callback_kwargs['corr_step'] = 0
                    elif option_name == 'plot_pred_and_true':
                        if option_value:
                            self.callback_kwargs['plot_pred_and_true'] = True
                        else:
                            self.callback_kwargs['plot_pred_and_true'] = False
            self.callback_kwargs['data_tag'] = self.DTS.tags[0]
            self.callback_kwargs['dataset'] = self.DTS
            self.initialized_callback = self.clbck_object(**self.callback_kwargs)
            self.callbacks = ([self.initialized_callback])

        else:

            if self.callbacks:
                self.callback_kwargs['data_tag'] = self.DTS.tags[0]
                self.callback_kwargs['dataset'] = self.DTS
                self.initialized_callback = self.clbck_object(**self.callback_kwargs)
                if self.clbck_object in self.callbacks:
                    self.callbacks.remove(self.clbck_object)
                    self.callbacks.append(self.initialized_callback)

    pass

    def show_training_params(self) -> None:
        """
        output the parameters of the neural network: batch_size, epochs, shuffle, callbacks, loss, metrics,
        x_train_shape, num_classes
        """
        msg = f'num_classes = {self.DTS.num_classes}, shape = {self.DTS.x_Train.shape}, epochs = {self.epochs},\n' \
              f'learning_rate={self.learning_rate}, callbacks = {self.callbacks}, batch_size = {self.batch_size},\n' \
              f'shuffle = {self.shuffle}, loss = {self.loss}, metrics = {self.metrics}\n'

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
            model_name = f'model_{self.nn_name}_ep_{self.best_epoch_num:002d}_m_{self.best_metric_result:.4f}'
            file_path_model: str = os.path.join(self.experiment_path, f'{model_name}.h5')
            self.model.save(file_path_model)
            self.Exch.print_2status_bar(('Info', f'Model is saved as {file_path_model}'))
        else:
            self.Exch.print_error(('Error', 'Cannot save. The model is not trained'))
            if not self.django_flag:
                sys.exit()
        pass

    def terra_fit(self, nnmodel, verbose: int = 0) -> None:
        """
        This method created for using wth externally compiled models

        Args:
            verbose:    verbose arg from tensorflow.keras.model.fit

        Return:
            None
        """
        self.model = nnmodel
        self.nn_name = f'{self.model.name}'
        if self.django_flag:
            self.epochs = self.Exch.get_epochs_from_django()
            self.batch_size = self.Exch.get_batch_size_from_django()
            if self.debug_verbose > 1:
                verbose = 2
                print('self.loss', self.loss)
                print('self.metrics', self.metrics)
                print('self.batch_size', self.batch_size)
                print('self.epochs', self.epochs)

        self.prepare_callbacks()
        if self.debug_verbose > 1:
            print('self.callbacks', self.callbacks)
            print('self.callbacks_kwargs', self.callback_kwargs)
            print('self.clbck_object', self.clbck_object)

        self.show_training_params()
        self.history = self.model.fit(self.DTS.x_Train,
                                      self.DTS.y_Train,
                                      batch_size=self.batch_size,
                                      shuffle=self.shuffle,
                                      validation_data=(self.DTS.x_Val, self.DTS.y_Val),
                                      epochs=self.epochs,
                                      verbose=verbose,
                                      callbacks=self.callbacks)
        self.model_is_trained = True

        self.best_epoch, self.best_epoch_num, self.stop_epoch = \
            self._search_best_epoch_data(history=self.history,
                                         monitor=self.monitor,
                                         monitor2=self.monitor2
                                         )
        self.best_metric_result = self.best_epoch[self.monitor]

        self.save_nnmodel()
        pass
