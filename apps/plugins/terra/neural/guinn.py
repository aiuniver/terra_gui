from typing import Tuple
import numpy as np
import sys
import os.path
import os
import operator
from tensorflow import keras

__version__ = 0.1


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
        """
        For testing in different setups and environment
        """
        self.debug_mode: bool = True
        self.debug_verbose = 3
        self.default_projects_folder = "TerraProjects"
        self.default_user_model_plans_folder = "ModelPlans"

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
        Checking setup environment
        """
        self.env_setup = "colab"
        if self.Exch.is_it_colab():
            self.env_setup = "colab"

        self.mounted_drive_writable = False

        if not self.Exch.is_google_drive_connected():
            self.Exch.print_2status_bar(
                ("Warning:", f"Google Drive is not connected! Using drive on VM!")
            )
            if self.debug_mode:
                self.mounted_drive_name = ""
                self.mounted_drive_path = "./TerraAI/projects"
                self.mounted_drive_writable = True
        else:
            """
            Setting location for TerraProjects - Home for _current_ user
            """
            (
                self.mounted_drive_name,
                self.mounted_drive_path,
            ) = self.Exch.get_google_drive_name_path()
            self.mounted_drive_writable = True

        self.HOME = os.path.join(self.mounted_drive_path, self.default_projects_folder)
        self.checking_HOME()
        self.default_user_model_plans_path = os.path.join(
            self.HOME, self.default_user_model_plans_folder
        )
        if not os.access(self.default_user_model_plans_path, os.F_OK):
            os.mkdir(self.default_user_model_plans_path)
            self.Exch.print_2status_bar(
                (
                    "Info",
                    f"Created the Home directory "
                    f"{self.default_user_model_plans_path} for keeping projects data",
                )
            )
        else:
            self.Exch.print_2status_bar(
                (
                    "Info",
                    f"The Home directory "
                    f"{self.default_user_model_plans_path} for keeping projects data, already "
                    f"exists",
                )
            )

        pass

        self.nn_name: str = ''
        self.model = None
        # self.external_model: bool = False

        if self.mounted_drive_writable:

            """
            Setting location for Projects in Home directory for _current_ user
            """
            self.project_name: str = ''
            self.project_path: str = ''
            self.set_project_name(self.project_name)

            """
            Setting location for task_name in current project for _current_ user 
            if task_type is currently = ''
            it's setting to None   
            """
            self.task_name: str = ''
            self.task_type: str = ''
            self.task_path: str = ''
            self.set_task_type()

            """
            Setting experiment_UUID and experiment_name 
            """
            self.experiment_name: str = ''
            self.experiment_UUID: str = ''
            self.experiment_path: str = ''
            self.set_experiment_UUID()
            self.set_experiment_name(str(self.experiment_UUID))

            self.best_epoch: dict = {}
            self.best_epoch_num: int = 0
            self.stop_epoch: int = 0
            self.model_is_trained: bool = True
            self.history: dict = {}
            self.best_metric_result = "0000"

            self.learning_rate = 1e-3
            self.optimizer_name: str = ''
            self.loss: dict = {}
            self.metrics: dict = {}
            self.batch_size = 32
            self.epochs = 20
            self.shuffle: bool = True

            self.monitor: str = 'input_1_accuracy'
            self.monitor2: str = "input_1_loss"

            # if not isinstance(self.metrics[0], str):
            #     self.monitor = str(self.metrics[0])
            # else:
            #     self.monitor = self.metrics[0]

    def set_main_params(self, output_params: dict = None, clbck_options: dict = None, callb_chekp: dict = None,
                        shuffle: bool = True, epoch: int = 10, batch_size: int = 32, ) -> None:
        self.output_params = output_params
        for output_key in self.output_params.keys():
            self.metrics.update({output_key: self.output_params[output_key]['metrics']})
            self.loss.update({output_key:self.output_params[output_key]['loss']})



        pass

    def set_dataset(self, dts_obj: object) -> None:
        """
        Setting task nn_name

        Args:
            dts_obj (object): setting task_name
        """
        self.DTS = dts_obj
        pass

    def set_callback(self, callback_obj: object) -> None:
        """
        Setting task nn_name

        Args:
            callback_obj (object): setting callbacks
        """
        self.callbacks = callback_obj
        pass

    def checking_HOME(self) -> None:
        """
        Checking mounted drive for write access and if it's writable,
        checking HOME directory for self.default_projects_folder
        if its not found, create this folder
        Also set the flag self.mounted_drive_writable
        Printing info and error message about write status

        """
        if os.access(self.mounted_drive_path, os.W_OK):
            self.mounted_drive_writable = True
            self.Exch.set_mounted_drive_status(True)
            if not os.access(self.HOME, os.F_OK):
                os.mkdir(self.HOME)
                self.Exch.print_2status_bar(
                    (
                        "info",
                        f"Created the Home directory {self.HOME} for keeping projects data",
                    )
                )
            else:
                if self.debug_verbose >= 3:
                    self.Exch.print_2status_bar(
                        (
                            "info",
                            f"The Home directory {self.HOME} for keeping projects data, already "
                            f"exists",
                        )
                    )
        else:
            self.Exch.print_error(
                (
                    "Error",
                    f"The mounted drive {self.mounted_drive_path} is not writable. "
                    f"Check mounted drive for write access",
                )
            )
            os.makedirs(self.mounted_drive_path)
            # sys.exit()
        pass

    def set_project_name(self, project_name: str) -> None:
        """
        Setting project nn_name

        Args:
            project_name (str):   nn_name of the project, also used as sub directory
        """
        if project_name == "":
            self.project_name = "noname_project"
        else:
            self.project_name = project_name
        self.project_path = os.path.join(self.HOME, self.project_name)
        pass

    def set_task_type(self) -> None:
        """
        Setting task_type to crete logistic 'pipe' from start to end
        also set the task_name to same string value if it's not changed from default
        """
        self.task_name = self.Exch.task_name
        self.task_type = self.task_name

        if self.task_type == "":
            self.task_type = None
        else:
            self.task_path = os.path.join(self.project_path, self.task_type)
        pass

    def set_task_name(self, task_name: str) -> None:
        """
        Setting task nn_name

        Args:
            task_name (str): setting task_name
        """
        self.task_name = task_name
        self.Exch.set_task_name(self.task_name)
        pass

    def set_experiment_UUID(self) -> None:
        """
        Setting experiment UUID
        """

        self.experiment_UUID = self.Exch.experiment_UUID
        self.experiment_path = os.path.join(self.task_path, str(self.experiment_UUID))
        pass

    def set_experiment_name(self, experiment_name: str) -> None:
        """
        Setting experiment nn_name

        Args:
            experiment_name (str): setting experiment nn_name
        """
        self.experiment_name = experiment_name
        pass

    # def load_dataset(self, dataset_obj: trds.DTS, task_type: str) -> None:
    #     """
    #     Load dataset object
    #
    #     Args:
    #         dataset_obj (object):   trds.DTS dataset object
    #         task_type (str):        task type for NN
    #
    #     Returns:
    #         None
    #     """
    #     # print('___nn___NN___load_dataset___', task_type)
    #     self.DTS = dataset_obj
        self.input_datatype = self.DTS.input_datatype
    #     self._reinit(task_type)
    #     pass
    #
    # def _reinit(self, task_type) -> None:
    #     # print('___nn___NN___reinit___', task_type)
    #     self.set_task_type(task_type)
    #     self.set_experiment_UUID()
    #     if self.django_flag:
    #         task_type_defaults_kwargs = self.task_type_defaults_dict.get(task_type)
    #         self.Exch.set_task_type_defaults(task_type_defaults_kwargs)
    #     self._reset()
    #     pass
    #
    # def _reset(self):
    #     # print('___nn___NN___reset___')
    #     self.callbacks = []
    #     pass

    def show_training_params(self) -> None:
        """
        output the parameters of the neural network: batch_size, epochs, shuffle, callbacks, loss, metrics,
        x_train_shape, num_classes
        """
        x_shape = []
        v_shape = []
        t_shape = []

        for i in self.DTS.X.keys():
            x_shape.append([i, self.DTS.X[i]['data'][0].shape])
            v_shape.append([i, self.DTS.X[i]['data'][1].shape])
            t_shape.append([i, self.DTS.X[i]['data'][2].shape])

        msg = f'num_classes = {self.DTS.num_classes}, x_Train_shape = {x_shape}, x_Val_shape = {v_shape}, \n'\
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
            model_name = f"model_{self.nn_name}_ep_{self.best_epoch_num:002d}_m_{self.best_metric_result:.4f}"
            file_path_model: str = os.path.join(
                self.experiment_path, f"{model_name}_last.h5"
            )
            self.model.save(file_path_model)
            self.Exch.print_2status_bar(
                ("Info", f"Model is saved as {file_path_model}")
            )
        else:
            self.Exch.print_error(("Error", "Cannot save. The model is not trained"))
            sys.exit()
        pass

    def save_model_weights(self) -> None:
        """
        Saving model weights if the model is trained

        Returns:
            None
        """

        if self.model_is_trained:
            model_weights_name = f'weights_{self.nn_name}_ep_{self.best_epoch_num:002d}_m_{self.best_metric_result:.4f}'
            file_path_weights: str = os.path.join(self.experiment_path, f'{model_weights_name}.h5')
            self.model.save_weights(file_path_weights)
            self.Exch.print_2status_bar(('info', f'Weights are saved as {file_path_weights}'))
        else:
            self.Exch.print_error(('Error', 'Cannot save. The model is not trained'))

        pass

    def prepare_dataset(self) -> None:
        """
        reformat samples of dataset

        Returns:
            None
        """

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
        pass

    def terra_fit(self, nnmodel, verbose: int = 0) -> None:
        """
        This method created for using wth externally compiled models

        Args:
            nnmodel (obj): keras model for fit
            verbose:    verbose arg from tensorflow.keras.model.fit

        Return:
            None
        """
        self.model = nnmodel
        self.nn_name = f"{self.model.name}"
        self.shuffle = self.Exch.shuffle
        self.loss = self.Exch.get_loss_from_django()
        self.metrics = self.Exch.get_metrics_from_django()
        self.epochs = self.Exch.get_epochs_from_django()
        self.batch_size = self.Exch.get_batch_size_from_django()
        if self.debug_verbose > 1:
            verbose = 2
            print("self.loss", self.loss)
            print("self.metrics", self.metrics)
            print("self.batch_size", self.batch_size)
            print("self.epochs", self.epochs)

        if self.debug_verbose > 1:
            print("self.callbacks", self.callbacks)

        self.prepare_dataset()
        self.show_training_params()

        if self.x_Val['input_1'] is not None:

            self.history = self.model.fit(
                self.x_Train,
                self.y_Train,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                validation_data=(self.x_Val, self.y_Val),
                epochs=self.epochs,
                verbose=verbose,
                callbacks=[self.callbacks, keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.experiment_path, f'{self.nn_name}_best.h5'), verbose=1,
                    save_best_only=True, save_weights_only=True, monitor='loss', mode='min')]
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
                callbacks=[self.callbacks, keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.experiment_path, f'{self.nn_name}_best.h5'), verbose=1,
                    save_best_only=True, save_weights_only=True, monitor='loss', mode='min')]
            )
        self.model_is_trained = True

        for n_out in self.DTS.Y.keys():
            for _ in self.loss[n_out]:
                for metric_out in self.metrics[n_out]:
                    self.monitor = f'{n_out}_{metric_out}'
                    self.monitor2 = f'{n_out}_loss'
                    self.best_epoch, self.best_epoch_num, self.stop_epoch = self._search_best_epoch_data(
                        history=self.history, monitor=self.monitor, monitor2=self.monitor2
                    )
                    self.best_metric_result = self.best_epoch[self.monitor]

        try:
            self.save_nnmodel()
        except RuntimeError:
            self.Exch.print_2status_bar(('Warning', 'Save model failed'))
        self.save_model_weights()

        pass

    @staticmethod
    def _search_best_epoch_data(
            history, monitor="output_1_val_accuracy", monitor2="output_1_loss"
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
        max_monitors = ["accuracy", "dice_coef"]
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
