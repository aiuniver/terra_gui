from typing import Tuple
import matplotlib.pyplot as plt
import os
import re
import tensorflow.keras.optimizers
from IPython import get_ipython

__version__ = 0.036


class Exchange:
    """
    Class for exchange data in Google Colab between django and terra
    """

    def __init__(self):
        """
        Notes:
            property_of = 'TERRA' flag for understanding what kind of object we are using now
        """
        self.property_of = "TERRA"
        self.stop_flag = False
        self.mounted_drive_name = ""
        self.mounted_drive_path = "/content"
        self.mounted_drive_writable = False
        self.nn_init_fail = True
        self.stop_flag = False
        self.epoch = 0

        self.experiment_UUID = 'unic_uiid'
        self.experiment_name = ""
        self.task_name = ""

        self.optimizer_name = "Adam"
        self.loss = "categorical_crossentropy"
        self.metrics = ["accuracy"]
        self.batch_size = 32
        self.epochs = 20
        self.shuffle = True
        self.callbacks = None
        self.load_option = "best"
        self.callback_show_options_switches = {
            "classification": {
                "show_every_epoch": False,  # выводить каждую эпоху
                "plot_loss_metric": False,  # выводить loss
                "plot_metric": False,  # выводить данные метрики
                "plot_loss_for_classes": False,  # выводить loss по каждому классу
                "plot_metric_for_classes": False,  # выводить данные метрики по каждому классу
                "show_worst_images": False,  # выводить худшие изображения по метрике
                "show_best_images": False,  # выводить лучшие изображения по метрике
                "plot_final": False,  # выводить графики в конце
            },
            "segmentation": {
                "show_every_epoch": False,  # выводить каждую эпоху
                "plot_loss_metric": False,  # выводить loss
                "plot_metric": False,  # выводить данные метрики
                "plot_loss_for_classes": False,  # выводить loss по каждому классу
                "plot_metric_for_classes": False,  # выводить данные метрики по каждому классу
                "show_worst_images": False,  # выводить худшие изображения по метрике
                "show_best_images": False,  # выводить лучшие изображения по метрике
                "plot_final": False,  # выводить графики в конце
            },
            "regression": {
                "show_every_epoch": False,  # выводить каждую эпоху
                "plot_loss_metric": False,  # выводить loss
                "plot_metric": False,  # выводить данные метрики
                "plot_scatter": False,  # выводить скаттер
                "plot_final": False,  # выводить графики в конце
            },
            "timeseries": {
                "show_every_epoch": False,  # выводить каждую эпоху
                "plot_loss_metric": False,  # выводить loss
                "plot_metric": False,  # выводить данные метрики
                "plot_autocorrelation": False,  # вывод графика автокорреляции
                "plot_pred_and_true": False,  # вывод графиков предсказания и истинного ряда
                "plot_final": False,  # выводить графики в конце
            },
        }
        self.models_list = []
        self.models_plans_path = str()
        pass

    def set_experiment_UUID(self, experiment_UUID) -> None:
        self.experiment_UUID = experiment_UUID
        pass

    def set_experiment_name(self, experiment_name) -> None:
        self.experiment_name = experiment_name
        pass

    def set_task_name(self, task_name):
        self.task_name = task_name
        pass

    def set_mounted_drive_status(self, status: bool) -> None:
        self.mounted_drive_writable = status
        pass

    def set_task_type_defaults(self, task_type_defaults_kwargs) -> None:
        for __var_name, __var_value in task_type_defaults_kwargs.items():
            setattr(self, __var_name, __var_value)
        pass

    def get_task_type_params_from_django(self, task_type_defaults_kwargs):
        """
        Get task_type params data to set it in terra
        Args:
            task_type_defaults_kwargs (dict): dictionary with keys we need to create new dictionary

        Returns:
            task_type_kwargs (dict): variables from GUI specific for task_type
        """
        task_type_kwargs = {}
        for __var_name in task_type_defaults_kwargs.keys():
            task_type_kwargs.update({__var_name, getattr(self, __var_name)})
        return task_type_kwargs

    def get_callback_show_options_from_django(self, task_type) -> dict:
        """
        Get callback options from django to set it in terra

        Returns:
            self.callback_options (dict):      List with True and False, for each switch button
        """
        return self.callback_show_options_switches[task_type]

    @staticmethod
    def get_losses_dict(task_type):
        available_losses = re.findall(
            "([A-Z]+[a-z]*)", " ".join(dir(tensorflow.keras.losses))
        )
        available_losses.remove("loss")
        # dict with known losses
        losses_dict = {
            "classification": {
                "squared_hinge",
                "hinge",
                "categorical_hinge",
                "categorical_crossentropy",
                "sparse_categorical_crossentropy",
                "binary_crossentropy",
                "kl_divergence",
                "poisson",
            },
            "regression": {
                "mse",
                "mae",
                "mape",
                "msle",
                "log_cosh",
                "cosine_similarity",
            },
            "segmentation": {
                "dice_coef",
                "squared_hinge",
                "hinge",
                "categorical_hinge",
                "categorical_crossentropy",
                "sparse_categorical_crossentropy",
                "binary_crossentropy",
                "kl_divergence",
                "poisson",
            },
            "timeseries": {
                "mse",
                "mae",
                "mape",
                "msle",
                "log_cosh",
                "cosine_similarity",
            },
        }

        # kill disappeared optimizers
        remove = [k for k in losses_dict[task_type] if k not in available_losses]
        for k in remove:
            del losses_dict[task_type][k]
        return losses_dict[task_type]

    def get_loss_from_django(self):
        """
        Get loss data to set it in terra

        Returns:
            self.loss (str):      loss name
        """
        return self.loss

    def get_epochs_from_django(self):
        """
        Get epochs q-ty to set it in terra

        Returns:
            self.epochs (int):  epochs q-ty
        """
        return self.epochs

    def get_batch_size_from_django(self):
        """
        Get batch_size q-ty to set it in terra

        Returns:
            self.batch_size (int):  batch_size q-ty
        """
        return self.batch_size

    def get_models_list_from_terra(self):
        """
        Get models list from models directory

        Returns:
            models_list (list): list from directory with plans
        """
        current_path = os.path.abspath(os.path.dirname(__file__))
        self.models_plans_path = os.path.join(
            "/".join(current_path.split("/")[:-1]), "networks/plans/"
        )
        self.models_list = []
        for filename in os.listdir(self.models_plans_path):
            if filename.endswith(".yaml"):
                self.models_list.append(filename)
        pass

    @staticmethod
    def get_metrics_dict(task_type):
        available_metrics = re.findall(
            "([A-Z]+[a-z]*)", " ".join(dir(tensorflow.keras.metrics))
        )
        if "metrics" in available_metrics:
            available_metrics.remove("metrics")
        # print(available_metrics)
        # dict with known metrics
        metrics_dict = {
            "classification": [
                "accuracy",
                "binary_accuracy",
                "binary_crossentropy",
                "categorical_accuracy",
                "categorical_crossentropy",
                "sparse_categorical_accuracy",
                "sparse_categorical_crossentropy",
                "top_k_categorical_accuracy",
                "sparse_top_k_categorical_accuracy",
                "hinge",
                "kl_divergence",
                "binary_crossentropy",
                "poisson",
            ],
            "regression": ["accuracy", "mae", "mse", "mape", "msle", "log_cosh"],
            "segmentation": [
                "dice_coef",
                "meanIoU",
                "accuracy",
                "binary_accuracy",
                "binary_crossentropy",
                "categorical_accuracy",
                "categorical_crossentropy",
                "sparse_categorical_accuracy",
                "sparse_categorical_crossentropy",
                "top_k_categorical_accuracy",
                "sparse_top_k_categorical_accuracy",
                "hinge",
                "kl_divergence",
                "binary_crossentropy",
                "poisson",
            ],
            "timeseries": ["accuracy", "mse", "mae", "mape", "msle", "log_cosh"],
        }

        # kill disappeared metrics
        remove = [
            k for k in metrics_dict[task_type] if k.title() not in available_metrics
        ]
        for k in remove:
            metrics_dict[task_type].remove(k)
        return metrics_dict[task_type]

    def get_metrics_from_django(self):
        """
        Get metrics data to set it in terra

        Returns:
            self.metrics (list):      list with metrics
        """
        return self.metrics

    @staticmethod
    def get_optimizers_dict_from_terra():
        available_optimizers = re.findall(
            "([A-Z]+[a-z]*)", " ".join(dir(tensorflow.keras.optimizers))
        )
        available_optimizers.remove("Optimizer")

        """
        # "kwargs": ["clipnorm", "clipvalue"] - was removed for future development
        """

        # dict with known optimizers
        optimizers_dict = {
            "SGD": {"learning_rate": 0.01, "momentum": 0.0, "nesterov": False},
            "RMSprop": {
                "learning_rate": 0.001,
                "rho": 0.9,
                "momentum": 0.0,
                "epsilon": 1e-07,
                "centered": False,
            },
            "Adam": {
                "learning_rate": 0.001,
                "beta_1": 0.9,
                "beta_2": 0.999,
                "epsilon": 1e-07,
                "amsgrad": False,
            },
            "Adadelta": {"learning_rate": 0.001, "rho": 0.95, "epsilon": 1e-07},
            "Adagrad": {
                "learning_rate": 0.001,
                "initial_accumulator_value": 0.1,
                "epsilon": 1e-07,
            },
            "Adamax": {
                "learning_rate": 0.001,
                "beta_1": 0.9,
                "beta_2": 0.999,
                "epsilon": 1e-07,
            },
            "Nadam": {
                "learning_rate": 0.001,
                "beta_1": 0.9,
                "beta_2": 0.999,
                "epsilon": 1e-07,
            },
            "Ftrl": {
                "learning_rate": 0.001,
                "learning_rate_power": -0.5,
                "initial_accumulator_value": 0.1,
                "l1_regularization_strength": 0.0,
                "l2_regularization_strength": 0.0,
                "l2_shrinkage_regularization_strength": 0.0,
                "beta": 0.0,
            },
        }

        # kill disappeared optimizers
        remove = [k for k in optimizers_dict if k not in available_optimizers]
        for k in remove:
            del optimizers_dict[k]
        return optimizers_dict

    def get_optimizer_params_from_django(self):
        """
        Get optimizer data to set it in terra

        Returns:
            self.optimizer_name (str):      name of optimizer
            optimizer_object (object):      optimizer object itself
            self.optimizers_kwargs (dict):  optimizers kwargs (from GUI)
        """
        optimizer_object = getattr(tensorflow.keras.optimizers, self.optimizer_name)
        return self.optimizer_name, optimizer_object, self.optimizers_kwargs

    @staticmethod
    def is_google_drive_connected():
        """
        Boolean indicator of google drive mounting state

        Returns:
            (bool): true if drive is on otherwise false
        """
        if os.access("/content/drive/", os.F_OK):
            return True
        return False

    def get_google_drive_name_path(self) -> Tuple[str, str]:
        """
        Getting mounted Google Drive name for current user if drive mounted
        Setting this variable to Exchange object as self.mounted_drive_name
        Setting path to Exchange object as self.mounted_drive_path
        and return both

        Return:
            mounted_drive_name (str): Only name of mounted drive
            mounted_drive_path (str): path to root of the mounted Google Drive
        """
        if self.is_google_drive_connected():
            _, folders, _ = next(os.walk("/content/drive/"))
            for folder in folders:
                if not folder.startswith("."):
                    self.mounted_drive_name = folder
                    self.mounted_drive_path = os.path.join("/content/drive/", folder)
        return self.mounted_drive_name, self.mounted_drive_path

    def print_progress_bar(self, data: tuple, stop_flag=False) -> None:
        """
        Print progress bar in status bar

        Args:
            data (tuple):       data[0] string with explanation, data[1] float, data[3] str usually time & etc,
            stop_flag (bool):   added for django
        """
        # print(f"{data[0]}: {data[1]} : {data[2]}")
        pass

    def print_2status_bar(self, data: tuple, stop_flag=False) -> None:
        """
        Print important messages in status bar

        Args:
            data (tuple):       data[0] string with Method, Class name etc, data[1] string with message
            stop_flag (bool):   added for django
        """
        # print(f"{data[0]}: {data[1]}")
        pass

    def print_error(self, data: tuple, stop_flag=False) -> None:
        """
        Print important messages: errors, warnings & etc

        Args:
            data (tuple):       data[0] string with message type, data[1] string with message
            stop_flag (bool):   added for django

        Example:
            data = ('Error', 'Project directory not found')
        """
        # print(f"{data[0]}: {data[1]}")
        pass

    def print_epoch_monitor(self, one_string, stop_flag=False) -> None:
        """
        Print block of text

        Args:
            one_string (str):   one string. can be separated by \n
            stop_flag (bool):   added for django

        Returns:
            None
        """
        # print(one_string)
        pass

    def print_training_status(self, data, stop_flag=False) -> None:
        """
        Print messages about training progress in training window

        Args:
            data (tuple):       data[0] string with Method, Class name etc, data[1] string with message
            stop_flag (bool):   added for django
        """
        # print(f"{data[0]}: {data[1]}")
        pass

    def set_nn_init_fail(self):
        self.nn_init_fail = True
        self.stop_flag = True
        pass

    def show_plot_data(self, data, stop_flag=False) -> None:
        """
        Plot line charts

        Args:
            data (dict):        dict of view (graph title, iterable of lists of tuples (x_data, y_data, label))
            stop_flag (bool):   added for django

        Examples:
            data (dict): {('title for graph1', 'x axis label1', 'y axis label1'): [([1, 2, 3], [10, 20, 30], 'label1'),
                          ([1, 2, 3], [40, 50, 60], 'label2')],
                          ('title for graph2', 'x axis label2', 'y axis label2'): [(...)]}

        Returns:
            None
        """
        length = len(data)
        fig = plt.figure(figsize=(6 * length, 6))
        for i, (labels_list, data_list) in enumerate(data.items()):
            title, xlabel, ylabel = labels_list
            ax = fig.add_subplot(1, length, i + 1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_axisbelow(True)
            ax.set_title(title)
            ax.minorticks_on()
            ax.grid(which="major", linestyle="-", linewidth="0.5", color="gray")
            ax.grid(which="minor", linestyle=":", linewidth="0.5", color="gray")
            for x_data, y_data, label in data_list:
                ax.plot(x_data, y_data, label=label)
            plt.legend()
        plt.tight_layout()
        plt.show()
        pass

    def show_scatter_data(self, data, stop_flag=False) -> None:
        """
        Plot scattered charts

        Args:
            data (dict):        ((graph title, x label, y label), iterable of lists of tuples (x_data, y_data, label))
            stop_flag (bool):   added for django

        Examples:
            data (dict): {('title for graph1', 'x axis label1', 'y axis label1'): [([1, 2, 3], [10, 20, 30], 'label1'),
                          ([1, 2, 3], [40, 50, 60], 'label2')],
                          ('title for graph2', 'x axis label2', 'y axis label2'): [(...)]}

        Returns:
            None
        """
        length = len(data)
        fig = plt.figure(figsize=(6 * length, 6))
        for i, (labels_list, data_list) in enumerate(data.items()):
            title, xlabel, ylabel = labels_list
            ax = fig.add_subplot(1, length, i + 1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            for x_data, y_data, label in data_list:
                ax.scatter(x_data, y_data, label=label)
            plt.legend()
        plt.tight_layout()
        plt.show()
        pass

    def show_image_data(self, data, stop_flag=False) -> None:
        """
        Plot numpy arrays containing images (3 rows maximum)

        Args:
            data (list):        iterable of tuples (image, title)
            stop_flag (bool):   added for django

        Returns:
            None

        Notes:
            image must be numpy array
        """
        length = len(data)
        # Count of rows to show and images in row
        rows = 3
        columns = length // rows if length % 3 == 0 else length // rows + 1

        fig = plt.figure(figsize=(5 * columns, 5 * rows))
        for i, (image, title) in enumerate(data):
            ax = fig.add_subplot(rows, columns, i + 1)
            ax.imshow(image)
            ax.set_title(title)
        plt.tight_layout()
        plt.show()
        pass

    def show_text_data(self, data, stop_flag=False) -> None:
        """
        Args:
            data:               strings separated with \n
            stop_flag (bool):   added for django

        Returns:
            None
        """
        # print(data)
        pass

    @staticmethod
    def is_it_colab() -> bool:
        """
        Checking google colab presence

        Returns:
            (bool): True if running in colab, False if is not
        """
        # if "google.colab" in str(get_ipython()):
        #     return True
        # else:
        #     return False
        try:
            _ = os.environ["COLAB_GPU"]
            return True
        except KeyError:
            return False

    @staticmethod
    def is_it_jupyter() -> bool:
        """
        Checking jupyter presence

        Returns:
            (bool): True if running in jupyter, False if is not
        """
        if "ipykernel" in str(get_ipython()):
            return True
        else:
            return False

    @staticmethod
    def get_hardware_accelerator_type() -> str:
        """
        Check and return accelerator
        Possible values: 'CPU', 'GPU', 'TPU'

        Returns:
            res_type (str): name of current accelerator type
        """
        import tensorflow as tf

        # Check if GPU is active
        device_name = tf.test.gpu_device_name()
        if device_name != "/device:GPU:0":
            if Exchange.is_it_colab():
                try:
                    # Try TPU initialize
                    _ = tf.distribute.cluster_resolver.TPUClusterResolver()
                    res_type = "TPU"
                except ValueError:
                    res_type = "CPU"
            else:
                res_type = "CPU"
        else:
            res_type = "GPU"
        return res_type


