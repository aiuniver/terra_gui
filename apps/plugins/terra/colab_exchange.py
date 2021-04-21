import gc
import os
import re

import dill as dill
from IPython import get_ipython
from django.conf import settings

from terra_ai.trds import DTS


class Exchange:
    """
    Class for exchange data in google colab between django and terra in training mode

    Notes:
        property_of = 'DJANGO' flag for understanding what kind of object we are using now
    """

    def __init__(self):
        # data for output current state of model training process
        self.out_data = {
            "stop_flag": False,
            "status_string": "",
            "progress_status": {
                "percents": 100,
                "progress_text": "",
                "iter_count": 5,
            },
            "errors": "",
            "prints": [],
            "plots": [],
            "scatters": [],
            "images": [],
            "texts": [],
        }

        self.callback_show_options_switches_front = {
            "classification": {
                "show_every_epoch": {"value": False, "label": "Выводить каждую эпоху"},
                "plot_loss_metric": {"value": False, "label": "Выводить loss"},
                "plot_metric": {"value": False, "label": "Выводить данные метрики"},
                "plot_loss_for_classes": {
                    "value": False,
                    "label": "Выводить loss по каждому классу",
                },
                "plot_metric_for_classes": {
                    "value": False,
                    "label": "Выводить данные метрики по каждому классу",
                },
                "show_worst_images": {
                    "value": False,
                    "label": "Выводить худшие изображения по метрике",
                },
                "show_best_images": {
                    "value": False,
                    "label": "Выводить лучшие изображения по метрике",
                },
                "plot_final": {"value": False, "label": "Выводить графики в конце"},
            },
            "segmentation": {
                "show_every_epoch": {"value": False, "label": "Выводить каждую эпоху"},
                "plot_loss_metric": {"value": False, "label": "Выводить loss"},
                "plot_metric": {"value": False, "label": "Выводить данные метрики"},
                "plot_loss_for_classes": {
                    "value": False,
                    "label": "Выводить loss по каждому классу",
                },
                "plot_metric_for_classes": {
                    "value": False,
                    "label": "Выводить данные метрики по каждому классу",
                },
                "show_worst_images": {
                    "value": False,
                    "label": "Выводить худшие изображения по метрике",
                },
                "show_best_images": {
                    "value": False,
                    "label": "Выводить лучшие изображения по метрике",
                },
                "plot_final": {"value": False, "label": "Выводить графики в конце"},
            },
            "regression": {
                "show_every_epoch": {"value": False, "label": "Выводить каждую эпоху"},
                "plot_loss_metric": {"value": False, "label": "Выводить loss"},
                "plot_metric": {"value": False, "label": "Выводить данные метрики"},
                "plot_scatter": {"value": False, "label": "Выводить скаттер"},
                "plot_final": {"value": False, "label": "Выводить графики в конце"},
            },
            "timeseries": {
                "show_every_epoch": {"value": False, "label": "Выводить каждую эпоху"},
                "plot_loss_metric": {"value": False, "label": "Выводить loss"},
                "plot_metric": {"value": False, "label": "Выводить данные метрики"},
                "plot_autocorrelation": {
                    "value": False,
                    "label": "Вывод графика автокорреляции",
                },
                "plot_pred_and_true": {
                    "value": False,
                    "label": "Вывод графиков предсказания и истинного ряда",
                },
                "plot_final": {"value": False, "label": "Выводить графики в конце"},
            },
        }
        self.property_of = "DJANGO"
        self.stop_training_flag = False
        self.process_flag = "dataset"
        self.hardware_accelerator_type = self.get_hardware_accelerator_type()
        self.dts = DTS(exch_obj=self)  # dataset init
        self.custom_datasets = []
        self.custom_datasets_path = f"{settings.TERRA_AI_DATA_PATH}/datasets"
        self.dts_name = None
        self.nn = None  # neural network init
        self.is_trained = False
        self.debug_verbose = 0
        self.model = None
        self.epochs = 20
        self.batch_size = 32
        self.epoch = 1

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

    def _set_data(self, key_name: str, data, stop_flag: bool) -> None:
        """
        Set data to self out data in pozition with key_name
        Args:
            key_name: name of data type, str
            data: formatting recieved data from terra, Any
            stop_flag: flag to stop JS monitor
        """
        if key_name == "plots":
            self.out_data["plots"] = self._reformatting_graphics_data(
                mode="lines", data=data
            )
        elif key_name == "scatters":
            self.out_data["scatters"] = self._reformatting_graphics_data(
                mode="markers", data=data
            )
        elif key_name == "progress_status":
            self.out_data["progress_status"]["progress_text"] = data[0]
            self.out_data["progress_status"]["percents"] = int(float(data[1]) * 100)
            self.out_data["progress_status"]["iter_count"] = data[2]
        elif key_name == "prints":
            self.out_data["prints"].append(data)
        else:
            self.out_data[key_name] = data
        self._check_stop_flag(stop_flag)

    @staticmethod
    def _reformatting_graphics_data(mode: str, data: dict) -> list:
        """
        This method is reformatting input data for graphics to JS format
        Args:
            mode: graphic type: 'lines' or 'markers' (scatter)
            data: graphic data

        Returns: list of lists with graphics data,
                every nested list can include some tuple for different lines in graphic

        """

        out_graphs = []
        current_graph = []
        for title, graph_data in data.items():
            for graph in graph_data:
                current_graph.append(
                    {
                        "x": graph[0],
                        "y": graph[1],
                        "name": graph[2],
                        "mode": mode,
                    }
                )
            out_graphs.append(
                {
                    "list": current_graph,
                    "title": title[0],
                    "xaxis": {"title": title[1]},
                    "yaxis": {"title": title[2]},
                }
            )
            current_graph = []
        return out_graphs

    def _get_custom_datasets_from_google_drive(self):
        custom_datasets_dict = {}
        if os.path.exists(self.custom_datasets_path):
            self.custom_datasets = os.listdir(self.custom_datasets_path)
            for dataset in self.custom_datasets:
                dataset_path = os.path.join(self.custom_datasets_path, dataset)
                with open(dataset_path, 'rb') as f:
                    custom_dts = dill.load(f)
                tags = custom_dts.tags
                name = custom_dts.name
                source = custom_dts.source
                custom_datasets_dict[name] = [tags, None, source]
                del custom_dts
        return custom_datasets_dict

    def _create_datasets_data(self) -> dict:
        """
        Create dataset unique tags
        Returns:
            dict of all datasets and their tags
            "datasets": datasets
            "tags": datasets tags

        """
        tags = set()
        datasets = self.dts.get_datasets_dict()
        custom_datasets = self._get_custom_datasets_from_google_drive()
        datasets.update(custom_datasets)

        for params in datasets.values():
            for i in range(len(params[0])):
                tags.add(params[0][i])
            for param in params[1:]:
                if param:
                    tags.add(param)

        # TODO for next relise step:

        # methods = self.dts.get_datasets_methods()
        # content = {
        #     'datasets': datasets,
        #     'tags': tags,
        #     'methods': methods,
        # }
        tags = dict(
            map(
                lambda item: (self._reformat_tags([item])[0], item),
                list(tags),
            )
        )
        # tags = self._reformat_tags(list(tags))

        content = {
            "datasets": datasets,
            "tags": tags,
        }

        return content

    def _prepare_dataset(self, **options) -> tuple:
        """
        prepare dataset for load to nn
        Args:
            **options: dataset options, such as dataset name, type of task, etc.

        Returns:
            changed dataset and its tags
        """
        self._reset_out_data()
        self.dts = DTS(exch_obj=self)
        gc.collect()
        # if options.get("dataset_name") == "mnist":
        #     self.dts.keras_datasets(dataset="mnist", net="conv", one_hot_encoding=True)
        # else:
        self.dts.prepare_dataset(**options)
        self._set_dts_name(self.dts.name)
        return self.dts.tags, self.dts.name

    def _create_custom_dataset(self, **options):
        dataset = options.get('dataset_name')
        dataset_path = os.path.join(self.custom_datasets_path, dataset)
        with open(dataset_path, 'rb') as f:
            custom_dts = dill.load(f)
        self.dts = custom_dts
        self._set_dts_name(self.dts.name)
        return self.dts.tags, self.dts.name

    @staticmethod
    def _reformat_tags(tags: list) -> list:
        return list(
            map(lambda tag: re.sub("[^a-z^A-Z^а-я^А-Я]+", "_", tag).lower(), tags)
        )

    def _check_stop_flag(self, flag: bool) -> None:
        """
        Checking flag state for JS monitor
        Args:
            flag: bool, recieved from terra
        """
        if flag:
            self.out_data["stop_flag"] = True

    def _reset_out_data(self):
        self.out_data = {
            "stop_flag": False,
            "status_string": "status_string",
            "progress_status": {
                "percents": 0,
                "progress_text": "No some progress",
                "iter_count": None,
            },
            "errors": "error_string",
            "prints": [],
            "plots": [],
            "scatters": [],
            "images": [],
            "texts": [],
        }

    def _set_dts_name(self, dts_name):
        self.dts_name = dts_name

    def prepare_dataset(self, **options):
        self.process_flag = "dataset"
        custom_flag = options.get('source')
        if custom_flag and custom_flag == 'custom':
            return self._create_custom_dataset()
        return self._prepare_dataset(**options)

    def set_stop_training_flag(self):
        """
        Set stop_training_flag in True if STOP button in interface is clicked
        """
        self.stop_training_flag = True

    def set_callbacks_switches(self, task: str, switches: dict):
        for switch, value in switches.items():
            self.callback_show_options_switches_front[task][switch]["value"] = value

    def print_progress_bar(self, data: tuple, stop_flag=False) -> None:
        """
        Print progress bar in status bar

        Args:
            data (tuple):       data[0] string with explanation, data[1] float, data[3] str usually time & etc,
            stop_flag (bool):   added for django
        """
        self._set_data("progress_status", data, stop_flag)

    def print_2status_bar(self, data: tuple, stop_flag=False) -> None:
        """
        Print important messages in status bar

        Args:
            data (tuple):       data[0] string with Method, Class name etc, data[1] string with message
            stop_flag (bool):   added for django
        """
        self._set_data("status_string", f"{data[0]}: {data[1]}", stop_flag)
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
        self._set_data("errors", f"{data[0]}: {data[1]}", stop_flag)
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
        self._set_data("prints", one_string, stop_flag)
        pass

    def show_plot_data(self, data, stop_flag=False) -> None:
        """
        Plot line charts

        Args:
            data (list):        iterable of tuples (x_data, y_data, label)
            stop_flag (bool):   added for django

        Example:
            data: [([1, 2, 3], [10, 20, 30], 'label'), ...]

        Returns:
            None
        """
        self._set_data("plots", data, stop_flag)
        pass

    def show_scatter_data(self, data, stop_flag=False) -> None:
        """
        Plot scattered charts

        Args:
            data (list):        iterable of tuples (x_data, y_data, label)
            stop_flag (bool):   added for django

        Examples:
            data: [([1, 2, 3], [10, 20, 30], 'label'), ...]

        Returns:
            None
        """
        self._set_data("scatters", data, stop_flag)
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
        self._set_data("images", data, stop_flag)
        pass

    def show_text_data(self, data, stop_flag=False) -> None:
        """
        Args:
            data:               strings separated with \n
            stop_flag (bool):   added for django

        Returns:
            None
        """
        self._set_data("texts", data, stop_flag)
        pass

    def get_stop_training_flag(self):
        return self.stop_training_flag

    def get_datasets_data(self):
        return self._create_datasets_data()

    def get_hardware_env(self):
        return self.hardware_accelerator_type

    def get_callbacks_switches(self, task: str) -> dict:
        return self.callback_show_options_switches_front[task]

    def get_state(self, task: str) -> dict:
        data = self.get_datasets_data()
        data.update(
            {
                # "layers_types": self.get_layers_type_list(),
                # "optimizers": self.get_optimizers_list(),
                "callbacks": self.callback_show_options_switches_front.get(task, {}),
                "hardware": self.get_hardware_env(),
            }
        )
        return data

    def get_data(self):
        if self.process_flag == "train":
            self.out_data["progress_status"]["progress_text"] = "Train progress"
            self.out_data["progress_status"]["percents"] = (
                self.epoch / self.epochs
            ) * 100
            self.out_data["progress_status"]["iter_count"] = self.epochs
        return self.out_data

    # def start_training(self, model_plan: object):
    #     if self.debug_verbose == 3:
    #         print(f"Dataset name: {self.dts.name}")
    #         print(f"Dataset shape: {self.dts.input_shape}")
    #         print(f"Plan: ")
    #         for idx, l in enumerate(model_plan.plan, start=1):
    #             print(f"Layer {idx}: {l}")
    #         print(f"x_Train: {self.nn.DTS.x_Train.shape}")
    #         print(f"y_Train: {self.nn.DTS.y_Train.shape}")
    #     self.nn.fit_model_plan(model_plan)
    #     self.out_data["stop_flag"] = True
    #
    # def start_evaluate(self):
    #     self.nn.evaluate()
    #     return self.out_data
    #
    # def start_nn_train(self, batch=32, epoch=20):
    #     if self.is_trained:
    #         self.nn.nn_cleaner()
    #         gc.collect()
    #         self.nn = NN(exch_obj=self)
    #     self.process_flag = "train"
    #     self._reset_out_data()
    #     self.nn.load_dataset(self.dts, task_type=self.current_state["task"])
    # TEST SETTINGS DELETE FOR PROD
    # if self.nn.env_setup == 'raw' and self.dts.name == 'mnist':
    #     self.dts.x_Train = self.dts.x_Train[:1000, :, :]
    #     self.dts.y_Train = self.dts.y_Train[:1000, :]
    #     self.dts.x_Val = self.dts.x_Val[:1000, :, :]
    #     self.dts.y_Val = self.dts.y_Val[:1000, :]

    # @dataclass
    # class MyPlan(LayersDef):
    #     framework = "keras"
    #     input_datatype = self.dts.input_datatype  # Type of data
    #     plan_name = self.current_state.get("model")
    #     num_classes = self.dts.num_classes
    #     input_shape = self.dts.input_shape
    #     plan = self.model_plan
    #
    # self.epochs = int(epoch)
    # self.batch_size = int(batch)

    # TEST PARAMS DELETE FOR PROD
    # if self.nn.env_setup == 'raw':
    #     self.epochs = 1
    #     self.batch_size = 64

    # training = Thread(target=self.start_training, args=(MyPlan,))
    # training.start()
    # training.join()
    # self.is_trained = True
    # return self.out_data


if __name__ == "__main__":
    b = Exchange()
    pass
