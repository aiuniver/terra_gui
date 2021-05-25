import base64
import gc
import os
import re
import tempfile
import zipfile

import dill as dill
import yaml
from IPython import get_ipython
from django.conf import settings
from tensorflow.keras.models import load_model

from terra_ai.trds import DTS
from terra_ai.guiexchange import Exchange as GuiExch
from apps.plugins.terra.neural.guinn import GUINN

from .layers_dataclasses import LayersDef, GUILayersDef
from .data import (
    LayerLocation,
    LayerType,
    OptimizerParams,
    ModelPlan,
    TerraExchangeProject,
)


# Dense
# Conv2D
# Maxpooling2D
# Conv1D
# Maxpooling1D
# LSTM
# Flatten
# Dropout
# Batchnormalization
# Concatenate
# Embedding
# Conv2DTranspose
# Conv1DTranspose
# Upsampling2D
# Upsampling1D


class StatesData:
    def __init__(self):
        self.optimizers_dict = {
            "SGD": {
                "main": {"learning_rate": {"type": "float", "default": 0.01}},
                "extra": {
                    "momentum": {"type": "float", "default": 0.0},
                    "nesterov": {"type": "bool", "default": False},
                },
            },
            "RMSprop": {
                "main": {"learning_rate": {"type": "float", "default": 0.001}},
                "extra": {
                    "rho": {"type": "float", "default": 0.9},
                    "momentum": {"type": "float", "default": 0.0},
                    "epsilon": {"type": "float", "default": 1e-07},
                    "centered": {"type": "bool", "default": False},
                },
            },
            "Adam": {
                "main": {"learning_rate": {"type": "float", "default": 0.001}},
                "extra": {
                    "beta_1": {"type": "float", "default": 0.9},
                    "beta_2": {"type": "float", "default": 0.999},
                    "epsilon": {"type": "float", "default": 1e-07},
                    "amsgrad": {"type": "bool", "default": False},
                },
            },
            "Adadelta": {
                "main": {"learning_rate": {"type": "float", "default": 0.001}},
                "extra": {
                    "rho": {"type": "float", "default": 0.95},
                    "epsilon": {"type": "float", "default": 1e-07},
                },
            },
            "Adagrad": {
                "main": {"learning_rate": {"type": "float", "default": 0.001}},
                "extra": {
                    "initial_accumulator_value": {"type": "float", "default": 0.1},
                    "epsilon": {"type": "float", "default": 1e-07},
                },
            },
            "Adamax": {
                "main": {"learning_rate": {"type": "float", "default": 0.001}},
                "extra": {
                    "beta_1": {"type": "float", "default": 0.9},
                    "beta_2": {"type": "float", "default": 0.999},
                    "epsilon": {"type": "float", "default": 1e-07},
                },
            },
            "Nadam": {
                "main": {"learning_rate": {"type": "float", "default": 0.001}},
                "extra": {
                    "beta_1": {"type": "float", "default": 0.9},
                    "beta_2": {"type": "float", "default": 0.999},
                    "epsilon": {"type": "float", "default": 1e-07},
                },
            },
            "Ftrl": {
                "main": {"learning_rate": {"type": "float", "default": 0.001}},
                "extra": {
                    "lr_power": {"type": "float", "default": -0.5},
                    "initial_accumulator_value": {"type": "float", "default": 0.1},
                    "l1_regularization_strength": {"type": "float", "default": 0.0},
                    "l2_regularization_strength": {"type": "float", "default": 0.0},
                    "l2_shrinkage_regularization_strength": {
                        "type": "float",
                        "default": 0.0,
                    },
                    "beta": {"type": "float", "default": 0.0},
                },
            },
        }

        # list of values for activation attribute of layer
        self.activation_values = [
            None,
            "sigmoid",
            "softmax",
            "tanh",
            "relu",
            "elu",
            "selu",
        ]

        # list of values for padding attribute of layer
        self.padding_values = ["valid", "same"]

        # dict of layers attributes in format for front
        self.layers_params_source = GUILayersDef
        self.layers_params = self.layers_params_source.layers_params

        self.states_for_outputs = {
            "classification": {
                "losses": [
                    "squared_hinge",
                    "hinge",
                    "categorical_hinge",
                    "categorical_crossentropy",
                    "sparse_categorical_crossentropy",
                    "binary_crossentropy",
                    "kl_divergence",
                    "poisson",
                ],
                "metrics": [
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
            },
            "segmentation": {
                "losses": [
                    "dice_coef",
                    "squared_hinge",
                    "hinge",
                    "categorical_hinge",
                    "categorical_crossentropy",
                    "sparse_categorical_crossentropy",
                    "binary_crossentropy",
                    "kl_divergence",
                    "poisson",
                ],
                "metrics": [
                    "dice_coef",
                    "mean_io_u",
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
            },
            "regression": {
                "losses": [
                    "mse",
                    "mae",
                    "mape",
                    "msle",
                    "log_cosh",
                    "cosine_similarity",
                ],
                "metrics": [
                    "accuracy",
                    "mae",
                    "mse",
                    "mape",
                    "msle",
                    "log_cosh",
                ],
            },
            "timeseries": {
                "losses": [
                    "mse",
                    "mae",
                    "mape",
                    "msle",
                    "log_cosh",
                    "cosine_similarity",
                ],
                "metrics": [
                    "accuracy",
                    "mse",
                    "mae",
                    "mape",
                    "msle",
                    "log_cosh",
                ],
            },
        }

        self.callback_show_options_switches_front = {
            "classification": {
                "show_every_epoch": {
                    "type": "bool",
                    "default": False,
                    "label": "каждую эпоху",
                },
                "plot_loss_metric": {"type": "bool", "default": False, "label": "loss"},
                "plot_metric": {
                    "type": "bool",
                    "default": False,
                    "label": "данные метрики",
                },
                "plot_loss_for_classes": {
                    "type": "bool",
                    "default": False,
                    "label": "loss по каждому классу",
                },
                "plot_metric_for_classes": {
                    "type": "bool",
                    "default": False,
                    "label": "данные метрики по каждому классу",
                },
                "show_worst_images": {
                    "type": "bool",
                    "default": False,
                    "label": "худшие изображения по метрике",
                },
                "show_best_images": {
                    "type": "bool",
                    "default": False,
                    "label": "лучшие изображения по метрике",
                },
                "plot_final": {
                    "type": "bool",
                    "default": False,
                    "label": "графики в конце",
                },
            },
            "segmentation": {
                "show_every_epoch": {
                    "type": "bool",
                    "default": False,
                    "label": "каждую эпоху",
                },
                "plot_loss_metric": {"type": "bool", "default": False, "label": "loss"},
                "plot_metric": {
                    "type": "bool",
                    "default": False,
                    "label": "данные метрики",
                },
                "plot_loss_for_classes": {
                    "type": "bool",
                    "default": False,
                    "label": "loss по каждому классу",
                },
                "plot_metric_for_classes": {
                    "type": "bool",
                    "default": False,
                    "label": "данные метрики по каждому классу",
                },
                "show_worst_images": {
                    "type": "bool",
                    "default": False,
                    "label": "худшие изображения по метрике",
                },
                "show_best_images": {
                    "type": "bool",
                    "default": False,
                    "label": "лучшие изображения по метрике",
                },
                "plot_final": {
                    "type": "bool",
                    "default": False,
                    "label": "графики в конце",
                },
            },
            "regression": {
                "show_every_epoch": {
                    "type": "bool",
                    "default": False,
                    "label": "каждую эпоху",
                },
                "plot_loss_metric": {"type": "bool", "default": False, "label": "loss"},
                "plot_metric": {
                    "type": "bool",
                    "default": False,
                    "label": "данные метрики",
                },
                "plot_scatter": {"type": "bool", "default": False, "label": "скаттер"},
                "plot_final": {
                    "type": "bool",
                    "default": False,
                    "label": "графики в конце",
                },
            },
            "timeseries": {
                "show_every_epoch": {
                    "type": "bool",
                    "default": False,
                    "label": "каждую эпоху",
                },
                "plot_loss_metric": {"type": "bool", "default": False, "label": "loss"},
                "plot_metric": {
                    "type": "bool",
                    "default": False,
                    "label": "данные метрики",
                },
                "plot_autocorrelation": {
                    "type": "bool",
                    "default": False,
                    "label": "график автокорреляции",
                },
                "plot_pred_and_true": {
                    "type": "bool",
                    "default": False,
                    "label": "графики предсказания и истинного ряда",
                },
                "plot_final": {
                    "type": "bool",
                    "default": False,
                    "label": "графики в конце",
                },
            },
        }

        self.paths_obj = TerraExchangeProject()


class Exchange(StatesData, GuiExch):
    """
    Class for exchange data in google colab between django and terra in training mode

    Notes:
        property_of = 'DJANGO' flag for understanding what kind of object we are using now
    """

    def __init__(self):
        StatesData.__init__(self)
        GuiExch.__init__(self)
        # data for output current state of model training process
        self.out_data = {
            "stop_flag": True,
            "status_string": "",
            "progress_status": {
                "percents": 0,
                "progress_text": "",
                "iter_count": 0,
            },
            "errors": "",
            "prints": [],
            "plots": [],
            "scatters": [],
            "images": [],
            "texts": [],
        }

        self.property_of = "DJANGO"
        self.stop_training_flag = False
        self.process_flag = "dataset"
        self.hardware_accelerator_type = self.get_hardware_accelerator_type()
        self.layers_list = self._set_layers_list()
        self.start_layers = {}
        self.dts = DTS(exch_obj=self)  # dataset init
        self.custom_datasets = []
        self.custom_datasets_path = self.paths_obj.gd.datasets
        self.dts_name = None
        self.task_name = ""
        self.mounted_drive_path = ''
        self.nn = GUINN(exch_obj=self)  # neural network init
        self.is_trained = False
        self.debug_verbose = 0
        self.model = None
        self.loss = "categorical_crossentropy"
        self.metrics = ["accuracy"]
        self.batch_size = 32
        self.epochs = 20
        self.shuffle = True
        self.epoch = 1
        self.optimizers = self._set_optimizers()
        self.dir_paths = self.paths_obj.dir
        self.gd_paths = self.paths_obj.gd

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
    def is_google_drive_connected():
        """
        Boolean indicator of google drive mounting state

        Returns:
            (bool): true if drive is on otherwise false
        """
        if os.access("/content/drive/", os.F_OK):
            return True
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

    def get_metrics_from_django(self):
        """
        Get metrics data to set it in terra

        Returns:
            self.metrics (list):      list with metrics
        """
        return self.metrics

    def get_loss_from_django(self):
        """
        Get loss data to set it in terra

        Returns:
            self.loss (str):      loss name
        """
        return self.loss

    def get_states_for_outputs(self) -> dict:
        """
        This method send some parametres for output layers, such as losses, metrics and tasks
        For example:
        "timeseries": {
                "losses": [
                    "mse",
                    "mae",
                    "mape",
                    "msle",
                    "log_cosh",
                    "cosine_similarity",
                ],
                "metrics": [
                    "accuracy",
                    "mse",
                    "mae",
                    "mape",
                    "msle",
                    "log_cosh",
                ]
            },
        :return: dict of parameteres
        """
        return self.states_for_outputs

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

    def _set_data(self, key_name: str, data, stop_flag: bool) -> None:
        """
        Set data to self out data in pozition with key_name
        Args:
            key_name: name of data type, str
            data: formatting recieved data from terra, Any
            stop_flag: flag to stop JS monitor
        """
        if key_name == 'images':
            print(data)
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
        elif key_name == "texts":
            self.out_data["texts"].append(data)
        # elif key_name == 'images':
        #     output = []
        #     for image_data in data:
        #         output.append({'image': image_data[0], 'title': image_data[1]})
        #     self.out_data['images'] = output
        else:
            self.out_data[key_name] = data
        self._check_stop_flag(stop_flag)
        # print(self.out_data)

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
                with open(dataset_path, "rb") as f:
                    custom_dts = dill.load(f)
                tags = list(custom_dts.tags.values())
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
        output = {"datasets": [], "tags": {}}

        datasets_dict = self.dts.get_datasets_dict()
        datasets_dict.update(self._get_custom_datasets_from_google_drive())

        for name, data in datasets_dict.items():
            dataset_tags = dict(
                map(
                    lambda item: (self._reformat_tags([item])[0], item),
                    sum(
                        list(
                            map(
                                lambda value: value
                                if isinstance(value, list)
                                else [value],
                                list(filter(None, data)),
                            )
                        ),
                        [],
                    ),
                )
            )
            output["tags"].update(dataset_tags)
            output["datasets"].append({"name": name, "tags": dataset_tags})

        # TODO for next relise step:

        # methods = self.dts.get_datasets_methods()
        # content = {
        #     'datasets': datasets,
        #     'tags': tags,
        #     'methods': methods,
        # }
        # tags = self._reformat_tags(list(tags))

        return output

    def _prepare_dataset(self, dataset_name: str, source: str) -> tuple:
        """
        prepare dataset for load to nn
        Args:
            **options: dataset options, such as dataset name, type of task, etc.

        Returns:
            changed dataset and its tags
        """
        self._reset_out_data()
        if source == "custom":
            self.dts = self._read_trds(dataset_name)
        else:
            self.dts = DTS(exch_obj=self)
            gc.collect()
            self.dts.prepare_dataset(dataset_name=dataset_name, source=source)
        self._set_dts_name(self.dts.name)
        self.out_data["stop_flag"] = True
        self._set_start_layers()
        return self.dts.tags, self.dts.name, self.start_layers

    def _read_trds(self, dataset_name: str) -> DTS:
        filename = f"{dataset_name}.trds"
        filepath = os.path.join(self.custom_datasets_path, filename)
        with open(filepath, "rb") as f:
            dts = dill.load(f)
        return dts

    def _set_start_layers(self):
        self.start_layers = {}

        def _create(dts_data: dict, location: LayerLocation):
            available = [data["data_name"] for name, data in dts_data.items()]
            for name, data in dts_data.items():
                index = len(self.start_layers.keys()) + 1
                data_name = data.get("data_name", "")
                self.start_layers[index] = {
                    "config": {
                        "name": f"l{index}_{data_name}",
                        "dts_layer_name": name,
                        "type": LayerType.Input
                        if location == LayerLocation.input
                        else LayerType.Dense,
                        "location_type": location,
                        "up_link": [],
                        "input_shape": list(self.dts.input_shape.get(name, [])),
                        "output_shape": [],
                        "data_name": data_name,
                        "data_available": available,
                        "params": {},
                    }
                }

        _create(self.dts.X, LayerLocation.input)
        _create(self.dts.Y, LayerLocation.output)

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
        self.start_layers = {}
        self.out_data = {
            "stop_flag": False,
            "status_string": "",
            "progress_status": {
                "percents": 0,
                "progress_text": "",
                "iter_count": 0,
            },
            "errors": "",
            "prints": [],
            "plots": [],
            "scatters": [],
            "images": [],
            "texts": [],
        }

    def _set_optimizers(self):
        return self.optimizers_dict

    def _set_dts_name(self, dts_name):
        self.dts_name = dts_name

    def _write_zip(self, write_path, file_list):
        try:
            zip_model = zipfile.ZipFile(write_path, 'w')
            for any_file in file_list:
                file_path = os.path.join(self.dir_paths.modeling, any_file)
                zip_model.write(file_path)
            return ''
        except Exception as e:
            return e.__str__()

    def _load_unzip(self, load_path, file_name):
        try:
            zip_model = zipfile.ZipFile(load_path, 'r')
            file_path = os.path.join(self.dir_paths.modeling, file_name)
            zip_model.write(file_path)
            return ''
        except Exception as e:
            return e.__str__()

    @staticmethod
    def _set_layers_list() -> list:
        """
        Create list of layers types for front (field Тип слоя)
        Returns:
            list of layers types
        """
        ep = LayersDef()
        layers_list = []
        layers = [
            [layer for layer in group.values()] for group in ep.layers_dict.values()
        ]
        for group in layers:
            layers_list.extend(group)
        return layers_list

    def _set_current_task(self, task):
        self.task_name = task

    def load_dataset(self, **kwargs):
        self._reset_out_data()
        dataset_name = kwargs.get('dataset_name', '')
        dataset_link = kwargs.get('dataset_link', '')
        inputs_count = kwargs.get('inputs_count', None)
        outputs_count = kwargs.get('outputs_count', None)
        if dataset_name:
            self.dts.load_data(name=dataset_name, link=dataset_link)
            self._set_dts_name(self.dts.name)
            return self.dts.get_parameters_dict()
        else:
            self.out_data["errors"] = 'Не указано наименование датасета'
        self.out_data["stop_flag"] = True


    def prepare_dataset(self, dataset_name: str, source: str = ""):
        self.process_flag = "dataset"
        return self._prepare_dataset(dataset_name=dataset_name, source=source)

    def get_default_datasets_params(self):
        return self.dts.get_parameters_dict()

    def set_callbacks_switches(self, task: str, switches: dict):
        for switch, value in switches.items():
            self.callback_show_options_switches_front[task][switch]["value"] = value

    def set_paths(self, **kwargs):
        paths = kwargs
        print(paths)

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

    def show_current_epoch(self, epoch: int):
        self.epoch = epoch + 1
        pass

    def get_stop_training_flag(self):
        return self.stop_training_flag

    def get_datasets_data(self):
        return self._create_datasets_data()

    def get_models(self):
        output = []
        files_for_unzipping = os.listdir(self.gd_paths.modeling)
        for arch_files in files_for_unzipping:
            output.append(arch_files[:-6])
        return output

    def get_custom_model(self, model_name, input_shape, output_shape=None):
        load_model_path = os.path.join(self.gd_paths.modeling, f'{model_name}.model')
        if os.path.exists(load_model_path):
            files_for_unzipping = os.listdir(self.gd_paths.modeling)
        is_write = True
        message = ''
        # if is_overwrite or not os.path.exists(write_model_path):
        #     message = self._write_zip(write_model_path, files_for_zipping)
        #     if message:
        #         is_write = False
        # else:
        #     if os.path.exists(write_model_path):
        #         message = 'This model is exists'
        #         is_write = False
        return is_write, message

    def _prepare_custom_model(self, model_name, input_shape, output_shape=None):
        preview = {}
        yaml_file = os.path.join(self.models_plans_path, f"{model_name}")
        with open(yaml_file) as f:
            templates = yaml.full_load(f)
        if input_shape:
            model_input_shape = input_shape
        else:
            model_input_shape = templates.get("input_shape")
        if output_shape:
            model_output_shape = output_shape
        else:
            model_output_shape = templates.get("output_shape", None)
        plan_name = templates.get('plan_name', 'No info')
        datatype_name = templates.get('input_datatype', 'No info')
        shape_data = templates.get("input_shape", '')
        preview.update({
            'name': plan_name,
            'input_shape': shape_data,
            'datatype': datatype_name,
            'preview_image': 'some_image'
        })

        self.model_plan = templates.get("plan")
        output = self.get_validated_plan(self.model_plan, model_input_shape, output_shape=model_output_shape,
                                         method='load')
        output.update({'preview': preview})
        return output

    def get_dataset_input_shape(self):
        return self.dts.input_shape

    def get_hardware_env(self):
        return self.hardware_accelerator_type

    def get_callbacks_switches(self, task: str) -> dict:
        return self.callback_show_options_switches_front[task]

    def get_state(self) -> dict:
        data = self.get_datasets_data()
        data.update(
            {
                "layers_types": self.get_layers_type_list(),
                "optimizers": self.get_optimizers(),
                "callbacks": self.callback_show_options_switches_front,
                "hardware": self.get_hardware_env(),
                "compile": self.get_states_for_outputs(),
            }
        )
        return data

    def get_layers_type_list(self):
        print(self.layers_params)
        return self.layers_params

    def get_optimizers(self):
        return self.optimizers

    def get_model_plan(self, plan=None, model_name=''):
        model_plan = ModelPlan()
        model_plan.input_datatype = self.dts.input_datatype
        model_plan.input_shape = self.dts.input_shape
        model_plan.output_shape = {}
        model_plan.plan = plan if plan else []
        model_plan.plan_name = model_name
        return model_plan.dict()

    def get_optimizer_kwargs(self, optimizer_name):
        optimizer_params = {"main": {}, "extra": {}}
        for name, params in self.optimizers.get(optimizer_name).items():
            for _param_name, values in params.items():
                optimizer_params[name][_param_name] = values.get("default")
        optimizer_kwargs = OptimizerParams(**optimizer_params)
        return optimizer_kwargs.dict()

    def get_data(self):
        if self.process_flag == "train":
            self.out_data["progress_status"]["progress_text"] = "Train progress"
            self.out_data["progress_status"]["percents"] = (
                self.epoch / self.epochs
            ) * 100
            self.out_data["progress_status"]["iter_count"] = self.epochs
            print(self.is_trained)
        return self.out_data

    def reset_training(self):
        self.nn.nn_cleaner()

    def start_training(self, model: bytes, **kwargs) -> None:
        if self.stop_training_flag:
            self.stop_training_flag = False
        self.process_flag = "train"
        self._reset_out_data()
        training = kwargs

        model_file = tempfile.NamedTemporaryFile(prefix='model_', suffix='tmp.h5', delete=False)
        self.nn.training_path = training.get('pathname', '')

        with open(model_file.name, 'wb') as f:
            f.write(base64.b64decode(model))

        self.nn.set_dataset(self.dts)
        nn_model = load_model(model_file.name)
        model_file.close()

        output_optimizer_params = {'op_name': "", 'op_kwargs': {}}

        output_params = training.get("outputs", {})
        clbck_chp = training.get("checkpoint", {})
        self.epochs = training.get("epochs_count", 10)
        batch_size = training.get("batch_sizes", 32)
        optimizer_params = training.get('optimizer', {})
        output_optimizer_params['op_name'] = optimizer_params.get('name')
        for key, val in optimizer_params.get('params', {}).items():
            output_optimizer_params['op_kwargs'].update(val)

        self.nn.set_main_params(
            output_params=output_params,
            clbck_chp=clbck_chp,
            epochs=self.epochs,
            batch_size=batch_size,
            optimizer_params=output_optimizer_params
        )
        try:
            # training = Process(target=self.nn.terra_fit, name='TRAIN_PROCESS', args=(nn_model,))
            self.nn.terra_fit(nn_model)
            # training.start()
        except Exception as e:
            self.out_data["stop_flag"] = True
            self.out_data["errors"] = e.__str__()
        self.out_data["stop_flag"] = True

    # def start_training(self, model: bytes, **kwargs) -> dict:
    #     training = Thread(target=self._start_training, name='TRAIN_PROCESS', daemon=False, args=(model,), kwargs=kwargs)
    #     training.start()
    #     self.is_trained = True
    #     time.sleep(30)
    #     self.stop_training()
    #     # training.join()
    #     self.is_trained = False
    #     return {}

    def stop_training(self):
        self.stop_training_flag = True
        self.out_data["stop_flag"] = True

    def save_model(self, **kwargs):
        model_name = kwargs.get('name')
        is_overwrite = kwargs.get('overwrite')
        write_model_path = os.path.join(self.gd_paths.modeling, f'{model_name}.model')
        files_for_zipping = os.listdir(self.dir_paths.modeling)
        is_write = True
        message = ''
        if is_overwrite or not os.path.exists(write_model_path):
            message = self._write_zip(write_model_path, files_for_zipping)
            if message:
                is_write = False
        else:
            if os.path.exists(write_model_path):
                message = 'This model is exists'
                is_write = False
        return is_write, message




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
    b.prepare_dataset(dataset_name="mnist")
    print(b.dts.task_type)
    pass
