import gc
import os
import re
from dataclasses import dataclass

import dill as dill
from IPython import get_ipython
from django.conf import settings

from terra_ai.trds import DTS
from terra_ai.guiexchange import Exchange as GuiExch
from apps.plugins.terra.neural.guinn import GUINN


@dataclass
class LayersDef:
    """Model Plan layers defaults"""

    """ Head
    """
    framework = "keras"
    input_datatype = None  # Type of data
    plan_name = "empty_string"
    num_classes = 0
    input_shape = None
    plan = []

    """
    Conv2D kwargs defaults for information
    -----------------------
    conv2d_kwargs = {
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
    }
    """

    """ Layers dictionary
    """
    layers_dict = {
        # Layers Main
        1: {
            1: "Dense",
            2: "Conv1D",
            3: "Conv2D",
            4: "Conv3D",
            5: "SeparableConv1D",
            6: "SeparableConv2D",
            7: "DepthwiseConv2D",
        },
        #   Layers UpScaling
        2: {
            1: "Conv1DTranspose",  # Conv1DTranspose tensorflow 2.3
            2: "Conv2DTranspose",
            3: "UpSampling1D",
            4: "UpSampling2D",
        },
        #   Layers DownScaling
        3: {
            1: "MaxPooling1D",
            2: "MaxPooling2D",
            3: "AveragePooling1D",
            4: "AveragePooling2D",
        },
        # Layers Connections
        4: {
            1: "Concatenate",
            2: "Add",
            3: "Multiply",
        },
        # Layers and functions Activations
        5: {
            1: "sigmoid",
            2: "softmax",
            3: "tanh",
            4: "relu",
            5: "LeakyReLU",
            6: "elu",
            7: "selu",
            8: "PReLU",
        },
        # Layers Optimization
        6: {
            1: "Dropout",
            2: "BatchNormalization",
        },
        # Layers Special
        7: {1: "Embedding", 2: "LSTM", 3: "GRU"},
        # Blocks
        8: {
            1: "Flatten",
            2: "Reshape",
            3: "GlobalMaxPooling1D",
            4: "GlobalMaxPooling2D",
            5: "GlobalAveragePooling1D",
            6: "GlobalAveragePooling2D",
            7: "RepeatVector",
        },
        # Input - Output custom layers
        9: {
            1: "Input",
            2: "assignment",
            # 3: 'out'
        },
    }

    """ Default layers kwargs with min, max
    param_lh: 
        param_name_lh: (min, max), (iterable int or str) for random generator 
    """
    filters_lh = (1, 1024)
    units_lh = (1, 512)
    kernel_size_lh = (1, 7)
    pool_size_lh = (2, 4, 6)
    strides_lh = (2, 4, 6)
    padding_lh = ("same", "valid")
    activation_lh = ("relu", "sigmoid", "softmax")
    size_lh = (2, 2)
    rate_lh = (0.1, 0.5)
    axis_lh = (0, 1)

    """ Layers defaults 
    """
    # Input_defaults = \
    #     {'shape': None,
    #      }

    Conv1D_defaults = {
        "filters": None,
        "kernel_size": None,
        "strides": 1,
        "padding": "valid",
        "data_format": "channels_last",
        "dilation_rate": 1,
        "groups": 1,
        "activation": None,
        "use_bias": True,
        "kernel_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
    }

    Conv2D_defaults = {
        "filters": None,
        "kernel_size": None,
        "strides": (1, 1),
        "padding": "valid",
        "data_format": None,
        "dilation_rate": (1, 1),
        "groups": 1,
        "activation": None,
        "use_bias": True,
        "kernel_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
    }

    Conv3D_defaults = {
        "filters": None,
        "kernel_size": None,
        "strides": (1, 1, 1),
        "padding": "valid",
        "data_format": None,
        "dilation_rate": (1, 1, 1),
        "groups": 1,
        "activation": None,
        "use_bias": True,
        "kernel_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
    }

    Conv1DTranspose_defaults = {
        "filters": None,
        "kernel_size": None,
        "strides": 1,
        "padding": "valid",
        "output_padding": None,
        "data_format": None,
        "dilation_rate": 1,
        "activation": None,
        "use_bias": True,
        "kernel_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
    }

    Conv2DTranspose_defaults = {
        "filters": None,
        "kernel_size": None,
        "strides": (1, 1),
        "padding": "valid",
        "output_padding": None,
        "data_format": None,
        "dilation_rate": (1, 1),
        "activation": None,
        "use_bias": True,
        "kernel_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
    }

    SeparableConv1D_defaults = {
        "filters": None,
        "kernel_size": None,
        "strides": 1,
        "padding": "valid",
        "data_format": None,
        "dilation_rate": 1,
        "depth_multiplier": 1,
        "activation": None,
        "use_bias": True,
        "depthwise_initializer": "glorot_uniform",
        "pointwise_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
        "depthwise_regularizer": None,
        "pointwise_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "depthwise_constraint": None,
        "pointwise_constraint": None,
        "bias_constraint": None,
    }

    SeparableConv2D_defaults = {
        "filters": None,
        "kernel_size": None,
        "strides": (1, 1),
        "padding": "valid",
        "data_format": None,
        "dilation_rate": (1, 1),
        "depth_multiplier": 1,
        "activation": None,
        "use_bias": True,
        "depthwise_initializer": "glorot_uniform",
        "pointwise_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
        "depthwise_regularizer": None,
        "pointwise_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "depthwise_constraint": None,
        "pointwise_constraint": None,
        "bias_constraint": None,
    }

    DepthwiseConv2D_defaults = {
        "kernel_size": None,
        "strides": (1, 1),
        "padding": "valid",
        "depth_multiplier": 1,
        "data_format": None,
        "dilation_rate": (1, 1),
        "activation": None,
        "use_bias": True,
        "depthwise_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
        "depthwise_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "depthwise_constraint": None,
        "bias_constraint": None,
    }

    MaxPooling1D_defaults = {
        "pool_size": 2,
        "strides": None,
        "padding": "valid",
        "data_format": "channels_last",
    }

    MaxPooling2D_defaults = {
        "pool_size": (2, 2),
        "strides": None,
        "padding": "valid",
        "data_format": None,
    }
    AveragePooling1D_defaults = {
        "pool_size": 2,
        "strides": None,
        "padding": "valid",
        "data_format": None,
    }

    AveragePooling2D_defaults = {
        "pool_size": (2, 2),
        "strides": None,
        "padding": "valid",
        "data_format": None,
    }

    UpSampling1D_defaults = {"size": 2}

    UpSampling2D_defaults = {
        "size": (2, 2),
        "data_format": None,
        "interpolation": "nearest",
    }

    LeakyReLU_defaults = {"alpha": 0.3}

    Dropout_defaults = {"rate": None, "noise_shape": None, "seed": None}

    Dense_defaults = {
        "units": None,
        "activation": None,
        "use_bias": True,
        "kernel_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
    }

    Add_defaults = {}

    Multiply_defaults = {}

    Flatten_defaults = {"data_format": None}

    Concatenate_defaults = {"axis": -1}

    Reshape_defaults = {"target_shape": None}

    sigmoid_defaults = {}

    softmax_defaults = {}

    tanh_defaults = {}

    relu_defaults = {}

    elu_defaults = {}

    selu_defaults = {}

    PReLU_defaults = {
        "alpha_initializer": "zeros",
        "alpha_regularizer": None,
        "alpha_constraint": None,
        "shared_axes": None,
    }

    GlobalMaxPooling1D_defaults = {"data_format": "channels_last"}

    GlobalMaxPooling2D_defaults = {"data_format": None}

    GlobalAveragePooling1D_defaults = {"data_format": "channels_last"}

    GlobalAveragePooling2D_defaults = {"data_format": None}

    GRU_defaults = {
        "units": None,
        "activation": "tanh",
        "recurrent_activation": "sigmoid",
        "use_bias": True,
        "kernel_initializer": "glorot_uniform",
        "recurrent_initializer": "orthogonal",
        "bias_initializer": "zeros",
        "kernel_regularizer": None,
        "recurrent_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "recurrent_constraint": None,
        "bias_constraint": None,
        "dropout": 0.0,
        "recurrent_dropout": 0.0,
        "return_sequences": False,
        "return_state": False,
        "go_backwards": False,
        "stateful": False,
        "unroll": False,
        "time_major": False,
        "reset_after": True,
    }

    LSTM_defaults = {
        "units": None,
        "activation": "tanh",
        "recurrent_activation": "sigmoid",
        "use_bias": True,
        "kernel_initializer": "glorot_uniform",
        "recurrent_initializer": "orthogonal",
        "bias_initializer": "zeros",
        "unit_forget_bias": True,
        "kernel_regularizer": None,
        "recurrent_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "recurrent_constraint": None,
        "bias_constraint": None,
        "dropout": 0.0,
        "recurrent_dropout": 0.0,
        "return_sequences": False,
        "return_state": False,
        "go_backwards": False,
        "stateful": False,
        "unroll": False,
    }

    Embedding_defaults = {
        "input_dim": None,
        "output_dim": None,
        "embeddings_initializer": "uniform",
        "embeddings_regularizer": None,
        "activity_regularizer": None,
        "embeddings_constraint": None,
        "mask_zero": False,
        "input_length": None,
    }

    RepeatVector_defaults = {
        "n": None,
    }
    pass


class StatesData:
    def __init__(self):
        self.django_optimizers_dict = {
            "SGD": {
                "learning_rate": {"type": "float", "value": 0.01},
                "momentum": {"type": "float", "value": 0.0},
                "nesterov": {"type": "bool", "value": False},
            },
            "RMSprop": {
                "learning_rate": {"type": "float", "value": 0.001},
                "rho": {"type": "float", "value": 0.9},
                "momentum": {"type": "float", "value": 0.0},
                "epsilon": {"type": "float", "value": 1e-07},
                "centered": {"type": "bool", "value": False},
            },
            "Adam": {
                "learning_rate": {"type": "float", "value": 0.001},
                "beta_1": {"type": "float", "value": 0.9},
                "beta_2": {"type": "float", "value": 0.999},
                "epsilon": {"type": "float", "value": 1e-07},
                "amsgrad": {"type": "bool", "value": False},
            },
            "Adadelta": {
                "learning_rate": {"type": "float", "value": 0.001},
                "rho": {"type": "float", "value": 0.95},
                "epsilon": {"type": "float", "value": 1e-07},
            },
            "Adagrad": {
                "learning_rate": {"type": "float", "value": 0.001},
                "initial_accumulator_value": {"type": "float", "value": 0.1},
                "epsilon": {"type": "float", "value": 1e-07},
            },
            "Adamax": {
                "learning_rate": {"type": "float", "value": 0.001},
                "beta_1": {"type": "float", "value": 0.9},
                "beta_2": {"type": "float", "value": 0.999},
                "epsilon": {"type": "float", "value": 1e-07},
            },
            "Nadam": {
                "learning_rate": {"type": "float", "value": 0.001},
                "beta_1": {"type": "float", "value": 0.9},
                "beta_2": {"type": "float", "value": 0.999},
                "epsilon": {"type": "float", "value": 1e-07},
            },
            "Ftrl": {
                "learning_rate": {"type": "float", "value": 0.001},
                "learning_rate_power": {"type": "float", "value": -0.5},
                "initial_accumulator_value": {"type": "float", "value": 0.1},
                "l1_regularization_strength": {"type": "float", "value": 0.0},
                "l2_regularization_strength": {"type": "float", "value": 0.0},
                "l2_shrinkage_regularization_strength": {"type": "float", "value": 0.0},
                "beta": {"type": "float", "value": 0.0},
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
        self.layers_params = {
            "Conv1D": {
                "filters": {"type": "int", "default": None},
                "kernel_size": {"type": "int", "default": None},
                "strides": {"type": "int", "default": 1},
                "padding": {
                    "type": "str",
                    "default": "valid",
                    "list": True,
                    "available": self.padding_values,
                },
                "activation": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": self.activation_values,
                },
            },
            "Conv2D": {
                "filters": {"type": "int", "default": None},
                "kernel_size": {"type": "tuple", "default": None},
                "strides": {"type": "tuple", "default": (1, 1)},
                "padding": {
                    "type": "str",
                    "default": "valid",
                    "list": True,
                    "available": self.padding_values,
                },
                "activation": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": self.activation_values,
                },
            },
            "Conv3D": {
                "filters": {"type": "int", "default": None},
                "kernel_size": {"type": "tuple", "default": None},
                "strides": {"type": "tuple", "default": (1, 1, 1)},
                "padding": {
                    "type": "str",
                    "default": "valid",
                    "list": True,
                    "available": self.padding_values,
                },
                "activation": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": self.activation_values,
                },
            },
            "Conv1DTranspose": {
                "filters": {"type": "int", "default": None},
                "kernel_size": {"type": "int", "default": None},
                "strides": {"type": "int", "default": 1},
                "padding": {
                    "type": "str",
                    "default": "valid",
                    "list": True,
                    "available": self.padding_values,
                },
                "activation": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": self.activation_values,
                },
            },
            "Conv2DTranspose": {
                "filters": {"type": "int", "default": None},
                "kernel_size": {"type": "tuple", "default": None},
                "strides": {"type": "tuple", "default": (1, 1)},
                "padding": {
                    "type": "str",
                    "default": "valid",
                    "list": True,
                    "available": self.padding_values,
                },
                "activation": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": self.activation_values,
                },
            },
            "SeparableConv1D": {
                "filters": {"type": "int", "default": None},
                "kernel_size": {"type": "int", "default": None},
                "strides": {"type": "int", "default": 1},
                "padding": {
                    "type": "str",
                    "default": "valid",
                    "list": True,
                    "available": self.padding_values,
                },
                "activation": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": self.activation_values,
                },
            },
            "SeparableConv2D": {
                "filters": {"type": "int", "default": None},
                "kernel_size": {"type": "tuple", "default": None},
                "strides": {"type": "tuple", "default": (1, 1)},
                "padding": {
                    "type": "str",
                    "default": "valid",
                    "list": True,
                    "available": self.padding_values,
                },
                "activation": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": self.activation_values,
                },
            },
            "DepthwiseConv2D": {
                "kernel_size": {"type": "tuple", "default": None},
                "strides": {"type": "tuple", "default": (1, 1)},
                "padding": {
                    "type": "str",
                    "default": "valid",
                    "list": True,
                    "available": self.padding_values,
                },
                "activation": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": self.activation_values,
                },
            },
            "MaxPooling1D": {
                "pool_size": {"type": "int", "default": 2},
                "strides": {"type": "int", "default": None},
                "padding": {
                    "type": "str",
                    "default": "valid",
                    "list": True,
                    "available": self.padding_values,
                },
            },
            "MaxPooling2D": {
                "pool_size": {"type": "tuple", "default": (2, 2)},
                "strides": {"type": "tuple", "default": None},
                "padding": {
                    "type": "str",
                    "default": "valid",
                    "list": True,
                    "available": self.padding_values,
                },
            },
            "AveragePooling1D": {
                "pool_size": {"type": "int", "default": 2},
                "strides": {"type": "int", "default": None},
                "padding": {
                    "type": "str",
                    "default": "valid",
                    "list": True,
                    "available": self.padding_values,
                },
            },
            "AveragePooling2D": {
                "pool_size": {"type": "tuple", "default": (2, 2)},
                "strides": {"type": "tuple", "default": None},
                "padding": {
                    "type": "str",
                    "default": "valid",
                    "list": True,
                    "available": self.padding_values,
                },
            },
            "UpSampling1D": {"size": {"type": "int", "default": 2}},
            "UpSampling2D": {"size": {"type": "tuple", "default": (2, 2)}},
            "LeakyReLU": {"alpha": {"type": "float", "default": 0.3}},
            "Dropout": {"rate": {"type": "float", "default": None}},
            "Dense": {
                "units": {"type": "int", "default": None},
                "activation": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": self.activation_values,
                },
                "use_bias": {"type": "bool", "default": True},
            },
            "Add": {},
            "Multiply": {},
            "Flatten": {},
            "Concatenate": {},
            "Reshape": {},
            "sigmoid": {},
            "softmax": {},
            "tanh": {},
            "relu": {},
            "elu": {},
            "selu": {},
            "PReLU": {},
            "GlobalMaxPooling1D": {},
            "GlobalMaxPooling2D": {},
            "GlobalAveragePooling1D": {},
            "GlobalAveragePooling2D": {},
            "GRU": {
                "units": {"type": "int", "default": None},
                "dropout": {"type": "float", "default": 0.0},
                "recurrent_dropout": {"type": "float", "default": 0.0},
                "return_sequences": {
                    "type": "bool",
                    "default": False,
                    "list": True,
                    "available": [False, True],
                },
                "return_state": {
                    "type": "bool",
                    "default": False,
                    "list": True,
                    "available": [False, True],
                },
            },
            "LSTM": {
                "units": {"type": "int", "default": None},
                "dropout": {"type": "float", "default": 0.0},
                "recurrent_dropout": {"type": "float", "default": 0.0},
                "return_sequences": {
                    "type": "bool",
                    "default": False,
                    "list": True,
                    "available": [False, True],
                },
                "return_state": {
                    "type": "bool",
                    "default": False,
                    "list": True,
                    "available": [False, True],
                },
            },
            "Embedding": {
                "input_dim": {"type": "int", "default": None},
                "output_dim": {"type": "int", "default": None},
                "input_length": {"type": "int", "default": None},
            },
            "RepeatVector": {"n": {"type": "int", "default": None}},
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

        self.property_of = "DJANGO"
        self.stop_training_flag = False
        self.process_flag = "dataset"
        self.hardware_accelerator_type = self.get_hardware_accelerator_type()
        self.layers_list = self._set_layers_list()
        self.start_layers = {}
        self.layers_data_state = {}
        self.dts = DTS(exch_obj=self)  # dataset init
        self.custom_datasets = []
        self.custom_datasets_path = f"{settings.TERRA_AI_DATA_PATH}/datasets"
        self.dts_name = None
        self.task_name = ""
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
        self.out_data["stop_flag"] = True
        return self.dts.tags, self.dts.name

    def _create_custom_dataset(self, **options):
        dataset = f'{options.get("dataset_name")}.trds'
        dataset_path = os.path.join(self.custom_datasets_path, dataset)
        with open(dataset_path, "rb") as f:
            custom_dts = dill.load(f)
        self.dts = custom_dts
        # print(
        #     "DTS",
        #     self.dts,
        #     "\n",
        #     self.dts.name,
        #     self.dts.X,
        #     self.dts.Y,
        # )
        self._set_dts_name(self.dts.name)
        self.out_data["stop_flag"] = True
        self._set_start_layers()
        print(self.start_layers)
        print(self.layers_data_state)
        return self.dts.tags, self.dts.name

    def _set_start_layers(self):
        inputs = self.dts.X
        outputs = self.dts.Y
        self.__create_start_layer(inputs, 'Input')
        self.__create_start_layer(outputs, 'Output')

    def __create_start_layer(self, dts_data: dict, layer_type: str):
        available = [data['data_name'] for name, data in dts_data.items()]
        for name, data in dts_data.items():
            idx = name.split['_'][1]
            print('idx', idx)
            layer_name = idx
            data_name = data['data_name']
            if layer_type == 'Input':
                input_shape = list(self.dts.input_shape[name])
                print('input_shape', input_shape)
            else:
                input_shape = []
            current_layer = {
                "name": layer_name,
                "type": layer_type,
                "data_name": data_name,
                "data_available": available,
                "params": {},
                "up_link": [
                    0
                ],
                "inp_shape": input_shape,
                "out_shape": []
            }
            print('CUR_LAYER: ', current_layer)
            self.start_layers[idx] = current_layer
            self.layers_data_state[idx] = {"data_name": data_name, "data_available": available}

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

    def prepare_dataset(self, **options):
        self.process_flag = "dataset"
        custom_flag = options.get("source")
        if custom_flag and custom_flag == "custom":
            self._set_current_task(options.get("task_type"))
            return self._create_custom_dataset(**options)
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
                "layers_types": self.get_layers_type_list(),
                "optimizers": self.get_optimizers_list(),
                "callbacks": self.callback_show_options_switches_front.get(task, {}),
                "hardware": self.get_hardware_env(),
            }
        )
        return data

    def get_layers_type_list(self):
        return self.layers_list

    def get_optimizers_list(self):
        return list(self.django_optimizers_dict.keys())

    def get_data(self):
        if self.process_flag == "train":
            self.out_data["progress_status"]["progress_text"] = "Train progress"
            self.out_data["progress_status"]["percents"] = (
                                                                   self.epoch / self.epochs
                                                           ) * 100
            self.out_data["progress_status"]["iter_count"] = self.epochs
        return self.out_data

    def start_training(self, model: object, callback: object):
        # if self.debug_verbose == 3:
        #     print(f"Dataset name: {self.dts.name}")
        #     print(f"Dataset shape: {self.dts.input_shape}")
        #     print(f"Plan: ")
        #     for idx, l in enumerate(model_plan.plan, start=1):
        #         print(f"Layer {idx}: {l}")
        #     print(f"x_Train: {self.nn.DTS.x_Train.shape}")
        #     print(f"y_Train: {self.nn.DTS.y_Train.shape}")
        self.nn.set_dataset(self.dts)
        nn_callback = dill.loads(callback)
        nn_model = dill.loads(model)
        self.nn.set_callback(nn_callback)
        self.nn.terra_fit(nn_model)
        self.out_data["stop_flag"] = True

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
    b.prepare_dataset(dataset_name="заболевания", task_type="classification")
    pass
