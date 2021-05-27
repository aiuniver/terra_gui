from dataclasses import dataclass


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
            # 5: "Conv3DTranspose"
            # 6: "UpSampling3D"
            # 7: "ZeroPadding1D"
            # 8: "ZeroPadding2D",
            # 9: "ZeroPadding3D"
        },
        #   Layers DownScaling
        3: {
            1: "MaxPooling1D",
            2: "MaxPooling2D",
            3: "AveragePooling1D",
            4: "AveragePooling2D",
            # 5: "MaxPooling3D "
            # 6: "AveragePooling3D"
        },
        # Layers Connections
        4: {
            1: "Concatenate",
            2: "Add",
            3: "Multiply",
            # 4: "Average",
            # 5: "Maximum",
            # 6: "Minimum",
            # 7: "Subtract",
            # 8: "Dot"
        },
        # Layers and functions Activations
        5: {
            1: "sigmoid",
            2: "softmax",
            3: "tanh",
            4: "relu",
            5: "selu",
            6: "elu",
            # 7: "LeakyReLU",
            # 8: "PReLU",
            # 9: "Activation",
            # 10: "ReLU",
            # 11: "Softmax",
            # 12: "ELU",
            # 13: "ThresholdedReLU"
        },
        # Layers Optimization
        6: {
            1: "Dropout",
            2: "BatchNormalization",
            # 3: "Masking",
            # 4: "LayerNormalization",
            # 5: "SpatialDropout1D",
            # 6: "SpatialDropout2D",
            # 7: "SpatialDropout3D",
            # 8: "GaussianDropout",
            # 9: "GaussianNoise",
            # 10: "ActivityRegularization",
            # 11: "AlphaDropout",
            # 12: "MultiHeadAttention",
            # 13: "Attention",
            # 14: "AdditiveAttention"
        },
        # Recurrent layers
        7: {1: "Embedding",
            2: "LSTM",
            3: "GRU",
            # 4: "SimpleRNN",
            # 5: "TimeDistributed",
            # 6: "Bidirectional",
            # 7: "ConvLSTM2D",
            # 8: "RNN"
            },
        # Shape-shifters
        8: {
            1: "Flatten",
            2: "Reshape",
            3: "GlobalMaxPooling1D",
            4: "GlobalMaxPooling2D",
            5: "GlobalAveragePooling1D",
            6: "GlobalAveragePooling2D",
            7: "RepeatVector",
            # 8: "GlobalMaxPooling3D ",
            # 9: "Permute",
            # 10: "Cropping1D",
            # 11: "Cropping2D",
            # 12: "Cropping3D",
        },
        # Input - Output custom layers
        9: {
            1: "Input",
            2: "assignment",
            # 3: 'out'
            # 4: "Lambda"
        },
        # Preprocessing layers
        # 10: {
        #     1: "TextVectorization",
        #     2: "Normalization",
        #     3: "CategoryEncoding",
        #     4: "Hashing",
        #     5: "Discretization",
        #     6: "StringLookup",
        #     7: "IntegerLookup",
        #     8: "CategoryCrossing"
        #     9: "Resizing",
        #     10: "Rescaling",
        #     11: "CenterCrop",
        #     12: "RandomCrop",
        #     13: "RandomFlip",
        #     14: "RandomTranslation",
        #     15: "RandomRotation",
        #     16: "RandomZoom",
        #     17: "RandomHeight",
        #     18: "RandomWidth"
        # }
    }

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


@dataclass
class GUILayersDef:
    filters_lh = (1, 1024)
    units_lh = (1, 512)
    kernel_size_lh = (1, 7)
    pool_size_lh = (2, 4, 6)
    strides_lh = (2, 4, 6)
    padding_lh = ("same", "valid")
    activation_lh = (None, "relu", "sigmoid", "softmax",
                     "softplus", "softsign", "tanh",
                     "selu", "elu", "exponential", "leaner")
    initializer_lh = ("random_normal", "random_uniform", "truncated_normal",
                      "zeros", "ones", "glorot_normal", "glorot_uniform", "uniform",
                      "identity", "orthogonal", "constant", "variance_scaling")
    regularizer_lh = (None, "l1", "l2", "l1_l2")
    constraint_lh = (None, "max_norm", "min_max_norm", "non_neg", "unit_norm", "radial_constraint")
    data_format_lh = ("channels_last", "channels_first")
    size_lh = (2, 2)
    rate_lh = (0.1, 0.5)
    axis_lh = (0, 1)

    initializers = {}

    regularizers = {}

    constraints = {}

    layers_params = {
        "Input": {"main": {},
                  "extra": {
                      # "shape": {"type": "tuple", "default": None},
                      # "batch_size": {"type": "int", "default": None},
                      # "name": {"type": "str", "default": None},
                      # "dtype": {"type": "str", "default": None},
                      # "sparse": {
                      #     "type": "bool",
                      #     "default": False,
                      # },
                      # "tensor": {"type": "tensor", "default": None},
                      # "ragged": {
                      #     "type": "bool",
                      #     "default": False,
                  }
                  },
        "Conv1D": {
            "main":
                {
                    "filters": {"type": "int", "default": 32},
                    "kernel_size": {"type": "int", "default": 2},
                    "strides": {"type": "int", "default": 1},
                    "padding": {
                        "type": "str",
                        "default": "same",
                        "list": True,
                        "available": padding_lh,
                    },
                    "activation": {
                        "type": "str",
                        "default": 'relu',
                        "list": True,
                        "available": activation_lh,
                    },
                },
            'extra': {
                "data_format": {
                    "type": "str",
                    "default": "channels_last",
                    "list": True,
                    "available": data_format_lh,
                },
                "dilation_rate": {"type": "int", "default": 1},  # has exceptions
                "groups": {"type": "int", "default": 1},  # has exceptions,
                "use_bias": {
                    "type": "bool",
                    "default": True,
                },
                "kernel_initializer": {
                    "type": "str",
                    "default": "glorot_uniform",
                    "list": True,
                    "available": initializer_lh,
                },
                "bias_initializer": {
                    "type": "str",
                    "default": "zeros",
                    "list": True,
                    "available": initializer_lh,
                },
                "kernel_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "bias_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "activity_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "kernel_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "bias_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                }
            }
        },
        "Conv2D": {
            "main": {
                "filters": {"type": "int", "default": 32},
                "kernel_size": {"type": "tuple", "default": (1, 1)},
                "strides": {"type": "tuple", "default": (1, 1)},
                "padding": {
                    "type": "str",
                    "default": "same",
                    "list": True,
                    "available": padding_lh,
                },
                "activation": {
                    "type": "str",
                    "default": 'relu',
                    "list": True,
                    "available": activation_lh,
                },
            },
            'extra': {
                "data_format": {
                    "type": "str",
                    "default": "channels_last",
                    "list": True,
                    "available": data_format_lh,
                },
                "dilation_rate": {"type": "tuple", "default": (1, 1)},  # has exceptions
                "groups": {"type": "int", "default": 1},  # has exceptions,
                "use_bias": {
                    "type": "bool",
                    "default": True,
                },
                "kernel_initializer": {
                    "type": "str",
                    "default": "glorot_uniform",
                    "list": True,
                    "available": initializer_lh,
                },
                "bias_initializer": {
                    "type": "str",
                    "default": "zeros",
                    "list": True,
                    "available": initializer_lh,
                },
                "kernel_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "bias_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "activity_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "kernel_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "bias_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                }
            }
        },
        "Conv3D": {
            "main":
                {
                    "filters": {"type": "int", "default": 32},
                    "kernel_size": {"type": "tuple", "default": (1, 1, 1)},
                    "strides": {"type": "tuple", "default": (1, 1, 1)},
                    "padding": {
                        "type": "str",
                        "default": "same",
                        "list": True,
                        "available": padding_lh,
                    },
                    "activation": {
                        "type": "str",
                        "default": 'relu',
                        "list": True,
                        "available": activation_lh,
                    },
                },
            'extra': {
                "data_format": {
                    "type": "str",
                    "default": "channels_last",
                    "list": True,
                    "available": data_format_lh,
                },
                "dilation_rate": {"type": "tuple", "default": (1, 1, 1)},  # has exceptions
                "groups": {"type": "int", "default": 1},  # has exceptions,
                "use_bias": {
                    "type": "bool",
                    "default": True,
                },
                "kernel_initializer": {
                    "type": "str",
                    "default": "glorot_uniform",
                    "list": True,
                    "available": initializer_lh,
                },
                "bias_initializer": {
                    "type": "str",
                    "default": "zeros",
                    "list": True,
                    "available": initializer_lh,
                },
                "kernel_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "bias_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "activity_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "kernel_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "bias_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                }
            }
        },
        "Conv1DTranspose": {
            "main":
                {
                    "filters": {"type": "int", "default": 32},
                    "kernel_size": {"type": "int", "default": 1},
                    "strides": {"type": "int", "default": 1},
                    "padding": {
                        "type": "str",
                        "default": "same",
                        "list": True,
                        "available": padding_lh,
                    },
                    "activation": {
                        "type": "str",
                        "default": 'relu',
                        "list": True,
                        "available": activation_lh
                    },
                },
            'extra': {
                "output_padding": {"type": "int", "default": None},
                "data_format": {
                    "type": "str",
                    "default": "channels_last",
                    "list": True,
                    "available": data_format_lh,
                },
                "dilation_rate": {"type": "int", "default": 1},  # has exceptions
                "use_bias": {
                    "type": "bool",
                    "default": True,
                },
                "kernel_initializer": {
                    "type": "str",
                    "default": "glorot_uniform",
                    "list": True,
                    "available": initializer_lh,
                },
                "bias_initializer": {
                    "type": "str",
                    "default": "zeros",
                    "list": True,
                    "available": initializer_lh,
                },
                "kernel_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "bias_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "activity_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "kernel_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "bias_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                }
            }
        },
        "Conv2DTranspose": {
            "main":
                {
                    "filters": {"type": "int", "default": 32},
                    "kernel_size": {"type": "tuple", "default": (1, 1)},
                    "strides": {"type": "tuple", "default": (1, 1)},
                    "padding": {
                        "type": "str",
                        "default": "same",
                        "list": True,
                        "available": padding_lh,
                    },
                    "activation": {
                        "type": "str",
                        "default": 'relu',
                        "list": True,
                        "available": activation_lh,
                    },
                },
            'extra': {
                "output_padding": {"type": "tuple", "default": None},
                "data_format": {
                    "type": "str",
                    "default": "channels_last",
                    "list": True,
                    "available": data_format_lh,
                },
                "dilation_rate": {"type": "tuple", "default": (1, 1)},  # has exceptions
                "use_bias": {
                    "type": "bool",
                    "default": True,
                },
                "kernel_initializer": {
                    "type": "str",
                    "default": "glorot_uniform",
                    "list": True,
                    "available": initializer_lh,
                },
                "bias_initializer": {
                    "type": "str",
                    "default": "zeros",
                    "list": True,
                    "available": initializer_lh,
                },
                "kernel_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "bias_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "activity_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "kernel_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "bias_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                }
            }
        },
        "SeparableConv1D": {
            "main":
                {
                    "filters": {"type": "int", "default": 32},
                    "kernel_size": {"type": "int", "default": 1},
                    "strides": {"type": "int", "default": 1},
                    "padding": {
                        "type": "str",
                        "default": "same",
                        "list": True,
                        "available": padding_lh,
                    },
                    "activation": {
                        "type": "str",
                        "default": 'relu',
                        "list": True,
                        "available": activation_lh,
                    },
                },
            'extra': {
                "data_format": {
                    "type": "str",
                    "default": "channels_last",
                    "list": True,
                    "available": data_format_lh,
                },
                "dilation_rate": {"type": "int", "default": 1},  # has exceptions
                "depth_multiplier": {"type": "int", "default": 1},
                "use_bias": {
                    "type": "bool",
                    "default": True,
                },
                "depthwise_initializer": {
                    "type": "str",
                    "default": "glorot_uniform",
                    "list": True,
                    "available": initializer_lh,
                },
                "pointwise_initializer": {
                    "type": "str",
                    "default": "glorot_uniform",
                    "list": True,
                    "available": initializer_lh,
                },
                "bias_initializer": {
                    "type": "str",
                    "default": "zeros",
                    "list": True,
                    "available": initializer_lh,
                },
                "depthwise_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "pointwise_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "bias_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "activity_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "depthwise_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "pointwise_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "bias_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                }
            }
        },
        "SeparableConv2D": {
            "main":
                {
                    "filters": {"type": "int", "default": 32},
                    "kernel_size": {"type": "tuple", "default": (1, 1)},
                    "strides": {"type": "tuple", "default": (1, 1)},
                    "padding": {
                        "type": "str",
                        "default": "same",
                        "list": True,
                        "available": padding_lh,
                    },
                    "activation": {
                        "type": "str",
                        "default": 'relu',
                        "list": True,
                        "available": activation_lh,
                    },
                },
            'extra': {
                "data_format": {
                    "type": "str",
                    "default": "channels_last",
                    "list": True,
                    "available": data_format_lh,
                },
                "dilation_rate": {"type": "tuple", "default": (1, 1)},  # has exceptions
                "depth_multiplier": {"type": "int", "default": 1},
                "use_bias": {
                    "type": "bool",
                    "default": True,
                },
                "depthwise_initializer": {
                    "type": "str",
                    "default": "glorot_uniform",
                    "list": True,
                    "available": initializer_lh,
                },
                "pointwise_initializer": {
                    "type": "str",
                    "default": "glorot_uniform",
                    "list": True,
                    "available": initializer_lh,
                },
                "bias_initializer": {
                    "type": "str",
                    "default": "zeros",
                    "list": True,
                    "available": initializer_lh,
                },
                "depthwise_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "pointwise_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "bias_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "activity_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "depthwise_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "pointwise_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "bias_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                }
            }
        },
        "DepthwiseConv2D": {
            "main":
                {
                    "kernel_size": {"type": "tuple", "default": (1, 1)},
                    "strides": {"type": "tuple", "default": (1, 1)},
                    "padding": {
                        "type": "str",
                        "default": "same",
                        "list": True,
                        "available": padding_lh,
                    },
                    "activation": {
                        "type": "str",
                        "default": 'relu',
                        "list": True,
                        "available": activation_lh,
                    },
                },
            'extra': {
                "data_format": {
                    "type": "str",
                    "default": "channels_last",
                    "list": True,
                    "available": data_format_lh,
                },
                "dilation_rate": {"type": "tuple", "default": (1, 1)},  # has exceptions
                "depth_multiplier": {"type": "int", "default": 1},
                "use_bias": {
                    "type": "bool",
                    "default": True,
                },
                "depthwise_initializer": {
                    "type": "str",
                    "default": "glorot_uniform",
                    "list": True,
                    "available": initializer_lh,
                },
                "bias_initializer": {
                    "type": "str",
                    "default": "zeros",
                    "list": True,
                    "available": initializer_lh,
                },
                "depthwise_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "bias_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "activity_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "depthwise_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "bias_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                }
            }
        },
        "MaxPooling1D": {
            "main":
                {
                    "pool_size": {"type": "int", "default": 2},
                    "strides": {"type": "int", "default": None},
                    "padding": {
                        "type": "str",
                        "default": "same",
                        "list": True,
                        "available": padding_lh,
                    },
                },
            'extra': {
                "data_format": {
                    "type": "str",
                    "default": "channels_last",
                    "list": True,
                    "available": data_format_lh,
                }
            }
        },
        "MaxPooling2D": {
            "main":
                {
                    "pool_size": {"type": "tuple", "default": (2, 2)},
                    "strides": {"type": "tuple", "default": None},
                    "padding": {
                        "type": "str",
                        "default": "same",
                        "list": True,
                        "available": padding_lh,
                    },
                },
            'extra': {
                "data_format": {
                    "type": "str",
                    "default": "channels_last",
                    "list": True,
                    "available": data_format_lh,
                }
            }
        },
        "AveragePooling1D": {
            "main":
                {
                    "pool_size": {"type": "int", "default": 2},
                    "strides": {"type": "int", "default": None},
                    "padding": {
                        "type": "str",
                        "default": "same",
                        "list": True,
                        "available": padding_lh,
                    },
                },
            'extra': {
                "data_format": {
                    "type": "str",
                    "default": "channels_last",
                    "list": True,
                    "available": data_format_lh,
                }
            }
        },
        "AveragePooling2D": {
            "main":
                {
                    "pool_size": {"type": "tuple", "default": (2, 2)},
                    "strides": {"type": "tuple", "default": None},
                    "padding": {
                        "type": "str",
                        "default": "same",
                        "list": True,
                        "available": padding_lh,
                    },
                },
            'extra': {
                "data_format": {
                    "type": "str",
                    "default": "channels_last",
                    "list": True,
                    "available": data_format_lh,
                }
            }
        },
        "UpSampling1D": {
            "main":
                {"size": {"type": "int", "default": 2}},
            'extra': {}
        },
        "UpSampling2D": {
            "main":
                {"size": {"type": "tuple", "default": (2, 2)}},
            'extra': {
                "data_format": {
                    "type": "str",
                    "default": "channels_last",
                    "list": True,
                    "available": data_format_lh,
                },
                "interpolation": {
                    "type": "str",
                    "default": "nearest",
                    "list": True,
                    "available": ["nearest", "bilinear"],
                }
            }
        },
        "LeakyReLU": {
            "main":
                {"alpha": {"type": "float", "default": 0.3}},
            'extra': {}
        },
        "Dropout": {
            "main":
                {"rate": {"type": "float", "default": 0.1}},
            'extra': {
                "noise_shape": {"type": "tensor", "default": None},
                "seed": {"type": "int", "default": None}
            }
        },
        "Dense": {
            "main":
                {
                    "units": {"type": "int", "default": 32},
                    "activation": {
                        "type": "str",
                        "default": 'relu',
                        "list": True,
                        "available": activation_lh,
                    },
                },
            'extra': {
                "use_bias": {
                    "type": "bool",
                    "default": True,
                },
                "kernel_initializer": {
                    "type": "str",
                    "default": "glorot_uniform",
                    "list": True,
                    "available": initializer_lh,
                },
                "bias_initializer": {
                    "type": "str",
                    "default": "zeros",
                    "list": True,
                    "available": initializer_lh,
                },
                "kernel_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "bias_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "activity_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "kernel_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "bias_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                # "name": {"type": "str", "default": None},
            }
        },
        "Add": {'main': {}, 'extra': {}},
        "Multiply": {'main': {}, 'extra': {}},
        "Flatten": {
            'main': {},
            'extra': {
                "data_format": {
                    "type": "str",
                    "default": "channels_last",
                    "list": True,
                    "available": data_format_lh,
                }
            }
        },
        "Concatenate": {
            'main': {
                "axis": {"type": "int", "default": -1}
            },
            'extra': {}
        },
        "Reshape": {
            'main': {
                "target_shape": {"type": "tuple", "default": None}
            },
            'extra': {}
        },
        "sigmoid": {'main': {}, 'extra': {}},
        "softmax": {'main': {}, 'extra': {}},
        "tanh": {'main': {}, 'extra': {}},
        "relu": {'main': {}, 'extra': {}},
        "elu": {'main': {}, 'extra': {}},
        "selu": {'main': {}, 'extra': {}},
        "PReLU": {
            'main': {},
            'extra': {
                "alpha_initializer": {
                    "type": "str",
                    "default": "zeros",
                    "list": True,
                    "available": initializer_lh,
                },
                "alpha_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "alpha_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "shared_axes": {"type": "list", "default": None}
            }
        },
        "GlobalMaxPooling1D": {'main': {},
                               'extra': {
                                   "data_format": {
                                       "type": "str",
                                       "default": "channels_last",
                                       "list": True,
                                       "available": data_format_lh,
                                   }}},
        "GlobalMaxPooling2D": {'main': {},
                               'extra': {
                                   "data_format": {
                                       "type": "str",
                                       "default": "channels_last",
                                       "list": True,
                                       "available": data_format_lh,
                                   }}},
        "GlobalAveragePooling1D": {'main': {},
                                   'extra': {
                                       "data_format": {
                                           "type": "str",
                                           "default": "channels_last",
                                           "list": True,
                                           "available": data_format_lh,
                                       }}},
        "GlobalAveragePooling2D": {'main': {},
                                   'extra': {
                                       "data_format": {
                                           "type": "str",
                                           "default": "channels_last",
                                           "list": True,
                                           "available": data_format_lh,
                                       }}},
        "GRU": {
            "main":
                {
                    "units": {"type": "int", "default": 32},
                    "return_sequences": {
                        "type": "bool",
                        "default": False,
                    },
                    "return_state": {
                        "type": "bool",
                        "default": False,
                    },
                },
            'extra': {
                "activation": {
                    "type": "str",
                    "default": "tanh",
                    "list": True,
                    "available": activation_lh,
                },
                "recurrent_activation": {
                    "type": "str",
                    "default": "sigmoid",
                    "list": True,
                    "available": activation_lh,
                },
                "use_bias": {
                    "type": "bool",
                    "default": True,
                },
                "kernel_initializer": {
                    "type": "str",
                    "default": "glorot_uniform",
                    "list": True,
                    "available": initializer_lh,
                },
                "recurrent_initializer": {
                    "type": "str",
                    "default": "orthogonal",
                    "list": True,
                    "available": initializer_lh,
                },
                "bias_initializer": {
                    "type": "str",
                    "default": "zeros",
                    "list": True,
                    "available": initializer_lh,
                },
                "kernel_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "recurrent_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "bias_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "activity_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "kernel_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "recurrent_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "bias_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "dropout": {"type": "float", "default": 0.0},
                "recurrent_dropout": {"type": "float", "default": 0.0},
                "go_backwards": {
                    "type": "bool",
                    "default": False,
                },
                "stateful": {
                    "type": "bool",
                    "default": False,
                },
                "time_major": {
                    "type": "bool",
                    "default": False,
                },
                "unroll": {
                    "type": "bool",
                    "default": False,
                },
                "reset_after": {
                    "type": "bool",
                    "default": True,
                }
            }
        },
        "LSTM": {
            "main":
                {
                    "units": {"type": "int", "default": 32},
                    "return_sequences": {
                        "type": "bool",
                        "default": False,
                    },
                    "return_state": {
                        "type": "bool",
                        "default": False,
                    },
                },
            'extra': {
                "activation": {
                    "type": "str",
                    "default": "tanh",
                    "list": True,
                    "available": activation_lh,
                },
                "recurrent_activation": {
                    "type": "str",
                    "default": "sigmoid",
                    "list": True,
                    "available": activation_lh,
                },
                "use_bias": {
                    "type": "bool",
                    "default": True,
                },
                "kernel_initializer": {
                    "type": "str",
                    "default": "glorot_uniform",
                    "list": True,
                    "available": initializer_lh,
                },
                "recurrent_initializer": {
                    "type": "str",
                    "default": "orthogonal",
                    "list": True,
                    "available": initializer_lh,
                },
                "bias_initializer": {
                    "type": "str",
                    "default": "zeros",
                    "list": True,
                    "available": initializer_lh,
                },
                "unit_forget_bias": {
                    "type": "bool",
                    "default": True,
                },
                "kernel_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "recurrent_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "bias_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "activity_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "kernel_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "recurrent_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "bias_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "dropout": {"type": "float", "default": 0.0},
                "recurrent_dropout": {"type": "float", "default": 0.0},
                "go_backwards": {
                    "type": "bool",
                    "default": False,
                },
                "stateful": {
                    "type": "bool",
                    "default": False,
                },
                "time_major": {
                    "type": "bool",
                    "default": False,
                },
                "unroll": {
                    "type": "bool",
                    "default": False,
                }
            }
        },
        "Embedding": {
            "main":
                {
                    "input_dim": {"type": "int", "default": None},
                    "output_dim": {"type": "int", "default": None},
                    "input_length": {"type": "int", "default": None},
                },
            'extra': {
                "embeddings_initializer": {
                    "type": "str",
                    "default": "uniform",
                    "list": True,
                    "available": initializer_lh,
                },
                "embeddings_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "activity_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "embeddings_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "mask_zero": {"type": "bool", "default": False},
            }
        },
        "RepeatVector": {
            "main":
                {"n": {"type": "int", "default": 8}},
            'extra': {}
        },
        "BatchNormalization ": {
            'main': {},
            'extra': {
                "axis": {"type": "int", "default": -1},
                "momentum": {"type": "float", "default": 0.99},
                "epsilon": {"type": "float", "default": 0.001},
                "center": {
                    "type": "bool",
                    "default": True,
                },
                "scale": {
                    "type": "bool",
                    "default": True,
                },
                "beta_initializer": {
                    "type": "str",
                    "default": "zeros",
                    "list": True,
                    "available": initializer_lh,
                },
                "gamma_initializer": {
                    "type": "str",
                    "default": "ones",
                    "list": True,
                    "available": initializer_lh,
                },
                "moving_mean_initializer": {
                    "type": "str",
                    "default": "zeros",
                    "list": True,
                    "available": initializer_lh,
                },
                "moving_variance_initializer": {
                    "type": "str",
                    "default": "ones",
                    "list": True,
                    "available": initializer_lh,
                },
                "beta_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "gamma_regularizer": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": regularizer_lh,
                },
                "beta_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "gamma_constraint": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": constraint_lh,
                },
                "renorm": {
                    "type": "bool",
                    "default": False,
                },
                "renorm_clipping": {"type": "dict", "default": None},
                "renorm_momentum": {"type": "float", "default": 0.99},
                "fused": {
                    "type": "bool",
                    "default": None,
                },
                "trainable": {
                    "type": "bool",
                    "default": True,
                },
                "virtual_batch_size": {"type": "int", "default": None},
                "adjustment": {"type": "func", "default": None},
                "name": {"type": "str", "default": None}
            }
        }
    }
    pass
