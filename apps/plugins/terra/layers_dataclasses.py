import sys
from dataclasses import dataclass

# import keras_contrib
import tensorflow



@dataclass
class GUILayersDef:
    filters_lh = (1, 1024)
    units_lh = (1, 512)
    kernel_size_lh = (1, 7)
    pool_size_lh = (2, 4, 6)
    strides_lh = (2, 4, 6)
    padding_lh = (
        "same", "valid"
    )
    activation_lh = (
        None, "relu", "sigmoid", "softmax",
        "softplus", "softsign", "tanh", "linear",
        "selu", "elu", "exponential", "hard_sigmoid",
        "gelu", "swish",  # <- activations from tf 2.5.0
    )
    initializer_lh = (
        "random_normal", "random_uniform", "truncated_normal",
        "zeros", "ones", "glorot_normal", "glorot_uniform", "uniform",
        "identity",  # identity only with 2D tensors
        "orthogonal", "constant", "variance_scaling",
        "lecun_normal", "lecun_uniform",
        "variance_scaling", "he_normal", "he_uniform"
    )
    regularizer_lh = (
        None, "l1", "l2", "l1_l2"
    )
    constraint_lh = (
        None, "max_norm", "min_max_norm", "non_neg",
        "unit_norm", "radial_constraint"
    )
    data_format_lh = (
        "channels_last", "channels_first"
    )
    size_lh = (2, 2)
    rate_lh = (0.1, 0.5)
    axis_lh = (0, 1)

    tf_2_5_0_layers = [
        'Activation',  # added
        # 'ActivityRegularization', # added
        'Add',
        # 'AdditiveAttention',      # added
        # 'AlphaDropout',           # added
        # 'Attention',              # added
        'Average',  # added
        'AveragePooling1D',
        'AveragePooling2D',
        # 'AveragePooling3D',       # added
        'BatchNormalization',
        # 'Bidirectional',          # added
        'Concatenate',
        'Conv1D',
        'Conv1DTranspose',
        'Conv2D',
        'Conv2DTranspose',
        'Conv3D',
        # 'Conv3DTranspose',        # added
        # 'ConvLSTM1D',             # added
        # 'ConvLSTM2D',             # added
        # 'ConvLSTM3D',             # added
        # 'Cropping1D',             # added
        # 'Cropping2D',             # added
        # 'Cropping3D',             # added
        'Dense',
        # 'DenseFeatures',
        'DepthwiseConv2D',
        # 'Dot',                    # added
        'Dropout',
        'ELU',  # added
        'Embedding',
        'Flatten',
        'GRU',
        # 'GRUCell',
        # 'GaussianDropout',        # added
        # 'GaussianNoise',          # added
        'GlobalAveragePooling1D',
        'GlobalAveragePooling2D',
        # 'GlobalAveragePooling3D', # added
        'GlobalMaxPooling1D',
        'GlobalMaxPooling2D',
        # 'GlobalMaxPooling3D',     # added
        # 'InputLayer',
        # 'InputSpec',
        'LSTM',
        # 'LSTMCell',
        # 'Lambda',                 # added
        # 'LayerNormalization',     # added
        'LeakyReLU',
        # 'LocallyConnected1D',
        # 'LocallyConnected2D',
        # 'Masking',                # added
        'MaxPooling1D',
        'MaxPooling2D',
        # 'MaxPooling3D',           # added
        # 'Maximum',                # added
        # 'Minimum',                # added
        # 'MultiHeadAttention',     # added
        'Multiply',
        'PReLU',
        # 'Permute',                # added
        # 'RNN',                    # added
        'ReLU',  # added
        'RepeatVector',
        'Reshape',
        'SeparableConv1D',
        'SeparableConv2D',
        # 'SimpleRNN',              # added
        # 'SimpleRNNCell',
        'Softmax',  # added
        # 'SpatialDropout1D',       # added
        # 'SpatialDropout2D',       # added
        # 'SpatialDropout3D',       # added
        # 'StackedRNNCells',
        # 'Subtract',               # added
        # 'ThresholdedReLU',        # added
        # 'TimeDistributed',        # added
        'UpSampling1D',
        'UpSampling2D',
        # 'UpSampling3D',           # added
        # 'Wrapper',
        # 'ZeroPadding1D',          # added
        'ZeroPadding2D',  # added
        # 'ZeroPadding3D',          # added
    ]

    # дефолты обновлены по tf 2.5.0
    layers_params = {
        # Main Layers
        "Dense": {
            "main":
                {
                    "units": {
                        "type": "int",
                        "default": 32},
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
            }
        },
        "Conv1D": {
            "main": {
                "filters": {
                    "type": "int",
                    "default": 32
                },
                "kernel_size": {
                    "type": "int",
                    "default": 5
                },
                "strides": {
                    "type": "int",
                    "default": 1
                },
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
                "dilation_rate": {
                    "type": "int",
                    "default": 1
                },
                "groups": {
                    "type": "int",
                    "default": 1
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
                "filters": {
                    "type": "int",
                    "default": 32
                },
                "kernel_size": {
                    "type": "tuple",
                    "default": (3, 3)
                },
                "strides": {
                    "type": "tuple",
                    "default": (1, 1)
                },
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
                "dilation_rate": {
                    "type": "tuple",
                    "default": (1, 1)
                },
                "groups": {
                    "type": "int",
                    "default": 1
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
                    "filters": {
                        "type": "int",
                        "default": 32
                    },
                    "kernel_size": {
                        "type": "tuple",
                        "default": (3, 3, 3)
                    },
                    "strides": {
                        "type": "tuple",
                        "default": (1, 1, 1)
                    },
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
                "dilation_rate": {
                    "type": "tuple",
                    "default": (1, 1, 1)
                },
                "groups": {
                    "type": "int",
                    "default": 1
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
                    "filters": {
                        "type": "int",
                        "default": 32
                    },
                    "kernel_size": {
                        "type": "int",
                        "default": 3
                    },
                    "strides": {
                        "type": "int",
                        "default": 1
                    },
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
                "dilation_rate": {
                    "type": "int",
                    "default": 1
                },
                "depth_multiplier": {
                    "type": "int",
                    "default": 1
                },
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
                    "filters": {
                        "type": "int",
                        "default": 32
                    },
                    "kernel_size": {
                        "type": "tuple",
                        "default": (3, 3)
                    },
                    "strides": {
                        "type": "tuple",
                        "default": (1, 1)
                    },
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
                "dilation_rate": {
                    "type": "tuple",
                    "default": (1, 1)
                },
                "depth_multiplier": {
                    "type": "int",
                    "default": 1
                },
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
                    "kernel_size": {
                        "type": "tuple",
                        "default": (1, 1)},
                    "strides": {
                        "type": "tuple",
                        "default": (1, 1)
                    },
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
                "dilation_rate": {
                    "type": "tuple",
                    "default": (1, 1)
                },
                "depth_multiplier": {
                    "type": "int",
                    "default": 1
                },
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

        # UpScaling Layers
        "Conv1DTranspose": {
            "main":
                {
                    "filters": {
                        "type": "int",
                        "default": 32
                    },
                    "kernel_size": {
                        "type": "int",
                        "default": 3
                    },
                    "strides": {
                        "type": "int",
                        "default": 1
                    },
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
                "output_padding": {
                    "type": "int",
                    "default": None
                },
                "data_format": {
                    "type": "str",
                    "default": "channels_last",
                    "list": True,
                    "available": data_format_lh,
                },
                "dilation_rate": {
                    "type": "int",
                    "default": 1
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
                    "filters": {
                        "type": "int",
                        "default": 32
                    },
                    "kernel_size": {
                        "type": "tuple",
                        "default": (3, 3)
                    },
                    "strides": {
                        "type": "tuple",
                        "default": (1, 1)
                    },
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
                "output_padding": {
                    "type": "tuple",
                    "default": None
                },
                "data_format": {
                    "type": "str",
                    "default": "channels_last",
                    "list": True,
                    "available": data_format_lh,
                },
                "dilation_rate": {
                    "type": "tuple",
                    "default": (1, 1)
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
        "UpSampling1D": {
            "main":
                {"size": {
                    "type": "int",
                    "default": 2}
                },
            'extra': {}
        },
        "UpSampling2D": {
            "main": {
                "size": {
                    "type": "tuple",
                    "default": (2, 2)
                }
            },
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
        # "Conv3DTranspose": {
        #     "main":{
        #             "filters": {
        #                 "type": "int",
        #                 "default": 32
        #             },
        #             "kernel_size": {
        #                 "type": "tuple",
        #                 "default": (3, 3, 3)
        #             },
        #             "strides": {
        #                 "type": "tuple",
        #                 "default": (1, 1, 1)
        #             },
        #             "padding": {
        #                 "type": "str",
        #                 "default": "same",
        #                 "list": True,
        #                 "available": padding_lh,
        #             },
        #             "activation": {
        #                 "type": "str",
        #                 "default": 'relu',
        #                 "list": True,
        #                 "available": activation_lh,
        #             },
        #         },
        #     'extra': {
        #         "output_padding": {
        #             "type": "tuple",
        #             "default": None
        #         },
        #         "data_format": {
        #             "type": "str",
        #             "default": "channels_last",
        #             "list": True,
        #             "available": data_format_lh,
        #         },
        #         "dilation_rate": {
        #             "type": "tuple",
        #             "default": (1, 1, 1)
        #         },
        #         "use_bias": {
        #             "type": "bool",
        #             "default": True,
        #         },
        #         "kernel_initializer": {
        #             "type": "str",
        #             "default": "glorot_uniform",
        #             "list": True,
        #             "available": initializer_lh,
        #         },
        #         "bias_initializer": {
        #             "type": "str",
        #             "default": "zeros",
        #             "list": True,
        #             "available": initializer_lh,
        #         },
        #         "kernel_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "bias_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "activity_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "kernel_constraint": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": constraint_lh,
        #         },
        #         "bias_constraint": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": constraint_lh,
        #         }
        #     }
        # },

        # "UpSampling3D": {
        #     "main": {
        #         "size": {
        #             "type": "tuple",
        #             "default": (2, 2, 2)
        #         }
        #     },
        #     'extra': {
        #         "data_format": {
        #             "type": "str",
        #             "default": "channels_last",
        #             "list": True,
        #             "available": data_format_lh,
        #         },
        #     }
        # },
        # 'ZeroPadding1D': {
        #     "main":
        #         {"padding": {
        #             "type": "int",
        #             "default": 1}
        #         },
        #     'extra': {}
        # },
        "ZeroPadding2D": {
            "main": {
                "padding": {
                    "type": "tuple",
                    "default": ((1, 1), (1, 1))
                }
            },
            'extra': {
                "data_format": {
                    "type": "str",
                    "default": "channels_last",
                    "list": True,
                    "available": data_format_lh,
                },
            }
        },
        # "ZeroPadding3D": {
        #     "main": {
        #         "padding": {
        #             "type": "tuple",
        #             "default": (1, 1, 1)
        #         }
        #     },
        #     'extra': {
        #         "data_format": {
        #             "type": "str",
        #             "default": "channels_last",
        #             "list": True,
        #             "available": data_format_lh,
        #         },
        #     }
        # },

        # DownScaling Layers
        "MaxPooling1D": {
            "main":
                {
                    "pool_size": {
                        "type": "int",
                        "default": 2
                    },
                    "strides": {
                        "type": "int",
                        "default": None
                    },
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
                    "pool_size": {
                        "type": "tuple",
                        "default": (2, 2)
                    },
                    "strides": {
                        "type": "tuple",
                        "default": None
                    },
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
                    "pool_size": {
                        "type": "int",
                        "default": 2
                    },
                    "strides": {
                        "type": "int",
                        "default": None
                    },
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
                    "pool_size": {
                        "type": "tuple",
                        "default": (2, 2)
                    },
                    "strides": {
                        "type": "tuple",
                        "default": None},
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
        # "MaxPooling3D": {
        #     "main":
        #         {
        #             "pool_size": {
        #                 "type": "tuple",
        #                 "default": (2, 2, 2)
        #             },
        #             "strides": {
        #                 "type": "tuple",
        #                 "default": None
        #             },
        #             "padding": {
        #                 "type": "str",
        #                 "default": "same",
        #                 "list": True,
        #                 "available": padding_lh,
        #             },
        #         },
        #     'extra': {
        #         "data_format": {
        #             "type": "str",
        #             "default": "channels_last",
        #             "list": True,
        #             "available": data_format_lh,
        #         }
        #     }
        # },
        # "AveragePooling3D": {
        #     "main":
        #         {
        #             "pool_size": {
        #                 "type": "tuple",
        #                 "default": (2, 2, 2)
        #             },
        #             "strides": {
        #                 "type": "tuple",
        #                 "default": None},
        #             "padding": {
        #                 "type": "str",
        #                 "default": "same",
        #                 "list": True,
        #                 "available": padding_lh,
        #             },
        #         },
        #     'extra': {
        #         "data_format": {
        #             "type": "str",
        #             "default": "channels_last",
        #             "list": True,
        #             "available": data_format_lh,
        #         }
        #     }
        # },
        # "Cropping1D": {
        #     'main': {
        #         "cropping": {
        #             "type": "tuple",
        #             "default": (1, 1),
        #         }
        #     },
        #     'extra': {}
        # },
        "Cropping2D": {
            'main': {
                "cropping": {
                    "type": "tuple",
                    "default": (0, 0),
                }
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
        # "Cropping3D": {
        #     'main': {
        #         "cropping": {
        #             "type": "tuple",
        #             "default": ((1, 1), (1, 1), (1, 1)),
        #         }
        #     },
        #     'extra': {
        #         "data_format": {
        #             "type": "str",
        #             "default": "channels_last",
        #             "list": True,
        #             "available": data_format_lh,
        #         }
        #     }
        # },

        # Layers Connections
        "Concatenate": {
            'main': {
                "axis": {
                    "type": "int",
                    "default": -1
                }
            },
            'extra': {}
        },
        "Add": {
            'main': {},
            'extra': {}
        },
        "Multiply": {
            'main': {},
            'extra': {}
        },
        "Average": {
            'main': {},
            'extra': {}
        },
        # "Maximum": {
        #     'main': {},
        #     'extra': {}
        # },
        # "Minimum": {
        #     'main': {},
        #     'extra': {}
        # },
        # "Subtract": {
        #     'main': {},
        #     'extra': {}
        # },
        # "Dot": {
        #     'main': {
        #         'axes': {
        #             "type": "tuple",
        #             "default": (1,)
        #         }
        #     },
        #     'extra': {
        #         'normilize': {
        #             "type": "bool",
        #             "default": False
        #     }
        # }

        # Activations Layers
        # "sigmoid": {
        #     'main': {},
        #     'extra': {}
        # },  # убрать так как он указывается через Activation
        # "softmax": {
        #     'main': {},
        #     'extra': {}
        # },  # убрать так как он указывается через Activation
        # "tanh": {
        #     'main': {},
        #     'extra': {}
        # },  # убрать так как он указывается через Activation
        # "relu": {
        #     'main': {},
        #     'extra': {}
        # },  # убрать так как он указывается через Activation
        # "elu": {
        #     'main': {},
        #     'extra': {}
        # },  # убрать так как он указывается через Activation
        # "selu": {
        #     'main': {},
        #     'extra': {}
        # },  # убрать так как он указывается через Activation
        "Activation": {
            'main': {
                "activation": {
                    "type": "str",
                    "default": 'relu',
                    "list": True,
                    "available": activation_lh,
                }
            },
            'extra': {}
        },
        "LeakyReLU": {
            "main": {
                "alpha": {
                    "type": "float",
                    "default": 0.3
                }
            },
            'extra': {}
        },
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
                "shared_axes": {
                    "type": "list",
                    "default": None
                }
            }
        },
        "ReLU": {
            "main": {},
            'extra': {
                "max_value": {
                    "type": "float",
                    "default": None
                },
                "negative_slope": {
                    "type": "float",
                    "default": 0.0
                },
                "threshold": {
                    "type": "float",
                    "default": 0.0
                },
            }
        },
        "Softmax": {
            'main': {},
            'extra': {"axis": {
                "type": "int",
                "default": -1}
            }
        },
        "ELU": {
            'main': {},
            'extra': {
                "alpha": {
                    "type": "float",
                    "default": 1.0
                }
            }
        },
        "ThresholdedReLU": {
            'main': {},
            'extra': {
                "theta": {
                    "type": "float",
                    "default": 1.0
                }
            }
        },

        # Optimization Layers
        "Dropout": {
            "main": {
                "rate": {
                    "type": "float",
                    "default": 0.1
                }
            },
            'extra': {
                "noise_shape": {
                    "type": "tensor",
                    "default": None
                },
                "seed": {
                    "type": "int",
                    "default": None
                }
            }
        },
        "BatchNormalization": {
            'main': {},
            'extra': {
                "axis": {
                    "type": "int",
                    "default": -1
                },
                "momentum": {
                    "type": "float",
                    "default": 0.99
                },
                "epsilon": {
                    "type": "float",
                    "default": 0.001
                },
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
            }
        },
        # "Masking": {
        #     "main": {
        #         "mask_value": {
        #             "type": "float",
        #             "default": 0.0
        #         }
        #     },
        #     'extra': {}
        # },
        # "LayerNormalization": {
        #     'main': {},
        #     'extra': {
        #         "axis": {
        #             "type": "int",
        #             "default": -1
        #         },
        #         "epsilon": {
        #             "type": "float",
        #             "default": 0.001
        #         },
        #         "center": {
        #             "type": "bool",
        #             "default": True,
        #         },
        #         "scale": {
        #             "type": "bool",
        #             "default": True,
        #         },
        #         "beta_initializer": {
        #             "type": "str",
        #             "default": "zeros",
        #             "list": True,
        #             "available": initializer_lh,
        #         },
        #         "gamma_initializer": {
        #             "type": "str",
        #             "default": "ones",
        #             "list": True,
        #             "available": initializer_lh,
        #         },
        #         "beta_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "gamma_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "beta_constraint": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": constraint_lh,
        #         },
        #         "gamma_constraint": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": constraint_lh,
        #         },
        #     }
        # },
        # "SpatialDropout1D": {
        #     "main": {
        #         "rate": {
        #             "type": "float",
        #             "default": 0.1
        #         }
        #     },
        #     'extra': {}
        # },
        # "SpatialDropout2D": {
        #     "main": {
        #         "rate": {
        #             "type": "float",
        #             "default": 0.1
        #         }
        #     },
        #     'extra': {
        #         "data_format": {
        #             "type": "str",
        #             "default": "channels_last",
        #             "list": True,
        #             "available": data_format_lh,
        #         },
        #     }
        # },
        # "SpatialDropout3D": {
        #     "main": {
        #         "rate": {
        #             "type": "float",
        #             "default": 0.1
        #         }
        #     },
        #     'extra': {
        #         "data_format": {
        #             "type": "str",
        #             "default": "channels_last",
        #             "list": True,
        #             "available": data_format_lh,
        #         },
        #     }
        # },
        # "GaussianDropout": {
        #     "main": {
        #         "rate": {
        #             "type": "float",
        #             "default": 0.1
        #         }
        #     },
        #     'extra': {}
        # },
        # "GaussianNoise": {
        #     "main": {
        #         "stddev": {
        #             "type": "float",
        #             "default": 1.0
        #         }
        #     },
        #     'extra': {}
        # },
        # "ActivityRegularization": {
        #     "main": {
        #         "l1": {
        #             "type": "float",
        #             "default": 1.0
        #         },
        #         "l2": {
        #             "type": "float",
        #             "default": 1.0
        #         }
        #     },
        #     'extra': {}
        # },
        # "AlphaDropout": {
        #     "main": {
        #         "rate": {
        #             "type": "float",
        #             "default": 0.1
        #         }
        #     },
        #     'extra': {
        #         "noise_shape": {
        #             "type": "tensor",
        #             "default": None
        #         },
        #         "seed": {
        #             "type": "int",
        #             "default": None
        #         }
        #     }
        # },
        # "MultiHeadAttention": {
        #     "main": {
        #         "num_heads": {
        #             "type": "int",
        #             "default": 2
        #         },
        #         "key_dim": {
        #             "type": "int",
        #             "default": 2
        #         },
        #     },
        #     'extra': {
        #         "value_dim": {
        #             "type": "int",
        #             "default": None
        #         },
        #         "dropout": {
        #             "type": "float",
        #             "default": 0.0
        #         },
        #         "use_bias": {
        #             "type": "bool",
        #             "default": True
        #         },
        #         "kernel_initializer": {
        #             "type": "str",
        #             "default": "glorot_uniform",
        #             "list": True,
        #             "available": initializer_lh,
        #         },
        #         "bias_initializer": {
        #             "type": "str",
        #             "default": "zeros",
        #             "list": True,
        #             "available": initializer_lh,
        #         },
        #         "kernel_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "bias_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "activity_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "kernel_constraint": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": constraint_lh,
        #         },
        #         "bias_constraint": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": constraint_lh,
        #         }
        #     }
        # },
        # "Attention": {
        #     "main": {},
        #     'extra': {
        #         "use_scale": {
        #             "type": "bool",
        #             "default": False
        #         },
        #     }
        # },
        # "AdditiveAttention": {
        #     "main": {},
        #     'extra': {
        #         "use_scale": {
        #             "type": "bool",
        #             "default": True
        #         },
        #     }
        # },
        # "InstanceNormalization": {
        #     "main": {},
        #     'extra': {
        #         "axis": {
        #             "type": "int",
        #             "default": -1
        #         },
        #         "epsilon": {
        #             "type": "float",
        #             "default": 0.001
        #         },
        #         "center": {
        #             "type": "bool",
        #             "default": True,
        #         },
        #         "scale": {
        #             "type": "bool",
        #             "default": True,
        #         },
        #         "beta_initializer": {
        #             "type": "str",
        #             "default": "zeros",
        #             "list": True,
        #             "available": initializer_lh,
        #         },
        #         "gamma_initializer": {
        #             "type": "str",
        #             "default": "ones",
        #             "list": True,
        #             "available": initializer_lh,
        #         },
        #         "beta_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "gamma_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "beta_constraint": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": constraint_lh,
        #         },
        #         "gamma_constraint": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": constraint_lh,
        #         },
        #     }
        # },
        "Normalization": {
            'main': {},
            'extra': {
                "axis": {
                    "type": "int",
                    "default": -1
                },
                "mean": {
                    "type": "float",
                    "default": None
                },
                "variance": {
                    "type": "float",
                    "default": None
                },
            }
        },

        # Recurrent layers
        "Embedding": {
            "main":
                {
                    "input_dim": {
                        "type": "int",
                        "default": None
                    },
                    "output_dim": {
                        "type": "int",
                        "default": None
                    },
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
                "mask_zero": {
                    "type": "bool",
                    "default": False
                },
                "input_length": {
                    "type": "int",
                    "default": None
                },
            }
        },
        "LSTM": {
            "main":
                {
                    "units": {
                        "type": "int",
                        "default": 32
                    },
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
                "dropout": {
                    "type": "float",
                    "default": 0.0
                },
                "recurrent_dropout": {
                    "type": "float",
                    "default": 0.0
                },
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
        "GRU": {
            "main":
                {
                    "units": {
                        "type": "int",
                        "default": 32
                    },
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
                "dropout": {
                    "type": "float",
                    "default": 0.0
                },
                "recurrent_dropout": {
                    "type": "float",
                    "default": 0.0
                },
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
        # "SimpleRNN": {
        #     "main": {
        #         "units": {
        #             "type": "int",
        #             "default": 32
        #         },
        #         "return_sequences": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #         "return_state": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #     },
        #     'extra': {
        #         "activation": {
        #             "type": "str",
        #             "default": "tanh",
        #             "list": True,
        #             "available": activation_lh,
        #         },
        #         "kernel_initializer": {
        #             "type": "str",
        #             "default": "glorot_uniform",
        #             "list": True,
        #             "available": initializer_lh,
        #         },
        #         "recurrent_initializer": {
        #             "type": "str",
        #             "default": "orthogonal",
        #             "list": True,
        #             "available": initializer_lh,
        #         },
        #         "bias_initializer": {
        #             "type": "str",
        #             "default": "zeros",
        #             "list": True,
        #             "available": initializer_lh,
        #         },
        #         "kernel_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "recurrent_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "bias_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "activity_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "kernel_constraint": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": constraint_lh,
        #         },
        #         "recurrent_constraint": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": constraint_lh,
        #         },
        #         "bias_constraint": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": constraint_lh,
        #         },
        #         "dropout": {
        #             "type": "float",
        #             "default": 0.0
        #         },
        #         "recurrent_dropout": {
        #             "type": "float",
        #             "default": 0.0
        #         },
        #         "go_backwards": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #         "stateful": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #         "unroll": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #     }
        # },
        # "TimeDistributed": {
        #     "main": {
        #         "layer": {
        #             "type": "obj",
        #             "default": None
        #         },
        #     },
        #     'extra': {}
        # },
        # "Bidirectional": {
        #     "main": {
        #         "layer": {
        #             "type": "obj",
        #             "default": None
        #         },
        #     },
        #     'extra': {
        #         "merge_mode": {
        #             "type": "str",
        #             "default": 'concat',
        #             "list": True,
        #             "available": ['sum', 'mul', 'concat', 'ave', None],
        #         },
        #         "backward_layer": {
        #             "type": "obj",
        #             "default": None
        #         },
        #     }
        # },
        # "ConvLSTM1D": {
        #     "main": {
        #         "filters": {
        #             "type": "int",
        #             "default": 32
        #         },
        #         "kernel_size": {
        #             "type": "int",
        #             "default": 5
        #         },
        #         "strides": {
        #             "type": "int",
        #             "default": 1
        #         },
        #         "padding": {
        #             "type": "str",
        #             "default": "same",
        #             "list": True,
        #             "available": padding_lh,
        #         },
        #         "return_sequences": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #         "return_state": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #     },
        #     'extra': {
        #         "data_format": {
        #             "type": "str",
        #             "default": "channels_last",
        #             "list": True,
        #             "available": data_format_lh,
        #         },
        #         "dilation_rate": {
        #             "type": "int",
        #             "default": 1
        #         },
        #         "activation": {
        #             "type": "str",
        #             "default": "tanh",
        #             "list": True,
        #             "available": activation_lh,
        #         },
        #         "recurrent_activation": {
        #             "type": "str",
        #             "default": "hard_sigmoid",
        #             "list": True,
        #             "available": activation_lh,
        #         },
        #         "use_bias": {
        #             "type": "bool",
        #             "default": True,
        #         },
        #         "kernel_initializer": {
        #             "type": "str",
        #             "default": "glorot_uniform",
        #             "list": True,
        #             "available": initializer_lh,
        #         },
        #         "recurrent_initializer": {
        #             "type": "str",
        #             "default": "orthogonal",
        #             "list": True,
        #             "available": initializer_lh,
        #         },
        #         "bias_initializer": {
        #             "type": "str",
        #             "default": "zeros",
        #             "list": True,
        #             "available": initializer_lh,
        #         },
        #         "unit_forget_bias": {
        #             "type": "bool",
        #             "default": True,
        #         },
        #         "kernel_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "recurrent_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "bias_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "activity_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "kernel_constraint": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": constraint_lh,
        #         },
        #         "recurrent_constraint": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": constraint_lh,
        #         },
        #         "bias_constraint": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": constraint_lh,
        #         },
        #         "dropout": {
        #             "type": "float",
        #             "default": 0.0
        #         },
        #         "recurrent_dropout": {
        #             "type": "float",
        #             "default": 0.0
        #         },
        #         "go_backwards": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #         "stateful": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #     }
        # },
        # "ConvLSTM2D": {
        #     "main": {
        #         "filters": {
        #             "type": "int",
        #             "default": 32
        #         },
        #         "kernel_size": {
        #             "type": "tuple",
        #             "default": (3, 3)
        #         },
        #         "strides": {
        #             "type": "tuple",
        #             "default": (1, 1)
        #         },
        #         "padding": {
        #             "type": "str",
        #             "default": "same",
        #             "list": True,
        #             "available": padding_lh,
        #         },
        #         "return_sequences": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #         "return_state": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #     },
        #     'extra': {
        #         "data_format": {
        #             "type": "str",
        #             "default": "channels_last",
        #             "list": True,
        #             "available": data_format_lh,
        #         },
        #         "dilation_rate": {
        #             "type": "tuple",
        #             "default": (1, 1)
        #         },
        #         "activation": {
        #             "type": "str",
        #             "default": "tanh",
        #             "list": True,
        #             "available": activation_lh,
        #         },
        #         "recurrent_activation": {
        #             "type": "str",
        #             "default": "hard_sigmoid",
        #             "list": True,
        #             "available": activation_lh,
        #         },
        #         "use_bias": {
        #             "type": "bool",
        #             "default": True,
        #         },
        #         "kernel_initializer": {
        #             "type": "str",
        #             "default": "glorot_uniform",
        #             "list": True,
        #             "available": initializer_lh,
        #         },
        #         "recurrent_initializer": {
        #             "type": "str",
        #             "default": "orthogonal",
        #             "list": True,
        #             "available": initializer_lh,
        #         },
        #         "bias_initializer": {
        #             "type": "str",
        #             "default": "zeros",
        #             "list": True,
        #             "available": initializer_lh,
        #         },
        #         "unit_forget_bias": {
        #             "type": "bool",
        #             "default": True,
        #         },
        #         "kernel_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "recurrent_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "bias_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "activity_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "kernel_constraint": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": constraint_lh,
        #         },
        #         "recurrent_constraint": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": constraint_lh,
        #         },
        #         "bias_constraint": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": constraint_lh,
        #         },
        #         "dropout": {
        #             "type": "float",
        #             "default": 0.0
        #         },
        #         "recurrent_dropout": {
        #             "type": "float",
        #             "default": 0.0
        #         },
        #         "go_backwards": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #         "stateful": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #     }
        # },
        # "ConvLSTM3D": {
        #     "main": {
        #         "filters": {
        #             "type": "int",
        #             "default": 32
        #         },
        #         "kernel_size": {
        #             "type": "tuple",
        #             "default": (3, 3, 3)
        #         },
        #         "strides": {
        #             "type": "tuple",
        #             "default": (1, 1, 1)
        #         },
        #         "padding": {
        #             "type": "str",
        #             "default": "same",
        #             "list": True,
        #             "available": padding_lh,
        #         },
        #         "return_sequences": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #         "return_state": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #     },
        #     'extra': {
        #         "data_format": {
        #             "type": "str",
        #             "default": "channels_last",
        #             "list": True,
        #             "available": data_format_lh,
        #         },
        #         "dilation_rate": {
        #             "type": "tuple",
        #             "default": (1, 1, 1)
        #         },
        #         "activation": {
        #             "type": "str",
        #             "default": "tanh",
        #             "list": True,
        #             "available": activation_lh,
        #         },
        #         "recurrent_activation": {
        #             "type": "str",
        #             "default": "hard_sigmoid",
        #             "list": True,
        #             "available": activation_lh,
        #         },
        #         "use_bias": {
        #             "type": "bool",
        #             "default": True,
        #         },
        #         "kernel_initializer": {
        #             "type": "str",
        #             "default": "glorot_uniform",
        #             "list": True,
        #             "available": initializer_lh,
        #         },
        #         "recurrent_initializer": {
        #             "type": "str",
        #             "default": "orthogonal",
        #             "list": True,
        #             "available": initializer_lh,
        #         },
        #         "bias_initializer": {
        #             "type": "str",
        #             "default": "zeros",
        #             "list": True,
        #             "available": initializer_lh,
        #         },
        #         "unit_forget_bias": {
        #             "type": "bool",
        #             "default": True,
        #         },
        #         "kernel_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "recurrent_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "bias_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "activity_regularizer": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": regularizer_lh,
        #         },
        #         "kernel_constraint": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": constraint_lh,
        #         },
        #         "recurrent_constraint": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": constraint_lh,
        #         },
        #         "bias_constraint": {
        #             "type": "str",
        #             "default": None,
        #             "list": True,
        #             "available": constraint_lh,
        #         },
        #         "dropout": {
        #             "type": "float",
        #             "default": 0.0
        #         },
        #         "recurrent_dropout": {
        #             "type": "float",
        #             "default": 0.0
        #         },
        #         "go_backwards": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #         "stateful": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #     }
        # },
        # "RNN": {
        #     "main": {
        #         "cell": {
        #             "type": "obj",
        #             "default": None
        #         },
        #         "return_sequences": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #         "return_state": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #     },
        #     'extra': {
        #         "go_backwards": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #         "stateful": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #         "unroll": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #         "time_major": {
        #             "type": "bool",
        #             "default": False,
        #         },
        #     }
        # },

        # Shape-shifters
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
        "Reshape": {
            'main': {
                "target_shape": {
                    "type": "tuple",
                    "default": None
                }
            },
            'extra': {}
        },
        "GlobalMaxPooling1D": {
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
        "GlobalMaxPooling2D": {
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
        "GlobalAveragePooling1D": {
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
        "GlobalAveragePooling2D": {
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
        "RepeatVector": {
            "main": {
                "n": {
                    "type": "int",
                    "default": 8
                }
            },
            'extra': {}
        },
        # "GlobalMaxPooling3D": {
        #     'main': {},
        #     'extra': {
        #         "data_format": {
        #             "type": "str",
        #             "default": "channels_last",
        #             "list": True,
        #             "available": data_format_lh,
        #         }
        #     }
        # },
        # "GlobalAveragePooling3D": {
        #     'main': {},
        #     'extra': {
        #         "data_format": {
        #             "type": "str",
        #             "default": "channels_last",
        #             "list": True,
        #             "available": data_format_lh,
        #         }
        #     }
        # },
        # "Permute": {
        #     'main': {
        #         "dims": {
        #             "type": "tuple",
        #             "default": None,
        #         }
        #     },
        #     'extra': {}
        # },

        # Input - Output custom layers
        "Input": {
            "main": {},
            "extra": {
                # "shape": {"type": "tuple", "default": None},
                # "batch_size": {"type": "int", "default": None},
                # "name": {"type": "str", "default": None},
                # "dtype": {"type": "str", "default": None},
                # "sparse": {"type": "bool", "default": False},
                # "tensor": {"type": "tensor", "default": None},
                # "ragged": {"type": "bool", "default": False},
                # "type_spec": {"type": "odj", "default": None}
            }
        },
        # "Lambda": {
        #     "main": {
        #         "function": {
        #             "type": "func",
        #             "default": None
        #         },
        #     },
        #     "extra": {
        #         "output_shape": {
        #             "type": "tuple",
        #             "default": None
        #         },
        #         "mask": {
        #             "type": "???",
        #             "default": None},
        #         "arguments": {
        #             "type": "dict",
        #             "default": None
        #         },
        #     }
        # },

        # Preprocessing layers
        #  'CategoryCrossing': {},
        #  'CategoryEncoding': {},
        #  'CenterCrop': {},
        #  'Discretization': {},
        #  'DiscretizingCombiner': {},
        #  'Hashing': {},
        #  'IntegerLookup': {},
        #  'Normalization': {},
        #  'PreprocessingLayer': {},
        #  'RandomContrast': {}
        #  'RandomCrop': {},
        #  'RandomFlip': {},
        #  'RandomHeight': {},
        #  'RandomRotation': {},
        #  'RandomTranslation': {},
        #  'RandomWidth': {},
        #  'RandomZoom': {},
        'Rescaling': {
            'main': {
                "scale": {
                    "type": "float",
                    "default": 1.0,
                },
                "offset": {
                    "type": "float",
                    "default": 0.0,
                },
            },
            'extra': {}
        },
        'Resizing': {
            'main': {
                "height": {
                    "type": "int",
                    "default": 224,
                },
                "width": {
                    "type": "int",
                    "default": 224,
                },
            },
            'extra': {
                "interpolation": {
                    "type": "str",
                    "default": "bilinear",
                    "list": True,
                    "available": ['bilinear', 'nearest', 'bicubic', 'area',
                                  'lanczos3', 'lanczos5', 'gaussian', 'mitchellcubic'],
                }
            }
        },
        #  'StringLookup': {},
        #  'TextVector': {},

        # Pretrained blocks
        # 1: 'MobileNetV3Large',
        #  2: 'MobileNetV3Small',
        #  3: 'DenseNet121',
        #  4: 'DenseNet169',
        #  5: 'DenseNet201',
        #  6: 'EfficientNetB0',
        #  7: 'EfficientNetB1',
        #  8: 'EfficientNetB2',
        #  9: 'EfficientNetB3',
        #  10: 'EfficientNetB4',
        #  11: 'EfficientNetB5',
        #  12: 'EfficientNetB6',
        #  13: 'EfficientNetB7',
        #  14: 'InceptionResNetV2',
        #  15: 'InceptionV3',
        #  16: 'MobileNet',
        #  17: 'MobileNetV2',
        #  18: 'NASNetLarge',
        #  19: 'NASNetMobile',
        #  20: 'ResNet101',
        #  21: 'ResNet152',
        #  22: 'ResNet50',
        #  23: 'ResNet101V2',
        #  24: 'ResNet152V2',
        #  25: 'ResNet50V2',
        'VGG16': {
            'main': {
                "include_top": {
                    "type": "bool",
                    "default": True,
                },
                "weights": {
                    "type": "str",
                    "default": 'imagenet',
                    "list": True,
                    "available": [None, 'imagenet'],
                },
                "trainable": {
                    "type": "bool",
                    "default": False,
                },
                # "input_layer": {
                #     "type": "str",
                #     "default": "block1_conv1",
                #     "list": True,
                #     "available": ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"],
                # },
                "output_layer": {
                    "type": "str",
                    "default": "last",
                    "list": True,
                    "available": ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3",
                                  "last"],
                },
                "pooling": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": [None, "avg", "max"],
                },
                "classes": {
                    "type": "int",
                    "default": 1000,
                },
            },
            'extra': {
                "classifier_activation": {
                    "type": "str",
                    "default": 'softmax',
                    "list": True,
                    "available": activation_lh,
                },
            }
        },
        #  27: 'VGG19',
        #  28: 'Xception',
    }
    pass


def get_def_parameters_dict(layer_name):
    new_dict = {}
    if len(GUILayersDef.layers_params[layer_name]['main'].keys()) != 0:
        for key, value in GUILayersDef.layers_params[layer_name]['main'].items():
            new_dict[key] = value['default']
    if len(GUILayersDef.layers_params[layer_name]['extra'].keys()) != 0:
        for key, value in GUILayersDef.layers_params[layer_name]['extra'].items():
            new_dict[key] = value['default']
    return new_dict


@dataclass
class LayersDef:
    """Model Plan layers defaults"""

    """ Head
    """
    framework = "keras"
    input_datatype = ""  # Type of data
    plan_name = ""
    num_classes = 0
    input_shape = {}
    output_shape = {}
    plan = []
    model_schema = []
    multiplan = False

    """ Layers dictionary
    """
    layers_dict = {
        # Main Layers
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
            1: "Conv1DTranspose",
            2: "Conv2DTranspose",
            3: "UpSampling1D",
            4: "UpSampling2D",
            # 5: "Conv3DTranspose"
            # 6: "UpSampling3D"
            # 7: "ZeroPadding1D"
            8: "ZeroPadding2D",
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
            # 7: "Cropping1D",
            8: "Cropping2D",
            # 9: "Cropping3D",
        },
        # Layers Connections
        4: {
            1: "Concatenate",
            2: "Add",
            3: "Multiply",
            4: "Average",
            # 5: "Maximum",
            # 6: "Minimum",
            # 7: "Subtract",
            # 8: "Dot"
        },
        # Layers and functions Activations
        5: {
            # 1: "sigmoid",  # убрать так как он указывается через Activation
            # 2: "softmax",  # убрать так как он указывается через Activation
            # 3: "tanh",  # убрать так как он указывается через Activation
            # 4: "relu",  # убрать так как он указывается через Activation
            # 5: "selu",  # убрать так как он указывается через Activation
            # 6: "elu",  # убрать так как он указывается через Activation
            1: "Activation",
            2: "LeakyReLU",
            3: "PReLU",
            4: "ReLU",
            5: "Softmax",
            6: "ELU",
            7: "ThresholdedReLU"
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
            # 15: "InstanceNormalization",
            16: "Normalization",
        },
        # Recurrent layers
        7: {1: "Embedding",
            2: "LSTM",
            3: "GRU",
            # 4: "SimpleRNN",
            # 5: "TimeDistributed",
            # 6: "Bidirectional",
            # 7: "ConvLSTM1D",
            # 8: "ConvLSTM2D",
            # 9: "ConvLSTM3D",
            # 10: "RNN",
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
            # 9: "GlobalAveragePooling3D ",
            # 10: "Permute",
        },
        # Input - Output custom layers
        9: {
            1: "Input",
            # 2: "Lambda"
        },
        # Preprocessing layers
        10: {
            # 1: 'CategoryCrossing',
            #  2: 'CategoryEncoding',
            #  3: 'CenterCrop',
            #  4: 'Discretization',
            #  5: 'DiscretizingCombiner',
            #  6: 'Hashing',
            #  7: 'IntegerLookup',
            #  8: 'Normalization',
            #  9: 'PreprocessingLayer',
            #  10: 'RandomContrast'
            #  11: 'RandomCrop',
            #  12: 'RandomFlip',
            #  13: 'RandomHeight',
            #  14: 'RandomRotation',
            #  15: 'RandomTranslation',
            #  16: 'RandomWidth',
            #  17: 'RandomZoom',
            18: 'Rescaling',
            19: 'Resizing',
            #  20: 'StringLookup',
            #  21: 'TextVector,
        },
        # Pretrained blocks
        11: {
            # 0: 'Custom_Block'
            # 1: 'MobileNetV3Large',
            #  2: 'MobileNetV3Small',
            #  3: 'DenseNet121',
            #  4: 'DenseNet169',
            #  5: 'DenseNet201',
            #  6: 'EfficientNetB0',
            #  7: 'EfficientNetB1',
            #  8: 'EfficientNetB2',
            #  9: 'EfficientNetB3',
            #  10: 'EfficientNetB4',
            #  11: 'EfficientNetB5',
            #  12: 'EfficientNetB6',
            #  13: 'EfficientNetB7',
            #  14: 'InceptionResNetV2',
            #  15: 'InceptionV3',
            #  16: 'MobileNet',
            #  17: 'MobileNetV2',
            #  18: 'NASNetLarge',
            #  19: 'NASNetMobile',
            #  20: 'ResNet101',
            #  21: 'ResNet152',
            #  22: 'ResNet50',
            #  23: 'ResNet101V2',
            #  24: 'ResNet152V2',
            #  25: 'ResNet50V2',
            26: 'VGG16',
            #  27: 'VGG19',
            #  28: 'Xception',
        }
    }

    layers_links = {
        # Main Layers
        "Dense": tensorflow.keras.layers,
        "Conv1D": tensorflow.keras.layers,
        "Conv2D": tensorflow.keras.layers,
        "Conv3D": tensorflow.keras.layers,
        "SeparableConv1D": tensorflow.keras.layers,
        "SeparableConv2D": tensorflow.keras.layers,
        "DepthwiseConv2D": tensorflow.keras.layers,

        # Layers UpScaling
        "Conv1DTranspose": tensorflow.keras.layers,
        "Conv2DTranspose": tensorflow.keras.layers,
        "UpSampling1D": tensorflow.keras.layers,
        "UpSampling2D": tensorflow.keras.layers,
        # "Conv3DTranspose": tensorflow.keras.layers,
        # "UpSampling3D": tensorflow.keras.layers,
        # "ZeroPadding1D": tensorflow.keras.layers,
        "ZeroPadding2D": tensorflow.keras.layers,
        # "ZeroPadding3D": tensorflow.keras.layers,

        #   Layers DownScaling
        "MaxPooling1D": tensorflow.keras.layers,
        "MaxPooling2D": tensorflow.keras.layers,
        "AveragePooling1D": tensorflow.keras.layers,
        "AveragePooling2D": tensorflow.keras.layers,
        # "MaxPooling3D": tensorflow.keras.layers,
        # "AveragePooling3D": tensorflow.keras.layers,

        # Layers Connections
        "Concatenate": tensorflow.keras.layers,
        "Add": tensorflow.keras.layers,
        "Multiply": tensorflow.keras.layers,
        "Average": tensorflow.keras.layers,
        # "Maximum": tensorflow.keras.layers,
        # "Minimum": tensorflow.keras.layers,
        # "Subtract": tensorflow.keras.layers,
        # "Dot": tensorflow.keras.layers,

        # Layers and functions Activations
        "Activation": tensorflow.keras.layers,
        "LeakyReLU": tensorflow.keras.layers,
        "PReLU": tensorflow.keras.layers,
        "ReLU": tensorflow.keras.layers,
        "Softmax": tensorflow.keras.layers,
        "ELU": tensorflow.keras.layers,
        "ThresholdedReLU": tensorflow.keras.layers,

        # Layers Optimization
        "Dropout": tensorflow.keras.layers,
        "BatchNormalization": tensorflow.keras.layers,
        # "Masking": tensorflow.keras.layers,
        # "LayerNormalization": tensorflow.keras.layers,
        # "SpatialDropout1D": tensorflow.keras.layers,
        # "SpatialDropout2D": tensorflow.keras.layers,
        # "SpatialDropout3D": tensorflow.keras.layers,
        # "GaussianDropout": tensorflow.keras.layers,
        # "GaussianNoise": tensorflow.keras.layers,
        # "ActivityRegularization": tensorflow.keras.layers,
        # "AlphaDropout": tensorflow.keras.layers,
        # "MultiHeadAttention": tensorflow.keras.layers,
        # "Attention": tensorflow.keras.layers,
        # "AdditiveAttention": tensorflow.keras.layers,
        # "InstanceNormalization": keras_contrib.layers.normalization.instancenormalization,  # pip install keras_contrib
        "Normalization": tensorflow.keras.layers.experimental.preprocessing,

        # Recurrent layers
        "Embedding": tensorflow.keras.layers,
        "LSTM": tensorflow.keras.layers,
        "GRU": tensorflow.keras.layers,
        # SimpleRNN": tensorflow.keras.layers,
        # "TimeDistributed": tensorflow.keras.layers,
        # "Bidirectional": tensorflow.keras.layers,
        # "ConvLSTM1D": tensorflow.keras.layers,
        # "ConvLSTM2D": tensorflow.keras.layers,
        # "ConvLSTM3D": tensorflow.keras.layers,
        # "RNN": tensorflow.keras.layers,

        # Shape-shifters
        "Flatten": tensorflow.keras.layers,
        "Reshape": tensorflow.keras.layers,
        "GlobalMaxPooling1D": tensorflow.keras.layers,
        "GlobalMaxPooling2D": tensorflow.keras.layers,
        "GlobalAveragePooling1D": tensorflow.keras.layers,
        "GlobalAveragePooling2D": tensorflow.keras.layers,
        "RepeatVector": tensorflow.keras.layers,
        # "GlobalMaxPooling3D": tensorflow.keras.layers,
        # "GlobalAveragePooling3D": tensorflow.keras.layers,
        # "Permute": tensorflow.keras.layers,
        # "Cropping1D": tensorflow.keras.layers,
        "Cropping2D": tensorflow.keras.layers,
        # "Cropping3D": tensorflow.keras.layers,

        # Input - Output custom layers
        "Input": tensorflow.keras.layers,
        # "Lambda": tensorflow.keras.layers,

        # Preprocessing layers
        #  'CategoryCrossing': tensorflow.keras.layers.experimental.preprocessing,
        #  'CategoryEncoding': tensorflow.keras.layers.experimental.preprocessing,
        #  'CenterCrop': tensorflow.keras.layers.experimental.preprocessing,
        #  'Discretization': tensorflow.keras.layers.experimental.preprocessing,
        #  'DiscretizingCombiner': tensorflow.keras.layers.experimental.preprocessing.Discretization,
        #  'Hashing': tensorflow.keras.layers.experimental.preprocessing,
        #  'IntegerLookup': tensorflow.keras.layers.experimental.preprocessing,
        #  'Normalization': tensorflow.keras.layers.experimental.preprocessing,
        #  'PreprocessingLayer': tensorflow.keras.layers.experimental.preprocessing,
        #  'RandomContrast': tensorflow.keras.layers.experimental.preprocessing,
        #  'RandomCrop': tensorflow.keras.layers.experimental.preprocessing,
        #  'RandomFlip': tensorflow.keras.layers.experimental.preprocessing,
        #  'RandomHeight': tensorflow.keras.layers.experimental.preprocessing,
        #  'RandomRotation': tensorflow.keras.layers.experimental.preprocessing,
        #  'RandomTranslation': tensorflow.keras.layers.experimental.preprocessing,
        #  'RandomWidth': tensorflow.keras.layers.experimental.preprocessing,
        #  'RandomZoom': tensorflow.keras.layers.experimental.preprocessing,
        'Rescaling': tensorflow.keras.layers.experimental.preprocessing,
        'Resizing': tensorflow.keras.layers.experimental.preprocessing,
        #  'StringLookup': tensorflow.keras.layers.experimental.preprocessing,
        #  'TextVectorization': tensorflow.keras.layers.experimental.preprocessing,

        # Pretrained blocks
        # 'MobileNetV3Large': tensorflow.keras.applications,
        #  'MobileNetV3Small': tensorflow.keras.applications,
        #  'DenseNet121': tensorflow.keras.applications.densenet,
        #  'DenseNet169': tensorflow.keras.applications.densenet,
        #  'DenseNet201': tensorflow.keras.applications.densenet,
        #  'EfficientNetB0': tensorflow.keras.applications.efficientnet,
        #  'EfficientNetB1': tensorflow.keras.applications.efficientnet,
        #  'EfficientNetB2': tensorflow.keras.applications.efficientnet,
        #  'EfficientNetB3': tensorflow.keras.applications.efficientnet,
        #  'EfficientNetB4': tensorflow.keras.applications.efficientnet,
        #  'EfficientNetB5': tensorflow.keras.applications.efficientnet,
        #  'EfficientNetB6': tensorflow.keras.applications.efficientnet,
        #  'EfficientNetB7': tensorflow.keras.applications.efficientnet,
        #  'InceptionResNetV2': tensorflow.keras.applications.inception_resnet_v2,
        #  'InceptionV3': tensorflow.keras.applications.inception_v3,
        #  'MobileNet': tensorflow.keras.applications.mobilenet,
        #  'MobileNetV2': tensorflow.keras.applications.mobilenet_v2,
        #  'NASNetLarge': tensorflow.keras.applications.nasnet,
        #  'NASNetMobile': tensorflow.keras.applications.nasnet,
        #  'ResNet101': tensorflow.keras.applications.resnet,
        #  'ResNet152': tensorflow.keras.applications.resnet,
        #  'ResNet50': tensorflow.keras.applications.resnet50,
        #  'ResNet101V2': tensorflow.keras.applications.resnet_v2,
        #  'ResNet152V2': tensorflow.keras.applications.resnet_v2,
        #  'ResNet50V2': tensorflow.keras.applications.resnet_v2,
        'VGG16': tensorflow.keras.applications.vgg16,
        #  'VGG19': tensorflow.keras.applications.vgg19,
        #  'Xception': tensorflow.keras.applications.xception,
    }

    """ Default layers kwargs with min, max
    param_lh: 
        param_name_lh: (min, max), (iterable int or str) for random generator 
    """
    filters_lh = GUILayersDef.filters_lh
    units_lh = GUILayersDef.units_lh
    kernel_size_lh = GUILayersDef.kernel_size_lh
    pool_size_lh = GUILayersDef.pool_size_lh
    strides_lh = GUILayersDef.strides_lh
    padding_lh = GUILayersDef.padding_lh
    activation_lh = GUILayersDef.activation_lh
    initializer_lh = GUILayersDef.initializer_lh
    regularizer_lh = GUILayersDef.regularizer_lh
    constraint_lh = GUILayersDef.constraint_lh
    data_format_lh = GUILayersDef.data_format_lh
    size_lh = GUILayersDef.size_lh
    rate_lh = GUILayersDef.rate_lh
    axis_lh = GUILayersDef.axis_lh

    """ Layers defaults 
    """

    Input_defaults = get_def_parameters_dict('Input')
    Conv1D_defaults = get_def_parameters_dict('Conv1D')
    Conv2D_defaults = get_def_parameters_dict('Conv2D')
    Conv3D_defaults = get_def_parameters_dict('Conv3D')
    Conv1DTranspose_defaults = get_def_parameters_dict('Conv1DTranspose')
    Conv2DTranspose_defaults = get_def_parameters_dict('Conv2DTranspose')
    SeparableConv1D_defaults = get_def_parameters_dict('SeparableConv1D')
    SeparableConv2D_defaults = get_def_parameters_dict('SeparableConv2D')
    DepthwiseConv2D_defaults = get_def_parameters_dict('DepthwiseConv2D')
    MaxPooling1D_defaults = get_def_parameters_dict('MaxPooling1D')
    MaxPooling2D_defaults = get_def_parameters_dict('MaxPooling2D')
    AveragePooling1D_defaults = get_def_parameters_dict('AveragePooling1D')
    AveragePooling2D_defaults = get_def_parameters_dict('AveragePooling2D')
    UpSampling1D_defaults = get_def_parameters_dict('UpSampling1D')
    UpSampling2D_defaults = get_def_parameters_dict('UpSampling2D')
    LeakyReLU_defaults = get_def_parameters_dict('LeakyReLU')
    Dropout_defaults = get_def_parameters_dict('Dropout')
    Dense_defaults = get_def_parameters_dict('Dense')
    Add_defaults = get_def_parameters_dict('Add')
    Multiply_defaults = get_def_parameters_dict('Multiply')
    Flatten_defaults = get_def_parameters_dict('Flatten')
    Concatenate_defaults = get_def_parameters_dict('Concatenate')
    Reshape_defaults = get_def_parameters_dict('Reshape')
    PReLU_defaults = get_def_parameters_dict('PReLU')
    GlobalMaxPooling1D_defaults = get_def_parameters_dict('GlobalMaxPooling1D')
    GlobalMaxPooling2D_defaults = get_def_parameters_dict('GlobalMaxPooling2D')
    GlobalAveragePooling1D_defaults = get_def_parameters_dict('GlobalAveragePooling1D')
    GlobalAveragePooling2D_defaults = get_def_parameters_dict('GlobalAveragePooling1D')
    GRU_defaults = get_def_parameters_dict('GRU')
    LSTM_defaults = get_def_parameters_dict('LSTM')
    Embedding_defaults = get_def_parameters_dict('Embedding')
    RepeatVector_defaults = get_def_parameters_dict('RepeatVector')
    BatchNormalization_defaults = get_def_parameters_dict('BatchNormalization')
    Activation_defaults = get_def_parameters_dict('Activation')
    ReLU_defaults = get_def_parameters_dict('ReLU')
    Softmax_defaults = get_def_parameters_dict('Softmax')
    ELU_defaults = get_def_parameters_dict('ELU')
    ThresholdedReLU_defaults = get_def_parameters_dict('ThresholdedReLU')
    Average_defaults = get_def_parameters_dict('Average')
    # InstanceNormalization_defaults = get_def_parameters_dict('InstanceNormalization')
    Rescaling_defaults = get_def_parameters_dict('Rescaling')
    ZeroPadding2D_defaults = get_def_parameters_dict('ZeroPadding2D')
    Normalization_defaults = get_def_parameters_dict('Normalization')
    Cropping2D_defaults = get_def_parameters_dict('Cropping2D')
    VGG16_defaults = get_def_parameters_dict('VGG16')
    Resizing_defaults = get_def_parameters_dict('Resizing')

    # Pretrained model plans
    pretrained_model_plans = {
        # 1: 'MobileNetV3Large',
        #  2: 'MobileNetV3Small',
        #  3: 'DenseNet121',
        #  4: 'DenseNet169',
        #  5: 'DenseNet201',
        #  6: 'EfficientNetB0',
        #  7: 'EfficientNetB1',
        #  8: 'EfficientNetB2',
        #  9: 'EfficientNetB3',
        #  10: 'EfficientNetB4',
        #  11: 'EfficientNetB5',
        #  12: 'EfficientNetB6',
        #  13: 'EfficientNetB7',
        #  14: 'InceptionResNetV2',
        #  15: 'InceptionV3',
        #  16: 'MobileNet',
        #  17: 'MobileNetV2',
        #  18: 'NASNetLarge',
        #  19: 'NASNetMobile',
        #  20: 'ResNet101',
        #  21: 'ResNet152',
        #  22: 'ResNet50',
        #  23: 'ResNet101V2',
        #  24: 'ResNet152V2',
        #  25: 'ResNet50V2',
        'VGG16': [
            [1, 9, 1, 0, {}, 0, 0],
            [2, 1, 3, 0, {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu',
                       'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
                       'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
                       'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
                       'bias_constraint': None, 'name': 'block1_conv1'}, 1, 0],
            [3, 1, 3, 0, {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu',
                       'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
                       'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
                       'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
                       'bias_constraint': None, 'name': 'block1_conv2'}, 2, 0],
            [4, 3, 2, 0, {'pool_size': (2, 2), 'strides': (2, 2), 'padding': 'valid', 'data_format': 'channels_last',
                       'name': 'block1_pool'}, 3, 0],
            [5, 1, 3, 0,
             {'filters': 128, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'block2_conv1'}, 4, 0],
            [6, 1, 3, 0,
             {'filters': 128, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'block2_conv2'}, 5, 0],
            [7, 3, 2, 0, {'pool_size': (2, 2), 'strides': (2, 2), 'padding': 'valid', 'data_format': 'channels_last',
                       'name': 'block2_pool'}, 6, 0],
            [8, 1, 3, 0,
             {'filters': 256, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'block3_conv1'}, 7, 0],
            [9, 1, 3, 0,
             {'filters': 256, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'block3_conv2'}, 8, 0],
            [10, 1, 3, 0,
             {'filters': 256, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'block3_conv3'}, 9, 0],
            [11, 3, 2, 0, {'pool_size': (2, 2), 'strides': (2, 2), 'padding': 'valid', 'data_format': 'channels_last',
                        'name': 'block3_pool'}, 10, 0],
            [12, 1, 3, 0,
             {'filters': 512, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'block4_conv1'}, 11, 0],
            [13, 1, 3, 0,
             {'filters': 512, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'block4_conv2'}, 12, 0],
            [14, 1, 3, 0,
             {'filters': 512, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'block4_conv3'}, 13, 0],
            [15, 3, 2, 0, {'pool_size': (2, 2), 'strides': (2, 2), 'padding': 'valid', 'data_format': 'channels_last',
                        'name': 'block4_pool'}, 14, 0],
            [16, 1, 3, 0,
             {'filters': 512, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'block5_conv1'}, 15, 0],
            [17, 1, 3, 0,
             {'filters': 512, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'block5_conv2'}, 16, 0],
            [18, 1, 3, 0,
             {'filters': 512, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'block5_conv3'}, 17, 0],
            [19, 3, 2, 0, {'pool_size': (2, 2), 'strides': (2, 2), 'padding': 'valid', 'data_format': 'channels_last',
                        'name': 'block5_pool'}, 18, 0],
            [20, 8, 1, 0, {'data_format': 'channels_last', 'name': 'flatten'}, 19, 0],
            [21, 1, 1, 0, {'units': 4096, 'activation': 'relu', 'use_bias': True, 'kernel_initializer': 'GlorotUniform',
                        'bias_initializer': 'Zeros', 'kernel_regularizer': None, 'bias_regularizer': None,
                        'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None,
                        'name': 'fc1'}, 20, 0],
            [22, 1, 1, 0, {'units': 4096, 'activation': 'relu', 'use_bias': True, 'kernel_initializer': 'GlorotUniform',
                        'bias_initializer': 'Zeros', 'kernel_regularizer': None, 'bias_regularizer': None,
                        'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None,
                        'name': 'fc2'}, 21, 0],
            [23, 1, 1, 0, {'units': 1000, 'activation': 'softmax', 'use_bias': True, 'kernel_initializer': 'GlorotUniform',
                        'bias_initializer': 'Zeros', 'kernel_regularizer': None, 'bias_regularizer': None,
                        'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None,
                        'name': 'predictions'}, 22, 0]
        ],
        #  27: 'VGG19',
        #  28: 'Xception',
    }
    pass


if __name__ == "__main__":
    pass