import copy
import sys
from dataclasses import dataclass
from terra_ai import customLayers
# import keras_contrib
import tensorflow

__version__ = 0.025


def check_datatype(in_shape):
    dim = len(in_shape)
    if dim == 1:
        result = "DIM"
    elif dim == 2:
        result = "1D"
    elif dim == 3:
        result = "2D"
    elif dim == 4:
        result = "3D"
        msg = f"Error: {result} dimensions arrays is not supported! input_shape = {in_shape}"
        sys.exit(msg)
    elif dim == 5:
        result = "4D"
        msg = f"Error: {result} dimensions arrays is not supported! input_shape = {in_shape}"
        sys.exit(msg)
    else:
        msg = f"Error: More than 6 dimensions arrays is not supported! input_shape = {in_shape}"
        sys.exit(msg)
    return result


def get_def_parameters_dict(layer_name):
    new_dict = {}
    if len(GUILayersDef.layers_params[layer_name]['main'].keys()) != 0:
        for key, value in GUILayersDef().layers_params[layer_name]['main'].items():
            new_dict[key] = value['default']
    if len(GUILayersDef.layers_params[layer_name]['extra'].keys()) != 0:
        for key, value in GUILayersDef().layers_params[layer_name]['extra'].items():
            new_dict[key] = value['default']
    return new_dict


def get_block_params_from_plan(plan, layers_dict, layer_params, short_plan=False) -> object:
    """
    Extract parameters from block plan to default dict:
    plan:           ex,     [(1, 1, 3, 0, {'filters': 32, 'activation': 'relu', 'kernel_size': (3, 3),
                                            'padding': 'same', 'strides': (2, 2)}, 0, 0),
                            (2, 6, 2, 0, {}, 1, 0),
                            (3, 6, 1, 0, {'rate': 0.2}, 2, 0)],
    default dict:   ex,     {'L1_Conv2D_filters': 32,
                            'L1_Conv2D_kernel_size': (3, 3),
                            'L1_Conv2D_strides': (2, 2),
                            'L1_Conv2D_padding': 'same',
                            'L1_Conv2D_activation': 'relu',
                            'L3_Dropout_rate': 0.2}
    """
    # print(plan)
    get_params = {}
    for layer in plan:
        layer_type = layers_dict.get(layer[1], None).get(layer[2], None)
        if layer_type:
            def_params = copy.deepcopy(layer_params.get(layer_type, {}).get("main", {}))
            def_params["kernel_regularizer"] = layer_params.get(layer_type, {}).get("extra", {}).get(
                "kernel_regularizer", {})
            def_params["use_bias"] = layer_params.get(layer_type, {}).get("extra", {}).get(
                "use_bias", {})

            if short_plan:
                for curr_param, val in layer[3].items():
                    def_params[curr_param]['default'] = val
                for param, val in def_params.items():
                    if val:
                        get_params[f"L{layer[0]}_{layer_type}_{param}"] = val
            else:
                # print(layer_type, layer[4].items())
                for curr_param, val in layer[4].items():
                    # print(layer_type, curr_param, val)
                    def_params[curr_param]['default'] = val
                for param, val in def_params.items():
                    if val:
                        get_params[f"L{layer[0]}_{layer_type}_{param}"] = val
    return get_params


def set_block_params_to_plan(plan, block_params) -> object:
    """
    Put parameters from model parameters dict to block plan:
    plan:           ex,     (1, "Conv2D", {'filters': 32, 'activation': 'relu', 'kernel_size': (3, 3),
                            'padding': 'same', 'strides': (2, 2)}, [2]),
                            (2, "BatchNormalization", {}, [3]),
                            (3, "Dropout", {'rate': 0.2}, [])],
    default dict:   ex,     {'L1_Conv2D_filters': 32, 'L1_Conv2D_kernel_size': (3, 3),
                            'L1_Conv2D_strides': (2, 2),
                            'L1_Conv2D_padding': 'same',
                            'L1_Conv2D_activation': 'relu',
                            'L3_Dropout_rate': 0.2}
    """
    aux_plan = []
    for layer in plan:
        aux_plan.append(list(layer))

    for param, val in block_params.items():
        layer_idx = int(param.split('_')[0][1:])
        def_param = param[len(f"{param.split('_')[0]}_{param.split('_')[1]}_"):]
        aux_plan[layer_idx - 1][2][def_param] = val

    update_plan = []
    for layer in aux_plan:
        update_plan.append(tuple(layer))
    return update_plan


@dataclass
class PlanLinkLibrary:
    custom_block_plan = {
        'Conv2DBNDrop': [
            (1, 1, 3, 0, {'filters': 32, 'activation': 'relu', 'kernel_size': (3, 3),
                          'padding': 'same', 'strides': (2, 2)}, 0, 0),
            (2, 6, 2, 0, {}, 1, 0),
            (3, 6, 1, 0, {'rate': 0.2}, 2, 0)
        ],
        'Conv2DBNLeaky': [
            (1, 1, 3, 0, {'filters': 32, 'activation': 'relu', 'kernel_size': (3, 3),
                          'padding': 'same', 'strides': (2, 2)}, 0, 0),
            (2, 6, 2, 0, {}, 1, 0),
            (3, 5, 2, 0, {'alpha': 0.1}, 2, 0)
        ],
        'CustomResBlock': [
            (1, 1, 3, 0, {'filters': 32, 'activation': 'linear', 'kernel_size': (3, 3),
                          'padding': 'same', 'strides': (1, 1), 'use_bias': False}, 0, 0),
            (2, 6, 2, 0, {}, 1, 0),
            (3, 5, 2, 0, {'alpha': 0.1}, 2, 0),
            (4, 1, 3, 0, {'filters': 64, 'activation': 'linear', 'kernel_size': (1, 1),
                          'padding': 'same', 'strides': (1, 1), 'use_bias': False}, 3, 0),
            (5, 6, 2, 0, {}, 4, 0),
            (6, 5, 2, 0, {'alpha': 0.1}, 5, 0),
            (7, 4, 2, 0, {}, 6, [0]),
        ],
        'Resnet50Block': [
            (1, 1, 3, 0, {'filters': 64, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid',
                          'activation': 'linear'}, 0, 0),
            (2, 6, 2, 0, {}, 1, 0),
            (3, 5, 1, 0, {'activation': 'relu'}, 2, 0),
            (4, 1, 3, 0, {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same',
                          'activation': 'linear'}, 3, 0),
            (5, 6, 2, 0, {}, 4, 0),
            (6, 5, 1, 0, {'activation': 'relu'}, 5, 0),
            (7, 1, 3, 0, {'filters': 256, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid',
                          'activation': 'linear'}, 6, 0),
            (8, 6, 2, 0, {}, 7, 0),
            (9, 4, 2, 0, {}, 8, [0]),
            (10, 5, 1, 0, {'activation': 'relu'}, 9, 0),
        ],
        'PSPBlock': [
            (1, 1, 3, 0, {'filters': 32, 'activation': 'relu', 'kernel_size': (3, 3), 'padding': 'same'}, 0, 0),
            (2, 3, 2, 0, {'pool_size': (2, 2)}, 1, 0),
            (3, 1, 3, 0, {'filters': 64, 'activation': 'relu', 'kernel_size': (3, 3), 'padding': 'same'}, 2, 0),
            (4, 2, 2, 0, {'filters': 64, 'activation': 'relu', 'kernel_size': (3, 3),
                          'padding': 'same', 'strides': (2, 2)}, 3, 0),
            (5, 3, 2, 0, {'pool_size': (4, 4)}, 1, 0),
            (6, 1, 3, 0, {'filters': 128, 'activation': 'relu', 'kernel_size': (3, 3), 'padding': 'same'}, 5, 0),
            (7, 2, 2, 0, {'filters': 128, 'activation': 'relu', 'kernel_size': (3, 3),
                          'padding': 'same', 'strides': (4, 4)}, 6, 0),
            (8, 3, 2, 0, {'pool_size': (8, 8)}, 1, 0),
            (9, 1, 3, 0, {'filters': 256, 'activation': 'relu', 'kernel_size': (3, 3), 'padding': 'same'}, 8, 0),
            (10, 2, 2, 0, {'filters': 256, 'activation': 'relu', 'kernel_size': (3, 3),
                           'padding': 'same', 'strides': (8, 8)}, 9, 0),
            (11, 4, 1, 0, {}, 1, [4, 7, 10]),
            (12, 1, 3, 0, {'filters': 32, 'activation': 'relu', 'kernel_size': (3, 3), 'padding': 'same'}, 11, 0),
        ],
        'UNETBlock': [
            (1, 1, 3, 0, {'filters': 64, 'activation': 'relu', 'kernel_size': (3, 3), 'strides': (1, 1),
                          'padding': 'same'}, 0, 0),
            (2, 6, 2, 0, {}, 1, 0),
            (3, 3, 2, 0, {'padding': 'same', 'pool_size': (2, 2), 'strides': (2, 2)}, 2, 0),
            (4, 1, 3, 0, {'filters': 128, 'activation': 'relu', 'kernel_size': (3, 3),
                          'strides': (1, 1), 'padding': 'same'}, 3, 0),
            (5, 6, 2, 0, {}, 4, 0),
            (6, 3, 2, 0, {'padding': 'same', 'pool_size': (2, 2), 'strides': (2, 2)}, 5, 0),
            (7, 1, 3, 0, {'filters': 256, 'activation': 'relu', 'kernel_size': (3, 3),
                          'strides': (1, 1), 'padding': 'same'}, 6, 0),
            (8, 6, 2, 0, {}, 7, 0),
            (9, 1, 3, 0, {'filters': 256, 'activation': 'relu', 'kernel_size': (3, 3),
                          'strides': (1, 1), 'padding': 'same'}, 8, 0),
            (10, 6, 2, 0, {}, 9, 0),
            (11, 2, 2, 0, {'filters': 128, 'activation': 'relu', 'kernel_size': (3, 3),
                           'padding': 'same', 'strides': (2, 2)}, 10, 0),
            (12, 6, 2, 0, {}, 11, 0),
            (13, 4, 1, 0, {}, 12, [5]),
            (14, 1, 3, 0, {'filters': 128, 'activation': 'relu', 'kernel_size': (3, 3),
                           'strides': (1, 1), 'padding': 'same'}, 13, 0),
            (15, 6, 2, 0, {}, 14, 0),
            (16, 2, 2, 0, {'filters': 64, 'activation': 'relu', 'kernel_size': (3, 3),
                           'padding': 'same', 'strides': (2, 2)}, 15, 0),
            (17, 6, 2, 0, {}, 16, 0),
            (18, 4, 1, 0, {}, 17, [2]),
            (19, 1, 3, 0, {'filters': 64, 'activation': 'relu', 'kernel_size': (3, 3),
                           'strides': (1, 1), 'padding': 'same'}, 18, 0),
        ],
        'XceptionBlock': [
            (1, 1, 6, 0, {'filters': 128, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same',
                          'activation': 'linear'}, 0, 0),
            (2, 6, 2, 0, {}, 1, 0),
            (3, 5, 1, 0, {'activation': 'relu'}, 2, 0),
            (4, 1, 6, 0, {'filters': 128, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same',
                          'activation': 'linear'}, 3, 0),
            (5, 6, 2, 0, {}, 4, 0),
            (6, 1, 3, 0, {'filters': 128, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'same',
                          'activation': 'linear'}, 0, 0),
            (7, 3, 2, 0, {'pool_size': (3, 3), 'strides': (2, 2), 'padding': 'same'}, 5, 0),
            (8, 6, 2, 0, {}, 6, 0),
            (9, 4, 2, 0, {}, 7, [8]),
        ],
        'InceptionV3block': [
            (1, 1, 3, 0, {'filters': 64, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same',
                          'activation': 'linear'}, 0, 0),
            (2, 6, 2, 0, {}, 1, 0),
            (3, 5, 1, 0, {'activation': 'relu'}, 2, 0),
            (4, 1, 3, 0, {'filters': 96, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same',
                          'activation': 'linear'}, 3, 0),
            (5, 6, 2, 0, {}, 4, 0),
            (6, 5, 1, 0, {'activation': 'relu'}, 5, 0),
            (7, 1, 3, 0, {'filters': 96, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same',
                          'activation': 'linear'}, 6, 0),
            (8, 6, 2, 0, {}, 7, 0),
            (9, 1, 3, 0, {'filters': 48, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same',
                          'activation': 'linear'}, 0, 0),
            (10, 6, 2, 0, {}, 9, 0),
            (11, 5, 1, 0, {'activation': 'relu'}, 10, 0),
            (12, 1, 3, 0, {'filters': 64, 'kernel_size': (5, 5), 'strides': (1, 1), 'padding': 'same',
                           'activation': 'linear'}, 11, 0),
            (13, 6, 2, 0, {}, 12, 0),
            (14, 5, 1, 0, {'activation': 'relu'}, 13, 0),
            (15, 3, 4, 0, {'pool_size': (3, 3), 'strides': (1, 1), 'padding': 'same'}, 0, 0),
            (16, 1, 3, 0, {'filters': 32, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same',
                           'activation': 'linear'}, 15, 0),
            (17, 6, 2, 0, {}, 16, 0),
            (18, 5, 1, 0, {'activation': 'relu'}, 17, 0),
            (19, 1, 3, 0, {'filters': 64, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same',
                           'activation': 'linear'}, 0, 0),
            (20, 6, 2, 0, {}, 19, 0),
            (21, 5, 1, 0, {'activation': 'relu'}, 20, 0),
            (22, 4, 1, 0, {'axis': -1}, 8, [14, 18, 21]),
        ]
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
        "Attention": tensorflow.keras.layers,

        # Layers and functions Activations
        "Activation": tensorflow.keras.layers,
        "LeakyReLU": tensorflow.keras.layers,
        "PReLU": tensorflow.keras.layers,
        "ReLU": tensorflow.keras.layers,
        "Softmax": tensorflow.keras.layers,
        "ELU": tensorflow.keras.layers,
        "ThresholdedReLU": tensorflow.keras.layers,
        "Mish": customLayers,

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
        # "AdditiveAttention": tensorflow.keras.layers,
        "InstanceNormalization": customLayers,
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
        'InceptionV3': tensorflow.keras.applications.inception_v3,
        #  'MobileNet': tensorflow.keras.applications.mobilenet,
        #  'MobileNetV2': tensorflow.keras.applications.mobilenet_v2,
        #  'NASNetLarge': tensorflow.keras.applications.nasnet,
        #  'NASNetMobile': tensorflow.keras.applications.nasnet,
        #  'ResNet101': tensorflow.keras.applications.resnet,
        #  'ResNet152': tensorflow.keras.applications.resnet,
        'ResNet50': tensorflow.keras.applications.resnet50,
        #  'ResNet101V2': tensorflow.keras.applications.resnet_v2,
        #  'ResNet152V2': tensorflow.keras.applications.resnet_v2,
        #  'ResNet50V2': tensorflow.keras.applications.resnet_v2,
        'VGG16': tensorflow.keras.applications.vgg16,
        #  'VGG19': tensorflow.keras.applications.vgg19,
        'Xception': tensorflow.keras.applications.xception,

        # Custom terra blocks as subclasses
        'CustomUNETBlock': customLayers,
        'YOLOResBlock': customLayers,
        'YOLOConvBlock': customLayers,
    }


@dataclass
class GUILayersDef:
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
        # UpScaling Layers
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
        # DownScaling Layers
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
        # Connections Layers
        4: {
            1: "Concatenate",
            2: "Add",
            3: "Multiply",
            4: "Average",
            # 5: "Maximum",
            # 6: "Minimum",
            # 7: "Subtract",
            # 8: "Dot"
            9: "Attention",
        },
        # Activations Layers
        5: {
            1: "Activation",
            2: "LeakyReLU",
            3: "PReLU",
            4: "ReLU",
            5: "Softmax",
            6: "ELU",
            7: "ThresholdedReLU",
            8: "Mish",
        },
        # Optimization Layers
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
            15: "InstanceNormalization",
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
        # Input - Output layers
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
            15: 'InceptionV3',
            #  16: 'MobileNet',
            #  17: 'MobileNetV2',
            #  18: 'NASNetLarge',
            #  19: 'NASNetMobile',
            #  20: 'ResNet101',
            #  21: 'ResNet152',
            22: 'ResNet50',
            #  23: 'ResNet101V2',
            #  24: 'ResNet152V2',
            #  25: 'ResNet50V2',
            26: 'VGG16',
            #  27: 'VGG19',
            28: 'Xception',
            29: 'YOLOResBlock',
            30: 'YOLOConvBlock',
            31: 'CustomUNETBlock',

        },
        # Custom_Block
        12: {
            # 0: 'Custom_Block',
            1: 'Conv2DBNDrop',
            2: 'Conv2DBNLeaky',
            3: 'CustomResBlock',
            4: 'Resnet50Block',
            5: 'PSPBlock',
            6: 'UNETBlock',
            7: 'XceptionBlock',
            8: 'InceptionV3block'
        },
        # 13: {
        #     1: 'CustomUNETBlock',
        #     2: 'YOLOResBlock',
        #     3: 'YOLOConvBlock',
        # }
    }

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
        'Activation',               # added
        # 'ActivityRegularization', # added
        'Add',
        # 'AdditiveAttention',      # added
        # 'AlphaDropout',           # added
        'Attention',                # added
        'Average',                  # added
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
        'Cropping2D',               # added
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
        'ReLU',                     # added
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
        'ZeroPadding2D',            # added
        # 'ZeroPadding3D',          # added
    ]

    # дефолты обновлены по tf 2.5.0
    layer_params = {
        # Main Layers
        "Dense": {
            "main": {
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
            },
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
            "main": {
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
            "main": {
                "size": {
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
            "main": {
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
            "main": {
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
                    "default": ((0, 0), (0, 0)),
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
        # },
        "Attention": {
                "main": {},
                'extra': {
                    "use_scale": {
                        "type": "bool",
                        "default": False
                    },
                }
            },

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
        "Mish": {
            'main': {},
            'extra': {}
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
        #
        # "AdditiveAttention": {
        #     "main": {},
        #     'extra': {
        #         "use_scale": {
        #             "type": "bool",
        #             "default": True
        #         },
        #     }
        # },
        "InstanceNormalization": {
            "main": {},
            'extra': {
                "axis": {
                    "type": "int",
                    "default": -1
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
            "main": {
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
            "main": {
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
        'InceptionV3': {
            'main': {
                "include_top": {
                    "type": "bool",
                    "default": False,
                },
                "weights": {
                    "type": "str",
                    "default": 'imagenet',
                    "list": True,
                    "available": [None, 'imagenet'],
                },
                "output_layer": {
                    "type": "str",
                    "default": "last",
                    "list": True,
                    "available": ["mixed0",
                                  "mixed1",
                                  "mixed2",
                                  "mixed3",
                                  "mixed4",
                                  "mixed5",
                                  "mixed6",
                                  "mixed7",
                                  "mixed8",
                                  "mixed9",
                                  "last"],
                },
                "pooling": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": [None, "avg", "max"],
                },
                "trainable": {
                    "type": "bool",
                    "default": False,
                },
            },
            'extra': {
                "classes": {
                    "type": "int",
                    "default": 1000,
                },
                "classifier_activation": {
                    "type": "str",
                    "default": 'softmax',
                    "list": True,
                    "available": activation_lh,
                },
            }
        },
        #  16: 'MobileNet',
        #  17: 'MobileNetV2',
        #  18: 'NASNetLarge',
        #  19: 'NASNetMobile',
        #  20: 'ResNet101',
        #  21: 'ResNet152',
        'ResNet50': {
            'main': {
                "include_top": {
                    "type": "bool",
                    "default": False,
                },
                "weights": {
                    "type": "str",
                    "default": 'imagenet',
                    "list": True,
                    "available": [None, 'imagenet'],
                },
                "output_layer": {
                    "type": "str",
                    "default": "last",
                    "list": True,
                    "available": ["conv2_block3_out",
                                  "conv3_block4_out",
                                  "conv4_block6_out",
                                  "last"],
                },
                "pooling": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": [None, "avg", "max"],
                },
                "trainable": {
                    "type": "bool",
                    "default": False,
                },
            },
            'extra': {
                "classes": {
                    "type": "int",
                    "default": 1000,
                },
                "classifier_activation": {
                    "type": "str",
                    "default": 'softmax',
                    "list": True,
                    "available": activation_lh,
                },
            }
        },
        #  23: 'ResNet101V2',
        #  24: 'ResNet152V2',
        #  25: 'ResNet50V2',
        'VGG16': {
            'main': {
                "include_top": {
                    "type": "bool",
                    "default": False,
                },
                "weights": {
                    "type": "str",
                    "default": 'imagenet',
                    "list": True,
                    "available": [None, 'imagenet'],
                },
                "output_layer": {
                    "type": "str",
                    "default": "last",
                    "list": True,
                    "available": ["block1_conv2",
                                  "block2_conv2",
                                  "block3_conv3",
                                  "block4_conv3",
                                  "block5_conv3",
                                  "last"],
                },
                "pooling": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": [None, "avg", "max"],
                },
                "trainable": {
                    "type": "bool",
                    "default": False,
                },
            },
            'extra': {
                "classes": {
                    "type": "int",
                    "default": 1000,
                },
                "classifier_activation": {
                    "type": "str",
                    "default": 'softmax',
                    "list": True,
                    "available": activation_lh,
                },
            }
        },
        #  27: 'VGG19',
        'Xception': {
            'main': {
                "include_top": {
                    "type": "bool",
                    "default": False,
                },
                "weights": {
                    "type": "str",
                    "default": 'imagenet',
                    "list": True,
                    "available": [None, 'imagenet'],
                },
                "output_layer": {
                    "type": "str",
                    "default": "last",
                    "list": True,
                    "available": ["last"],
                },
                "pooling": {
                    "type": "str",
                    "default": None,
                    "list": True,
                    "available": [None, "avg", "max"],
                },
                "trainable": {
                    "type": "bool",
                    "default": False,
                },
            },
            'extra': {
                "classes": {
                    "type": "int",
                    "default": 1000,
                },
                "classifier_activation": {
                    "type": "str",
                    "default": 'softmax',
                    "list": True,
                    "available": activation_lh,
                },
            }
        },
        'CustomUNETBlock': {
            'main': {
                "filters": {
                    "type": "int",
                    "default": 32
                },
                "activation": {
                    "type": "str",
                    "default": 'relu',
                    "list": True,
                    "available": activation_lh,
                }
            },
            'extra': {}
        },
        'YOLOResBlock': {
            'main': {
                "mode": {
                    "type": "str",
                    "default": 'YOLOv3',
                    "list": True,
                    "available": ['YOLOv3', 'YOLOv4'],
                },
                "filters": {
                    "type": "int",
                    "default": 32
                },
                "num_resblocks": {
                    "type": "int",
                    "default": 1,
                }
            },
            'extra': {
                "use_bias": {
                    "type": "bool",
                    "default": False
                },
                "activation": {
                    "type": "str",
                    "default": 'LeakyReLU',
                    "list": True,
                    "available": ['LeakyReLU', 'Mish'],
                },
                "include_head": {
                    "type": "bool",
                    "default": True
                },
                "all_narrow": {
                    "type": "bool",
                    "default": True
                },
            }
        },
        'YOLOConvBlock': {
            'main': {
                "mode": {
                    "type": "str",
                    "default": 'YOLOv3',
                    "list": True,
                    "available": ['YOLOv3', 'YOLOv4'],
                },
                "filters": {
                    "type": "int",
                    "default": 32
                },
                "num_conv": {
                    "type": "int",
                    "default": 1,
                },
            },
            'extra': {
                "use_bias": {
                    "type": "bool",
                    "default": False
                },
                "activation": {
                    "type": "str",
                    "default": 'LeakyReLU',
                    "list": True,
                    "available": ['LeakyReLU', 'Mish'],
                },
                "first_conv_kernel": {
                    "type": "tuple",
                    "default": (3, 3)
                },
                "first_conv_strides": {
                    "type": "tuple",
                    "default": (1, 1)
                },
                "first_conv_padding": {
                    "type": "str",
                    "default": "same",
                    "list": True,
                    "available": padding_lh,
                },
            }
        },
    }

    block_params = {
        'Conv2DBNDrop': {
            'main': {},
            'extra': get_block_params_from_plan(PlanLinkLibrary.custom_block_plan.get('Conv2DBNDrop', {}),
                                               layers_dict, layer_params),
        },
        'Conv2DBNLeaky': {
            'main': {},
            'extra': get_block_params_from_plan(PlanLinkLibrary.custom_block_plan.get('Conv2DBNLeaky', {}),
                                               layers_dict, layer_params),
        },
        'CustomResBlock': {
            'main': {},
            'extra': get_block_params_from_plan(PlanLinkLibrary.custom_block_plan.get('CustomResBlock', {}),
                                               layers_dict, layer_params),
        },
        'Resnet50Block': {
            'main': {},
            'extra': get_block_params_from_plan(PlanLinkLibrary.custom_block_plan.get('Resnet50Block', {}),
                                               layers_dict, layer_params),
        },
        'PSPBlock': {
            'main': {},
            'extra': get_block_params_from_plan(PlanLinkLibrary.custom_block_plan.get('PSPBlock', {}),
                                               layers_dict, layer_params),
        },
        'UNETBlock': {
            'main': {},
            'extra': get_block_params_from_plan(PlanLinkLibrary.custom_block_plan.get('UNETBlock', {}),
                                               layers_dict, layer_params),
        },
        'XceptionBlock': {
            'main': {},
            'extra': get_block_params_from_plan(PlanLinkLibrary.custom_block_plan.get('XceptionBlock', {}),
                                               layers_dict, layer_params),
        },
        'InceptionV3block': {
            'main': {},
            'extra': get_block_params_from_plan(PlanLinkLibrary.custom_block_plan.get('InceptionV3block', {}),
                                               layers_dict, layer_params),
        }
    }

    layers_params = {}
    layers_params.update(layer_params)
    layers_params.update(block_params)
    pass


@dataclass
class LayersDef(GUILayersDef, PlanLinkLibrary):
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

    """ Layers defaults 
    """
    for key in GUILayersDef.layers_params.keys():
        locals()[f"{key}_defaults"] = get_def_parameters_dict(key)
    pass


if __name__ == "__main__":
    # params = get_block_params_for_gui(LayersDef.custom_block_plan['Conv2DBNDrop'])
    # params = GUILayersDef.layers_params.get('Conv2DBNDrop', {}).get('main')
    cblocks = ['Conv2DBNDrop', 'Conv2DBNLeaky', 'CustomResBlock', 'Resnet50Block', 'PSPBlock', 'UNETBlock']
    # print(PlanLinkLibrary.custom_block_plan.keys())
    layer = 'InceptionV3block'
    params = getattr(LayersDef, f'{layer}_defaults')
    # print(params)
    # print(getattr(LayersDef, f"{'Conv2DBNLeaky'}_defaults"))
    for k, v in params.items():
        if type(v) == str:
            print(f"'{k}'", ': ', f"'{v}'", ',', sep='')
        else:
            print(f"'{k}'", ': ', v, ',', sep='')
    # print(LayersDef.custom_block_plan.get(layer, None))
    # print(layers_params.get("Conv2DBNDrop"))
    x = LayersDef()
    # print(LayersDef().Dense_defaults)
    pass
