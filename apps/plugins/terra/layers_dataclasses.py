import copy
import sys
from dataclasses import dataclass

# import keras_contrib
import tensorflow

__version__ = 0.022


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


def set_block_params_to_plan(plan, block_params, short_plan=False) -> object:
    """
    Put parameters from model parameters dict to block plan:
    plan:           ex,     [(1, 1, 3, 0, {'filters': 32, 'activation': 'relu', 'kernel_size': (3, 3),
                                            'padding': 'same', 'strides': (2, 2)}, 0, 0),
                            (2, 6, 2, 0, {}, 1, 0),
                            (3, 6, 1, 0, {'rate': 0.2}, 2, 0)],
    default dict:   ex,     {'L1_Conv2D_filters': 32, 'L1_Conv2D_kernel_size': (3, 3),
                            'L1_Conv2D_strides': (2, 2),
                            'L1_Conv2D_padding': 'same',
                            'L1_Conv2D_activation': 'relu',
                            'L3_Dropout_rate': 0.2}
    """
    aux_plan = []
    for layer in plan:
        aux_plan.append(list(layer))
    # print('aux_plan', aux_plan)

    for param, val in block_params.items():
        print('set_block_params_to_plan__param', val, param, val)
        layer_idx = int(param.split('_')[0][1:])
        def_param = param[len(f"{param.split('_')[0]}_{param.split('_')[1]}_"):]
        # print(layer_idx, def_param)
        if short_plan:
            aux_plan[layer_idx - 1][3][def_param] = val
        else:
            aux_plan[layer_idx - 1][4][def_param] = val

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
            (1, 1, 3, 0, {'filters': 32, 'activation': 'relu', 'kernel_size': (3, 3),
                          'padding': 'same', 'strides': (1, 1)}, 0, 0),
            (2, 6, 2, 0, {}, 1, 0),
            (3, 5, 2, 0, {'alpha': 0.1}, 2, 0),
            (4, 1, 3, 0, {'filters': 32, 'activation': 'relu', 'kernel_size': (3, 3),
                          'padding': 'same', 'strides': (1, 1)}, 3, 0),
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
        'InceptionV3': [
            (1, 9, 1, 0, {}, 0, 0),
            (2, 1, 3, 0,
             {'filters': 32, 'kernel_size': (3, 3), 'strides': (2, 2), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d'}, 1, 0),
            (3, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
                          'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                          'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                          'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization'}, 2, 0),
            (4, 5, 1, 0, {'activation': 'relu', 'name': 'activation'}, 3, 0),
            (5, 1, 3, 0,
             {'filters': 32, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_1'}, 4, 0),
            (6, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
                          'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                          'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                          'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_1'}, 5, 0),
            (7, 5, 1, 0, {'activation': 'relu', 'name': 'activation_1'}, 6, 0),
            (8, 1, 3, 0,
             {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_2'}, 7, 0),
            (9, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
                          'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                          'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                          'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_2'}, 8, 0),
            (10, 5, 1, 0, {'activation': 'relu', 'name': 'activation_2'}, 9, 0),
            (11, 3, 2, 0, {'pool_size': (3, 3), 'strides': (2, 2), 'padding': 'valid', 'data_format': 'channels_last',
                           'name': 'max_pooling2d'}, 10, 0),
            (12, 1, 3, 0,
             {'filters': 80, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_3'}, 11, 0),
            (13, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_3'}, 12, 0),
            (14, 5, 1, 0, {'activation': 'relu', 'name': 'activation_3'}, 13, 0),
            (15, 1, 3, 0,
             {'filters': 192, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_4'}, 14, 0),
            (16, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_4'}, 15, 0),
            (17, 5, 1, 0, {'activation': 'relu', 'name': 'activation_4'}, 16, 0),
            (18, 3, 2, 0, {'pool_size': (3, 3), 'strides': (2, 2), 'padding': 'valid', 'data_format': 'channels_last',
                           'name': 'max_pooling2d_1'}, 17, 0),
            (19, 1, 3, 0,
             {'filters': 64, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_8'}, 18, 0),
            (20, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_8'}, 19, 0),
            (21, 5, 1, 0, {'activation': 'relu', 'name': 'activation_8'}, 20, 0),
            (22, 1, 3, 0,
             {'filters': 48, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_6'}, 18, 0),
            (23, 1, 3, 0,
             {'filters': 96, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_9'}, 21, 0),
            (24, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_6'}, 22, 0),
            (25, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_9'}, 23, 0),
            (26, 5, 1, 0, {'activation': 'relu', 'name': 'activation_6'}, 24, 0),
            (27, 5, 1, 0, {'activation': 'relu', 'name': 'activation_9'}, 25, 0),
            (28, 3, 4, 0, {'pool_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last',
                           'name': 'average_pooling2d'}, 18, 0),
            (29, 1, 3, 0,
             {'filters': 64, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_5'}, 18, 0),
            (30, 1, 3, 0,
             {'filters': 64, 'kernel_size': (5, 5), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_7'}, 26, 0),
            (31, 1, 3, 0,
             {'filters': 96, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_10'}, 27, 0),
            (32, 1, 3, 0,
             {'filters': 32, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_11'}, 28, 0),
            (33, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_5'}, 29, 0),
            (34, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_7'}, 30, 0),
            (35, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_10'}, 31, 0),
            (36, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_11'}, 32, 0),
            (37, 5, 1, 0, {'activation': 'relu', 'name': 'activation_5'}, 33, 0),
            (38, 5, 1, 0, {'activation': 'relu', 'name': 'activation_7'}, 34, 0),
            (39, 5, 1, 0, {'activation': 'relu', 'name': 'activation_10'}, 35, 0),
            (40, 5, 1, 0, {'activation': 'relu', 'name': 'activation_11'}, 36, 0),
            (41, 4, 1, 0, {'axis': 3, 'name': 'mixed0'}, 37, [38, 39, 40]),
            (42, 1, 3, 0,
             {'filters': 64, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_15'}, 41, 0),
            (43, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_15'}, 42, 0),
            (44, 5, 1, 0, {'activation': 'relu', 'name': 'activation_15'}, 43, 0),
            (45, 1, 3, 0,
             {'filters': 48, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_13'}, 41, 0),
            (46, 1, 3, 0,
             {'filters': 96, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_16'}, 44, 0),
            (47, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_13'}, 45, 0),
            (48, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_16'}, 46, 0),
            (49, 5, 1, 0, {'activation': 'relu', 'name': 'activation_13'}, 47, 0),
            (50, 5, 1, 0, {'activation': 'relu', 'name': 'activation_16'}, 48, 0),
            (51, 3, 4, 0, {'pool_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last',
                           'name': 'average_pooling2d_1'}, 41, 0),
            (52, 1, 3, 0,
             {'filters': 64, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_12'}, 41, 0),
            (53, 1, 3, 0,
             {'filters': 64, 'kernel_size': (5, 5), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_14'}, 49, 0),
            (54, 1, 3, 0,
             {'filters': 96, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_17'}, 50, 0),
            (55, 1, 3, 0,
             {'filters': 64, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_18'}, 51, 0),
            (56, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_12'}, 52, 0),
            (57, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_14'}, 53, 0),
            (58, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_17'}, 54, 0),
            (59, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_18'}, 55, 0),
            (60, 5, 1, 0, {'activation': 'relu', 'name': 'activation_12'}, 56, 0),
            (61, 5, 1, 0, {'activation': 'relu', 'name': 'activation_14'}, 57, 0),
            (62, 5, 1, 0, {'activation': 'relu', 'name': 'activation_17'}, 58, 0),
            (63, 5, 1, 0, {'activation': 'relu', 'name': 'activation_18'}, 59, 0),
            (64, 4, 1, 0, {'axis': 3, 'name': 'mixed1'}, 60, [61, 62, 63]),
            (65, 1, 3, 0,
             {'filters': 64, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_22'}, 64, 0),
            (66, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_22'}, 65, 0),
            (67, 5, 1, 0, {'activation': 'relu', 'name': 'activation_22'}, 66, 0),
            (68, 1, 3, 0,
             {'filters': 48, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_20'}, 64, 0),
            (69, 1, 3, 0,
             {'filters': 96, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_23'}, 67, 0),
            (70, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_20'}, 68, 0),
            (71, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_23'}, 69, 0),
            (72, 5, 1, 0, {'activation': 'relu', 'name': 'activation_20'}, 70, 0),
            (73, 5, 1, 0, {'activation': 'relu', 'name': 'activation_23'}, 71, 0),
            (74, 3, 4, 0, {'pool_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last',
                           'name': 'average_pooling2d_2'}, 64, 0),
            (75, 1, 3, 0,
             {'filters': 64, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_19'}, 64, 0),
            (76, 1, 3, 0,
             {'filters': 64, 'kernel_size': (5, 5), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_21'}, 72, 0),
            (77, 1, 3, 0,
             {'filters': 96, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_24'}, 73, 0),
            (78, 1, 3, 0,
             {'filters': 64, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_25'}, 74, 0),
            (79, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_19'}, 75, 0),
            (80, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_21'}, 76, 0),
            (81, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_24'}, 77, 0),
            (82, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_25'}, 78, 0),
            (83, 5, 1, 0, {'activation': 'relu', 'name': 'activation_19'}, 79, 0),
            (84, 5, 1, 0, {'activation': 'relu', 'name': 'activation_21'}, 80, 0),
            (85, 5, 1, 0, {'activation': 'relu', 'name': 'activation_24'}, 81, 0),
            (86, 5, 1, 0, {'activation': 'relu', 'name': 'activation_25'}, 82, 0),
            (87, 4, 1, 0, {'axis': 3, 'name': 'mixed2'}, 83, [84, 85, 86]),
            (88, 1, 3, 0,
             {'filters': 64, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_27'}, 87, 0),
            (89, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_27'}, 88, 0),
            (90, 5, 1, 0, {'activation': 'relu', 'name': 'activation_27'}, 89, 0),
            (91, 1, 3, 0,
             {'filters': 96, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_28'}, 90, 0),
            (92, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_28'}, 91, 0),
            (93, 5, 1, 0, {'activation': 'relu', 'name': 'activation_28'}, 92, 0),
            (94, 1, 3, 0,
             {'filters': 384, 'kernel_size': (3, 3), 'strides': (2, 2), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_26'}, 87, 0),
            (95, 1, 3, 0,
             {'filters': 96, 'kernel_size': (3, 3), 'strides': (2, 2), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_29'}, 93, 0),
            (96, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_26'}, 94, 0),
            (97, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_29'}, 95, 0),
            (98, 5, 1, 0, {'activation': 'relu', 'name': 'activation_26'}, 96, 0),
            (99, 5, 1, 0, {'activation': 'relu', 'name': 'activation_29'}, 97, 0),
            (100, 3, 2, 0, {'pool_size': (3, 3), 'strides': (2, 2), 'padding': 'valid', 'data_format': 'channels_last',
                            'name': 'max_pooling2d_2'}, 87, 0),
            (101, 4, 1, 0, {'axis': 3, 'name': 'mixed3'}, 98, [99, 100]),
            (102, 1, 3, 0,
             {'filters': 128, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_34'}, 101, 0),
            (103, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_34'}, 102, 0),
            (104, 5, 1, 0, {'activation': 'relu', 'name': 'activation_34'}, 103, 0),
            (105, 1, 3, 0,
             {'filters': 128, 'kernel_size': (7, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_35'}, 104, 0),
            (106, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_35'}, 105, 0),
            (107, 5, 1, 0, {'activation': 'relu', 'name': 'activation_35'}, 106, 0),
            (108, 1, 3, 0,
             {'filters': 128, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_31'}, 101, 0),
            (109, 1, 3, 0,
             {'filters': 128, 'kernel_size': (1, 7), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_36'}, 107, 0),
            (110, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_31'}, 108, 0),
            (111, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_36'}, 109, 0),
            (112, 5, 1, 0, {'activation': 'relu', 'name': 'activation_31'}, 110, 0),
            (113, 5, 1, 0, {'activation': 'relu', 'name': 'activation_36'}, 111, 0),
            (114, 1, 3, 0,
             {'filters': 128, 'kernel_size': (1, 7), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_32'}, 112, 0),
            (115, 1, 3, 0,
             {'filters': 128, 'kernel_size': (7, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_37'}, 113, 0),
            (116, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_32'}, 114, 0),
            (117, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_37'}, 115, 0),
            (118, 5, 1, 0, {'activation': 'relu', 'name': 'activation_32'}, 116, 0),
            (119, 5, 1, 0, {'activation': 'relu', 'name': 'activation_37'}, 117, 0),
            (120, 3, 4, 0, {'pool_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last',
                            'name': 'average_pooling2d_3'}, 101, 0),
            (121, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_30'}, 101, 0),
            (122, 1, 3, 0,
             {'filters': 192, 'kernel_size': (7, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_33'}, 118, 0),
            (123, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 7), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_38'}, 119, 0),
            (124, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_39'}, 120, 0),
            (125, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_30'}, 121, 0),
            (126, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_33'}, 122, 0),
            (127, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_38'}, 123, 0),
            (128, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_39'}, 124, 0),
            (129, 5, 1, 0, {'activation': 'relu', 'name': 'activation_30'}, 125, 0),
            (130, 5, 1, 0, {'activation': 'relu', 'name': 'activation_33'}, 126, 0),
            (131, 5, 1, 0, {'activation': 'relu', 'name': 'activation_38'}, 127, 0),
            (132, 5, 1, 0, {'activation': 'relu', 'name': 'activation_39'}, 128, 0),
            (133, 4, 1, 0, {'axis': 3, 'name': 'mixed4'}, 129, [130, 131, 132]),
            (134, 1, 3, 0,
             {'filters': 160, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_44'}, 133, 0),
            (135, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_44'}, 134, 0),
            (136, 5, 1, 0, {'activation': 'relu', 'name': 'activation_44'}, 135, 0),
            (137, 1, 3, 0,
             {'filters': 160, 'kernel_size': (7, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_45'}, 136, 0),
            (138, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_45'}, 137, 0),
            (139, 5, 1, 0, {'activation': 'relu', 'name': 'activation_45'}, 138, 0),
            (140, 1, 3, 0,
             {'filters': 160, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_41'}, 133, 0),
            (141, 1, 3, 0,
             {'filters': 160, 'kernel_size': (1, 7), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_46'}, 139, 0),
            (142, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_41'}, 140, 0),
            (143, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_46'}, 141, 0),
            (144, 5, 1, 0, {'activation': 'relu', 'name': 'activation_41'}, 142, 0),
            (145, 5, 1, 0, {'activation': 'relu', 'name': 'activation_46'}, 143, 0),
            (146, 1, 3, 0,
             {'filters': 160, 'kernel_size': (1, 7), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_42'}, 144, 0),
            (147, 1, 3, 0,
             {'filters': 160, 'kernel_size': (7, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_47'}, 145, 0),
            (148, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_42'}, 146, 0),
            (149, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_47'}, 147, 0),
            (150, 5, 1, 0, {'activation': 'relu', 'name': 'activation_42'}, 148, 0),
            (151, 5, 1, 0, {'activation': 'relu', 'name': 'activation_47'}, 149, 0),
            (152, 3, 4, 0, {'pool_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last',
                            'name': 'average_pooling2d_4'}, 133, 0),
            (153, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_40'}, 133, 0),
            (154, 1, 3, 0,
             {'filters': 192, 'kernel_size': (7, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_43'}, 150, 0),
            (155, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 7), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_48'}, 151, 0),
            (156, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_49'}, 152, 0),
            (157, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_40'}, 153, 0),
            (158, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_43'}, 154, 0),
            (159, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_48'}, 155, 0),
            (160, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_49'}, 156, 0),
            (161, 5, 1, 0, {'activation': 'relu', 'name': 'activation_40'}, 157, 0),
            (162, 5, 1, 0, {'activation': 'relu', 'name': 'activation_43'}, 158, 0),
            (163, 5, 1, 0, {'activation': 'relu', 'name': 'activation_48'}, 159, 0),
            (164, 5, 1, 0, {'activation': 'relu', 'name': 'activation_49'}, 160, 0),
            (165, 4, 1, 0, {'axis': 3, 'name': 'mixed5'}, 161, [162, 163, 164]),
            (166, 1, 3, 0,
             {'filters': 160, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_54'}, 165, 0),
            (167, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_54'}, 166, 0),
            (168, 5, 1, 0, {'activation': 'relu', 'name': 'activation_54'}, 167, 0),
            (169, 1, 3, 0,
             {'filters': 160, 'kernel_size': (7, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_55'}, 168, 0),
            (170, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_55'}, 169, 0),
            (171, 5, 1, 0, {'activation': 'relu', 'name': 'activation_55'}, 170, 0),
            (172, 1, 3, 0,
             {'filters': 160, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_51'}, 165, 0),
            (173, 1, 3, 0,
             {'filters': 160, 'kernel_size': (1, 7), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_56'}, 171, 0),
            (174, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_51'}, 172, 0),
            (175, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_56'}, 173, 0),
            (176, 5, 1, 0, {'activation': 'relu', 'name': 'activation_51'}, 174, 0),
            (177, 5, 1, 0, {'activation': 'relu', 'name': 'activation_56'}, 175, 0),
            (178, 1, 3, 0,
             {'filters': 160, 'kernel_size': (1, 7), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_52'}, 176, 0),
            (179, 1, 3, 0,
             {'filters': 160, 'kernel_size': (7, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_57'}, 177, 0),
            (180, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_52'}, 178, 0),
            (181, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_57'}, 179, 0),
            (182, 5, 1, 0, {'activation': 'relu', 'name': 'activation_52'}, 180, 0),
            (183, 5, 1, 0, {'activation': 'relu', 'name': 'activation_57'}, 181, 0),
            (184, 3, 4, 0, {'pool_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last',
                            'name': 'average_pooling2d_5'}, 165, 0),
            (185, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_50'}, 165, 0),
            (186, 1, 3, 0,
             {'filters': 192, 'kernel_size': (7, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_53'}, 182, 0),
            (187, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 7), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_58'}, 183, 0),
            (188, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_59'}, 184, 0),
            (189, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_50'}, 185, 0),
            (190, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_53'}, 186, 0),
            (191, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_58'}, 187, 0),
            (192, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_59'}, 188, 0),
            (193, 5, 1, 0, {'activation': 'relu', 'name': 'activation_50'}, 189, 0),
            (194, 5, 1, 0, {'activation': 'relu', 'name': 'activation_53'}, 190, 0),
            (195, 5, 1, 0, {'activation': 'relu', 'name': 'activation_58'}, 191, 0),
            (196, 5, 1, 0, {'activation': 'relu', 'name': 'activation_59'}, 192, 0),
            (197, 4, 1, 0, {'axis': 3, 'name': 'mixed6'}, 193, [194, 195, 196]),
            (198, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_64'}, 197, 0),
            (199, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_64'}, 198, 0),
            (200, 5, 1, 0, {'activation': 'relu', 'name': 'activation_64'}, 199, 0),
            (201, 1, 3, 0,
             {'filters': 192, 'kernel_size': (7, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_65'}, 200, 0),
            (202, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_65'}, 201, 0),
            (203, 5, 1, 0, {'activation': 'relu', 'name': 'activation_65'}, 202, 0),
            (204, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_61'}, 197, 0),
            (205, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 7), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_66'}, 203, 0),
            (206, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_61'}, 204, 0),
            (207, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_66'}, 205, 0),
            (208, 5, 1, 0, {'activation': 'relu', 'name': 'activation_61'}, 206, 0),
            (209, 5, 1, 0, {'activation': 'relu', 'name': 'activation_66'}, 207, 0),
            (210, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 7), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_62'}, 208, 0),
            (211, 1, 3, 0,
             {'filters': 192, 'kernel_size': (7, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_67'}, 209, 0),
            (212, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_62'}, 210, 0),
            (213, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_67'}, 211, 0),
            (214, 5, 1, 0, {'activation': 'relu', 'name': 'activation_62'}, 212, 0),
            (215, 5, 1, 0, {'activation': 'relu', 'name': 'activation_67'}, 213, 0),
            (216, 3, 4, 0, {'pool_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last',
                            'name': 'average_pooling2d_6'}, 197, 0),
            (217, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_60'}, 197, 0),
            (218, 1, 3, 0,
             {'filters': 192, 'kernel_size': (7, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_63'}, 214, 0),
            (219, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 7), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_68'}, 215, 0),
            (220, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_69'}, 216, 0),
            (221, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_60'}, 217, 0),
            (222, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_63'}, 218, 0),
            (223, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_68'}, 219, 0),
            (224, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_69'}, 220, 0),
            (225, 5, 1, 0, {'activation': 'relu', 'name': 'activation_60'}, 221, 0),
            (226, 5, 1, 0, {'activation': 'relu', 'name': 'activation_63'}, 222, 0),
            (227, 5, 1, 0, {'activation': 'relu', 'name': 'activation_68'}, 223, 0),
            (228, 5, 1, 0, {'activation': 'relu', 'name': 'activation_69'}, 224, 0),
            (229, 4, 1, 0, {'axis': 3, 'name': 'mixed7'}, 225, [226, 227, 228]),
            (230, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_72'}, 229, 0),
            (231, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_72'}, 230, 0),
            (232, 5, 1, 0, {'activation': 'relu', 'name': 'activation_72'}, 231, 0),
            (233, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 7), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_73'}, 232, 0),
            (234, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_73'}, 233, 0),
            (235, 5, 1, 0, {'activation': 'relu', 'name': 'activation_73'}, 234, 0),
            (236, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_70'}, 229, 0),
            (237, 1, 3, 0,
             {'filters': 192, 'kernel_size': (7, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_74'}, 235, 0),
            (238, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_70'}, 236, 0),
            (239, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_74'}, 237, 0),
            (240, 5, 1, 0, {'activation': 'relu', 'name': 'activation_70'}, 238, 0),
            (241, 5, 1, 0, {'activation': 'relu', 'name': 'activation_74'}, 239, 0),
            (242, 1, 3, 0,
             {'filters': 320, 'kernel_size': (3, 3), 'strides': (2, 2), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_71'}, 240, 0),
            (243, 1, 3, 0,
             {'filters': 192, 'kernel_size': (3, 3), 'strides': (2, 2), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_75'}, 241, 0),
            (244, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_71'}, 242, 0),
            (245, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_75'}, 243, 0),
            (246, 5, 1, 0, {'activation': 'relu', 'name': 'activation_71'}, 244, 0),
            (247, 5, 1, 0, {'activation': 'relu', 'name': 'activation_75'}, 245, 0),
            (248, 3, 2, 0, {'pool_size': (3, 3), 'strides': (2, 2), 'padding': 'valid', 'data_format': 'channels_last',
                            'name': 'max_pooling2d_3'}, 229, 0),
            (249, 4, 1, 0, {'axis': 3, 'name': 'mixed8'}, 246, [247, 248]),
            (250, 1, 3, 0,
             {'filters': 448, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_80'}, 249, 0),
            (251, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_80'}, 250, 0),
            (252, 5, 1, 0, {'activation': 'relu', 'name': 'activation_80'}, 251, 0),
            (253, 1, 3, 0,
             {'filters': 384, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_77'}, 249, 0),
            (254, 1, 3, 0,
             {'filters': 384, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_81'}, 252, 0),
            (255, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_77'}, 253, 0),
            (256, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_81'}, 254, 0),
            (257, 5, 1, 0, {'activation': 'relu', 'name': 'activation_77'}, 255, 0),
            (258, 5, 1, 0, {'activation': 'relu', 'name': 'activation_81'}, 256, 0),
            (259, 1, 3, 0,
             {'filters': 384, 'kernel_size': (1, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_78'}, 257, 0),
            (260, 1, 3, 0,
             {'filters': 384, 'kernel_size': (3, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_79'}, 257, 0),
            (261, 1, 3, 0,
             {'filters': 384, 'kernel_size': (1, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_82'}, 258, 0),
            (262, 1, 3, 0,
             {'filters': 384, 'kernel_size': (3, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_83'}, 258, 0),
            (263, 3, 4, 0, {'pool_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last',
                            'name': 'average_pooling2d_7'}, 249, 0),
            (264, 1, 3, 0,
             {'filters': 320, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_76'}, 249, 0),
            (265, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_78'}, 259, 0),
            (266, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_79'}, 260, 0),
            (267, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_82'}, 261, 0),
            (268, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_83'}, 262, 0),
            (269, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_84'}, 263, 0),
            (270, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_76'}, 264, 0),
            (271, 5, 1, 0, {'activation': 'relu', 'name': 'activation_78'}, 265, 0),
            (272, 5, 1, 0, {'activation': 'relu', 'name': 'activation_79'}, 266, 0),
            (273, 5, 1, 0, {'activation': 'relu', 'name': 'activation_82'}, 267, 0),
            (274, 5, 1, 0, {'activation': 'relu', 'name': 'activation_83'}, 268, 0),
            (275, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_84'}, 269, 0),
            (276, 5, 1, 0, {'activation': 'relu', 'name': 'activation_76'}, 270, 0),
            (277, 4, 1, 0, {'axis': 3, 'name': 'mixed9_0'}, 271, [272]),
            (278, 4, 1, 0, {'axis': 3, 'name': 'concatenate'}, 273, [274]),
            (279, 5, 1, 0, {'activation': 'relu', 'name': 'activation_84'}, 275, 0),
            (280, 4, 1, 0, {'axis': 3, 'name': 'mixed9'}, 276, [277, 278, 279]),
            (281, 1, 3, 0,
             {'filters': 448, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_89'}, 280, 0),
            (282, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_89'}, 281, 0),
            (283, 5, 1, 0, {'activation': 'relu', 'name': 'activation_89'}, 282, 0),
            (284, 1, 3, 0,
             {'filters': 384, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_86'}, 280, 0),
            (285, 1, 3, 0,
             {'filters': 384, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_90'}, 283, 0),
            (286, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_86'}, 284, 0),
            (287, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_90'}, 285, 0),
            (288, 5, 1, 0, {'activation': 'relu', 'name': 'activation_86'}, 286, 0),
            (289, 5, 1, 0, {'activation': 'relu', 'name': 'activation_90'}, 287, 0),
            (290, 1, 3, 0,
             {'filters': 384, 'kernel_size': (1, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_87'}, 288, 0),
            (291, 1, 3, 0,
             {'filters': 384, 'kernel_size': (3, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_88'}, 288, 0),
            (292, 1, 3, 0,
             {'filters': 384, 'kernel_size': (1, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_91'}, 289, 0),
            (293, 1, 3, 0,
             {'filters': 384, 'kernel_size': (3, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_92'}, 289, 0),
            (294, 3, 4, 0, {'pool_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'data_format': 'channels_last',
                            'name': 'average_pooling2d_8'}, 280, 0),
            (295, 1, 3, 0,
             {'filters': 320, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_85'}, 280, 0),
            (296, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_87'}, 290, 0),
            (297, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_88'}, 291, 0),
            (298, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_91'}, 292, 0),
            (299, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_92'}, 293, 0),
            (300, 1, 3, 0,
             {'filters': 192, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_93'}, 294, 0),
            (301, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_85'}, 295, 0),
            (302, 5, 1, 0, {'activation': 'relu', 'name': 'activation_87'}, 296, 0),
            (303, 5, 1, 0, {'activation': 'relu', 'name': 'activation_88'}, 297, 0),
            (304, 5, 1, 0, {'activation': 'relu', 'name': 'activation_91'}, 298, 0),
            (305, 5, 1, 0, {'activation': 'relu', 'name': 'activation_92'}, 299, 0),
            (306, 6, 2, 0,
             {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': False,
              'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
              'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
              'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_93'}, 300, 0),
            (307, 5, 1, 0, {'activation': 'relu', 'name': 'activation_85'}, 301, 0),
            (308, 4, 1, 0, {'axis': 3, 'name': 'mixed9_1'}, 302, [303]),
            (309, 4, 1, 0, {'axis': 3, 'name': 'concatenate_1'}, 304, [305]),
            (310, 5, 1, 0, {'activation': 'relu', 'name': 'activation_93'}, 306, 0),
            (311, 4, 1, 0, {'axis': 3, 'name': 'mixed10'}, 307, [308, 309, 310]),
            (312, 8, 6, 0, {'data_format': 'channels_last', 'name': 'avg_pool'}, 311, 0),
            (313, 1, 1, 0,
             {'units': 1000, 'activation': 'softmax', 'use_bias': True, 'kernel_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'kernel_regularizer': None, 'bias_regularizer': None,
              'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'name': 'predictions'},
             312, 0),
        ],
        #  16: 'MobileNet',
        #  17: 'MobileNetV2',
        #  18: 'NASNetLarge',
        #  19: 'NASNetMobile',
        #  20: 'ResNet101',
        #  21: 'ResNet152',
        'ResNet50': [
            (1, 9, 1, 0, {}, 0, 0),
            (2, 2, 8, 0, {'padding': ((3, 3), (3, 3)), 'data_format': 'channels_last', 'name': 'conv1_pad'}, 1, 0),
            (3, 1, 3, 0,
             {'filters': 64, 'kernel_size': (7, 7), 'strides': (2, 2), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv1_conv'}, 2, 0),
            (4, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                          'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                          'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                          'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv1_bn'}, 3, 0),
            (5, 5, 1, 0, {'activation': 'relu', 'name': 'conv1_relu'}, 4, 0),
            (6, 2, 8, 0, {'padding': ((1, 1), (1, 1)), 'data_format': 'channels_last', 'name': 'pool1_pad'}, 5, 0),
            (7, 3, 2, 0,
             {'pool_size': (3, 3), 'strides': (2, 2), 'padding': 'valid', 'data_format': 'channels_last',
              'name': 'pool1_pool'},
             6, 0),
            (8, 1, 3, 0,
             {'filters': 64, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2_block1_1_conv'}, 7, 0),
            (9, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                          'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                          'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                          'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv2_block1_1_bn'}, 8, 0),
            (10, 5, 1, 0, {'activation': 'relu', 'name': 'conv2_block1_1_relu'}, 9, 0),
            (11, 1, 3, 0,
             {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2_block1_2_conv'}, 10, 0),
            (12, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv2_block1_2_bn'}, 11, 0),
            (13, 5, 1, 0, {'activation': 'relu', 'name': 'conv2_block1_2_relu'}, 12, 0),
            (14, 1, 3, 0,
             {'filters': 256, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2_block1_0_conv'}, 7, 0),
            (15, 1, 3, 0,
             {'filters': 256, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2_block1_3_conv'}, 13, 0),
            (16, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv2_block1_0_bn'}, 14, 0),
            (17, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv2_block1_3_bn'}, 15, 0),
            (18, 4, 2, 0, {'name': 'conv2_block1_add'}, 16, [17]),
            (19, 5, 1, 0, {'activation': 'relu', 'name': 'conv2_block1_out'}, 18, 0),
            (20, 1, 3, 0,
             {'filters': 64, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2_block2_1_conv'}, 19, 0),
            (21, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv2_block2_1_bn'}, 20, 0),
            (22, 5, 1, 0, {'activation': 'relu', 'name': 'conv2_block2_1_relu'}, 21, 0),
            (23, 1, 3, 0,
             {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2_block2_2_conv'}, 22, 0),
            (24, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv2_block2_2_bn'}, 23, 0),
            (25, 5, 1, 0, {'activation': 'relu', 'name': 'conv2_block2_2_relu'}, 24, 0),
            (26, 1, 3, 0,
             {'filters': 256, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2_block2_3_conv'}, 25, 0),
            (27, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv2_block2_3_bn'}, 26, 0),
            (28, 4, 2, 0, {'name': 'conv2_block2_add'}, 19, [27]),
            (29, 5, 1, 0, {'activation': 'relu', 'name': 'conv2_block2_out'}, 28, 0),
            (30, 1, 3, 0,
             {'filters': 64, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2_block3_1_conv'}, 29, 0),
            (31, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv2_block3_1_bn'}, 30, 0),
            (32, 5, 1, 0, {'activation': 'relu', 'name': 'conv2_block3_1_relu'}, 31, 0),
            (33, 1, 3, 0,
             {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2_block3_2_conv'}, 32, 0),
            (34, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv2_block3_2_bn'}, 33, 0),
            (35, 5, 1, 0, {'activation': 'relu', 'name': 'conv2_block3_2_relu'}, 34, 0),
            (36, 1, 3, 0,
             {'filters': 256, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2_block3_3_conv'}, 35, 0),
            (37, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv2_block3_3_bn'}, 36, 0),
            (38, 4, 2, 0, {'name': 'conv2_block3_add'}, 29, [37]),
            (39, 5, 1, 0, {'activation': 'relu', 'name': 'conv2_block3_out'}, 38, 0),
            (40, 1, 3, 0,
             {'filters': 128, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv3_block1_1_conv'}, 39, 0),
            (41, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv3_block1_1_bn'}, 40, 0),
            (42, 5, 1, 0, {'activation': 'relu', 'name': 'conv3_block1_1_relu'}, 41, 0),
            (43, 1, 3, 0,
             {'filters': 128, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv3_block1_2_conv'}, 42, 0),
            (44, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv3_block1_2_bn'}, 43, 0),
            (45, 5, 1, 0, {'activation': 'relu', 'name': 'conv3_block1_2_relu'}, 44, 0),
            (46, 1, 3, 0,
             {'filters': 512, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv3_block1_0_conv'}, 39, 0),
            (47, 1, 3, 0,
             {'filters': 512, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv3_block1_3_conv'}, 45, 0),
            (48, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv3_block1_0_bn'}, 46, 0),
            (49, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv3_block1_3_bn'}, 47, 0),
            (50, 4, 2, 0, {'name': 'conv3_block1_add'}, 48, [49]),
            (51, 5, 1, 0, {'activation': 'relu', 'name': 'conv3_block1_out'}, 50, 0),
            (52, 1, 3, 0,
             {'filters': 128, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv3_block2_1_conv'}, 51, 0),
            (53, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv3_block2_1_bn'}, 52, 0),
            (54, 5, 1, 0, {'activation': 'relu', 'name': 'conv3_block2_1_relu'}, 53, 0),
            (55, 1, 3, 0,
             {'filters': 128, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv3_block2_2_conv'}, 54, 0),
            (56, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv3_block2_2_bn'}, 55, 0),
            (57, 5, 1, 0, {'activation': 'relu', 'name': 'conv3_block2_2_relu'}, 56, 0),
            (58, 1, 3, 0,
             {'filters': 512, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv3_block2_3_conv'}, 57, 0),
            (59, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv3_block2_3_bn'}, 58, 0),
            (60, 4, 2, 0, {'name': 'conv3_block2_add'}, 51, [59]),
            (61, 5, 1, 0, {'activation': 'relu', 'name': 'conv3_block2_out'}, 60, 0),
            (62, 1, 3, 0,
             {'filters': 128, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv3_block3_1_conv'}, 61, 0),
            (63, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv3_block3_1_bn'}, 62, 0),
            (64, 5, 1, 0, {'activation': 'relu', 'name': 'conv3_block3_1_relu'}, 63, 0),
            (65, 1, 3, 0,
             {'filters': 128, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv3_block3_2_conv'}, 64, 0),
            (66, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv3_block3_2_bn'}, 65, 0),
            (67, 5, 1, 0, {'activation': 'relu', 'name': 'conv3_block3_2_relu'}, 66, 0),
            (68, 1, 3, 0,
             {'filters': 512, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv3_block3_3_conv'}, 67, 0),
            (69, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv3_block3_3_bn'}, 68, 0),
            (70, 4, 2, 0, {'name': 'conv3_block3_add'}, 61, [69]),
            (71, 5, 1, 0, {'activation': 'relu', 'name': 'conv3_block3_out'}, 70, 0),
            (72, 1, 3, 0,
             {'filters': 128, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv3_block4_1_conv'}, 71, 0),
            (73, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv3_block4_1_bn'}, 72, 0),
            (74, 5, 1, 0, {'activation': 'relu', 'name': 'conv3_block4_1_relu'}, 73, 0),
            (75, 1, 3, 0,
             {'filters': 128, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv3_block4_2_conv'}, 74, 0),
            (76, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv3_block4_2_bn'}, 75, 0),
            (77, 5, 1, 0, {'activation': 'relu', 'name': 'conv3_block4_2_relu'}, 76, 0),
            (78, 1, 3, 0,
             {'filters': 512, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv3_block4_3_conv'}, 77, 0),
            (79, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv3_block4_3_bn'}, 78, 0),
            (80, 4, 2, 0, {'name': 'conv3_block4_add'}, 71, [79]),
            (81, 5, 1, 0, {'activation': 'relu', 'name': 'conv3_block4_out'}, 80, 0),
            (82, 1, 3, 0,
             {'filters': 256, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv4_block1_1_conv'}, 81, 0),
            (83, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv4_block1_1_bn'}, 82, 0),
            (84, 5, 1, 0, {'activation': 'relu', 'name': 'conv4_block1_1_relu'}, 83, 0),
            (85, 1, 3, 0,
             {'filters': 256, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv4_block1_2_conv'}, 84, 0),
            (86, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv4_block1_2_bn'}, 85, 0),
            (87, 5, 1, 0, {'activation': 'relu', 'name': 'conv4_block1_2_relu'}, 86, 0),
            (88, 1, 3, 0,
             {'filters': 1024, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None,
              'name': 'conv4_block1_0_conv'}, 81, 0),
            (89, 1, 3, 0,
             {'filters': 1024, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None,
              'name': 'conv4_block1_3_conv'}, 87, 0),
            (90, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv4_block1_0_bn'}, 88, 0),
            (91, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv4_block1_3_bn'}, 89, 0),
            (92, 4, 2, 0, {'name': 'conv4_block1_add'}, 90, [91]),
            (93, 5, 1, 0, {'activation': 'relu', 'name': 'conv4_block1_out'}, 92, 0),
            (94, 1, 3, 0,
             {'filters': 256, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv4_block2_1_conv'}, 93, 0),
            (95, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv4_block2_1_bn'}, 94, 0),
            (96, 5, 1, 0, {'activation': 'relu', 'name': 'conv4_block2_1_relu'}, 95, 0),
            (97, 1, 3, 0,
             {'filters': 256, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv4_block2_2_conv'}, 96, 0),
            (98, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv4_block2_2_bn'}, 97, 0),
            (99, 5, 1, 0, {'activation': 'relu', 'name': 'conv4_block2_2_relu'}, 98, 0),
            (100, 1, 3, 0,
             {'filters': 1024, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None,
              'name': 'conv4_block2_3_conv'}, 99, 0),
            (101, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv4_block2_3_bn'}, 100, 0),
            (102, 4, 2, 0, {'name': 'conv4_block2_add'}, 93, [101]),
            (103, 5, 1, 0, {'activation': 'relu', 'name': 'conv4_block2_out'}, 102, 0),
            (104, 1, 3, 0,
             {'filters': 256, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None,
              'name': 'conv4_block3_1_conv'}, 103, 0),
            (105, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv4_block3_1_bn'}, 104, 0),
            (106, 5, 1, 0, {'activation': 'relu', 'name': 'conv4_block3_1_relu'}, 105, 0),
            (107, 1, 3, 0,
             {'filters': 256, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv4_block3_2_conv'}, 106, 0),
            (108, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv4_block3_2_bn'}, 107, 0),
            (109, 5, 1, 0, {'activation': 'relu', 'name': 'conv4_block3_2_relu'}, 108, 0),
            (110, 1, 3, 0,
             {'filters': 1024, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None,
              'name': 'conv4_block3_3_conv'}, 109, 0),
            (111, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv4_block3_3_bn'}, 110, 0),
            (112, 4, 2, 0, {'name': 'conv4_block3_add'}, 103, [111]),
            (113, 5, 1, 0, {'activation': 'relu', 'name': 'conv4_block3_out'}, 112, 0),
            (114, 1, 3, 0,
             {'filters': 256, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None,
              'name': 'conv4_block4_1_conv'}, 113, 0),
            (115, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv4_block4_1_bn'}, 114, 0),
            (116, 5, 1, 0, {'activation': 'relu', 'name': 'conv4_block4_1_relu'}, 115, 0),
            (117, 1, 3, 0,
             {'filters': 256, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv4_block4_2_conv'}, 116, 0),
            (118, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv4_block4_2_bn'}, 117, 0),
            (119, 5, 1, 0, {'activation': 'relu', 'name': 'conv4_block4_2_relu'}, 118, 0),
            (120, 1, 3, 0,
             {'filters': 1024, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None,
              'name': 'conv4_block4_3_conv'}, 119, 0),
            (121, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv4_block4_3_bn'}, 120, 0),
            (122, 4, 2, 0, {'name': 'conv4_block4_add'}, 113, [121]),
            (123, 5, 1, 0, {'activation': 'relu', 'name': 'conv4_block4_out'}, 122, 0),
            (124, 1, 3, 0,
             {'filters': 256, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None,
              'name': 'conv4_block5_1_conv'}, 123, 0),
            (125, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv4_block5_1_bn'}, 124, 0),
            (126, 5, 1, 0, {'activation': 'relu', 'name': 'conv4_block5_1_relu'}, 125, 0),
            (127, 1, 3, 0,
             {'filters': 256, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv4_block5_2_conv'}, 126, 0),
            (128, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv4_block5_2_bn'}, 127, 0),
            (129, 5, 1, 0, {'activation': 'relu', 'name': 'conv4_block5_2_relu'}, 128, 0),
            (130, 1, 3, 0,
             {'filters': 1024, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None,
              'name': 'conv4_block5_3_conv'}, 129, 0),
            (131, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv4_block5_3_bn'}, 130, 0),
            (132, 4, 2, 0, {'name': 'conv4_block5_add'}, 123, [131]),
            (133, 5, 1, 0, {'activation': 'relu', 'name': 'conv4_block5_out'}, 132, 0),
            (134, 1, 3, 0,
             {'filters': 256, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None,
              'name': 'conv4_block6_1_conv'}, 133, 0),
            (135, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv4_block6_1_bn'}, 134, 0),
            (136, 5, 1, 0, {'activation': 'relu', 'name': 'conv4_block6_1_relu'}, 135, 0),
            (137, 1, 3, 0,
             {'filters': 256, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv4_block6_2_conv'}, 136, 0),
            (138, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv4_block6_2_bn'}, 137, 0),
            (139, 5, 1, 0, {'activation': 'relu', 'name': 'conv4_block6_2_relu'}, 138, 0),
            (140, 1, 3, 0,
             {'filters': 1024, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None,
              'name': 'conv4_block6_3_conv'}, 139, 0),
            (141, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv4_block6_3_bn'}, 140, 0),
            (142, 4, 2, 0, {'name': 'conv4_block6_add'}, 133, [141]),
            (143, 5, 1, 0, {'activation': 'relu', 'name': 'conv4_block6_out'}, 142, 0),
            (144, 1, 3, 0,
             {'filters': 512, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None,
              'name': 'conv5_block1_1_conv'}, 143, 0),
            (145, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv5_block1_1_bn'}, 144, 0),
            (146, 5, 1, 0, {'activation': 'relu', 'name': 'conv5_block1_1_relu'}, 145, 0),
            (147, 1, 3, 0,
             {'filters': 512, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv5_block1_2_conv'}, 146, 0),
            (148, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv5_block1_2_bn'}, 147, 0),
            (149, 5, 1, 0, {'activation': 'relu', 'name': 'conv5_block1_2_relu'}, 148, 0),
            (150, 1, 3, 0,
             {'filters': 2048, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None,
              'name': 'conv5_block1_0_conv'}, 143, 0),
            (151, 1, 3, 0,
             {'filters': 2048, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None,
              'name': 'conv5_block1_3_conv'}, 149, 0),
            (152, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv5_block1_0_bn'}, 150, 0),
            (153, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv5_block1_3_bn'}, 151, 0),
            (154, 4, 2, 0, {'name': 'conv5_block1_add'}, 152, [153]),
            (155, 5, 1, 0, {'activation': 'relu', 'name': 'conv5_block1_out'}, 154, 0),
            (156, 1, 3, 0,
             {'filters': 512, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None,
              'name': 'conv5_block2_1_conv'}, 155, 0),
            (157, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv5_block2_1_bn'}, 156, 0),
            (158, 5, 1, 0, {'activation': 'relu', 'name': 'conv5_block2_1_relu'}, 157, 0),
            (159, 1, 3, 0,
             {'filters': 512, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv5_block2_2_conv'}, 158, 0),
            (160, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv5_block2_2_bn'}, 159, 0),
            (161, 5, 1, 0, {'activation': 'relu', 'name': 'conv5_block2_2_relu'}, 160, 0),
            (162, 1, 3, 0,
             {'filters': 2048, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None,
              'name': 'conv5_block2_3_conv'}, 161, 0),
            (163, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv5_block2_3_bn'}, 162, 0),
            (164, 4, 2, 0, {'name': 'conv5_block2_add'}, 155, [163]),
            (165, 5, 1, 0, {'activation': 'relu', 'name': 'conv5_block2_out'}, 164, 0),
            (166, 1, 3, 0,
             {'filters': 512, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None,
              'name': 'conv5_block3_1_conv'}, 165, 0),
            (167, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv5_block3_1_bn'}, 166, 0),
            (168, 5, 1, 0, {'activation': 'relu', 'name': 'conv5_block3_1_relu'}, 167, 0),
            (169, 1, 3, 0,
             {'filters': 512, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv5_block3_2_conv'}, 168, 0),
            (170, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv5_block3_2_bn'}, 169, 0),
            (171, 5, 1, 0, {'activation': 'relu', 'name': 'conv5_block3_2_relu'}, 170, 0),
            (172, 1, 3, 0,
             {'filters': 2048, 'kernel_size': (1, 1), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None,
              'name': 'conv5_block3_3_conv'}, 171, 0),
            (173, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 1.001e-05, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros',
                            'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                            'beta_constraint': None, 'gamma_constraint': None, 'name': 'conv5_block3_3_bn'}, 172, 0),
            (174, 4, 2, 0, {'name': 'conv5_block3_add'}, 165, [173]),
            (175, 5, 1, 0, {'activation': 'relu', 'name': 'conv5_block3_out'}, 174, 0),
            (176, 8, 6, 0, {'data_format': 'channels_last', 'name': 'avg_pool'}, 175, 0),
            (177, 1, 1, 0,
             {'units': 1000, 'activation': 'softmax', 'use_bias': True, 'kernel_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'kernel_regularizer': None, 'bias_regularizer': None,
              'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None,
              'name': 'predictions'}, 176, 0),
        ],
        #  23: 'ResNet101V2',
        #  24: 'ResNet152V2',
        #  25: 'ResNet50V2',
        'VGG16': [
            [1, 9, 1, 0, {}, 0, 0],
            [2, 1, 3, 0,
             {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': True,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'block1_conv1'}, 1, 0],
            [3, 1, 3, 0,
             {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'relu',
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
            [23, 1, 1, 0,
             {'units': 1000, 'activation': 'softmax', 'use_bias': True, 'kernel_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'kernel_regularizer': None, 'bias_regularizer': None,
              'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None,
              'name': 'predictions'}, 22, 0]
        ],
        #  27: 'VGG19',
        'Xception': [
            (1, 9, 1, 0, {}, 0, 0),
            (2, 1, 3, 0,
             {'filters': 32, 'kernel_size': (3, 3), 'strides': (2, 2), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'block1_conv1'}, 1, 0),
            (3, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                          'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                          'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                          'beta_constraint': None, 'gamma_constraint': None, 'name': 'block1_conv1_bn'}, 2, 0),
            (4, 5, 1, 0, {'activation': 'relu', 'name': 'block1_conv1_act'}, 3, 0),
            (5, 1, 3, 0,
             {'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'valid', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'block1_conv2'}, 4, 0),
            (6, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                          'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                          'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                          'beta_constraint': None, 'gamma_constraint': None, 'name': 'block1_conv2_bn'}, 5, 0),
            (7, 5, 1, 0, {'activation': 'relu', 'name': 'block1_conv2_act'}, 6, 0),
            (8, 1, 6, 0,
             {'filters': 128, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block2_sepconv1'}, 7, 0),
            (9, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                          'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                          'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                          'beta_constraint': None, 'gamma_constraint': None, 'name': 'block2_sepconv1_bn'}, 8, 0),
            (10, 5, 1, 0, {'activation': 'relu', 'name': 'block2_sepconv2_act'}, 9, 0),
            (11, 1, 6, 0,
             {'filters': 128, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block2_sepconv2'}, 10, 0),
            (12, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block2_sepconv2_bn'}, 11, 0),
            (13, 1, 3, 0,
             {'filters': 128, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d'}, 7, 0),
            (14, 3, 2, 0, {'pool_size': (3, 3), 'strides': (2, 2), 'padding': 'same', 'data_format': 'channels_last',
                           'name': 'block2_pool'}, 12, 0),
            (15, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization'}, 13, 0),
            (16, 4, 2, 0, {'name': 'add'}, 14, [15]),
            (17, 5, 1, 0, {'activation': 'relu', 'name': 'block3_sepconv1_act'}, 16, 0),
            (18, 1, 6, 0,
             {'filters': 256, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block3_sepconv1'}, 17, 0),
            (19, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block3_sepconv1_bn'}, 18, 0),
            (20, 5, 1, 0, {'activation': 'relu', 'name': 'block3_sepconv2_act'}, 19, 0),
            (21, 1, 6, 0,
             {'filters': 256, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block3_sepconv2'}, 20, 0),
            (22, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block3_sepconv2_bn'}, 21, 0),
            (23, 1, 3, 0,
             {'filters': 256, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_1'}, 16, 0),
            (24, 3, 2, 0, {'pool_size': (3, 3), 'strides': (2, 2), 'padding': 'same', 'data_format': 'channels_last',
                           'name': 'block3_pool'}, 22, 0),
            (25, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_1'}, 23, 0),
            (26, 4, 2, 0, {'name': 'add_1'}, 24, [25]),
            (27, 5, 1, 0, {'activation': 'relu', 'name': 'block4_sepconv1_act'}, 26, 0),
            (28, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block4_sepconv1'}, 27, 0),
            (29, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block4_sepconv1_bn'}, 28, 0),
            (30, 5, 1, 0, {'activation': 'relu', 'name': 'block4_sepconv2_act'}, 29, 0),
            (31, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block4_sepconv2'}, 30, 0),
            (32, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block4_sepconv2_bn'}, 31, 0),
            (33, 1, 3, 0,
             {'filters': 728, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_2'}, 26, 0),
            (34, 3, 2, 0, {'pool_size': (3, 3), 'strides': (2, 2), 'padding': 'same', 'data_format': 'channels_last',
                           'name': 'block4_pool'}, 32, 0),
            (35, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'batch_normalization_2'}, 33, 0),
            (36, 4, 2, 0, {'name': 'add_2'}, 34, [35]),
            (37, 5, 1, 0, {'activation': 'relu', 'name': 'block5_sepconv1_act'}, 36, 0),
            (38, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block5_sepconv1'}, 37, 0),
            (39, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block5_sepconv1_bn'}, 38, 0),
            (40, 5, 1, 0, {'activation': 'relu', 'name': 'block5_sepconv2_act'}, 39, 0),
            (41, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block5_sepconv2'}, 40, 0),
            (42, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block5_sepconv2_bn'}, 41, 0),
            (43, 5, 1, 0, {'activation': 'relu', 'name': 'block5_sepconv3_act'}, 42, 0),
            (44, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block5_sepconv3'}, 43, 0),
            (45, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block5_sepconv3_bn'}, 44, 0),
            (46, 4, 2, 0, {'name': 'add_3'}, 45, [36]),
            (47, 5, 1, 0, {'activation': 'relu', 'name': 'block6_sepconv1_act'}, 46, 0),
            (48, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block6_sepconv1'}, 47, 0),
            (49, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block6_sepconv1_bn'}, 48, 0),
            (50, 5, 1, 0, {'activation': 'relu', 'name': 'block6_sepconv2_act'}, 49, 0),
            (51, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block6_sepconv2'}, 50, 0),
            (52, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block6_sepconv2_bn'}, 51, 0),
            (53, 5, 1, 0, {'activation': 'relu', 'name': 'block6_sepconv3_act'}, 52, 0),
            (54, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block6_sepconv3'}, 53, 0),
            (55, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block6_sepconv3_bn'}, 54, 0),
            (56, 4, 2, 0, {'name': 'add_4'}, 55, [46]),
            (57, 5, 1, 0, {'activation': 'relu', 'name': 'block7_sepconv1_act'}, 56, 0),
            (58, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block7_sepconv1'}, 57, 0),
            (59, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block7_sepconv1_bn'}, 58, 0),
            (60, 5, 1, 0, {'activation': 'relu', 'name': 'block7_sepconv2_act'}, 59, 0),
            (61, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block7_sepconv2'}, 60, 0),
            (62, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block7_sepconv2_bn'}, 61, 0),
            (63, 5, 1, 0, {'activation': 'relu', 'name': 'block7_sepconv3_act'}, 62, 0),
            (64, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block7_sepconv3'}, 63, 0),
            (65, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block7_sepconv3_bn'}, 64, 0),
            (66, 4, 2, 0, {'name': 'add_5'}, 65, [56]),
            (67, 5, 1, 0, {'activation': 'relu', 'name': 'block8_sepconv1_act'}, 66, 0),
            (68, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block8_sepconv1'}, 67, 0),
            (69, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block8_sepconv1_bn'}, 68, 0),
            (70, 5, 1, 0, {'activation': 'relu', 'name': 'block8_sepconv2_act'}, 69, 0),
            (71, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block8_sepconv2'}, 70, 0),
            (72, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block8_sepconv2_bn'}, 71, 0),
            (73, 5, 1, 0, {'activation': 'relu', 'name': 'block8_sepconv3_act'}, 72, 0),
            (74, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block8_sepconv3'}, 73, 0),
            (75, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block8_sepconv3_bn'}, 74, 0),
            (76, 4, 2, 0, {'name': 'add_6'}, 75, [66]),
            (77, 5, 1, 0, {'activation': 'relu', 'name': 'block9_sepconv1_act'}, 76, 0),
            (78, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block9_sepconv1'}, 77, 0),
            (79, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block9_sepconv1_bn'}, 78, 0),
            (80, 5, 1, 0, {'activation': 'relu', 'name': 'block9_sepconv2_act'}, 79, 0),
            (81, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block9_sepconv2'}, 80, 0),
            (82, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block9_sepconv2_bn'}, 81, 0),
            (83, 5, 1, 0, {'activation': 'relu', 'name': 'block9_sepconv3_act'}, 82, 0),
            (84, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block9_sepconv3'}, 83, 0),
            (85, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block9_sepconv3_bn'}, 84, 0),
            (86, 4, 2, 0, {'name': 'add_7'}, 85, [76]),
            (87, 5, 1, 0, {'activation': 'relu', 'name': 'block10_sepconv1_act'}, 86, 0),
            (88, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block10_sepconv1'}, 87, 0),
            (89, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block10_sepconv1_bn'}, 88, 0),
            (90, 5, 1, 0, {'activation': 'relu', 'name': 'block10_sepconv2_act'}, 89, 0),
            (91, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block10_sepconv2'}, 90, 0),
            (92, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block10_sepconv2_bn'}, 91, 0),
            (93, 5, 1, 0, {'activation': 'relu', 'name': 'block10_sepconv3_act'}, 92, 0),
            (94, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block10_sepconv3'}, 93, 0),
            (95, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block10_sepconv3_bn'}, 94, 0),
            (96, 4, 2, 0, {'name': 'add_8'}, 95, [86]),
            (97, 5, 1, 0, {'activation': 'relu', 'name': 'block11_sepconv1_act'}, 96, 0),
            (98, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block11_sepconv1'}, 97, 0),
            (99, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                           'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones', 'moving_mean_initializer': 'Zeros',
                           'moving_variance_initializer': 'Ones', 'beta_regularizer': None, 'gamma_regularizer': None,
                           'beta_constraint': None, 'gamma_constraint': None, 'name': 'block11_sepconv1_bn'}, 98, 0),
            (100, 5, 1, 0, {'activation': 'relu', 'name': 'block11_sepconv2_act'}, 99, 0),
            (101, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block11_sepconv2'}, 100, 0),
            (102, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros', 'moving_variance_initializer': 'Ones',
                            'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None,
                            'gamma_constraint': None, 'name': 'block11_sepconv2_bn'}, 101, 0),
            (103, 5, 1, 0, {'activation': 'relu', 'name': 'block11_sepconv3_act'}, 102, 0),
            (104, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block11_sepconv3'}, 103, 0),
            (105, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros', 'moving_variance_initializer': 'Ones',
                            'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None,
                            'gamma_constraint': None, 'name': 'block11_sepconv3_bn'}, 104, 0),
            (106, 4, 2, 0, {'name': 'add_9'}, 105, [96]),
            (107, 5, 1, 0, {'activation': 'relu', 'name': 'block12_sepconv1_act'}, 106, 0),
            (108, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block12_sepconv1'}, 107, 0),
            (109, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros', 'moving_variance_initializer': 'Ones',
                            'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None,
                            'gamma_constraint': None, 'name': 'block12_sepconv1_bn'}, 108, 0),
            (110, 5, 1, 0, {'activation': 'relu', 'name': 'block12_sepconv2_act'}, 109, 0),
            (111, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block12_sepconv2'}, 110, 0),
            (112, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros', 'moving_variance_initializer': 'Ones',
                            'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None,
                            'gamma_constraint': None, 'name': 'block12_sepconv2_bn'}, 111, 0),
            (113, 5, 1, 0, {'activation': 'relu', 'name': 'block12_sepconv3_act'}, 112, 0),
            (114, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block12_sepconv3'}, 113, 0),
            (115, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros', 'moving_variance_initializer': 'Ones',
                            'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None,
                            'gamma_constraint': None, 'name': 'block12_sepconv3_bn'}, 114, 0),
            (116, 4, 2, 0, {'name': 'add_10'}, 115, [106]),
            (117, 5, 1, 0, {'activation': 'relu', 'name': 'block13_sepconv1_act'}, 116, 0),
            (118, 1, 6, 0,
             {'filters': 728, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block13_sepconv1'}, 117, 0),
            (119, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros', 'moving_variance_initializer': 'Ones',
                            'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None,
                            'gamma_constraint': None, 'name': 'block13_sepconv1_bn'}, 118, 0),
            (120, 5, 1, 0, {'activation': 'relu', 'name': 'block13_sepconv2_act'}, 119, 0),
            (121, 1, 6, 0,
             {'filters': 1024, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block13_sepconv2'}, 120, 0),
            (122, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros', 'moving_variance_initializer': 'Ones',
                            'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None,
                            'gamma_constraint': None, 'name': 'block13_sepconv2_bn'}, 121, 0),
            (123, 1, 3, 0,
             {'filters': 1024, 'kernel_size': (1, 1), 'strides': (2, 2), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'groups': 1, 'use_bias': False,
              'kernel_initializer': 'GlorotUniform', 'bias_initializer': 'Zeros', 'kernel_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None,
              'bias_constraint': None, 'name': 'conv2d_3'}, 116, 0),
            (124, 3, 2, 0, {'pool_size': (3, 3), 'strides': (2, 2), 'padding': 'same', 'data_format': 'channels_last',
                            'name': 'block13_pool'}, 122, 0),
            (125, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros', 'moving_variance_initializer': 'Ones',
                            'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None,
                            'gamma_constraint': None, 'name': 'batch_normalization_3'}, 123, 0),
            (126, 4, 2, 0, {'name': 'add_11'}, 124, [125]),
            (127, 1, 6, 0,
             {'filters': 1536, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block14_sepconv1'}, 126, 0),
            (128, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros', 'moving_variance_initializer': 'Ones',
                            'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None,
                            'gamma_constraint': None, 'name': 'block14_sepconv1_bn'}, 127, 0),
            (129, 5, 1, 0, {'activation': 'relu', 'name': 'block14_sepconv1_act'}, 128, 0),
            (130, 1, 6, 0,
             {'filters': 2048, 'kernel_size': (3, 3), 'strides': (1, 1), 'padding': 'same', 'activation': 'linear',
              'data_format': 'channels_last', 'dilation_rate': (1, 1), 'depth_multiplier': 1, 'use_bias': False,
              'depthwise_initializer': 'GlorotUniform', 'pointwise_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'depthwise_regularizer': None, 'pointwise_regularizer': None,
              'bias_regularizer': None, 'activity_regularizer': None, 'depthwise_constraint': None,
              'pointwise_constraint': None, 'bias_constraint': None, 'name': 'block14_sepconv2'}, 129, 0),
            (131, 6, 2, 0, {'axis': 3, 'momentum': 0.99, 'epsilon': 0.001, 'center': True, 'scale': True,
                            'beta_initializer': 'Zeros', 'gamma_initializer': 'Ones',
                            'moving_mean_initializer': 'Zeros', 'moving_variance_initializer': 'Ones',
                            'beta_regularizer': None, 'gamma_regularizer': None, 'beta_constraint': None,
                            'gamma_constraint': None, 'name': 'block14_sepconv2_bn'}, 130, 0),
            (132, 5, 1, 0, {'activation': 'relu', 'name': 'block14_sepconv2_act'}, 131, 0),
            (133, 8, 6, 0, {'data_format': 'channels_last', 'name': 'avg_pool'}, 132, 0),
            (134, 1, 1, 0,
             {'units': 1000, 'activation': 'softmax', 'use_bias': True, 'kernel_initializer': 'GlorotUniform',
              'bias_initializer': 'Zeros', 'kernel_regularizer': None, 'bias_regularizer': None,
              'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None, 'name': 'predictions'},
             133, 0),
        ],
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
        # "InstanceNormalization": keras_contrib.layers.normalization.instancenormalization, # pip install keras_contrib
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
        },
        # Activations Layers
        5: {
            1: "Activation",
            2: "LeakyReLU",
            3: "PReLU",
            4: "ReLU",
            5: "Softmax",
            6: "ELU",
            7: "ThresholdedReLU"
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
        }
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
    layer_params = {
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
        #  28: 'Xception',
    }

    block_params = {
        'Conv2DBNDrop': {
            'main': get_block_params_from_plan(PlanLinkLibrary.custom_block_plan.get('Conv2DBNDrop', {}),
                                               layers_dict, layer_params),
            'extra': {}
        },
        'Conv2DBNLeaky': {
            'main': get_block_params_from_plan(PlanLinkLibrary.custom_block_plan.get('Conv2DBNLeaky', {}),
                                               layers_dict, layer_params),
            'extra': {}
        },
        'CustomResBlock': {
            'main': get_block_params_from_plan(PlanLinkLibrary.custom_block_plan.get('CustomResBlock', {}),
                                               layers_dict, layer_params),
            'extra': {}
        },
        'Resnet50Block': {
            'main': get_block_params_from_plan(PlanLinkLibrary.custom_block_plan.get('Resnet50Block', {}),
                                               layers_dict, layer_params),
            'extra': {}
        },
        'PSPBlock': {
            'main': get_block_params_from_plan(PlanLinkLibrary.custom_block_plan.get('PSPBlock', {}),
                                               layers_dict, layer_params),
            'extra': {}
        },
        'UNETBlock': {
            'main': get_block_params_from_plan(PlanLinkLibrary.custom_block_plan.get('UNETBlock', {}),
                                               layers_dict, layer_params),
            'extra': {}
        },
        'XceptionBlock': {
            'main': get_block_params_from_plan(PlanLinkLibrary.custom_block_plan.get('XceptionBlock', {}),
                                               layers_dict, layer_params),
            'extra': {}
        },
        'InceptionV3block': {
            'main': get_block_params_from_plan(PlanLinkLibrary.custom_block_plan.get('InceptionV3block', {}),
                                               layers_dict, layer_params),
            'extra': {}
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
