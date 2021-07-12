"""
## Дополнительные структуры данных параметров типов слоев
"""

from enum import Enum


class PaddingChoice(str, Enum):
    valid = "valid"
    same = "same"


class PaddingAddCausalChoice(str, Enum):
    valid = "valid"
    same = "same"
    causal = "causal"


class DataFormatChoice(str, Enum):
    channels_last = "channels_last"
    channels_first = "channels_first"


class InitializerChoice(str, Enum):
    random_normal = "random_normal"
    random_uniform = "random_uniform"
    truncated_normal = "truncated_normal"
    zeros = "zeros"
    ones = "ones"
    glorot_normal = "glorot_normal"
    glorot_uniform = "glorot_uniform"
    uniform = "uniform"
    identity = "identity"
    orthogonal = "orthogonal"
    constant = "constant"
    variance_scaling = "variance_scaling"
    lecun_normal = "lecun_normal"
    lecun_uniform = "lecun_uniform"
    he_normal = "he_normal"
    he_uniform = "he_uniform"


class RegularizerChoice(str, Enum):
    l1 = "l1"
    l2 = "l2"
    l1_l2 = "l1_l2"


class ConstraintChoice(str, Enum):
    max_norm = "max_norm"
    min_max_norm = "min_max_norm"
    non_neg = "non_neg"
    unit_norm = "unit_norm"
    radial_constraint = "radial_constraint"


class ActivationChoice(str, Enum):
    deserialize = "deserialize"
    elu = "elu"
    exponential = "exponential"
    gelu = "gelu"
    get = "get"
    hard_sigmoid = "hard_sigmoid"
    linear = "linear"
    relu = "relu"
    selu = "selu"
    serialize = "serialize"
    sigmoid = "sigmoid"
    softmax = "softmax"
    softplus = "softplus"
    softsign = "softsign"
    swish = "swish"
    tanh = "tanh"


class InterpolationChoice(str, Enum):
    nearest = "nearest"
    bilinear = "bilinear"


class ResizingInterpolationChoice(str, Enum):
    bilinear = "bilinear"
    nearest = "nearest"
    bicubic = "bicubic"
    area = "area"
    lanczos3 = "lanczos3"
    gaussian = "gaussian"
    mitchellcubic = "mitchellcubic"


class ModuleChoice(str, Enum):
    tensorflow_keras_layers = "tensorflow.keras.layers"
    terra_custom_layers = "customLayers"
    tensorflow_keras_layers_preprocessing = 'tensorflow.keras.layers.experimental.preprocessing'


class ModuleTypeChoice(str, Enum):
    keras = "keras"
    terra_layer = "terra_layer"