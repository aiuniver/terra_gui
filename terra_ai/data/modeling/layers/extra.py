"""
## Дополнительные структуры данных параметров типов слоев
"""

from enum import Enum
from typing import Optional, Union, Tuple
from pydantic import validator
from pydantic.types import PositiveInt

from ...mixins import BaseMixinData
from ...exceptions import LayerValueConfigException


class PaddingChoice(str, Enum):
    valid = "valid"
    same = "same"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, PaddingChoice))


class PaddingAddCausalChoice(str, Enum):
    valid = "valid"
    same = "same"
    causal = "causal"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, PaddingAddCausalChoice))


class DataFormatChoice(str, Enum):
    channels_last = "channels_last"
    channels_first = "channels_first"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, DataFormatChoice))


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

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, InitializerChoice))


class RegularizerChoice(str, Enum):
    l1 = "l1"
    l2 = "l2"
    l1_l2 = "l1_l2"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, RegularizerChoice))


class ConstraintChoice(str, Enum):
    max_norm = "max_norm"
    min_max_norm = "min_max_norm"
    non_neg = "non_neg"
    unit_norm = "unit_norm"
    radial_constraint = "radial_constraint"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, ConstraintChoice))


class ActivationChoice(str, Enum):
    elu = "elu"
    exponential = "exponential"
    gelu = "gelu"
    hard_sigmoid = "hard_sigmoid"
    linear = "linear"
    relu = "relu"
    selu = "selu"
    sigmoid = "sigmoid"
    softmax = "softmax"
    softplus = "softplus"
    softsign = "softsign"
    swish = "swish"
    tanh = "tanh"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, ActivationChoice))


class InterpolationChoice(str, Enum):
    nearest = "nearest"
    bilinear = "bilinear"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, InterpolationChoice))


class ResizingInterpolationChoice(str, Enum):
    bilinear = "bilinear"
    nearest = "nearest"
    bicubic = "bicubic"
    area = "area"
    lanczos3 = "lanczos3"
    gaussian = "gaussian"
    mitchellcubic = "mitchellcubic"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, ResizingInterpolationChoice))


class SpaceToDepthDataFormatChoice(str, Enum):
    NHWC = "NHWC"
    NCHW = "NCHW"
    NCHW_VECT_C = "NCHW_VECT_C"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, SpaceToDepthDataFormatChoice))


class PretrainedModelWeightsChoice(str, Enum):
    imagenet = "imagenet"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, PretrainedModelWeightsChoice))


class PretrainedModelPoolingChoice(str, Enum):
    max = "max"
    avg = "avg"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, PretrainedModelPoolingChoice))


class YOLOModeChoice(str, Enum):
    YOLOv3 = "YOLOv3"
    YOLOv4 = "YOLOv4"
    # YOLOv5 = "YOLOv5"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, YOLOModeChoice))


class YOLOActivationChoice(str, Enum):
    LeakyReLU = "LeakyReLU"
    Mish = "Mish"
    Swish = "Swish"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, YOLOActivationChoice))


class VAELatentRegularizerChoice(str, Enum):
    vae = "vae"
    bvae = "bvae"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, VAELatentRegularizerChoice))


class ModuleChoice(str, Enum):
    tensorflow_keras_layers = "tensorflow.keras.layers"
    terra_custom_layers = "terra_ai.custom_objects.customLayers"
    tensorflow_keras_layers_preprocessing = (
        "tensorflow.keras.layers.experimental.preprocessing"
    )
    tensorflow_nn = "tensorflow.nn"
    inception_v3 = "tensorflow.keras.applications.inception_v3"
    xception = "tensorflow.keras.applications.xception"
    vgg16 = "tensorflow.keras.applications.vgg16"
    vgg19 = "tensorflow.keras.applications.vgg19"
    resnet50 = "tensorflow.keras.applications.resnet50"
    resnet101 = "tensorflow.keras.applications.resnet"
    resnet152 = "tensorflow.keras.applications.resnet"
    resnet50v2 = "tensorflow.keras.applications.resnet_v2"
    resnet101v2 = "tensorflow.keras.applications.resnet_v2"
    resnet152v2 = "tensorflow.keras.applications.resnet_v2"
    densenet121 = "tensorflow.keras.applications.densenet"
    densenet169 = "tensorflow.keras.applications.densenet"
    densenet201 = "tensorflow.keras.applications.densenet"
    nasnetmobile = "tensorflow.keras.applications.nasnet"
    nasnetlarge = "tensorflow.keras.applications.nasnet"
    mobilenetv3small = "tensorflow.keras.applications"
    mobilenetv2 = "tensorflow.keras.applications.mobilenet_v2"
    efficientnetb0 = "tensorflow.keras.applications.efficientnet"


class ModuleTypeChoice(str, Enum):
    keras = "keras"
    tensorflow = "tensorflow"
    keras_pretrained_model = "keras_pretrained_model"
    terra_layer = "terra_layer"
    block_plan = "block_plan"


class LayerValidationMethodChoice(str, Enum):
    fixed = "fixed"
    minimal = "minimal"
    dependence_tuple2 = "dependence_tuple2"
    dependence_tuple3 = "dependence_tuple3"


class LayerValueConfig(BaseMixinData):
    value: Union[
        PositiveInt,
        Tuple[PositiveInt, PositiveInt],
        Tuple[PositiveInt, PositiveInt, PositiveInt],
    ]
    validation: LayerValidationMethodChoice

    @validator("validation")
    def _validate_validation(
        cls, value: LayerValidationMethodChoice, values
    ) -> LayerValidationMethodChoice:
        __value = values.get("value")
        if not __value:
            return value
        if value == LayerValidationMethodChoice.dependence_tuple2:
            if not (isinstance(__value, tuple) and len(__value) == 2):
                raise LayerValueConfigException(value, __value)
        if value == LayerValidationMethodChoice.dependence_tuple3:
            if not (isinstance(__value, tuple) and len(__value) == 3):
                raise LayerValueConfigException(value, __value)
        if value in [
            LayerValidationMethodChoice.fixed,
            LayerValidationMethodChoice.minimal,
        ]:
            if __value and not isinstance(__value, int):
                raise LayerValueConfigException(value, __value)
        return value


class LayerConfigData(BaseMixinData):
    num_uplinks: Optional[LayerValueConfig]
    input_dimension: LayerValueConfig
    module: Optional[ModuleChoice]
    module_type: ModuleTypeChoice


class CONVBlockConfigChoice(str, Enum):
    conv_conv_bn_lrelu_drop = "conv_conv_bn_lrelu_drop"
    conv_bn_lrelu_drop_conv_bn_lrelu_drop = "conv_bn_lrelu_drop_conv_bn_lrelu_drop"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, CONVBlockConfigChoice))
