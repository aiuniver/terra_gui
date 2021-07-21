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


class SpaceToDepthDataFormatChoice(str, Enum):
    NHWC = "NHWC"
    NCHW = "NCHW"
    NCHW_VECT_C = "NCHW_VECT_C"


class PretrainedModelWeightsChoice(str, Enum):
    imagenet = "imagenet"


class PretrainedModelPoolingChoice(str, Enum):
    max = "max"
    avg = "avg"


class YOLOModeChoice(str, Enum):
    YOLOv3 = "YOLOv3"
    YOLOv4 = "YOLOv4"
    YOLOv5 = "YOLOv5"


class YOLOActivationChoice(str, Enum):
    LeakyReLU = "LeakyReLU"
    Mish = "Mish"
    Swish = "Swish"


class VAELatentRegularizerChoice(str, Enum):
    vae = "vae"
    bvae = "bvae"


class ModuleChoice(str, Enum):
    tensorflow_keras_layers = "tensorflow.keras.layers"
    terra_custom_layers = "customLayers"
    tensorflow_keras_layers_preprocessing = (
        "tensorflow.keras.layers.experimental.preprocessing"
    )
    tensorflow_nn = "tensorflow.nn"
    inception_v3 = "tensorflow.keras.applications.inception_v3"
    xception = "tensorflow.keras.applications.xception"
    vgg16 = "tensorflow.keras.applications.vgg16"
    resnet50 = "tensorflow.keras.applications.resnet50"


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
                print(__value)
                raise LayerValueConfigException(value, __value)
        return value


class LayerConfigData(BaseMixinData):
    num_uplinks: Optional[LayerValueConfig]
    input_dimension: LayerValueConfig
    module: Optional[ModuleChoice]
    module_type: ModuleTypeChoice
