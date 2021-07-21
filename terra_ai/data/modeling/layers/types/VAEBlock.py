"""
## Тип слоя `VAEvBlock`
"""
from typing import Optional, Tuple

from pydantic import PositiveInt

from ..extra import LayerConfigData, LayerValidationMethodChoice, ModuleChoice, ModuleTypeChoice, ActivationChoice, \
    YOLOModeChoice, YOLOActivationChoice, PaddingChoice, VAELatentRegularizerChoice
from ....mixins import BaseMixinData

LayerConfig = LayerConfigData(
    **{
        "num_uplinks": {
            "value": 1,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "input_dimension": {
            "value": (2, 3, 4),
            "validation": LayerValidationMethodChoice.dependence_tuple3,
        },
        "module": ModuleChoice.terra_custom_layers,
        "module_type": ModuleTypeChoice.terra_layer,
    }
)


class ParametersMainData(BaseMixinData):
    latent_size: PositiveInt = 32
    latent_regularizer: Optional[VAELatentRegularizerChoice] = VAELatentRegularizerChoice.vae
    beta: float = 5.
    capacity: float = 128.
    randomSample: bool = True
    roll_up: bool = True


class ParametersExtraData(BaseMixinData):
    pass
