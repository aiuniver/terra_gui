"""
## Тип слоя `VAEvBlock`
"""
from typing import Optional

from pydantic import PositiveInt

from ..extra import (
    LayerConfigData,
    LayerValidationMethodChoice,
    ModuleChoice,
    ModuleTypeChoice,
    VAELatentRegularizerChoice,
)
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
        "module": ModuleChoice.terra_gan_custom_layers,
        "module_type": ModuleTypeChoice.terra_layer,
    }
)


class ParametersMainData(BaseMixinData):
    latent_size: PositiveInt = 32
    latent_regularizer: Optional[
        VAELatentRegularizerChoice
    ] = VAELatentRegularizerChoice.vae
    beta: float = 5.0
    capacity: float = 128.0
    random_sample: bool = True
    roll_up: bool = True


class ParametersExtraData(BaseMixinData):
    pass
