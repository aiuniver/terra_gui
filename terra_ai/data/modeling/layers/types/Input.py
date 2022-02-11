"""
## Тип слоя `Input`
"""

from typing import Optional, Tuple
from pydantic.types import PositiveInt

from ..extra import (
    LayerConfigData,
    LayerValidationMethodChoice,
    ModuleChoice,
    ModuleTypeChoice,
    DtypeInputLayerChoice,
)
from ....mixins import BaseMixinData

LayerConfig = LayerConfigData(
    **{
        "num_uplinks": {
            "value": 1,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "input_dimension": {
            "value": 2,
            "validation": LayerValidationMethodChoice.minimal,
        },
        "module": ModuleChoice.tensorflow_keras_layers,
        "module_type": ModuleTypeChoice.keras,
    }
)


class ParametersMainData(BaseMixinData):
    # shape: Optional[Tuple[PositiveInt, ...]]
    # batch_size: Optional[PositiveInt]
    # name: Optional[str]
    # dtype: Optional[DtypeInputLayerChoice] = DtypeInputLayerChoice.float32
    # sparse: Optional[bool]
    # # tensor: Optional[str]   # тут тензор а не str
    # ragged: Optional[bool]
    # type_spec: Optional[str]
    pass


class ParametersExtraData(BaseMixinData):
    pass
