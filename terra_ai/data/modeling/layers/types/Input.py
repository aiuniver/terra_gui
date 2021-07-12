"""
## Тип слоя `Input`
"""

from typing import Optional, Tuple
from pydantic.types import PositiveInt

from .extra import ModuleChoise, ModuleTypeChoice
from ....mixins import BaseMixinData
from ....types import ConstrainedIntValueGe2


class ParametersMainData(BaseMixinData):
    shape: Optional[Tuple[PositiveInt, ...]]
    batch_size: Optional[PositiveInt]
    name: Optional[str]
    dtype: Optional[str]
    sparse: Optional[bool]
    # tensor: Optional[str]   # тут тензор а не str
    ragged: Optional[bool]
    type_spec: Optional[str]


class ParametersExtraData(BaseMixinData):
    pass


class LayerConfig(BaseMixinData):
    num_uplinks: PositiveInt = 1
    input_dimension: ConstrainedIntValueGe2 = 2
    module: ModuleChoise = ModuleChoise.tensorflow_keras_layers
    module_type: ModuleTypeChoice = ModuleTypeChoice.keras
