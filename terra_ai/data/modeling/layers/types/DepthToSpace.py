"""
## Тип слоя `DepthToSpace`
"""

from ....mixins import BaseMixinData
from ..extra import (
    LayerConfigData,
    LayerValidationMethodChoice,
    ModuleChoice,
    ModuleTypeChoice,
    SpaceToDepthDataFormatChoice,
)
from ....types import ConstrainedIntValueGe2

LayerConfig = LayerConfigData(
    **{
        "num_uplinks": {
            "value": 1,
            "validation": LayerValidationMethodChoice.fixed,
        },
        "input_dimension": {
            "value": (4, 5),
            "validation": LayerValidationMethodChoice.dependence_tuple2,
        },
        "module": ModuleChoice.tensorflow_nn,
        "module_type": ModuleTypeChoice.tensorflow,
    }
)


class ParametersMainData(BaseMixinData):
    block_size: ConstrainedIntValueGe2 = 2


class ParametersExtraData(BaseMixinData):
    data_format: SpaceToDepthDataFormatChoice = SpaceToDepthDataFormatChoice.NHWC
