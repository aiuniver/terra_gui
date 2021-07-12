"""
## Тип слоя `Resizing`
"""

from ....mixins import BaseMixinData
from .extra import ResizingInterpolationChoice


class ParametersMainData(BaseMixinData):
    height: int = 224
    width: int = 224


class ParametersExtraData(BaseMixinData):
    interpolation: ResizingInterpolationChoice = ResizingInterpolationChoice.bilinear


class LayerConfig(BaseMixinData):
    num_uplinks: int or str or list = 1
    input_dimension: int or str = '2+'
    module: str = 'tensorflow.keras.layers.experimental.preprocessing'
    module_type: str = 'keras'


