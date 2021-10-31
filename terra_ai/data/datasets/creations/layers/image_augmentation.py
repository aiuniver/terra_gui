from typing import List, Tuple, Optional, Union

from pydantic.types import PositiveInt

from ....mixins import BaseMixinData
from ....types import ConstrainedFloatValueGe0, ConstrainedFloatValueGe0Le1, ConstrainedIntValueGe0Le2


class FliplrData(BaseMixinData):
    p: ConstrainedFloatValueGe0Le1


class FlipudData(BaseMixinData):
    p: ConstrainedFloatValueGe0Le1


class CropData(BaseMixinData):
    percent: Tuple[ConstrainedFloatValueGe0, ConstrainedFloatValueGe0Le1]


class GaussianBlurData(BaseMixinData):
    sigma: Tuple[ConstrainedFloatValueGe0, ConstrainedFloatValueGe0]


class LinearContrastData(BaseMixinData):
    alpha: Tuple[ConstrainedFloatValueGe0, ConstrainedFloatValueGe0]
    per_channel: bool = True


# Исключен в связи с плохой работой с типом float32
# class AdditiveGaussianNoiseData(BaseMixinData):
#     scale: Tuple[ConstrainedFloatValueGe0, ConstrainedFloatValueGe0]
#     # 'scale': (0.0, 0.05 * 255)
#     per_channel: bool = True


class MultiplyData(BaseMixinData):
    mul: Tuple[ConstrainedFloatValueGe0, ConstrainedFloatValueGe0]
    per_channel: bool = True


# Исключен в связи с невозможностью работать с типом float32
# class AddToHueAndSaturationData(BaseMixinData):
#     value: Tuple[int, int]
#     per_channel: bool = True


class ChannelShuffleData(BaseMixinData):
    p: ConstrainedFloatValueGe0Le1
    channels: List[ConstrainedIntValueGe0Le2]


class MotionBlurData(BaseMixinData):
    k: Union[PositiveInt, Tuple[PositiveInt, PositiveInt]]
    angle: List[int]


class ScaleData(BaseMixinData):
    x: Tuple[ConstrainedFloatValueGe0, ConstrainedFloatValueGe0]
    y: Tuple[ConstrainedFloatValueGe0, ConstrainedFloatValueGe0]


class TranslatePercentData(BaseMixinData):
    x: Tuple[float, float]
    y: Tuple[float, float]


class AffineData(BaseMixinData):
    scale: Optional[ScaleData]
    translate_percent: Optional[TranslatePercentData]


class AugmentationData(BaseMixinData):
    Fliplr: Optional[FliplrData]
    Flipud: Optional[FlipudData]
    Crop: Optional[CropData]
    GaussianBlur: Optional[GaussianBlurData]
    LinearContrast: Optional[LinearContrastData]
    # AdditiveGaussianNoise: Optional[AdditiveGaussianNoiseData]
    Multiply: Optional[MultiplyData]
    # AddToHueAndSaturation: Optional[AddToHueAndSaturationData]
    ChannelShuffle: Optional[ChannelShuffleData]
    MotionBlur: Optional[MotionBlurData]
    Affine: Optional[AffineData]
