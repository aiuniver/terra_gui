from typing import List, Tuple

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


class AdditiveGaussianNoiseData(BaseMixinData):
    scale: Tuple[ConstrainedFloatValueGe0, ConstrainedFloatValueGe0]
    # 'scale': (0.0, 0.05 * 255)
    per_channel: bool = True


class MultiplyData(BaseMixinData):
    mul: Tuple[ConstrainedFloatValueGe0, ConstrainedFloatValueGe0]
    per_channel: bool = True


class AddToHueAndSaturationData(BaseMixinData):
    value: Tuple[int, int]
    per_channel: bool = True


class ChannelShuffleData(BaseMixinData):
    p: ConstrainedFloatValueGe0Le1
    channels: List[ConstrainedIntValueGe0Le2, ConstrainedIntValueGe0Le2, ConstrainedIntValueGe0Le2]


class MotionBlurData(BaseMixinData):
    k: ConstrainedFloatValueGe0Le1
    angle: List[int, int]


class ScaleData(BaseMixinData):
    x: Tuple[ConstrainedFloatValueGe0, ConstrainedFloatValueGe0]
    y: Tuple[ConstrainedFloatValueGe0, ConstrainedFloatValueGe0]


class TranslatePercentData(BaseMixinData):
    x: Tuple[float, float]
    y: Tuple[float, float]


class AffineData(BaseMixinData):
    scale: ScaleData
    translate_percent: TranslatePercentData


class AugmentationData(BaseMixinData):
    Fliplr: FliplrData
    Flipud: FlipudData
    Crop: CropData
    GaussianBlur: GaussianBlurData
    LinearContrast: LinearContrastData
    AdditiveGaussianNoise: AdditiveGaussianNoiseData
    Multiply: MultiplyData
    AddToHueAndSaturation: AddToHueAndSaturationData
    ChannelShuffle: ChannelShuffleData
    MotionBlur: MotionBlurData
    Affine: AffineData
