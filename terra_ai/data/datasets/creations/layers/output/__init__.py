from enum import Enum

from .....mixins import BaseMixinData
from ....extra import LayerOutputTypeChoice
from . import types


class LayerBaseData(BaseMixinData):
    pass


class LayerImagesData(LayerBaseData, types.Images.ParametersData):
    pass


class LayerTextData(LayerBaseData, types.Text.ParametersData):
    pass


class LayerAudioData(LayerBaseData, types.Audio.ParametersData):
    pass


class LayerClassificationData(LayerBaseData, types.Classification.ParametersData):
    pass


class LayerSegmentationData(LayerBaseData, types.Segmentation.ParametersData):
    pass


class LayerTextSegmentationData(LayerBaseData, types.TextSegmentation.ParametersData):
    pass


class LayerRegressionData(LayerBaseData, types.Regression.ParametersData):
    pass


class LayerTimeseriesData(LayerBaseData, types.Timeseries.ParametersData):
    pass


Layer = Enum(
    "Layer",
    dict(
        map(
            lambda item: (item.name, f"Layer{item.name}Data"),
            list(LayerOutputTypeChoice),
        )
    ),
    type=str,
)
