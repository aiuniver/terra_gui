from enum import Enum

from .....mixins import BaseMixinData
from ....extra import LayerInputTypeChoice
from . import types


class LayerBaseData(BaseMixinData):
    pass


class LayerImagesData(LayerBaseData, types.Images.ParametersData):
    pass


class LayerTextData(LayerBaseData, types.Text.ParametersData):
    pass


class LayerAudioData(LayerBaseData, types.Audio.ParametersData):
    pass


class LayerDataframeData(LayerBaseData, types.Dataframe.ParametersData):
    pass


Layer = Enum(
    "Layer",
    dict(
        map(
            lambda item: (item.name, f"Layer{item.name}Data"),
            list(LayerInputTypeChoice),
        )
    ),
    type=str,
)
