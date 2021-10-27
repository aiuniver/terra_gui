from ....mixins import BaseMixinData
from ...outputs import OutputsList
from ..extra import YoloParameters


class ParametersData(BaseMixinData):
    outputs: OutputsList = OutputsList()
    yolo: YoloParameters = YoloParameters()
