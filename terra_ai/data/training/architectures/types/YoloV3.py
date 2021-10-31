from ....mixins import BaseMixinData
from ..extra import YoloParameters


class ParametersData(BaseMixinData):
    yolo: YoloParameters = YoloParameters()
