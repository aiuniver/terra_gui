from ..extra import YoloParameters
from . import Base


class ParametersData(Base.ParametersData):
    yolo: YoloParameters = YoloParameters()
