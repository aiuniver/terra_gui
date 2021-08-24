from ...extra import ParametersBaseData
from .....extra import YoloVersionChoice


class ParametersData(ParametersBaseData):
    yolo_version: YoloVersionChoice = YoloVersionChoice.yolo_v3
