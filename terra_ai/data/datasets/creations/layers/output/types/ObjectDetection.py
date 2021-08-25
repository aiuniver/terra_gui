from ...extra import SourcesPathsData
from .....extra import LayerYoloChoice


class ParametersData(SourcesPathsData):
    yolo: LayerYoloChoice = LayerYoloChoice.v4
