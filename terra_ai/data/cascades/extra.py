from enum import Enum


class BlockGroupChoice(str, Enum):
    InputData = "InputData"
    OutputData = "OutputData"
    Model = "Model"
    Function = "Function"
    Custom = "Custom"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, BlockGroupChoice))


class BlockFunctionGroupChoice(str, Enum):
    Audio = "Audio"
    Array = "Array"
    Image = "Image"
    ObjectDetection = "ObjectDetection"
    Segmentation = "Segmentation"
    TextSegmentation = "TextSegmentation"
    Text = "Text"
    Video = "Video"


class BlockCustomGroupChoice(str, Enum):
    Tracking = "Tracking"
