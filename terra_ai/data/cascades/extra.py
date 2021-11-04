from enum import Enum


class BlockGroupChoice(str, Enum):
    InputData = "InputData"
    OutputData = "OutputData"
    Model = "Model"
    Function = "Function"
    Custom = "Custom"


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
