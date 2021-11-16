from enum import Enum

from terra_ai.data.cascades.extra import BlockGroupChoice


class BlockOutputDataSaveAsChoice(str, Enum):
    source = "source"
    file = "file"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, BlockOutputDataSaveAsChoice))


class PostprocessBoxesMethodAvailableChoice(str, Enum):
    nms = "nms"
    soft_nms = "soft_nms"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, BlockFunctionGroupChoice))


class ChangeTypeAvailableChoice(str, Enum):
    int = "int"
    int8 = "int8"
    int32 = "int32"
    int64 = "int64"
    uint = "uint"
    uint8 = "uint8"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"
    float = "float"
    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    complex = "complex"
    complex64 = "complex64"
    complex128 = "complex128"
    bool = "bool"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, BlockFunctionGroupChoice))


class BlockFunctionGroupChoice(str, Enum):
    Image = "Image"
    Text = "Text"
    Audio = "Audio"
    Video = "Video"
    Array = "Array"
    Segmentation = "Segmentation"
    TextSegmentation = "TextSegmentation"
    ObjectDetection = "ObjectDetection"
    Tracker = "Tracker"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, BlockFunctionGroupChoice))


class BlockFunctionTypeChoice(str, Enum):
    ChangeType = "ChangeType"
    ChangeSize = "ChangeSize"
    MinMaxScale = "MinMaxScale"
    CropImage = "CropImage"
    MaskedImage = "MaskedImage"
    PlotMaskSegmentation = "PlotMaskSegmentation"
    PutTag = "PutTag"
    PostprocessBoxes = "PostprocessBoxes"
    PlotBBoxes = "PlotBBoxes"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, BlockFunctionGroupChoice))


class BlockCustomGroupChoice(str, Enum):
    Tracking = "Tracking"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, BlockCustomGroupChoice))


class BlockCustomTypeChoice(str, Enum):
    Sort = "Sort"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, BlockCustomTypeChoice))


class BlocksBindChoice(Enum):
    Model = ("Model", tuple(), 1)
    OutputData = ("OutputData", (BlockGroupChoice.Model,
                                 BlockFunctionTypeChoice.PlotBBoxes,
                                 BlockFunctionTypeChoice.PlotMaskSegmentation,
                                 BlockFunctionTypeChoice.PutTag), 1)

    Sort = ("Sort", (BlockFunctionTypeChoice.PostprocessBoxes, ), 1)
    ChangeType = ("ChangeType", tuple(), 1)
    ChangeSize = ("ChangeSize", tuple(), 1)
    MinMaxScale = ("MinMaxScale", tuple(), 1)
    CropImage = ("CropImage", tuple(), 1)
    MaskedImage = ("MaskedImage", tuple(), 1)
    PlotMaskSegmentation = ("PlotMaskSegmentation", tuple(), 1)
    PutTag = ("PutTag", tuple(), 1)
    PostprocessBoxes = ("PostprocessBoxes", (BlockGroupChoice.Model, BlockGroupChoice.InputData), 2)
    PlotBBoxes = ("PlotBBoxes", (BlockFunctionTypeChoice.PostprocessBoxes,
                                 BlockCustomTypeChoice.Sort,
                                 BlockGroupChoice.InputData), 2)

    def __init__(self, name, binds, bind_count):
        self._name = name
        self._binds = binds
        self._bind_count = bind_count

    @property
    def name(self):
        return self._name

    @property
    def binds(self):
        return self._binds

    @property
    def bind_count(self):
        return self._bind_count

    @staticmethod
    def checked_block(input_block):
        for block in BlocksBindChoice:
            if block.name == input_block:
                return block
