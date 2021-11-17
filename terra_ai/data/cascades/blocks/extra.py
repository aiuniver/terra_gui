from enum import Enum

from terra_ai.data.cascades.extra import BlockGroupChoice
from terra_ai.data.datasets.extra import LayerInputTypeChoice


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

    # Tracker = "Tracker"

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


class BlockServiceGroupChoice(str, Enum):
    Tracking = "Tracking"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, BlockCustomGroupChoice))


class BlockServiceTypeChoice(str, Enum):
    Sort = "Sort"
    DeepSort = "DeepSort"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, BlockCustomTypeChoice))


class BlocksBindChoice(Enum):
    Model = ("Model", tuple(), 1, tuple())
    OutputData = ("OutputData", (BlockGroupChoice.Model,
                                 BlockFunctionTypeChoice.PlotBBoxes,
                                 BlockFunctionTypeChoice.PlotMaskSegmentation,
                                 BlockFunctionTypeChoice.PutTag), 1, tuple())

    Sort = ("Sort", (BlockFunctionTypeChoice.PostprocessBoxes,), (LayerInputTypeChoice.Image,))
    DeepSort = ("DeepSort", (BlockFunctionTypeChoice.PostprocessBoxes,), (LayerInputTypeChoice.Image,))
    ChangeType = ("ChangeType", tuple(), 1, (LayerInputTypeChoice.Image,
                                             LayerInputTypeChoice.Audio,
                                             LayerInputTypeChoice.Video,
                                             LayerInputTypeChoice.Text))
    ChangeSize = ("ChangeSize", tuple(), 1, (LayerInputTypeChoice.Image,))
    MinMaxScale = ("MinMaxScale", tuple(), 1, (LayerInputTypeChoice.Image,
                                               LayerInputTypeChoice.Audio,
                                               LayerInputTypeChoice.Video,
                                               LayerInputTypeChoice.Text))
    CropImage = ("CropImage", tuple(), 1, tuple())
    MaskedImage = ("MaskedImage", tuple(), 1, (LayerInputTypeChoice.Image,))
    PlotMaskSegmentation = ("PlotMaskSegmentation", tuple(), 1, (LayerInputTypeChoice.Image,))
    PutTag = ("PutTag", tuple(), 1, (LayerInputTypeChoice.Text,))
    PostprocessBoxes = ("PostprocessBoxes", (BlockGroupChoice.Model,
                                             BlockGroupChoice.InputData), 2, (LayerInputTypeChoice.Image,))
    PlotBBoxes = ("PlotBBoxes", (BlockFunctionTypeChoice.PostprocessBoxes,
                                 BlockCustomTypeChoice.Sort,
                                 BlockGroupChoice.InputData), 2, (LayerInputTypeChoice.Image,))

    def __init__(self, name, binds, bind_count, data_type):
        self._name = name
        self._binds = binds
        self._bind_count = bind_count
        self._data_type = data_type

    @property
    def name(self):
        return self._name

    @property
    def binds(self):
        return self._binds

    @property
    def bind_count(self):
        return self._bind_count

    @property
    def data_type(self):
        return self._data_type

    @staticmethod
    def checked_block(input_block):
        for block in BlocksBindChoice:
            if block.name == input_block:
                return block
