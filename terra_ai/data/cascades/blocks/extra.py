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


class BlockServiceDeepSortMetricChoice(str, Enum):
    euclidean = "euclidean"
    сosine = "сosine"

    @staticmethod
    def values() -> list:
        return list(map(lambda item: item.value, BlockServiceDeepSortMetricChoice))


class BlocksBindChoice(Enum):
    Model = ("Model", ("Любой блок, совместимый с типом данных выбранной модели: ",), tuple())
    OutputData = ("OutputData", ((BlockGroupChoice.Model,
                                  BlockFunctionTypeChoice.PlotBBoxes,
                                  BlockFunctionTypeChoice.PlotMaskSegmentation,
                                  BlockFunctionTypeChoice.MaskedImage,
                                  BlockFunctionTypeChoice.PutTag),), tuple())

    Sort = ("Sort", (BlockFunctionTypeChoice.PostprocessBoxes,), (LayerInputTypeChoice.Image,))
    DeepSort = ("DeepSort", (BlockFunctionTypeChoice.PostprocessBoxes,), (LayerInputTypeChoice.Image,))
    ChangeType = ("ChangeType", tuple(), (LayerInputTypeChoice.Image,
                                          LayerInputTypeChoice.Audio,
                                          LayerInputTypeChoice.Video,
                                          LayerInputTypeChoice.Text))
    ChangeSize = ("ChangeSize", tuple(), (LayerInputTypeChoice.Image,))
    MinMaxScale = ("MinMaxScale", tuple(), (LayerInputTypeChoice.Image,
                                            LayerInputTypeChoice.Audio,
                                            LayerInputTypeChoice.Video,
                                            LayerInputTypeChoice.Text))
    CropImage = ("CropImage", tuple(), tuple())
    MaskedImage = ("MaskedImage", tuple(), (LayerInputTypeChoice.Image,))
    PlotMaskSegmentation = ("PlotMaskSegmentation", tuple(), (LayerInputTypeChoice.Image,))
    PutTag = ("PutTag", tuple(), (LayerInputTypeChoice.Text,))
    PostprocessBoxes = ("PostprocessBoxes", (BlockGroupChoice.Model,
                                             BlockGroupChoice.InputData), (LayerInputTypeChoice.Image,))
    PlotBBoxes = ("PlotBBoxes", ((BlockFunctionTypeChoice.PostprocessBoxes,
                                  BlockCustomTypeChoice.Sort),
                                 BlockGroupChoice.InputData), (LayerInputTypeChoice.Image,))

    def __init__(self, name, binds, data_type):
        self._name = name
        self._binds = self._get_binds(binds)
        self._required_binds = tuple([bind for bind in binds if not isinstance(bind, tuple)])
        self._bind_count = len(binds) if binds else 1
        self._data_type = ", ".join([type_name.value for type_name in data_type])

    @property
    def name(self):
        return self._name

    @property
    def binds(self):
        return self._binds

    @property
    def required_binds(self):
        return self._required_binds

    @property
    def bind_count(self):
        return self._bind_count

    @property
    def data_type(self):
        return self._data_type

    @staticmethod
    def _get_binds(binds):
        out = []
        for bind in binds:
            if isinstance(bind, tuple):
                out.extend(bind)
            else:
                out.append(bind)
        return tuple(out)

    @staticmethod
    def checked_block(input_block):
        for block in BlocksBindChoice:
            if block.name == input_block:
                return block


class FunctionParamsChoice(Enum):
    ChangeType = (BlockFunctionTypeChoice.ChangeType, ("change_type",))
    ChangeSize = (BlockFunctionTypeChoice.ChangeSize, ("shape",))
    MinMaxScale = (BlockFunctionTypeChoice.MinMaxScale, ("min_scale", "max_scale"))
    CropImage = (BlockFunctionTypeChoice.CropImage, tuple())
    MaskedImage = (BlockFunctionTypeChoice.MaskedImage, ("class_id",))
    PlotMaskSegmentation = (BlockFunctionTypeChoice.PlotMaskSegmentation, ("classes_colors",))
    PutTag = (BlockFunctionTypeChoice.PutTag, ("open_tag", "close_tag", "alpha"))
    PostprocessBoxes = (BlockFunctionTypeChoice.PostprocessBoxes, (
        "input_size", "score_threshold", "iou_threshold", "method", "sigma"
    ))
    PlotBBoxes = (BlockFunctionTypeChoice.PlotBBoxes, ("classes",))
    Sort = (BlockServiceTypeChoice.Sort, ("max_age", "min_hits"))

    def __init__(self, name, parameters):
        self._name = name
        self._parameters = parameters

    @property
    def name(self):
        return self._name

    @property
    def parameters(self):
        return self._parameters

    @staticmethod
    def get_parameters(input_block):
        for block in FunctionParamsChoice:
            if block.name == input_block:
                return block.parameters
