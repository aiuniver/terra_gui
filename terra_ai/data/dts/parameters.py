"""
## Структура данных для параметров слоев
"""

import sys

from enum import Enum
from typing import Optional, Any
from pydantic import validator, DirectoryPath, FilePath
from pydantic.color import Color


from ..mixins import BaseMixinData, UniqueListMixin
from ..validators import validate_positive_integer
from .extra import (
    LayerPrepareMethodChoice,
    LayerTaskTypeChoice,
    LayerNetChoice,
    LayerScalerChoice,
    LayerOutputMaskAssignmentChoice,
)


class LayerInputTypeImagesData(BaseMixinData):
    folder_path: Optional[DirectoryPath]
    width: int
    height: int
    net: LayerNetChoice = LayerNetChoice.Convolutional
    scaler: LayerScalerChoice = LayerScalerChoice.NoScaler

    _validate_positive_integer = validator("width", "height", allow_reuse=True)(
        validate_positive_integer
    )


class LayerInputTypeTextData(BaseMixinData):
    folder_path: Optional[DirectoryPath]
    delete_symbols: Optional[str]
    x_len: int
    step: int
    max_words_count: int
    pymorphy: Optional[bool] = False
    prepare_method: LayerPrepareMethodChoice = LayerPrepareMethodChoice.embedding
    word_to_vec_size: Optional[int]

    _validate_positive_integer = validator(
        "x_len", "step", "max_words_count", "word_to_vec_size", allow_reuse=True
    )(validate_positive_integer)

    @validator("prepare_method", allow_reuse=True)
    def _validate_prepare_method(
        cls, value: LayerPrepareMethodChoice
    ) -> LayerPrepareMethodChoice:
        if value == LayerPrepareMethodChoice.word_to_vec:
            cls.__fields__["word_to_vec_size"].required = True
        return value


class LayerInputTypeAudioData(BaseMixinData):
    folder_path: Optional[DirectoryPath]
    length: int
    step: int
    scaler: LayerScalerChoice = LayerScalerChoice.NoScaler
    audio_signal: Optional[bool] = True
    chroma_stft: Optional[bool] = False
    mfcc: Optional[bool] = False
    rms: Optional[bool] = False
    spectral_centroid: Optional[bool] = False
    spectral_bandwidth: Optional[bool] = False
    spectral_rolloff: Optional[bool] = False
    zero_crossing_rate: Optional[bool] = False

    _validate_positive_integer = validator("length", "step", allow_reuse=True)(
        validate_positive_integer
    )


class LayerInputTypeDataframeData(BaseMixinData):
    file_path: Optional[FilePath]
    separator: Optional[str]
    encoding: str = "utf-8"
    x_cols: Optional[int]
    scaler: LayerScalerChoice = LayerScalerChoice.NoScaler

    _validate_positive_integer = validator("x_cols", allow_reuse=True)(
        validate_positive_integer
    )


class LayerOutputTypeImagesData(BaseMixinData):
    folder_path: Optional[DirectoryPath]
    width: int
    height: int
    net: LayerNetChoice = LayerNetChoice.Convolutional
    scaler: LayerScalerChoice = LayerScalerChoice.NoScaler

    _validate_positive_integer = validator("width", "height", allow_reuse=True)(
        validate_positive_integer
    )


class LayerOutputTypeTextData(BaseMixinData):
    folder_path: Optional[DirectoryPath]
    delete_symbols: Optional[str]
    x_len: int
    step: int
    max_words_count: int
    pymorphy: Optional[bool] = False
    prepare_method: LayerPrepareMethodChoice = LayerPrepareMethodChoice.embedding
    word_to_vec_size: Optional[int]

    _validate_positive_integer = validator(
        "x_len", "step", "max_words_count", "word_to_vec_size", allow_reuse=True
    )(validate_positive_integer)

    @validator("prepare_method", allow_reuse=True)
    def _validate_prepare_method(
        cls, value: LayerPrepareMethodChoice
    ) -> LayerPrepareMethodChoice:
        if value == LayerPrepareMethodChoice.word_to_vec:
            cls.__fields__["word_to_vec_size"].required = True
        return value


class LayerOutputTypeAudioData(BaseMixinData):
    folder_path: Optional[DirectoryPath]
    length: int
    step: int
    scaler: LayerScalerChoice = LayerScalerChoice.NoScaler
    audio_signal: Optional[bool] = True
    chroma_stft: Optional[bool] = False
    mfcc: Optional[bool] = False
    rms: Optional[bool] = False
    spectral_centroid: Optional[bool] = False
    spectral_bandwidth: Optional[bool] = False
    spectral_rolloff: Optional[bool] = False
    zero_crossing_rate: Optional[bool] = False

    _validate_positive_integer = validator("length", "step", allow_reuse=True)(
        validate_positive_integer
    )


class LayerOutputTypeClassificationData(BaseMixinData):
    one_hot_encoding: Optional[bool] = True


class MaskSegmentationData(BaseMixinData):
    """
    Маска сегментации
    """

    name: str
    "Название класса"
    color: Color
    "Цвет класса"


class MasksSegmentationList(UniqueListMixin):
    """
    Список масок сегментации
    """

    class Meta:
        source = MaskSegmentationData
        identifier = "name"


class LayerOutputTypeSegmentationData(BaseMixinData):
    folder_path: Optional[DirectoryPath]
    mask_range: int
    mask_assignment: MasksSegmentationList

    _validate_positive_integer = validator("mask_range", allow_reuse=True)(
        validate_positive_integer
    )


class LayerOutputTypeTextSegmentationData(BaseMixinData):
    open_tags: Optional[str]
    close_tags: Optional[str]


class LayerOutputTypeRegressionData(BaseMixinData):
    y_col: Optional[int]

    _validate_positive_integer = validator("y_col", allow_reuse=True)(
        validate_positive_integer
    )


class LayerOutputTypeTimeseriesData(BaseMixinData):
    length: int
    y_cols: Optional[int]
    scaler: LayerScalerChoice = LayerScalerChoice.NoScaler
    task_type: LayerTaskTypeChoice = LayerTaskTypeChoice.timeseries

    _validate_positive_integer = validator("length", "y_cols", allow_reuse=True)(
        validate_positive_integer
    )


class LayerInputDatatype(str, Enum):
    """
    Список возможных типов данных для `input`-слоя в виде строк
    """

    images = "LayerInputTypeImagesData"
    text = "LayerInputTypeTextData"
    audio = "LayerInputTypeAudioData"
    dataframe = "LayerInputTypeDataframeData"


class LayerOutputDatatype(str, Enum):
    """
    Список возможных типов данных для `output`-слоя в виде строк
    """

    images = "LayerOutputTypeImagesData"
    text = "LayerOutputTypeTextData"
    audio = "LayerOutputTypeAudioData"
    classification = "LayerOutputTypeClassificationData"
    segmentation = "LayerOutputTypeSegmentationData"
    text_segmentation = "LayerOutputTypeTextSegmentationData"
    regression = "LayerOutputTypeRegressionData"
    timeseries = "LayerOutputTypeTimeseriesData"


LayerInputDatatypeUnion = tuple(
    map(
        lambda item: getattr(sys.modules[__name__], item),
        LayerInputDatatype,
    )
)
"""
Список возможных типов данных для `input`-слоя в виде классов
"""


LayerOutputDatatypeUnion = tuple(
    map(
        lambda item: getattr(sys.modules[__name__], item),
        LayerOutputDatatype,
    )
)
"""
Список возможных типов данных для `output`-слоя в виде классов
"""
