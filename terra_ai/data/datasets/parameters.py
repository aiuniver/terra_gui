"""
## Структура данных для параметров слоев
"""

import sys

from enum import Enum
from typing import Optional
from pydantic import validator, DirectoryPath, FilePath, PositiveInt
from pydantic.color import Color


from ..mixins import BaseMixinData, UniqueListMixin
from .extra import (
    LayerPrepareMethodChoice,
    LayerTaskTypeChoice,
    LayerNetChoice,
    LayerScalerChoice,
)


class LayerInputTypeImagesData(BaseMixinData):
    folder_path: Optional[DirectoryPath]
    width: PositiveInt
    height: PositiveInt
    net: LayerNetChoice = LayerNetChoice.Convolutional
    scaler: LayerScalerChoice = LayerScalerChoice.NoScaler


class LayerInputTypeTextData(BaseMixinData):
    folder_path: Optional[DirectoryPath]
    delete_symbols: Optional[str]
    x_len: PositiveInt
    step: PositiveInt
    max_words_count: PositiveInt
    pymorphy: Optional[bool] = False
    prepare_method: LayerPrepareMethodChoice = LayerPrepareMethodChoice.embedding
    word_to_vec_size: Optional[PositiveInt]

    @validator("prepare_method", allow_reuse=True)
    def _validate_prepare_method(
        cls, value: LayerPrepareMethodChoice
    ) -> LayerPrepareMethodChoice:
        if value == LayerPrepareMethodChoice.word_to_vec:
            cls.__fields__["word_to_vec_size"].required = True
        return value


class LayerInputTypeAudioData(BaseMixinData):
    folder_path: Optional[DirectoryPath]
    length: PositiveInt
    step: PositiveInt
    scaler: LayerScalerChoice = LayerScalerChoice.NoScaler
    audio_signal: Optional[bool] = True
    chroma_stft: Optional[bool] = False
    mfcc: Optional[bool] = False
    rms: Optional[bool] = False
    spectral_centroid: Optional[bool] = False
    spectral_bandwidth: Optional[bool] = False
    spectral_rolloff: Optional[bool] = False
    zero_crossing_rate: Optional[bool] = False


class LayerInputTypeDataframeData(BaseMixinData):
    file_path: Optional[FilePath]
    separator: Optional[str]
    encoding: str = "utf-8"
    x_cols: Optional[PositiveInt]
    scaler: LayerScalerChoice = LayerScalerChoice.NoScaler


class LayerOutputTypeImagesData(BaseMixinData):
    folder_path: Optional[DirectoryPath]
    width: PositiveInt
    height: PositiveInt
    net: LayerNetChoice = LayerNetChoice.Convolutional
    scaler: LayerScalerChoice = LayerScalerChoice.NoScaler


class LayerOutputTypeTextData(BaseMixinData):
    folder_path: Optional[DirectoryPath]
    delete_symbols: Optional[str]
    x_len: PositiveInt
    step: PositiveInt
    max_words_count: PositiveInt
    pymorphy: Optional[bool] = False
    prepare_method: LayerPrepareMethodChoice = LayerPrepareMethodChoice.embedding
    word_to_vec_size: Optional[PositiveInt]

    @validator("prepare_method", allow_reuse=True)
    def _validate_prepare_method(
        cls, value: LayerPrepareMethodChoice
    ) -> LayerPrepareMethodChoice:
        if value == LayerPrepareMethodChoice.word_to_vec:
            cls.__fields__["word_to_vec_size"].required = True
        return value


class LayerOutputTypeAudioData(BaseMixinData):
    folder_path: Optional[DirectoryPath]
    length: PositiveInt
    step: PositiveInt
    scaler: LayerScalerChoice = LayerScalerChoice.NoScaler
    audio_signal: Optional[bool] = True
    chroma_stft: Optional[bool] = False
    mfcc: Optional[bool] = False
    rms: Optional[bool] = False
    spectral_centroid: Optional[bool] = False
    spectral_bandwidth: Optional[bool] = False
    spectral_rolloff: Optional[bool] = False
    zero_crossing_rate: Optional[bool] = False


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
    mask_range: PositiveInt
    mask_assignment: MasksSegmentationList


class LayerOutputTypeTextSegmentationData(BaseMixinData):
    open_tags: Optional[str]
    close_tags: Optional[str]


class LayerOutputTypeRegressionData(BaseMixinData):
    y_col: Optional[PositiveInt]


class LayerOutputTypeTimeseriesData(BaseMixinData):
    length: PositiveInt
    y_cols: Optional[PositiveInt]
    scaler: LayerScalerChoice = LayerScalerChoice.NoScaler
    task_type: LayerTaskTypeChoice = LayerTaskTypeChoice.timeseries


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
