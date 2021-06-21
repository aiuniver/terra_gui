from datetime import datetime
from typing import Optional, Union
from pathlib import PosixPath
from pydantic import validator, FilePath, DirectoryPath, HttpUrl

from . import mixins, extra, validators


class DatasetTagsData(mixins.AliasMixinData):
    name: str


class DatasetTagsListData(mixins.UniqueListMixin):
    class Meta:
        source = DatasetTagsData
        identifier = "alias"


class DatasetData(mixins.AliasMixinData):
    name: str
    size: Optional[extra.SizeData]
    date: Optional[datetime]
    tags: DatasetTagsListData = DatasetTagsListData()


class DatasetsList(mixins.UniqueListMixin):
    class Meta:
        source = DatasetData
        identifier = "alias"


class DatasetSourceData(mixins.BaseMixinData):
    mode: extra.DatasetSourceModeChoice
    value: Union[FilePath, HttpUrl]

    @validator("value", allow_reuse=True)
    def _validate_mode_value(
        cls, value: Union[PosixPath, HttpUrl], **kwargs
    ) -> Union[PosixPath, HttpUrl]:
        if isinstance(value, PosixPath):
            split_value = str(value).split(".")
            if len(split_value) < 2 or split_value[-1].lower() != "zip":
                raise ValueError(f"{value}: Value must be a zip-file")
        mode = kwargs.get("values", {}).get("mode")
        if mode == extra.DatasetSourceModeChoice.google_drive:
            if not isinstance(value, PosixPath):
                raise ValueError(f'{value}: Value must be a "{PosixPath}"')
        if mode == extra.DatasetSourceModeChoice.url:
            if not isinstance(value, HttpUrl):
                raise ValueError(f'{value}: Value must be a "{HttpUrl}"')
        return value


class DatasetCreateInputsImagesParametersData(mixins.BaseMixinData):
    folder_path: Optional[DirectoryPath]
    width: int
    height: int
    net: extra.LayerNetChoice = extra.LayerNetChoice.Convolutional
    scaler: extra.LayerScalerChoice = extra.LayerScalerChoice.NoScaler

    _validate_positive_integer = validator("width", "height", allow_reuse=True)(
        validators.validate_positive_integer
    )


class DatasetCreateInputsTextParametersData(mixins.BaseMixinData):
    folder_path: Optional[DirectoryPath]
    delete_symbols: Optional[str]
    x_len: int
    step: int
    max_words_count: int
    pymorphy: Optional[bool] = False
    prepare_method: extra.LayerPrepareMethodChoice = (
        extra.LayerPrepareMethodChoice.embedding
    )
    word_to_vec_size: Optional[int]

    _validate_positive_integer = validator(
        "x_len", "step", "max_words_count", "word_to_vec_size", allow_reuse=True
    )(validators.validate_positive_integer)

    @validator("prepare_method", allow_reuse=True)
    def _validate_prepare_method(
        cls, value: extra.LayerPrepareMethodChoice
    ) -> extra.LayerPrepareMethodChoice:
        if value == extra.LayerPrepareMethodChoice.word_to_vec:
            cls.__fields__["word_to_vec_size"].required = True
        return value


class DatasetCreateInputsAudioParametersData(mixins.BaseMixinData):
    folder_path: Optional[DirectoryPath]
    length: int
    step: int
    scaler: extra.LayerScalerChoice = extra.LayerScalerChoice.NoScaler
    audio_signal: Optional[bool] = True
    chroma_stft: Optional[bool] = False
    mfcc: Optional[bool] = False
    rms: Optional[bool] = False
    spectral_centroid: Optional[bool] = False
    spectral_bandwidth: Optional[bool] = False
    spectral_rolloff: Optional[bool] = False
    zero_crossing_rate: Optional[bool] = False

    _validate_positive_integer = validator("length", "step", allow_reuse=True)(
        validators.validate_positive_integer
    )


class DatasetCreateInputsDataframeParametersData(mixins.BaseMixinData):
    file_path: Optional[FilePath]
    separator: Optional[str]
    encoding: str = "utf-8"
    x_cols: Optional[int]
    scaler: extra.LayerScalerChoice = extra.LayerScalerChoice.NoScaler

    _validate_positive_integer = validator("x_cols", allow_reuse=True)(
        validators.validate_positive_integer
    )


class DatasetCreateInputsData(mixins.AliasMixinData):
    name: str
    type: extra.InputTypeChoice
    parameters: Optional[
        Union[
            DatasetCreateInputsImagesParametersData,
            DatasetCreateInputsTextParametersData,
            DatasetCreateInputsAudioParametersData,
            DatasetCreateInputsDataframeParametersData,
        ]
    ]

    @validator("type", allow_reuse=True, pre=True)
    def _validate_type(cls, value: extra.InputTypeChoice) -> extra.InputTypeChoice:
        __type = Union[
            DatasetCreateInputsImagesParametersData,
            DatasetCreateInputsTextParametersData,
            DatasetCreateInputsAudioParametersData,
            DatasetCreateInputsDataframeParametersData,
        ]
        if value == extra.InputTypeChoice.images:
            __type = DatasetCreateInputsImagesParametersData
        if value == extra.InputTypeChoice.text:
            __type = DatasetCreateInputsTextParametersData
        if value == extra.InputTypeChoice.audio:
            __type = DatasetCreateInputsAudioParametersData
        if value == extra.InputTypeChoice.dataframe:
            __type = DatasetCreateInputsDataframeParametersData
        cls.__fields__["parameters"].type_ = __type
        cls.__fields__["parameters"].required = True
        return value

    @validator("parameters", allow_reuse=True, pre=True)
    def _validate_parameters(
        cls,
        value: Union[
            DatasetCreateInputsImagesParametersData,
            DatasetCreateInputsTextParametersData,
            DatasetCreateInputsAudioParametersData,
            DatasetCreateInputsDataframeParametersData,
        ],
        **kwargs,
    ) -> Union[
        DatasetCreateInputsImagesParametersData,
        DatasetCreateInputsTextParametersData,
        DatasetCreateInputsAudioParametersData,
        DatasetCreateInputsDataframeParametersData,
    ]:
        return kwargs.get("field").type_(**value)


class DatasetCreateInputsList(mixins.UniqueListMixin):
    class Meta:
        source = DatasetCreateInputsData
        identifier = "alias"


class DatasetCreateData(mixins.BaseMixinData):
    inputs: DatasetCreateInputsList = DatasetCreateInputsList()
