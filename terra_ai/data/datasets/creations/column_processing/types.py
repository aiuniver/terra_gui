from typing import Optional, List
from pydantic import validator
from pydantic.types import PositiveInt, PositiveFloat
from pydantic.color import Color

# from terra_ai.data.datasets.creations.layers.image_augmentation import AugmentationData
from terra_ai.data.types import ConstrainedIntValueGe0
from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.extra import (
    LayerNetChoice,
    LayerScalerImageChoice,
    LayerScalerAudioChoice,
    LayerScalerVideoChoice,
    LayerScalerRegressionChoice,
    LayerScalerTimeseriesChoice,
    LayerScalerDefaultChoice,
    LayerTextModeChoice,
    LayerPrepareMethodChoice,
    LayerAudioModeChoice,
    LayerAudioFillModeChoice,
    LayerAudioParameterChoice,
    LayerAudioResampleChoice,
    LayerVideoFillModeChoice,
    LayerVideoFrameModeChoice,
    LayerVideoModeChoice,
    LayerTypeProcessingClassificationChoice,
    LayerImageFrameModeChoice,
    LayerTypeProcessingClassificationChoice, LayerImageFrameModeChoice, LayerObjectDetectionModelChoice,
    LayerYoloChoice, LayerODDatasetTypeChoice, LayerTransformerMethodChoice,
)
from terra_ai.data.datasets.creations.layers.extra import MinMaxScalerData


class ParametersBaseData(BaseMixinData):
    pass


class ParametersImageData(ParametersBaseData, MinMaxScalerData):

    """
    Обработчик изображений.
    Inputs:
        width: int - ширина изображений (пикс)
        height: int - высота изображений (пикс)
        image_mode: str - режим обработки изображений. Варианты: 'stretch', 'fit', 'cut'
        net: str - режим обработки массивов. Варианты: 'convolutional', 'linear'
        scaler: str - тип скейлера. Варианты: 'no_scaler', 'min_max_scaler', 'terra_image_scaler'
    """

    width: PositiveInt
    height: PositiveInt
    image_mode: LayerImageFrameModeChoice = LayerImageFrameModeChoice.stretch
    net: LayerNetChoice = LayerNetChoice.convolutional
    scaler: LayerScalerImageChoice


class ParametersTextData(ParametersBaseData):

    """
    Обработчик текстовых данных.
    Inputs:
        filters: str - символы, подлежащие удалению. По умолчанию: –—!"#$%&()*+,-./:;<=>?@[\\]^«»№_`{|}~\t\n\xa0–\ufeff
        text_mode: str - режим обработки текстов. Варианты: 'completely', 'length_and_step'.
        max_words: int - \x1B[3mОПЦИОНАЛЬНО при text_mode==completely.\x1B[0m Максимальное количество слов, учитываемое
         из каждого текстового файла.
        length: int - \x1B[3mОПЦИОНАЛЬНО при text_mode==length_and_step.\x1B[0m Длина последовательности слов.
        step: int - \x1B[3mОПЦИОНАЛЬНО при text_mode==length_and_step.\x1B[0m Шаг последовательности слов.
        pymorphy: bool - перевод слов в инфинитив. По умолчанию: False
        prepare_method: str - режим формирования массивов. Варианты: 'embedding', 'bag_of_words', 'word_to_vec'
        word_to_vec_size: int - \x1B[3mОПЦИОНАЛЬНО при prepare_method==word_to_vec.\x1B[0m Размер Word2Vec пространства.
    """

    filters: str = '–—!"#$%&()*+,-./:;<=>?@[\\]^«»№_`{|}~\t\n\xa0–\ufeff'
    text_mode: LayerTextModeChoice
    max_words: Optional[PositiveInt]
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]
    pymorphy: bool = False
    prepare_method: LayerPrepareMethodChoice
    max_words_count: Optional[PositiveInt]
    word_to_vec_size: Optional[PositiveInt]
    transformer: LayerTransformerMethodChoice = LayerTransformerMethodChoice.none
    open_tags: Optional[str]
    close_tags: Optional[str]

    @validator("text_mode")
    def _validate_text_mode(cls, value: LayerTextModeChoice) -> LayerTextModeChoice:
        if value == LayerTextModeChoice.completely:
            cls.__fields__["max_words"].required = True
        elif value == LayerTextModeChoice.length_and_step:
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value

    @validator("prepare_method")
    def _validate_prepare_method(cls, value: bool) -> bool:
        if value in [
            LayerPrepareMethodChoice.embedding,
            LayerPrepareMethodChoice.bag_of_words,
        ]:
            cls.__fields__["max_words_count"].required = True
        elif value == LayerPrepareMethodChoice.word_to_vec:
            cls.__fields__["word_to_vec_size"].required = True
        return value


class ParametersAudioData(ParametersBaseData, MinMaxScalerData):

    """
    Обработчик аудиофайлов.
    Inputs:
        sample_rate: int - частота дискретизации при открытии файлов
        audio_mode: str - режим обработки аудиофайлов. Варианты: 'completely', 'length_and_step'
        max_seconds: int - \x1B[3mОПЦИОНАЛЬНО при audio_mode==completely.\x1B[0m Максимальное количество секунд,
        учитываемое из каждого аудиофайла
        length: int - \x1B[3mОПЦИОНАЛЬНО при audio_mode==length_and_step.\x1B[0m Длина последовательности аудиоряда (сек)
        step: int - \x1B[3mОПЦИОНАЛЬНО при audio_mode==length_and_step.\x1B[0m Шаг последовательности аудиоряда (сек)
        fill_mode: str - заполнение при недостаточной длине аудиофайла. Варианты: 'last_millisecond', 'loop'
        parameter: str - параметры обработки. Варианты: 'audio_signal', 'chroma_stft', 'mfcc', 'rms',
        'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'zero_crossing_rate'
        resample: str - режим ресемпла аудиофайлов (влияет на скорость обработки файлов).
        Варианты: 'kaiser_best', 'kaiser_fast', 'scipy'
        scaler: str - тип скейлера. Варианты: 'no_scaler', 'min_max_scaler', 'standard_scaler'
    """

    sample_rate: PositiveInt = 22050
    audio_mode: LayerAudioModeChoice = LayerAudioModeChoice.completely
    max_seconds: Optional[PositiveFloat]
    length: Optional[PositiveFloat]
    step: Optional[PositiveFloat]
    fill_mode: LayerAudioFillModeChoice = LayerAudioFillModeChoice.last_millisecond
    parameter: LayerAudioParameterChoice
    resample: LayerAudioResampleChoice = LayerAudioResampleChoice.kaiser_best
    scaler: LayerScalerAudioChoice

    @validator("audio_mode")
    def _validate_audio_mode(cls, value: LayerAudioModeChoice) -> LayerAudioModeChoice:
        if value == LayerAudioModeChoice.completely:
            cls.__fields__["max_seconds"].required = True
        elif value == LayerAudioModeChoice.length_and_step:
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value


class ParametersVideoData(ParametersBaseData, MinMaxScalerData):

    """
    Обработчик видеофайлов.
    Inputs:
        width: int - ширина видеокадра (пикс)
        height: int - высота видеокадра (пикс)
        frame_mode: str - режим обработки кадров. Варианты: 'stretch', 'fit', 'cut'
        video_mode: str - режим обработки кадров. Варианты: 'completely', 'length_and_step'
        max_frames: int - \x1B[3mОПЦИОНАЛЬНО при video_mode==completely.\x1B[0m Количество кадров для каждого видеофайла
        length: int - \x1B[3mОПЦИОНАЛЬНО при video_mode==length_and_step.\x1B[0m Длина окна выборки (кол-во кадров)
        step: int - \x1B[3mОПЦИОНАЛЬНО при video_mode==length_and_step.\x1B[0m Шаг окна выборки (кол-во кадров)
        fill_mode: str - режим обработки аудиофайлов. Варианты: 'last_frames', 'loop', 'average_value'
        scaler: str - тип скейлера. Варианты: 'no_scaler', 'min_max_scaler'
    """

    width: PositiveInt
    height: PositiveInt
    fill_mode: LayerVideoFillModeChoice = LayerVideoFillModeChoice.last_frames
    frame_mode: LayerVideoFrameModeChoice = LayerVideoFrameModeChoice.fit
    video_mode: LayerVideoModeChoice = LayerVideoModeChoice.completely
    max_frames: Optional[PositiveInt]
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]
    scaler: LayerScalerVideoChoice

    @validator("video_mode")
    def _validate_video_mode(cls, value: LayerVideoModeChoice) -> LayerVideoModeChoice:
        if value == LayerVideoModeChoice.completely:
            cls.__fields__["max_frames"].required = True
        elif value == LayerVideoModeChoice.length_and_step:
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value


class ParametersScalerData(ParametersBaseData, MinMaxScalerData):

    """
    Обработчик числовых значений.
    Inputs:
        scaler: str - тип скейлера. Варианты: 'standard_scaler', 'min_max_scaler'
    """

    scaler: LayerScalerDefaultChoice
    length: int = 0
    depth: int = 0
    step: int = 1

    def __init__(self, **data):
        try:
            data.pop("length")
            data.pop("depth")
            data.pop("step")
        except KeyError:
            pass
        super().__init__(**data)


class ParametersClassificationData(ParametersBaseData):

    """
    Обработчик типа задачи "классификация".
    Inputs:
        type_processing: str - режим обработки кадров. Варианты: 'categorical', 'ranges'. По умолчанию: 'categorical'
        ranges: int - \x1B[3mОПЦИОНАЛЬНО при type_processing==ranges.\x1B[0m Диапазоны разбивки на классы.
    """

    one_hot_encoding: bool = True
    type_processing: Optional[
        LayerTypeProcessingClassificationChoice
    ] = LayerTypeProcessingClassificationChoice.categorical
    ranges: Optional[str]
    length: int = 0
    depth: int = 0
    step: int = 1

    @validator("type_processing")
    def _validate_type_processing(
            cls, value: LayerTypeProcessingClassificationChoice
    ) -> LayerTypeProcessingClassificationChoice:
        if value == LayerTypeProcessingClassificationChoice.ranges:
            cls.__fields__["ranges"].required = True
        return value


class ParametersRegressionData(ParametersBaseData, MinMaxScalerData):

    """
    Обработчик типа задачи "регрессия".
    Inputs:
        scaler: str - тип скейлера. Варианты: 'standard_scaler', 'min_max_scaler'
    """

    scaler: LayerScalerRegressionChoice


class ParametersSegmentationData(ParametersBaseData):

    """
    Обработчик типа задачи "сегментация".
    Inputs:
        mask_range: int - диапазон для каждого из RGB каналов.
        classes_names: list - названия классов
        classes_colors: list - цвета классов в формате RGB
    """

    mask_range: ConstrainedIntValueGe0
    classes_names: List[str]
    classes_colors: List[Color]
    height: Optional[PositiveInt]
    width: Optional[PositiveInt]

    @validator("width", "height", pre=True)
    def _validate_size(cls, value: PositiveInt) -> PositiveInt:
        if not value:
            value = None
        return value


class ParametersTextSegmentationData(ParametersBaseData):

    """
    Обработчик типа задачи "сегментация текстов".
    Inputs:
        open_tags: str - открывающие теги (через пробел)
        close_tags: str - закрывающие теги (через пробел)
    """

    open_tags: str
    close_tags: str

    sources_paths: Optional[list]
    filters: Optional[str]
    text_mode: Optional[LayerTextModeChoice]
    max_words: Optional[PositiveInt]
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]
    prepare_method: Optional[LayerPrepareMethodChoice]
    max_words_count: Optional[PositiveInt]

    # def __init__(self\, **data):
    #     data.update({"cols_names": None})
    #     super().__init__(**data)

    # @validator("text_mode")
    # def _validate_text_mode(cls, value: LayerTextModeChoice) -> LayerTextModeChoice:
    #     if value == LayerTextModeChoice.completely:
    #         cls.__fields__["max_words"].required = True
    #     elif value == LayerTextModeChoice.length_and_step:
    #         cls.__fields__["length"].required = True
    #         cls.__fields__["step"].required = True
    #     return value


class ParametersTimeseriesData(ParametersBaseData, MinMaxScalerData):

    """
    Обработчик видеофайлов.
    Inputs:
        length: int - ***
        step: int - ***
        trend: bool - ***
        trend_limit: bool - \x1B[3mОПЦИОНАЛЬНО при trend==True.\x1B[0m ***
        depth: int - \x1B[3mОПЦИОНАЛЬНО при ***==***.\x1B[0m ***
        scaler: str - тип скейлера. Варианты: 'no_scaler', 'standard_scaler', 'min_max_scaler'
    """

    length: PositiveInt
    step: PositiveInt
    trend: bool
    trend_limit: Optional[str]
    depth: Optional[PositiveInt]
    scaler: Optional[LayerScalerTimeseriesChoice]

    def __init__(self, **data):
        try:
            if data.get("trend"):
                data.pop("depth")
                data.pop("scaler")
            else:
                data.pop("trend_limit")
        except KeyError:
            pass
        super().__init__(**data)

    @validator("trend")
    def _validate_trend(cls, value: bool) -> bool:
        if value:
            cls.__fields__["trend_limit"].required = True
        else:
            cls.__fields__["depth"].required = True
            cls.__fields__["scaler"].required = True
        return value


class ParametersObjectDetectionData(ParametersBaseData):

    """
    Обработчик типа задачи обнаружения объектов.
    """

    model: LayerObjectDetectionModelChoice = LayerObjectDetectionModelChoice.yolo
    yolo: LayerYoloChoice = LayerYoloChoice.v4
    classes_names: Optional[list]
    num_classes: Optional[PositiveInt]
    model_type: LayerODDatasetTypeChoice = LayerODDatasetTypeChoice.Yolo_terra
    frame_mode: LayerImageFrameModeChoice = LayerImageFrameModeChoice.stretch
    # put: Optional[PositiveInt]
    # cols_names: Optional[str]

    # def __init__(self, **data):
    #     data.update({"cols_names": None})
    #     super().__init__(**data)


class ParametersImageGANData(ParametersBaseData):

    """
    Обработчик типа задачи "ImageGAN".
    """

    pass


class ParametersImageCGANData(ParametersBaseData):

    """
    Обработчик типа задачи "ImageCGAN".
    """

    pass


class ParametersTextToImageGANData(ParametersBaseData):

    """
    Обработчик типа задачи "CGAN".
    """

    pass


class ParametersTransformerData(ParametersBaseData):

    """
    Обработчик типа задачи "Transformer".
    """

    pass


class ParametersGeneratorData(ParametersBaseData):
    shape: tuple


class ParametersDiscriminatorData(ParametersBaseData):
    shape: tuple


class ParametersNoiseData(ParametersBaseData):
    shape: tuple


class ParametersTrackerData(ParametersBaseData):
    pass


class ParametersText2SpeechData(ParametersBaseData):
    pass


class ParametersSpeech2TextData(ParametersBaseData):
    pass
