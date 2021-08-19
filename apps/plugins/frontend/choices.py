from enum import Enum


class LayerInputTypeChoice(str, Enum):
    Image = "Изображения"
    Text = "Текст"
    Audio = "Аудио"
    Video = "Видео"
    Dataframe = "Таблицы"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerInputTypeChoice))


class LayerOutputTypeChoice(str, Enum):
    Image = "Изображения"
    Text = "Текст"
    Audio = "Аудио"
    Classification = "Классификация"
    Segmentation = "Сегментация"
    TextSegmentation = "Сегментация текстов"
    Regression = "Регрессия"
    Timeseries = "Временные ряды"
    ObjectDetection = "Обнаружение объектов"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerOutputTypeChoice))


class LayerNetChoice(str, Enum):
    convolutional = "Сверточная"
    linear = "Линейная"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerNetChoice))


class LayerScalerChoice(str, Enum):
    no_scaler = "Не применять"
    min_max_scaler = "MinMaxScaler"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerScalerChoice))


class LayerAudioModeChoice(str, Enum):
    completely = "Целиком"
    length_and_step = "По длине и шагу"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerAudioModeChoice))


class LayerAudioParameterChoice(str, Enum):
    audio_signal = "Audio signal"
    zero_crossing_rate = "Zero-crossing rate"
    rms = "RMS"
    spectral_rolloff = "Spectral roll-off"
    spectral_bandwidth = "Spectral bandwidth"
    spectral_centroid = "Spectral centroid"
    mfcc = "MFCC"
    chroma_stft = "Chroma STFT"

    @staticmethod
    def items_tuple() -> list:
        return list(
            map(lambda item: (item.name, item.value), LayerAudioParameterChoice)
        )


class LayerTextModeChoice(str, Enum):
    completely = "Целиком"
    length_and_step = "По длине и шагу"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerTextModeChoice))


class LayerVideoFillModeChoice(str, Enum):
    black_frames = "Черными кадрами"
    average_value = "Средним значением"
    last_frames = "Последними кадрами"


class LayerVideoFrameModeChoice(str, Enum):
    keep_proportions = "Сохранить пропорции"
    stretch = "Растянуть"


class LayerVideoModeChoice(str, Enum):
    completely = "Целиком"
    length_and_step = "По длине и шагу"


class LayerPrepareMethodChoice(str, Enum):
    embedding = "Embedding"
    bag_of_words = "Bag of words"
    word_to_vec = "Word2Vec"

    @staticmethod
    def items_tuple() -> list:
        return list(map(lambda item: (item.name, item.value), LayerPrepareMethodChoice))


class LayerDataframeAlignBaseMethodChoice(str, Enum):
    pad_sequences = "Pad sequences"
    xlen_step = "XLen step"


class LayerDefineClassesChoice(str, Enum):
    handmade = "Ручной ввод"
    autosearch = "Автоматический поиск"
    annotation = "Файл аннотации"


class LayerYoloVersionChoice(str, Enum):
    v3 = "V3"
    v4 = "V4"
