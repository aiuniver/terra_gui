from apps.plugins.frontend.choices import (
    LayerScalerDefaultChoice,
    LayerScalerImageChoice,
    LayerScalerVideoChoice,
    LayerAudioModeChoice,
    LayerAudioFillModeChoice,
    LayerAudioParameterChoice,
    LayerAudioResampleChoice,
    LayerTypeProcessingChoice,
    LayerNetChoice,
    LayerImageModeChoice,
    LayerYoloVersionChoice,
    LayerODDatasetTypeChoice,
    LayerDefineClassesChoice,
    LayerTextModeChoice,
    LayerPrepareMethodChoice,
    LayerVideoFillModeChoice,
    LayerVideoFrameModeChoice,
    LayerVideoModeChoice,
)


MinScalerField = {
    "type": "number",
    "label": "Минимальный скейлер",
    "name": "min_scaler",
    "parse": "min_scaler",
    "value": 0,
}

MaxScalerField = {
    "type": "number",
    "label": "Максимальный скейлер",
    "name": "max_scaler",
    "parse": "max_scaler",
    "value": 1,
}

ScalerDefaultField = {
    "type": "select",
    "label": "Скейлер",
    "name": "scaler",
    "parse": "scaler",
    "value": LayerScalerDefaultChoice.min_max_scaler.name,
    "list": LayerScalerDefaultChoice.values(),
    "fields": {
        LayerScalerDefaultChoice.min_max_scaler.name: [
            MinScalerField,
            MaxScalerField,
        ]
    },
}

ScalerImageField = {
    "type": "select",
    "label": "Скейлер",
    "name": "scaler",
    "parse": "scaler",
    "value": LayerScalerImageChoice.min_max_scaler.name,
    "list": LayerScalerImageChoice.values(),
    "fields": {
        LayerScalerImageChoice.min_max_scaler.name: [
            MinScalerField,
            MaxScalerField,
        ],
        LayerScalerImageChoice.terra_image_scaler: [
            MinScalerField,
            MaxScalerField,
        ],
    },
}

ScalerVideoField = {
    "type": "select",
    "label": "Скейлер",
    "name": "scaler",
    "parse": "scaler",
    "value": LayerScalerVideoChoice.min_max_scaler.name,
    "list": LayerScalerVideoChoice.values(),
    "fields": {
        LayerScalerVideoChoice.min_max_scaler.name: [
            MinScalerField,
            MaxScalerField,
        ],
    },
}

SampleRateField = {
    "type": "number",
    "label": "Частота дискретизации",
    "name": "sample_rate",
    "parse": "sample_rate",
    "value": 22050,
}

AudioMaxSecondsField = {
    "type": "number",
    "label": "Длина аудио (сек.)",
    "name": "max_seconds",
    "parse": "max_seconds",
}

AudioLengthField = {
    "type": "number",
    "label": "Длина (сек.)",
    "name": "length",
    "parse": "length",
}

AudioStepField = {
    "type": "number",
    "label": "Шаг (сек.)",
    "name": "step",
    "parse": "step",
}

AudioModeField = {
    "type": "select",
    "label": "Формат аудио",
    "name": "audio_mode",
    "parse": "audio_mode",
    "value": LayerAudioModeChoice.completely.name,
    "list": LayerAudioModeChoice.values(),
    "fields": {
        LayerAudioModeChoice.completely.name: [
            AudioMaxSecondsField,
        ],
        LayerAudioModeChoice.length_and_step.name: [
            AudioLengthField,
            AudioStepField,
        ],
    },
}

AudioFillModeField = {
    "type": "select",
    "label": "Заполнение недостающих аудио-дорожек",
    "name": "fill_mode",
    "parse": "fill_mode",
    "value": LayerAudioFillModeChoice.last_millisecond.name,
    "list": LayerAudioFillModeChoice.values(),
}

AudioParameterField = {
    "type": "select",
    "label": "Параметр",
    "name": "parameter",
    "parse": "parameter",
    "value": LayerAudioParameterChoice.audio_signal.name,
    "list": LayerAudioParameterChoice.values(),
}

AudioResampleField = {
    "type": "select",
    "label": "Ресемпл",
    "name": "resample",
    "parse": "resample",
    "value": LayerAudioResampleChoice.kaiser_best.name,
    "list": LayerAudioResampleChoice.values(),
}

OneHotEncodingField = {
    "type": "checkbox",
    "label": "One-Hot encoding",
    "name": "one_hot_encoding",
    "parse": "one_hot_encoding",
    "value": True,
}

TypeProcessingRangeField = {
    "type": "text",
    "label": "Диапазоны/число диапазонов",
    "name": "ranges",
    "parse": "ranges",
}

TypeProcessingField = {
    "type": "select",
    "label": "Тип предобработки",
    "name": "type_processing",
    "parse": "type_processing",
    "value": LayerTypeProcessingChoice.categorical.name,
    "list": LayerTypeProcessingChoice.values(),
    "fields": {
        LayerTypeProcessingChoice.ranges.name: [
            TypeProcessingRangeField,
        ]
    },
}

WidthField = {
    "type": "number",
    "label": "Ширина",
    "name": "width",
    "parse": "width",
}

HeightField = {
    "type": "number",
    "label": "Высота",
    "name": "height",
    "parse": "height",
}

NetField = {
    "type": "select",
    "label": "Сеть",
    "name": "net",
    "parse": "net",
    "value": LayerNetChoice.convolutional.name,
    "list": LayerNetChoice.values(),
}

ImageModeField = {
    "type": "select",
    "label": "Режим изображения",
    "name": "image_mode",
    "parse": "image_mode",
    "value": LayerImageModeChoice.stretch.name,
    "list": LayerImageModeChoice.values(),
}

YoloVersionField = {
    "type": "select",
    "label": "Версия Yolo",
    "name": "yolo",
    "parse": "yolo",
    "value": LayerYoloVersionChoice.v4.name,
    "list": LayerYoloVersionChoice.values(),
}

ODModelTypeField = {
    "type": "select",
    "label": "Тип аннотации",
    "name": "model_type",
    "parse": "model_type",
    "value": LayerODDatasetTypeChoice.Yolo_terra.name,
    "list": LayerODDatasetTypeChoice.values(),
}

MaskRangeField = {
    "type": "number",
    "label": "Диапазон каналов",
    "name": "mask_range",
    "parse": "mask_range",
}

ClassesNamesField = {
    "type": "segmentation_manual",
    "label": "Название класса",
    "name": "classes_names",
    "parse": "classes_names[]",
}

ClassesSearchField = {
    "type": "segmentation_search",
    "label": "Найти",
    "name": "search",
    "parse": "search",
}

ClassesAnnotationField = {
    "type": "segmentation_annotation",
    "label": "Выберите файл",
    "name": "annotation",
    "parse": "annotation",
}

ClassesField = {
    "type": "select",
    "label": "Ввод данных",
    "name": "classes",
    "parse": "classes",
    "value": LayerDefineClassesChoice.handmade.name,
    "list": LayerDefineClassesChoice.values(),
    "fields": {
        LayerDefineClassesChoice.handmade.name: [
            ClassesNamesField,
        ],
        LayerDefineClassesChoice.autosearch.name: [
            ClassesSearchField,
        ],
        LayerDefineClassesChoice.annotation.name: [
            ClassesAnnotationField,
        ],
    },
}

TextMaxWordsField = {
    "type": "number",
    "label": "Количество слов",
    "name": "max_words",
    "parse": "max_words",
}

TextLengthField = {
    "type": "number",
    "label": "Длина",
    "name": "length",
    "parse": "length",
}

TextStepField = {
    "type": "number",
    "label": "Шаг",
    "name": "step",
    "parse": "step",
}

TextModeField = {
    "type": "select",
    "label": "Формат текстов",
    "name": "text_mode",
    "parse": "text_mode",
    "value": LayerTextModeChoice.completely.name,
    "list": LayerTextModeChoice.values(),
    "fields": {
        LayerTextModeChoice.completely.name: [
            TextMaxWordsField,
        ],
        LayerTextModeChoice.length_and_step.name: [
            TextLengthField,
            TextStepField,
        ],
    },
}

TextMaxWordsCountField = {
    "type": "number",
    "label": "Максимальное количество слов",
    "name": "max_words_count",
    "parse": "max_words_count",
    "value": 20000,
}

TextWordToVecSizeField = {
    "type": "number",
    "label": "Размер Word2Vec пространства",
    "name": "word_to_vec_size",
    "parse": "word_to_vec_size",
    "value": 200,
}

TextPrepareMethodField = {
    "type": "select",
    "label": "Метод подготовки",
    "name": "prepare_method",
    "parse": "prepare_method",
    "value": LayerPrepareMethodChoice.embedding.name,
    "list": LayerPrepareMethodChoice.values(),
    "fields": {
        LayerPrepareMethodChoice.embedding.name: [
            TextMaxWordsCountField,
        ],
        LayerPrepareMethodChoice.bag_of_words.name: [
            TextMaxWordsCountField,
        ],
        LayerPrepareMethodChoice.word_to_vec.name: [
            TextWordToVecSizeField,
        ],
    },
}

TextPymorphyField = {
    "type": "checkbox",
    "label": "Pymorphy",
    "name": "pymorphy",
    "parse": "pymorphy",
    "value": False,
}

TextFiltersField = {
    "type": "text",
    "label": "Фильтры",
    "name": "filters",
    "parse": "filters",
    "value": '–—!"#$%&()*+,-./:;<=>?@[\\]^«»№_`{|}~\t\n\xa0–\ufeff',
}

TextOpenTagsField = {
    "type": "text",
    "label": "Открывающие теги (через пробел)",
    "name": "open_tags",
    "parse": "open_tags",
}

TextCloseTagsField = {
    "type": "text",
    "label": "Закрывающие теги (через пробел)",
    "name": "close_tags",
    "parse": "close_tags",
}

TimeseriesLengthField = {
    "type": "number",
    "label": "Длина",
    "name": "length",
    "parse": "length",
}

TimeseriesStepField = {
    "type": "number",
    "label": "Шаг",
    "name": "step",
    "parse": "step",
}

TimeseriesTrendLimitField = {
    "type": "text",
    "label": "Отклонение нулевого тренда",
    "name": "trend_limit",
    "parse": "trend_limit",
}

TimeseriesDepthField = {
    "type": "number",
    "label": "Глубина предсказания",
    "name": "depth",
    "parse": "depth",
}

TimeseriesTrendField = {
    "type": "checkbox",
    "label": "Предсказывать тренд",
    "name": "trend",
    "parse": "trend",
    "value": False,
    "fields": {
        "true": [
            TimeseriesTrendLimitField,
        ],
        "false": [
            TimeseriesDepthField,
            ScalerDefaultField,
        ],
    },
}

VideoWidthField = {
    "type": "number",
    "label": "Ширина кадра",
    "name": "width",
    "parse": "width",
}

VideoHeightField = {
    "type": "number",
    "label": "Высота кадра",
    "name": "height",
    "parse": "height",
}

VideoFillModeField = {
    "type": "select",
    "label": "Заполнение недостающих кадров",
    "name": "fill_mode",
    "parse": "fill_mode",
    "value": LayerVideoFillModeChoice.average_value.name,
    "list": LayerVideoFillModeChoice.values(),
}

VideoFrameModeField = {
    "type": "select",
    "label": "Обработка кадров",
    "name": "frame_mode",
    "parse": "frame_mode",
    "value": LayerVideoFrameModeChoice.fit.name,
    "list": LayerVideoFrameModeChoice.values(),
}

VideoMaxFramesField = {
    "type": "number",
    "label": "Количество кадров",
    "name": "max_frames",
    "parse": "max_frames",
}

VideoLengthField = {
    "type": "number",
    "label": "Длина (кадров)",
    "name": "length",
    "parse": "length",
}

VideoStepField = {
    "type": "number",
    "label": "Шаг (кадров)",
    "name": "step",
    "parse": "step",
}

VideoModeField = {
    "type": "select",
    "label": "Формат видео",
    "name": "video_mode",
    "parse": "video_mode",
    "value": LayerVideoModeChoice.completely.name,
    "list": LayerVideoModeChoice.values(),
    "fields": {
        LayerVideoModeChoice.completely.name: [
            VideoMaxFramesField,
        ],
        LayerVideoModeChoice.length_and_step.name: [
            VideoLengthField,
            VideoStepField,
        ],
    },
}
