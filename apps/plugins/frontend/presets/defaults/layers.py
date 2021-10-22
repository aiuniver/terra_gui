from ...choices import (
    LayerNetChoice,
    LayerScalerImageChoice,
    LayerScalerAudioChoice,
    LayerAudioModeChoice,
    LayerAudioFillModeChoice,
    LayerAudioParameterChoice,
    LayerAudioResampleChoice,
    LayerTextModeChoice,
    LayerPrepareMethodChoice,
)


SourcesPaths = {
    "type": "multiselect_sources_paths",
    "label": "Выберите путь",
    "name": "sources_paths",
    "parse": "sources_paths",
    "value": [],
}


LayerImageDefaults = [
    SourcesPaths,
    {
        "type": "number",
        "label": "Ширина",
        "name": "width",
        "parse": "width",
    },
    {
        "type": "number",
        "label": "Высота",
        "name": "height",
        "parse": "height",
    },
    {
        "type": "select",
        "label": "Сеть",
        "name": "net",
        "parse": "net",
        "value": LayerNetChoice.convolutional.name,
        "list": list(
            map(
                lambda item: {
                    "value": item.name,
                    "label": item.value,
                },
                list(LayerNetChoice),
            )
        ),
    },
    {
        "type": "select",
        "label": "Скейлер",
        "name": "scaler",
        "parse": "scaler",
        "value": LayerScalerImageChoice.min_max_scaler.name,
        "list": list(
            map(
                lambda item: {
                    "value": item.name,
                    "label": item.value,
                },
                list(LayerScalerImageChoice),
            )
        ),
        "fields": {
            "min_max_scaler": [
                {
                    "type": "number",
                    "label": "Минимальный скейлер",
                    "name": "min_scaler",
                    "parse": "min_scaler",
                    "value": 0,
                },
                {
                    "type": "number",
                    "label": "Максимальный скейлер",
                    "name": "max_scaler",
                    "parse": "max_scaler",
                    "value": 1,
                },
            ],
            "terra_image_scaler": [
                {
                    "type": "number",
                    "label": "Минимальный скейлер",
                    "name": "min_scaler",
                    "parse": "min_scaler",
                    "value": 0,
                },
                {
                    "type": "number",
                    "label": "Максимальный скейлер",
                    "name": "max_scaler",
                    "parse": "max_scaler",
                    "value": 1,
                },
            ],
        },
    },
]


LayerTextDefaults = [
    SourcesPaths,
    {
        "type": "text",
        "label": "Фильтры",
        "name": "filters",
        "parse": "filters",
        "value": '–—!"#$%&()*+,-./:;<=>?@[\\]^«»№_`{|}~\t\n\xa0–\ufeff',
    },
    {
        "type": "select",
        "label": "Формат текстов",
        "name": "text_mode",
        "parse": "text_mode",
        "value": LayerTextModeChoice.completely.name,
        "list": list(
            map(
                lambda item: {
                    "value": item.name,
                    "label": item.value,
                },
                list(LayerTextModeChoice),
            )
        ),
        "fields": {
            "completely": [
                {
                    "type": "number",
                    "label": "Количество слов",
                    "name": "max_words",
                    "parse": "max_words",
                },
            ],
            "length_and_step": [
                {
                    "type": "number",
                    "label": "Длина",
                    "name": "length",
                    "parse": "length",
                },
                {
                    "type": "number",
                    "label": "Шаг",
                    "name": "step",
                    "parse": "step",
                },
            ],
        },
    },
    {
        "type": "checkbox",
        "label": "Pymorphy",
        "name": "pymorphy",
        "parse": "pymorphy",
        "value": False,
    },
    {
        "type": "select",
        "label": "Метод подготовки",
        "name": "prepare_method",
        "parse": "prepare_method",
        "value": LayerPrepareMethodChoice.embedding.name,
        "list": list(
            map(
                lambda item: {
                    "value": item.name,
                    "label": item.value,
                },
                list(LayerPrepareMethodChoice),
            )
        ),
        "fields": {
            "embedding": [
                {
                    "type": "number",
                    "label": "Максимальное количество слов",
                    "name": "max_words_count",
                    "parse": "max_words_count",
                    "value": 20000,
                },
            ],
            "bag_of_words": [
                {
                    "type": "number",
                    "label": "Максимальное количество слов",
                    "name": "max_words_count",
                    "parse": "max_words_count",
                    "value": 20000,
                },
            ],
            "word_to_vec": [
                {
                    "type": "number",
                    "label": "Размер Word2Vec пространства",
                    "name": "word_to_vec_size",
                    "parse": "word_to_vec_size",
                    "value": 200,
                },
            ],
        },
    },
]


LayerAudioDefaults = [
    SourcesPaths,
    {
        "type": "number",
        "label": "Частота дискретизации",
        "name": "sample_rate",
        "parse": "sample_rate",
        "value": 22050,
    },
    {
        "type": "select",
        "label": "Формат аудио",
        "name": "audio_mode",
        "parse": "audio_mode",
        "value": LayerAudioModeChoice.completely.name,
        "list": list(
            map(
                lambda item: {
                    "value": item.name,
                    "label": item.value,
                },
                list(LayerAudioModeChoice),
            )
        ),
        "fields": {
            "completely": [
                {
                    "type": "number",
                    "label": "Длина аудио (сек.)",
                    "name": "max_seconds",
                    "parse": "max_seconds",
                },
            ],
            "length_and_step": [
                {
                    "type": "number",
                    "label": "Длина (сек.)",
                    "name": "length",
                    "parse": "length",
                },
                {
                    "type": "number",
                    "label": "Шаг (сек.)",
                    "name": "step",
                    "parse": "step",
                },
            ],
        },
    },
    {
        "type": "select",
        "label": "Заполнение недостающих аудио-дорожек",
        "name": "fill_mode",
        "parse": "fill_mode",
        "value": LayerAudioFillModeChoice.last_millisecond.name,
        "list": list(
            map(
                lambda item: {
                    "value": item.name,
                    "label": item.value,
                },
                list(LayerAudioFillModeChoice),
            )
        ),
    },
    {
        "type": "select",
        "label": "Параметр",
        "name": "parameter",
        "parse": "parameter",
        "value": LayerAudioParameterChoice.audio_signal.name,
        "list": list(
            map(
                lambda item: {
                    "value": item.name,
                    "label": item.value,
                },
                list(LayerAudioParameterChoice),
            )
        ),
    },
    {
        "type": "select",
        "label": "Ресемпл",
        "name": "resample",
        "parse": "resample",
        "value": LayerAudioResampleChoice.kaiser_best.name,
        "list": list(
            map(
                lambda item: {
                    "value": item.name,
                    "label": item.value,
                },
                list(LayerAudioResampleChoice),
            )
        ),
    },
    {
        "type": "select",
        "label": "Скейлер",
        "name": "scaler",
        "parse": "scaler",
        "value": LayerScalerAudioChoice.min_max_scaler.name,
        "list": list(
            map(
                lambda item: {
                    "value": item.name,
                    "label": item.value,
                },
                list(LayerScalerAudioChoice),
            )
        ),
        "fields": {
            "min_max_scaler": [
                {
                    "type": "number",
                    "label": "Минимальный скейлер",
                    "name": "min_scaler",
                    "parse": "min_scaler",
                    "value": 0,
                },
                {
                    "type": "number",
                    "label": "Максимальный скейлер",
                    "name": "max_scaler",
                    "parse": "max_scaler",
                    "value": 1,
                },
            ]
        },
    },
]


LayerDataframeDefaults = [
    SourcesPaths,
]
