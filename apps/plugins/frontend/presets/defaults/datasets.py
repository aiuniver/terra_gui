from ...choices import (
    LayerInputTypeChoice,
    LayerOutputTypeChoice,
    LayerNetChoice,
    LayerScalerImageChoice,
    LayerScalerAudioChoice,
    LayerScalerVideoChoice,
    LayerScalerRegressionChoice,
    LayerScalerTimeseriesChoice,
    LayerScalerDefaultChoice,
    LayerAudioFillModeChoice,
    LayerAudioParameterChoice,
    LayerAudioResampleChoice,
    LayerVideoFillModeChoice,
    LayerVideoFrameModeChoice,
    LayerVideoModeChoice,
    LayerPrepareMethodChoice,
    LayerDefineClassesChoice,
    LayerYoloVersionChoice,
    LayerODDatasetTypeChoice,
    LayerTypeProcessingClassificationChoice,
    ColumnProcessingInputTypeChoice,
    ColumnProcessingOutputTypeChoice,
)
from .layers import (
    SourcesPaths,
    LayerImageDefaults,
    LayerTextDefaults,
    LayerAudioDefaults,
    LayerDataframeDefaults,
)


DataSetsColumnProcessing = [
    {
        "type": "text",
        "label": "Название обработчика",
        "name": "name",
        "parse": "name",
    },
    {
        "type": "select",
        "label": "Тип слоя",
        "name": "layer_type",
        "parse": "layer_type",
        "value": "input",
        "list": [
            {"value": "input", "label": "Входной"},
            {"value": "output", "label": "Выходной"},
        ],
        "fields": {
            "input": [
                {
                    "type": "select",
                    "label": "Тип обработчика",
                    "name": "type",
                    "parse": "type",
                    "value": ColumnProcessingInputTypeChoice.Image.name,
                    "list": list(
                        map(
                            lambda item: {
                                "value": item.name,
                                "label": item.value,
                            },
                            list(ColumnProcessingInputTypeChoice),
                        )
                    ),
                    "fields": {
                        "Image": [
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
                        ],
                        "Text": [
                            {
                                "type": "text",
                                "label": "Фильтры",
                                "name": "filters",
                                "parse": "filters",
                                "value": '–—!"#$%&()*+,-./:;<=>?@[\\]^«»№_`{|}~\t\n\xa0–\ufeff',
                            },
                            {
                                "type": "number",
                                "label": "Количество слов",
                                "name": "max_words",
                                "parse": "max_words",
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
                        ],
                        "Audio": [
                            {
                                "type": "number",
                                "label": "Частота дискретизации",
                                "name": "sample_rate",
                                "parse": "sample_rate",
                                "value": 22050,
                            },
                            {
                                "type": "number",
                                "label": "Длина аудио (сек.)",
                                "name": "max_seconds",
                                "parse": "max_seconds",
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
                        ],
                        "Video": [
                            {
                                "type": "number",
                                "label": "Ширина кадра",
                                "name": "width",
                                "parse": "width",
                            },
                            {
                                "type": "number",
                                "label": "Высота кадра",
                                "name": "height",
                                "parse": "height",
                            },
                            {
                                "type": "select",
                                "label": "Заполнение недостающих кадров",
                                "name": "fill_mode",
                                "parse": "fill_mode",
                                "value": LayerVideoFillModeChoice.last_frames.name,
                                "list": list(
                                    map(
                                        lambda item: {
                                            "value": item.name,
                                            "label": item.value,
                                        },
                                        list(LayerVideoFillModeChoice),
                                    )
                                ),
                            },
                            {
                                "type": "select",
                                "label": "Обработка кадров",
                                "name": "frame_mode",
                                "parse": "frame_mode",
                                "value": LayerVideoFrameModeChoice.fit.name,
                                "list": list(
                                    map(
                                        lambda item: {
                                            "value": item.name,
                                            "label": item.value,
                                        },
                                        list(LayerVideoFrameModeChoice),
                                    )
                                ),
                            },
                            {
                                "type": "number",
                                "label": "Количество кадров",
                                "name": "max_frames",
                                "parse": "max_frames",
                            },
                            {
                                "type": "select",
                                "label": "Скейлер",
                                "name": "scaler",
                                "parse": "scaler",
                                "value": LayerScalerVideoChoice.min_max_scaler.name,
                                "list": list(
                                    map(
                                        lambda item: {
                                            "value": item.name,
                                            "label": item.value,
                                        },
                                        list(LayerScalerVideoChoice),
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
                        ],
                        "Classification": [
                            {
                                "type": "checkbox",
                                "label": "One-Hot encoding",
                                "name": "one_hot_encoding",
                                "parse": "one_hot_encoding",
                                "value": True,
                            },
                            {
                                "type": "select",
                                "label": "Тип предобработки",
                                "name": "type_processing",
                                "parse": "type_processing",
                                "value": LayerTypeProcessingClassificationChoice.categorical.name,
                                "list": list(
                                    map(
                                        lambda item: {
                                            "value": item.name,
                                            "label": item.value,
                                        },
                                        list(LayerTypeProcessingClassificationChoice),
                                    )
                                ),
                                "fields": {
                                    "ranges": [
                                        {
                                            "type": "text",
                                            "label": "Диапазоны/число диапазонов",
                                            "name": "ranges",
                                            "parse": "ranges",
                                        }
                                    ]
                                },
                            },
                        ],
                        "Scaler": [
                            {
                                "type": "select",
                                "label": "Скейлер",
                                "name": "scaler",
                                "parse": "scaler",
                                "value": LayerScalerDefaultChoice.min_max_scaler.name,
                                "list": list(
                                    map(
                                        lambda item: {
                                            "value": item.name,
                                            "label": item.value,
                                        },
                                        list(LayerScalerDefaultChoice),
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
                        ],
                    },
                }
            ],
            "output": [
                {
                    "type": "select",
                    "label": "Тип обработчика",
                    "name": "type",
                    "parse": "type",
                    "value": ColumnProcessingOutputTypeChoice.Classification.name,
                    "list": list(
                        map(
                            lambda item: {
                                "value": item.name,
                                "label": item.value,
                            },
                            list(ColumnProcessingOutputTypeChoice),
                        )
                    ),
                    "fields": {
                        "Classification": [
                            {
                                "type": "checkbox",
                                "label": "One-Hot encoding",
                                "name": "one_hot_encoding",
                                "parse": "one_hot_encoding",
                                "value": True,
                            },
                            {
                                "type": "select",
                                "label": "Тип предобработки",
                                "name": "type_processing",
                                "parse": "type_processing",
                                "value": LayerTypeProcessingClassificationChoice.categorical.name,
                                "list": list(
                                    map(
                                        lambda item: {
                                            "value": item.name,
                                            "label": item.value,
                                        },
                                        list(LayerTypeProcessingClassificationChoice),
                                    )
                                ),
                                "fields": {
                                    "ranges": [
                                        {
                                            "type": "text",
                                            "label": "Диапазоны/число диапазонов",
                                            "name": "ranges",
                                            "parse": "ranges",
                                        }
                                    ]
                                },
                            },
                        ],
                        "Regression": [
                            {
                                "type": "select",
                                "label": "Скейлер",
                                "name": "scaler",
                                "parse": "scaler",
                                "value": LayerScalerRegressionChoice.min_max_scaler.name,
                                "list": list(
                                    map(
                                        lambda item: {
                                            "value": item.name,
                                            "label": item.value,
                                        },
                                        list(LayerScalerRegressionChoice),
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
                        ],
                        "Segmentation": [
                            {
                                "type": "number",
                                "label": "Диапазон каналов",
                                "name": "mask_range",
                                "parse": "mask_range",
                            },
                            {
                                "type": "select",
                                "label": "Ввод данных",
                                "name": "classes",
                                "parse": "classes",
                                "value": LayerDefineClassesChoice.handmade.name,
                                "list": list(
                                    map(
                                        lambda item: {
                                            "value": item.name,
                                            "label": item.value,
                                        },
                                        list(LayerDefineClassesChoice),
                                    )
                                ),
                                "fields": {
                                    "handmade": [
                                        {
                                            "type": "segmentation_manual",
                                            "label": "Название класса",
                                            "name": "classes_names",
                                            "parse": "classes_names[]",
                                        },
                                    ],
                                    "autosearch": [
                                        {
                                            "type": "segmentation_search",
                                            "label": "Найти",
                                            "name": "search",
                                            "parse": "search",
                                        },
                                    ],
                                    "annotation": [
                                        {
                                            "type": "segmentation_annotation",
                                            "label": "Выберите файл",
                                            "name": "annotation",
                                            "parse": "annotation",
                                        },
                                    ],
                                },
                            },
                        ],
                        "Timeseries": [
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
                            {
                                "type": "checkbox",
                                "label": "Предсказывать тренд",
                                "name": "trend",
                                "parse": "trend",
                                "value": False,
                                "fields": {
                                    "true": [
                                        {
                                            "type": "text",
                                            "label": "Отклонение нулевого тренда",
                                            "name": "trend_limit",
                                            "parse": "trend_limit",
                                        },
                                    ],
                                    "false": [
                                        {
                                            "type": "number",
                                            "label": "Глубина предсказания",
                                            "name": "depth",
                                            "parse": "depth",
                                        },
                                        {
                                            "type": "select",
                                            "label": "Скейлер",
                                            "name": "scaler",
                                            "parse": "scaler",
                                            "value": LayerScalerTimeseriesChoice.min_max_scaler.name,
                                            "list": list(
                                                map(
                                                    lambda item: {
                                                        "value": item.name,
                                                        "label": item.value,
                                                    },
                                                    list(LayerScalerTimeseriesChoice),
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
                                    ],
                                },
                            },
                        ],
                    },
                }
            ],
        },
    },
]

DataSetsInput = [
    {
        "type": "text",
        "label": "Название входа",
        "name": "name",
        "parse": "name",
    },
    {
        "type": "select",
        "label": "Тип данных",
        "name": "type",
        "parse": "type",
        "value": LayerInputTypeChoice.Image.name,
        "list": list(
            map(
                lambda item: {"value": item.name, "label": item.value},
                list(LayerInputTypeChoice),
            )
        ),
        "fields": {
            "Image": LayerImageDefaults,
            "Text": LayerTextDefaults,
            "Audio": LayerAudioDefaults,
            "Video": [
                SourcesPaths,
                {
                    "type": "number",
                    "label": "Ширина кадра",
                    "name": "width",
                    "parse": "width",
                },
                {
                    "type": "number",
                    "label": "Высота кадра",
                    "name": "height",
                    "parse": "height",
                },
                {
                    "type": "select",
                    "label": "Заполнение недостающих кадров",
                    "name": "fill_mode",
                    "parse": "fill_mode",
                    "value": LayerVideoFillModeChoice.average_value.name,
                    "list": list(
                        map(
                            lambda item: {
                                "value": item.name,
                                "label": item.value,
                            },
                            list(LayerVideoFillModeChoice),
                        )
                    ),
                },
                {
                    "type": "select",
                    "label": "Обработка кадров",
                    "name": "frame_mode",
                    "parse": "frame_mode",
                    "value": LayerVideoFrameModeChoice.fit.name,
                    "list": list(
                        map(
                            lambda item: {
                                "value": item.name,
                                "label": item.value,
                            },
                            list(LayerVideoFrameModeChoice),
                        )
                    ),
                },
                {
                    "type": "select",
                    "label": "Формат видео",
                    "name": "video_mode",
                    "parse": "video_mode",
                    "value": LayerVideoModeChoice.completely.name,
                    "list": list(
                        map(
                            lambda item: {
                                "value": item.name,
                                "label": item.value,
                            },
                            list(LayerVideoModeChoice),
                        )
                    ),
                    "fields": {
                        "completely": [
                            {
                                "type": "number",
                                "label": "Количество кадров",
                                "name": "max_frames",
                                "parse": "max_frames",
                            },
                        ],
                        "length_and_step": [
                            {
                                "type": "number",
                                "label": "Длина (кадров)",
                                "name": "length",
                                "parse": "length",
                            },
                            {
                                "type": "number",
                                "label": "Шаг (кадров)",
                                "name": "step",
                                "parse": "step",
                            },
                        ],
                    },
                },
                {
                    "type": "select",
                    "label": "Скейлер",
                    "name": "scaler",
                    "parse": "scaler",
                    "value": LayerScalerVideoChoice.min_max_scaler.name,
                    "list": list(
                        map(
                            lambda item: {
                                "value": item.name,
                                "label": item.value,
                            },
                            list(LayerScalerVideoChoice),
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
            ],
            "Dataframe": LayerDataframeDefaults,
        },
    },
]

DataSetsOutput = [
    {
        "type": "text",
        "name": "name",
        "label": "Название выхода",
        "parse": "name",
    },
    {
        "type": "select",
        "name": "type",
        "label": "Тип задачи",
        "parse": "type",
        "value": LayerOutputTypeChoice.Image.name,
        "list": list(
            map(
                lambda item: {"value": item.name, "label": item.value},
                list(LayerOutputTypeChoice),
            )
        ),
        "fields": {
            "Image": LayerImageDefaults,
            "Text": LayerTextDefaults,
            "Audio": LayerAudioDefaults,
            "Dataframe": LayerDataframeDefaults,
            "Classification": [],
            "Segmentation": [
                SourcesPaths,
                {
                    "type": "number",
                    "label": "Диапазон каналов",
                    "name": "mask_range",
                    "parse": "mask_range",
                },
                {
                    "type": "select",
                    "label": "Ввод данных",
                    "name": "classes",
                    "parse": "classes",
                    "value": LayerDefineClassesChoice.handmade.name,
                    "list": list(
                        map(
                            lambda item: {
                                "value": item.name,
                                "label": item.value,
                            },
                            list(LayerDefineClassesChoice),
                        )
                    ),
                    "fields": {
                        "handmade": [
                            {
                                "type": "segmentation_manual",
                                "label": "Название класса",
                                "name": "classes_names",
                                "parse": "classes_names[]",
                            },
                        ],
                        "autosearch": [
                            {
                                "type": "segmentation_search",
                                "label": "Найти",
                                "name": "search",
                                "parse": "search",
                            },
                        ],
                        "annotation": [
                            {
                                "type": "segmentation_annotation",
                                "label": "Выберите файл",
                                "name": "annotation",
                                "parse": "annotation",
                            },
                        ],
                    },
                },
            ],
            "TextSegmentation": [
                {
                    "type": "text",
                    "label": "Открывающие теги (через пробел)",
                    "name": "open_tags",
                    "parse": "open_tags",
                },
                {
                    "type": "text",
                    "label": "Закрывающие теги (через пробел)",
                    "name": "close_tags",
                    "parse": "close_tags",
                },
            ],
            "Regression": [
                SourcesPaths,
                {
                    "type": "select",
                    "label": "Скейлер",
                    "name": "scaler",
                    "parse": "scaler",
                    "value": LayerScalerRegressionChoice.min_max_scaler.name,
                    "list": list(
                        map(
                            lambda item: {
                                "value": item.name,
                                "label": item.value,
                            },
                            list(LayerScalerRegressionChoice),
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
            ],
            "Timeseries": [
                SourcesPaths,
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
                {
                    "type": "checkbox",
                    "label": "Предсказывать тренд",
                    "name": "trend",
                    "parse": "trend",
                    "value": False,
                    "fields": {
                        "true": [
                            {
                                "type": "text",
                                "label": "Отклонение нулевого тренда",
                                "name": "trend_limit",
                                "parse": "trend_limit",
                            },
                        ],
                        "false": [
                            {
                                "type": "number",
                                "label": "Глубина предсказания",
                                "name": "depth",
                                "parse": "depth",
                            },
                            {
                                "type": "select",
                                "label": "Скейлер",
                                "name": "scaler",
                                "parse": "scaler",
                                "value": LayerScalerTimeseriesChoice.min_max_scaler.name,
                                "list": list(
                                    map(
                                        lambda item: {
                                            "value": item.name,
                                            "label": item.value,
                                        },
                                        list(LayerScalerTimeseriesChoice),
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
                        ],
                    },
                },
            ],
            "ObjectDetection": [
                SourcesPaths,
                {
                    "type": "select",
                    "label": "Версия Yolo",
                    "name": "yolo",
                    "parse": "yolo",
                    "value": LayerYoloVersionChoice.v4.name,
                    "list": list(
                        map(
                            lambda item: {
                                "value": item.name,
                                "label": item.value,
                            },
                            list(LayerYoloVersionChoice),
                        )
                    ),
                },
                {
                    "type": "select",
                    "label": "Тип аннотации",
                    "name": "model_type",
                    "parse": "model_type",
                    "value": LayerODDatasetTypeChoice.Yolo_terra.name,
                    "list": list(
                        map(
                            lambda item: {
                                "value": item.name,
                                "label": item.value,
                            },
                            list(LayerODDatasetTypeChoice),
                        )
                    ),
                },
            ],
        },
    },
]
