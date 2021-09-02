from terra_ai.data.modeling.layers import Layer, types
from terra_ai.data.modeling.extra import LayerTypeChoice

from ..utils import prepare_pydantic_field
from ..choices import (
    LayerInputTypeChoice,
    LayerOutputTypeChoice,
    LayerNetChoice,
    LayerScalerChoice,
    LayerScalerImageChoice,
    LayerScalerAudioChoice,
    LayerScalerVideoChoice,
    LayerScalerRegressionChoice,
    LayerScalerTimeseriesChoice,
    LayerAudioModeChoice,
    LayerAudioParameterChoice,
    LayerTextModeChoice,
    LayerVideoFillModeChoice,
    LayerVideoFrameModeChoice,
    LayerVideoModeChoice,
    LayerPrepareMethodChoice,
    LayerDataframeAlignBaseMethodChoice,
    LayerDefineClassesChoice,
    LayerYoloVersionChoice,
    LayerTypeProcessingClassificationChoice,
)


SourcesPaths = {
    "type": "multiselect",
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
        "value": "convolutional",
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
        "value": "no_scaler",
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
            ]
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
        "value": "completely",
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
        "value": "embedding",
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
                },
            ],
            "bag_of_words": [
                {
                    "type": "number",
                    "label": "Максимальное количество слов",
                    "name": "max_words_count",
                    "parse": "max_words_count",
                },
            ],
            "word_to_vec": [
                {
                    "type": "number",
                    "label": "Размер Word2Vec пространства",
                    "name": "word_to_vec_size",
                    "parse": "word_to_vec_size",
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
    },
    {
        "type": "select",
        "label": "Формат аудио",
        "name": "audio_mode",
        "parse": "audio_mode",
        "value": "completely",
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
        "label": "Параметр",
        "name": "parameter",
        "parse": "parameter",
        "value": "audio_signal",
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
        "label": "Скейлер",
        "name": "scaler",
        "parse": "scaler",
        "value": "no_scaler",
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


Defaults = {
    "datasets": {
        "creation": {
            "input": [
                {
                    "type": "text",
                    "label": "Название входа",
                    "name": "name",
                    "parse": "name",
                },
                {
                    "type": "select",
                    "label": "Тип задачи",
                    "name": "type",
                    "parse": "type",
                    "value": "Image",
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
                                "value": "black_frames",
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
                                "value": "keep_proportions",
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
                                "value": "completely",
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
                                "value": "no_scaler",
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
                        "Dataframe": [
                            SourcesPaths,
                            {
                                "type": "text",
                                "label": "Сепаратор",
                                "name": "separator",
                                "parse": "separator",
                            },
                            {
                                "type": "checkbox",
                                "label": "Транспонирование",
                                "name": "transpose",
                                "parse": "transpose",
                                "value": False,
                            },
                            {
                                "type": "checkbox",
                                "label": "Выровнять базу",
                                "name": "align_base",
                                "parse": "align_base",
                                "value": False,
                                "fields": {
                                    "true": [
                                        {
                                            "type": "radio",
                                            "name": "align_base_method",
                                            "parse": "align_base_method",
                                            "value": "pad_sequences",
                                            "list": list(
                                                map(
                                                    lambda item: {
                                                        "value": item.name,
                                                        "label": item.value,
                                                    },
                                                    list(
                                                        LayerDataframeAlignBaseMethodChoice
                                                    ),
                                                )
                                            ),
                                            "fields": {
                                                "pad_sequences": [
                                                    {
                                                        "type": "number",
                                                        "label": "Длина примера",
                                                        "name": "example_length",
                                                        "parse": "example_length",
                                                    },
                                                ],
                                                "xlen_step": [
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
                                            "type": "select",
                                            "label": "Скейлер",
                                            "name": "scaler",
                                            "parse": "scaler",
                                            "value": "no_scaler",
                                            "list": list(
                                                map(
                                                    lambda item: {
                                                        "value": item.name,
                                                        "label": item.value,
                                                    },
                                                    list(LayerScalerChoice),
                                                )
                                            ),
                                        },
                                    ],
                                },
                            },
                        ],
                    },
                },
            ],
            "output": [
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
                    "value": "Image",
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
                        "Classification": [
                            SourcesPaths,
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
                                "value": "categorical",
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
                        "Segmentation": [
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
                                "value": "handmade",
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
                                            "api": "/api/v1/config/",
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
                                "label": "Открывающие теги",
                                "name": "open_tags",
                                "parse": "open_tags",
                            },
                            {
                                "type": "text",
                                "label": "Закрывающие теги",
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
                                "value": "no_scaler",
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
                                            "value": "no_scaler",
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
                                "value": "v4",
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
                        ],
                    },
                },
            ],
        },
    },
    "modeling": {
        "layer_form": [
            {
                "type": "text",
                "label": "Название слоя",
                "name": "name",
                "parse": "name",
            },
            {
                "type": "select",
                "label": "Тип слоя",
                "name": "type",
                "parse": "type",
                "list": list(
                    map(
                        lambda item: {"value": item.value, "label": item.name},
                        list(LayerTypeChoice),
                    )
                ),
            },
            {
                "type": "text_array",
                "label": "Размерность входных данных",
                "name": "input",
                "parse": "shape[input][]",
                "disabled": True,
            },
            {
                "type": "text_array",
                "label": "Размерность выходных данных",
                "name": "output",
                "parse": "shape[output][]",
                "disabled": True,
            },
        ],
        "layers_types": {},
    },
    "training": {
        "parameters": {},
    },
}


def __get_layer_type_params(data, group) -> list:
    output = []
    for name in data.__fields__:
        output.append(
            prepare_pydantic_field(
                data.__fields__[name], f"parameters[{group}][{name}]"
            )
        )
    return output


for layer in Layer:
    params = getattr(types, layer.name)
    Defaults["modeling"]["layers_types"].update(
        {
            layer.name: {
                "main": __get_layer_type_params(params.ParametersMainData, "main"),
                "extra": __get_layer_type_params(params.ParametersExtraData, "extra"),
            }
        }
    )
