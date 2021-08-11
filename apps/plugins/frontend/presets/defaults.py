from terra_ai.data.modeling.layers import Layer, types
from terra_ai.data.modeling.extra import LayerTypeChoice

from ..utils import prepare_pydantic_field
from ..choices import (
    LayerInputTypeChoice,
    LayerOutputTypeChoice,
    LayerNetChoice,
    LayerScalerChoice,
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
)


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
                    "label": "Тип данных",
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
                                        list(LayerScalerChoice),
                                    )
                                ),
                            },
                            {
                                "type": "checkbox",
                                "label": "Аргументация",
                                "name": "augmentation",
                                "parse": "augmentation",
                                "value": False,
                            },
                        ],
                        "Audio": [
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
                                            "label": "Длина аудио",
                                            "name": "max_seconds",
                                            "parse": "max_seconds",
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
                        ],
                        "Text": [
                            {
                                "type": "number",
                                "label": "Максимальное количество слов",
                                "name": "max_words_count",
                                "parse": "max_words_count",
                            },
                            {
                                "type": "text",
                                "label": "Фильтры",
                                "name": "delete_symbols",
                                "parse": "delete_symbols",
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
                                    "word_to_vec": [
                                        {
                                            "type": "number",
                                            "label": "Размер Word2Vec пространства",
                                            "name": "word_to_vec_size",
                                            "parse": "word_to_vec_size",
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
                        ],
                        "Dataframe": [
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
                    "label": "Тип данных",
                    "parse": "type",
                    "value": "Image",
                    "list": list(
                        map(
                            lambda item: {"value": item.name, "label": item.value},
                            list(LayerOutputTypeChoice),
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
                                        list(LayerScalerChoice),
                                    )
                                ),
                            },
                            {
                                "type": "checkbox",
                                "label": "Аргументация",
                                "name": "augmentation",
                                "parse": "augmentation",
                                "value": False,
                            },
                        ],
                        "Audio": [
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
                                            "label": "Длина аудио",
                                            "name": "max_seconds",
                                            "parse": "max_seconds",
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
                        ],
                        "Text": [
                            {
                                "type": "number",
                                "label": "Максимальное количество слов",
                                "name": "max_words_count",
                                "parse": "max_words_count",
                            },
                            {
                                "type": "text",
                                "label": "Фильтры",
                                "name": "delete_symbols",
                                "parse": "delete_symbols",
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
                                    "word_to_vec": [
                                        {
                                            "type": "number",
                                            "label": "Размер Word2Vec пространства",
                                            "name": "word_to_vec_size",
                                            "parse": "word_to_vec_size",
                                        },
                                    ]
                                },
                            },
                        ],
                        "Classification": [
                            {
                                "type": "checkbox",
                                "label": "Разбить на категории",
                                "name": "categorical",
                                "parse": "categorical",
                                "value": True,
                            },
                            {
                                "type": "checkbox",
                                "label": "Разбить на диапазоны",
                                "name": "categorical_ranges",
                                "parse": "categorical_ranges",
                                "value": False,
                                "fields": {
                                    "true": [
                                        {
                                            "type": "checkbox",
                                            "label": "Автоматически",
                                            "name": "auto_ranges",
                                            "parse": "auto_ranges",
                                            "value": True,
                                            "fields": {
                                                "false": [
                                                    {
                                                        "type": "text",
                                                        "label": "Диапазоны/число дипазонов",
                                                        "name": "ranges",
                                                        "parse": "ranges",
                                                    }
                                                ]
                                            },
                                        }
                                    ]
                                },
                            },
                            {
                                "type": "checkbox",
                                "label": "One-Hot encoding",
                                "name": "one_hot_encoding",
                                "parse": "one_hot_encoding",
                                "value": True,
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
                                "type": "number",
                                "label": "Количество классов",
                                "name": "num_classes",
                                "parse": "num_classes",
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
                                            "value": item.value,
                                            "label": item.name,
                                        },
                                        list(LayerDefineClassesChoice),
                                    )
                                ),
                                "fields": {
                                    "handmade": [
                                        {
                                            "type": "text",
                                            "label": "Название класса",
                                            "name": "classes_names",
                                            "parse": "classes_names[]",
                                        },
                                        {
                                            "type": "text",
                                            "label": "Цвет",
                                            "name": "classes_colors",
                                            "parse": "classes_colors[]",
                                        },
                                    ],
                                    "autosearch": [
                                        {
                                            "type": "button",
                                            "label": "Найти",
                                            "name": "search",
                                            "parse": "search",
                                        },
                                    ],
                                    "annotation": [
                                        {
                                            "type": "select",
                                            "label": "Выберите файл",
                                            "name": "annotation",
                                            "parse": "annotation",
                                            "list": [],
                                        },
                                        {
                                            "type": "button",
                                            "label": "Найти",
                                            "name": "search",
                                            "parse": "search",
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
                        "Timeseries": [
                            {
                                "type": "text",
                                "label": "Имена колонок",
                                "name": "cols_names",
                                "parse": "cols_names",
                            },
                            {
                                "type": "number",
                                "label": "Длина примера выборки",
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
                                                    list(LayerScalerChoice),
                                                )
                                            ),
                                        },
                                    ],
                                },
                            },
                        ],
                        "ObjectDetection": [
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
                "name": "name",
                "label": "Название слоя",
                "parse": "name",
            },
            {
                "type": "select",
                "name": "type",
                "label": "Тип слоя",
                "parse": "type",
                "list": list(
                    map(
                        lambda item: {"value": item.value, "label": item.name},
                        list(LayerTypeChoice),
                    )
                ),
            },
        ],
        "layers_types": {},
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
            layer.value: {
                "main": __get_layer_type_params(params.ParametersMainData, "main"),
                "extra": __get_layer_type_params(params.ParametersExtraData, "extra"),
            }
        }
    )
