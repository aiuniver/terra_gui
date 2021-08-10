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
                                "label": "Аугментация",
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
                                            "label": "Длина",
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
                                "type": "checkbox",
                                "label": "Embedding",
                                "name": "embedding",
                                "parse": "embedding",
                                "value": False,
                            },
                            {
                                "type": "checkbox",
                                "label": "Bag of words",
                                "name": "bag_of_words",
                                "parse": "bag_of_words",
                                "value": False,
                            },
                            {
                                "type": "checkbox",
                                "label": "Word2Vec",
                                "name": "word_to_vec",
                                "parse": "word_to_vec",
                                "value": False,
                                "fields": {
                                    "true": [
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
                                            "type": "checkbox",
                                            "label": "pad_sequences",
                                            "name": "pad_sequences",
                                            "parse": "pad_sequences",
                                            "value": True,
                                            "fields": {
                                                "true": [
                                                    {
                                                        "type": "number",
                                                        "label": "Длина примера",
                                                        "name": "example_length",
                                                        "parse": "example_length",
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
                                                ]
                                            },
                                        },
                                        {
                                            "type": "checkbox",
                                            "label": "xlen_step",
                                            "name": "xlen_step",
                                            "parse": "xlen_step",
                                            "value": False,
                                            "fields": {
                                                "true": [
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
                                                ]
                                            },
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
                                "label": "Аугментация",
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
                                            "label": "Длина",
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
                                "type": "checkbox",
                                "label": "Embedding",
                                "name": "embedding",
                                "parse": "embedding",
                                "value": False,
                            },
                            {
                                "type": "checkbox",
                                "label": "Bag of words",
                                "name": "bag_of_words",
                                "parse": "bag_of_words",
                                "value": False,
                            },
                            {
                                "type": "checkbox",
                                "label": "Word2Vec",
                                "name": "word_to_vec",
                                "parse": "word_to_vec",
                                "value": False,
                                "fields": {
                                    "true": [
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
