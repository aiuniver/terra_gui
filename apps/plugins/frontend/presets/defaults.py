from terra_ai.data.modeling.layers import Layer, types
from terra_ai.data.modeling.extra import LayerTypeChoice
from terra_ai.data.training.extra import (
    OptimizerChoice,
    CheckpointIndicatorChoice,
    CheckpointTypeChoice,
    CheckpointModeChoice,
    TasksGroupsList,
)
from terra_ai.data.presets.training import TasksGroups

from ..extra import FieldTypeChoice
from ..utils import prepare_pydantic_field
from ..choices import (
    LayerInputTypeChoice,
    LayerOutputTypeChoice,
    LayerNetChoice,
    LayerScalerImageChoice,
    LayerScalerAudioChoice,
    LayerScalerVideoChoice,
    LayerScalerRegressionChoice,
    LayerScalerTimeseriesChoice,
    LayerScalerDefaultChoice,
    LayerAudioModeChoice,
    LayerAudioFillModeChoice,
    LayerAudioParameterChoice,
    LayerAudioResampleChoice,
    LayerTextModeChoice,
    LayerVideoFillModeChoice,
    LayerVideoFrameModeChoice,
    LayerVideoModeChoice,
    LayerPrepareMethodChoice,
    LayerDefineClassesChoice,
    LayerYoloVersionChoice,
    LayerTypeProcessingClassificationChoice,
    ColumnProcessingTypeChoice,
)


TrainingTasksRelations = TasksGroupsList(TasksGroups)


TrainingLossSelect = {
    "type": FieldTypeChoice.select.value,
    "label": "Loss",
    "name": "architecture_parameters_outputs_%i_loss",
    "parse": "architecture[parameters][outputs][%i][loss]",
}


TrainingMetricSelect = {
    "type": FieldTypeChoice.multiselect.value,
    "label": "Выберите метрики",
    "name": "architecture_parameters_outputs_%i_metrics",
    "parse": "architecture[parameters][outputs][%i][metrics]",
}


TrainingClassesQuantitySelect = {
    "type": FieldTypeChoice.number.value,
    "label": "Количество классов",
    "name": "architecture_parameters_outputs_%i_classes_quantity",
    "parse": "architecture[parameters][outputs][%i][classes_quantity]",
    "disabled": True,
}


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


Defaults = {
    "datasets": {
        "creation": {
            "column_processing": [
                {
                    "type": "text",
                    "label": "Название обработчика",
                    "name": "name",
                    "parse": "name",
                },
                {
                    "type": "select",
                    "label": "Тип обработчика",
                    "name": "type",
                    "parse": "type",
                    "value": ColumnProcessingTypeChoice.Image.name,
                    "list": list(
                        map(
                            lambda item: {
                                "value": item.name,
                                "label": item.value,
                            },
                            list(ColumnProcessingTypeChoice),
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
                                    ]
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
                                "value": LayerVideoFrameModeChoice.keep_proportions.name,
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
                },
            ],
            "input": [
                {
                    "type": "text",
                    "label": "Название входа",
                    "name": "name",
                    "parse": "name",
                },
                {
                    "type": "select_creation_tasks",
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
                                "value": LayerVideoFrameModeChoice.keep_proportions.name,
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
            ],
            "output": [
                {
                    "type": "text",
                    "name": "name",
                    "label": "Название выхода",
                    "parse": "name",
                },
                {
                    "type": "select_creation_tasks",
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
        "base": {
            "main": {
                "fields": [
                    {
                        "type": "auto_complete",
                        "label": "Оптимизатор",
                        "name": "optimizer",
                        "parse": "optimizer[type]",
                        "value": OptimizerChoice.Adam.name,
                        "list": list(
                            map(
                                lambda item: {"value": item.name, "label": item.value},
                                list(OptimizerChoice),
                            )
                        ),
                    },
                ],
            },
            "fit": {
                "fields": [
                    {
                        "type": "number",
                        "label": "Размер батча",
                        "name": "batch",
                        "parse": "[batch]",
                        "value": 32,
                    },
                    {
                        "type": "number",
                        "label": "Количество эпох",
                        "name": "epochs",
                        "parse": "[epochs]",
                        "value": 20,
                    },
                    {
                        "type": "number",
                        "label": "Learning rate",
                        "name": "optimizer_learning_rate",
                        "parse": "optimizer[parameters][main][learning_rate]",
                        "value": 0.001,
                    },
                ],
            },
            "optimizer": {
                "name": "Параметры оптимизатора",
                "collapsable": True,
                "collapsed": True,
                "fields": {
                    OptimizerChoice.SGD.name: [
                        {
                            "type": "number",
                            "label": "Momentum",
                            "name": "optimizer_extra_momentum",
                            "parse": "optimizer[parameters][extra][momentum]",
                            "value": 0,
                        },
                        {
                            "type": "checkbox",
                            "label": "Nesterov",
                            "name": "optimizer_extra_nesterov",
                            "parse": "optimizer[parameters][extra][nesterov]",
                            "value": False,
                        },
                    ],
                    OptimizerChoice.RMSprop.name: [
                        {
                            "type": "number",
                            "label": "RHO",
                            "name": "optimizer_extra_rho",
                            "parse": "optimizer[parameters][extra][rho]",
                            "value": 0,
                        },
                        {
                            "type": "number",
                            "label": "RHO",
                            "name": "optimizer_extra_momentum",
                            "parse": "optimizer[parameters][extra][momentum]",
                            "value": 0,
                        },
                        {
                            "type": "number",
                            "label": "Epsilon",
                            "name": "optimizer_extra_epsilon",
                            "parse": "optimizer[parameters][extra][epsilon]",
                            "value": 1e-07,
                        },
                        {
                            "type": "checkbox",
                            "label": "Centered",
                            "name": "optimizer_extra_centered",
                            "parse": "optimizer[parameters][extra][centered]",
                            "value": False,
                        },
                    ],
                    OptimizerChoice.Adam.name: [
                        {
                            "type": "number",
                            "label": "Beta 1",
                            "name": "optimizer_extra_beta_1",
                            "parse": "optimizer[parameters][extra][beta_1]",
                            "value": 0.9,
                        },
                        {
                            "type": "number",
                            "label": "Beta 2",
                            "name": "optimizer_extra_beta_2",
                            "parse": "optimizer[parameters][extra][beta_2]",
                            "value": 0.999,
                        },
                        {
                            "type": "number",
                            "label": "Epsilon",
                            "name": "optimizer_extra_epsilon",
                            "parse": "optimizer[parameters][extra][epsilon]",
                            "value": 1e-07,
                        },
                        {
                            "type": "checkbox",
                            "label": "Amsgrad",
                            "name": "optimizer_extra_amsgrad",
                            "parse": "optimizer[parameters][extra][amsgrad]",
                            "value": False,
                        },
                    ],
                    OptimizerChoice.Adadelta.name: [
                        {
                            "type": "number",
                            "label": "RHO",
                            "name": "optimizer_extra_rho",
                            "parse": "optimizer[parameters][extra][rho]",
                            "value": 0.95,
                        },
                        {
                            "type": "number",
                            "label": "Epsilon",
                            "name": "optimizer_extra_epsilon",
                            "parse": "optimizer[parameters][extra][epsilon]",
                            "value": 1e-07,
                        },
                    ],
                    OptimizerChoice.Adagrad.name: [
                        {
                            "type": "number",
                            "label": "Initial accumulator value",
                            "name": "optimizer_extra_initial_accumulator_value",
                            "parse": "optimizer[parameters][extra][initial_accumulator_value]",
                            "value": 0.1,
                        },
                        {
                            "type": "number",
                            "label": "Epsilon",
                            "name": "optimizer_extra_epsilon",
                            "parse": "optimizer[parameters][extra][epsilon]",
                            "value": 1e-07,
                        },
                    ],
                    OptimizerChoice.Adamax.name: [
                        {
                            "type": "number",
                            "label": "Beta 1",
                            "name": "optimizer_extra_beta_1",
                            "parse": "optimizer[parameters][extra][beta_1]",
                            "value": 0.9,
                        },
                        {
                            "type": "number",
                            "label": "Beta 2",
                            "name": "optimizer_extra_beta_2",
                            "parse": "optimizer[parameters][extra][beta_2]",
                            "value": 0.999,
                        },
                        {
                            "type": "number",
                            "label": "Epsilon",
                            "name": "optimizer_extra_epsilon",
                            "parse": "optimizer[parameters][extra][epsilon]",
                            "value": 1e-07,
                        },
                    ],
                    OptimizerChoice.Nadam.name: [
                        {
                            "type": "number",
                            "label": "Beta 1",
                            "name": "optimizer_extra_beta_1",
                            "parse": "optimizer[parameters][extra][beta_1]",
                            "value": 0.9,
                        },
                        {
                            "type": "number",
                            "label": "Beta 2",
                            "name": "optimizer_extra_beta_2",
                            "parse": "optimizer[parameters][extra][beta_2]",
                            "value": 0.999,
                        },
                        {
                            "type": "number",
                            "label": "Epsilon",
                            "name": "optimizer_extra_epsilon",
                            "parse": "optimizer[parameters][extra][epsilon]",
                            "value": 1e-07,
                        },
                    ],
                    OptimizerChoice.Ftrl.name: [
                        {
                            "type": "number",
                            "label": "Learning rate power",
                            "name": "optimizer_extra_learning_rate_power",
                            "parse": "optimizer[parameters][extra][learning_rate_power]",
                            "value": -0.5,
                        },
                        {
                            "type": "number",
                            "label": "Initial accumulator value",
                            "name": "optimizer_extra_initial_accumulator_value",
                            "parse": "optimizer[parameters][extra][initial_accumulator_value]",
                            "value": 0.1,
                        },
                        {
                            "type": "number",
                            "label": "L1 regularization strength",
                            "name": "optimizer_extra_l1_regularization_strength",
                            "parse": "optimizer[parameters][extra][l1_regularization_strength]",
                            "value": 0,
                        },
                        {
                            "type": "number",
                            "label": "L2 regularization strength",
                            "name": "optimizer_extra_l2_regularization_strength",
                            "parse": "optimizer[parameters][extra][l2_regularization_strength]",
                            "value": 0,
                        },
                        {
                            "type": "number",
                            "label": "L2 shrinkage regularization strength",
                            "name": "optimizer_extra_l2_shrinkage_regularization_strength",
                            "parse": "optimizer[parameters][extra][l2_shrinkage_regularization_strength]",
                            "value": 0,
                        },
                        {
                            "type": "number",
                            "label": "Beta",
                            "name": "optimizer_extra_beta",
                            "parse": "optimizer[parameters][extra][beta]",
                            "value": 0,
                        },
                    ],
                },
            },
            "outputs": {
                "name": "Параметры выходных слоев",
                "collapsable": True,
                "collapsed": False,
                "fields": [],
            },
            "checkpoint": {
                "name": "Чекпоинты",
                "collapsable": True,
                "collapsed": False,
                "fields": [
                    {
                        "type": "select",
                        "label": "Монитор",
                        "name": "architecture_parameters_checkpoint_layer",
                        "parse": "architecture[parameters][checkpoint][layer]",
                    },
                    {
                        "type": "select",
                        "label": "Indicator",
                        "name": "architecture_parameters_checkpoint_indicator",
                        "parse": "architecture[parameters][checkpoint][indicator]",
                        "value": CheckpointIndicatorChoice.Val.name,
                        "list": list(
                            map(
                                lambda item: {"value": item.name, "label": item.value},
                                list(CheckpointIndicatorChoice),
                            )
                        ),
                    },
                    {
                        "type": "select",
                        "label": "Тип",
                        "name": "architecture_parameters_checkpoint_type",
                        "parse": "architecture[parameters][checkpoint][type]",
                        "value": CheckpointTypeChoice.Metrics.name,
                        "list": list(
                            map(
                                lambda item: {"value": item.name, "label": item.value},
                                list(CheckpointTypeChoice),
                            )
                        ),
                    },
                    {
                        "type": "select",
                        "label": "Режим",
                        "name": "architecture_parameters_checkpoint_mode",
                        "parse": "architecture[parameters][checkpoint][mode]",
                        "value": CheckpointModeChoice.Max.name,
                        "list": list(
                            map(
                                lambda item: {"value": item.name, "label": item.value},
                                list(CheckpointModeChoice),
                            )
                        ),
                    },
                    {
                        "type": "checkbox",
                        "label": "Сохранить лучшее",
                        "name": "architecture_parameters_checkpoint_save_best",
                        "parse": "architecture[parameters][checkpoint][save_best]",
                        "value": True,
                    },
                    {
                        "type": "checkbox",
                        "label": "Сохранить веса",
                        "name": "architecture_parameters_checkpoint_save_weights",
                        "parse": "architecture[parameters][checkpoint][save_weights]",
                        "value": False,
                    },
                ],
            },
        },
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
            LayerTypeChoice[layer.name].value: {
                "main": __get_layer_type_params(params.ParametersMainData, "main"),
                "extra": __get_layer_type_params(params.ParametersExtraData, "extra"),
            }
        }
    )
