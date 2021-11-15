from terra_ai.data.cascades.extra import BlockGroupChoice
from terra_ai.data.cascades.blocks.extra import (
    ChangeTypeAvailableChoice,
    PostprocessBoxesMethodAvailableChoice,
    BlockCustomGroupChoice,
    BlockCustomTypeChoice,
)

from ...choices import (
    BlockFunctionGroupChoice,
    BlockFunctionTypeChoice,
    LayerInputTypeChoice,
)


CascadesBlockForm = [
    {
        "type": "text",
        "label": "Название блока",
        "name": "name",
        "parse": "name",
    },
    {
        "type": "select",
        "label": "Тип блока",
        "name": "group",
        "parse": "group",
        "list": list(
            map(
                lambda item: {"value": item.name, "label": item.value},
                list(BlockGroupChoice),
            )
        ),
        "visible": False,
    },
]


FunctionTypesFields = {
    BlockFunctionTypeChoice.CropImage: [],
    BlockFunctionTypeChoice.ChangeType: [
        {
            "type": "select",
            "name": "change_type",
            "parse": "parameters[main][change_type]",
            "label": "Выбор типа",
            "value": ChangeTypeAvailableChoice.int,
            "list": list(
                map(
                    lambda item: {"value": item.name, "label": item.value},
                    list(ChangeTypeAvailableChoice),
                )
            ),
        },
    ],
    BlockFunctionTypeChoice.ChangeSize: [
        {
            "type": "text_array",
            "name": "shape",
            "parse": "parameters[main][shape]",
            "label": "Размерность",
            "value": "",
        },
    ],
    BlockFunctionTypeChoice.MinMaxScale: [
        {
            "type": "number",
            "name": "min_scale",
            "parse": "parameters[main][min_scale]",
            "label": "Минимальное значение",
            "value": 0,
        },
        {
            "type": "number",
            "name": "max_scale",
            "parse": "parameters[main][max_scale]",
            "label": "Максимальное значение",
            "value": 1,
        },
    ],
    BlockFunctionTypeChoice.MaskedImage: [],
    BlockFunctionTypeChoice.PlotMaskSegmentation: [],
    BlockFunctionTypeChoice.PutTag: [],
    BlockFunctionTypeChoice.PostprocessBoxes: [
        {
            "type": "number",
            "name": "score_threshold",
            "parse": "parameters[main][score_threshold]",
            "label": "Порог вероятности классов",
            "value": 0.3,
        },
        {
            "type": "number",
            "name": "iou_threshold",
            "parse": "parameters[main][iou_threshold]",
            "label": "Порог пересечения",
            "value": 0.45,
        },
        {
            "type": "select",
            "name": "method",
            "parse": "parameters[main][method]",
            "label": "Метод подавления немаксимумов",
            "value": PostprocessBoxesMethodAvailableChoice.nms,
            "list": list(
                map(
                    lambda item: {"value": item.name, "label": item.value},
                    list(PostprocessBoxesMethodAvailableChoice),
                )
            ),
            "fields": {
                PostprocessBoxesMethodAvailableChoice.soft_nms: [
                    {
                        "type": "number",
                        "name": "sigma",
                        "parse": "parameters[main][sigma]",
                        "label": "Коэффициент сглаживания",
                        "value": 0.3,
                    },
                ],
            },
        },
    ],
    BlockFunctionTypeChoice.PlotBBoxes: [],
}


FunctionGroupTypeRel = {
    BlockFunctionGroupChoice.Image: [
        BlockFunctionTypeChoice.CropImage,
    ],
    BlockFunctionGroupChoice.Text: [],
    BlockFunctionGroupChoice.Audio: [],
    BlockFunctionGroupChoice.Video: [],
    BlockFunctionGroupChoice.Array: [
        BlockFunctionTypeChoice.ChangeType,
        BlockFunctionTypeChoice.ChangeSize,
        BlockFunctionTypeChoice.MinMaxScale,
    ],
    BlockFunctionGroupChoice.Segmentation: [
        BlockFunctionTypeChoice.MaskedImage,
        BlockFunctionTypeChoice.PlotMaskSegmentation,
    ],
    BlockFunctionGroupChoice.TextSegmentation: [
        BlockFunctionTypeChoice.PutTag,
    ],
    BlockFunctionGroupChoice.ObjectDetection: [
        BlockFunctionTypeChoice.PostprocessBoxes,
        BlockFunctionTypeChoice.PlotBBoxes,
    ],
}


def get_function_type_field(type_name) -> dict:
    items = list(FunctionGroupTypeRel.get(type_name))
    if not len(items):
        return []
    return [
        {
            "type": "select",
            "name": "type",
            "label": "Тип",
            "parse": "parameters[main][type]",
            "value": items[0].name,
            "list": list(
                map(
                    lambda item: {"value": item.name, "label": item.value},
                    items,
                )
            ),
            "fields": dict(
                map(lambda item: (item.name, FunctionTypesFields.get(item, [])), items)
            ),
        }
    ]


CascadesBlocksTypes = {
    BlockGroupChoice.InputData: {
        "main": [
            {
                "type": "select",
                "label": "Тип данных",
                "name": "type",
                "parse": "parameters[main][type]",
                "value": LayerInputTypeChoice.Image.name,
                "list": list(
                    map(
                        lambda item: {"value": item.name, "label": item.value},
                        list(LayerInputTypeChoice),
                    )
                ),
                "fields": {
                    "Video": [
                        {
                            "type": "number",
                            "name": "width",
                            "label": "Ширина",
                            "parse": "parameters[main][width]",
                        },
                        {
                            "type": "number",
                            "name": "height",
                            "label": "Высота",
                            "parse": "parameters[main][height]",
                        },
                    ]
                },
            },
        ]
    },
    BlockGroupChoice.OutputData: {
        "main": [
            {
                "type": "select",
                "label": "Тип данных",
                "name": "type",
                "parse": "parameters[main][type]",
                "value": LayerInputTypeChoice.Image.name,
                "list": list(
                    map(
                        lambda item: {"value": item.name, "label": item.value},
                        list(LayerInputTypeChoice),
                    )
                ),
                "fields": {
                    "Video": [
                        {
                            "type": "number",
                            "name": "width",
                            "label": "Ширина",
                            "parse": "parameters[main][width]",
                        },
                        {
                            "type": "number",
                            "name": "height",
                            "label": "Высота",
                            "parse": "parameters[main][height]",
                        },
                    ]
                },
            },
        ]
    },
    BlockGroupChoice.Model: {
        "main": [
            {
                "type": "select",
                "name": "path",
                "label": "Обучение",
                "parse": "parameters[main][path]",
            },
            {
                "type": "checkbox",
                "name": "postprocess",
                "label": "Использовать постобработку",
                "parse": "parameters[main][postprocess]",
                "value": True,
            },
        ]
    },
    BlockGroupChoice.Function: {
        "main": [
            {
                "type": "select",
                "name": "group",
                "label": "Группа",
                "parse": "parameters[main][group]",
                "value": BlockFunctionGroupChoice.Image,
                "list": list(
                    map(
                        lambda item: {"value": item.name, "label": item.value},
                        list(BlockFunctionGroupChoice),
                    )
                ),
                "fields": {
                    BlockFunctionGroupChoice.Image: get_function_type_field(
                        BlockFunctionGroupChoice.Image
                    ),
                    BlockFunctionGroupChoice.Text: get_function_type_field(
                        BlockFunctionGroupChoice.Text
                    ),
                    BlockFunctionGroupChoice.Audio: get_function_type_field(
                        BlockFunctionGroupChoice.Audio
                    ),
                    BlockFunctionGroupChoice.Video: get_function_type_field(
                        BlockFunctionGroupChoice.Video
                    ),
                    BlockFunctionGroupChoice.Array: get_function_type_field(
                        BlockFunctionGroupChoice.Array
                    ),
                    BlockFunctionGroupChoice.ObjectDetection: get_function_type_field(
                        BlockFunctionGroupChoice.ObjectDetection
                    ),
                    BlockFunctionGroupChoice.Segmentation: get_function_type_field(
                        BlockFunctionGroupChoice.Segmentation
                    ),
                    BlockFunctionGroupChoice.TextSegmentation: get_function_type_field(
                        BlockFunctionGroupChoice.TextSegmentation
                    ),
                },
            },
        ]
    },
    BlockGroupChoice.Custom: {
        "main": [
            {
                "type": "select",
                "name": "group",
                "label": "Группа",
                "parse": "parameters[main][group]",
                "value": BlockCustomGroupChoice.Tracking,
                "list": list(
                    map(
                        lambda item: {"value": item.name, "label": item.value},
                        list(BlockCustomGroupChoice),
                    )
                ),
                "fields": {
                    BlockCustomGroupChoice.Tracking: [
                        {
                            "type": "select",
                            "name": "type",
                            "label": "Выбор типа",
                            "parse": "parameters[main][type]",
                            "value": BlockCustomTypeChoice.Sort,
                            "list": list(
                                map(
                                    lambda item: {
                                        "value": item.name,
                                        "label": item.value,
                                    },
                                    list(BlockCustomTypeChoice),
                                )
                            ),
                        },
                    ]
                },
            },
            {
                "type": "number",
                "name": "max_age",
                "label": "Количество кадров для остановки слежения",
                "parse": "parameters[main][max_age]",
                "value": 4,
            },
            {
                "type": "number",
                "name": "min_hits",
                "label": "Количество кадров для возобновления отслеживания",
                "parse": "parameters[main][min_hits]",
                "value": 4,
            },
        ]
    },
}
