from terra_ai.data.cascades.extra import BlockGroupChoice
from terra_ai.data.cascades.blocks.extra import (
    BlockServiceGroupChoice,
    BlockServiceTypeChoice,
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
    BlockFunctionTypeChoice.PlotBBoxes: [
        {
            "type": "number",
            "name": "line_thickness",
            "parse": "parameters[main][line_thickness]",
            "label": "Толщина линии рамки",
            "value": 1,
        },
    ],
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
                            "type": "checkbox",
                            "name": "switch_on_frame",
                            "label": "Switch on frame",
                            "parse": "parameters[main][switch_on_frame]",
                        },
                    ]
                },
                "manual": {
                    "Video": """
                        <b>Видео</b> - тип данных, который будет <i>использоваться для обработки</i> последующими блоками и каскадом в целом. 
                        Необходимые <a href="https://google.com" target="_blank">связи</a> с другими блоками на входе: 
                        None
                        Возможные связи с другими блоками на выходе:
                        <ol> 
                        <li>блок Model модель object detection или сегментации</li>  
                        <li>блок Function Наложение bbox на изображение</li>
                        <li>блок Function Постобработка yolo</li>
                        </ol>
                        Возвращает на выходе: Фреймы из видео
                    """
                }
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
                "manual": {
                    "Video": """
                        <b>Сохранение</b> результата в видео файл 
                        Необходимые <a href="https://google.com" target="_blank">связи</a> с другими блоками на входе:
                        <ol>
                        <li>блок Function Наложение bbox на изображение</li>
                        </ol>
                        Возможные <a href="https://google.com" target="_blank">связи</a> с другими блоками на выходе: 
                        None
                        Возвращает на выходе: Сохраняет переданные фреймы исходного видео в видеофайл с выставленными параметрами
                    """,
                    "Image": """
                        <b>Сохранение</b> результата в файл изображения
                        Необходимые <a href="https://google.com" target="_blank">связи</a> с другими блоками на входе:
                        <ol>
                        <li>блок Function Наложение bbox на изображение</li>
                        <li>блок Function Наложение маски по классу на изображение</li>
                        <li>блок Function Наложение маски всех классов по цветам</li>
                        </ol>
                        Возможные <a href="https://google.com" target="_blank">связи</a> с другими блоками на выходе: 
                        None
                        Возвращает на выходе: Сохраняет переданные обработанные изображения
                    """,
                    "Text": """
                        <b>Сохранение</b> результата в текстовый файл 
                        Необходимые <a href="https://google.com" target="_blank">связи</a> с другими блоками на входе:
                        <ol>
                        <li>блок Function Расстановка тэгов по вероятностям из модели</li>
                        <li>блок Service  speech_to_text</li>
                        </ol>
                        Возможные <a href="https://google.com" target="_blank">связи</a> с другими блоками на выходе: 
                        None
                        Возвращает на выходе: Сохраняет обработанный текст
                    """,
                    "Audio": """
                        <b>Сохранение</b> результата в текстовый файл 
                        Необходимые <a href="https://google.com" target="_blank">связи</a> с другими блоками на входе:
                        <ol>
                        <li>блок Service  text_to_speech</li>
                        </ol>
                        Возможные <a href="https://google.com" target="_blank">связи</a> с другими блоками на выходе: 
                        None
                        Возвращает на выходе: Сохраняет переданное аудио
                    """,
                }
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
                "manual": {
                    "ChangeSize": """
                        <b>Изменение</b> размера изображения по указанным параметрам. 
                        Необходимые <a href="https://google.com" target="_blank">связи</a> с другими блоками на входе:
                        <ol>
                        <li>любой блок возвращающий изображение</li>
                        </ol>
                        Возможные связи с другими блоками на выходе:
                        <ol>
                        <li>блок Output метод Видео или Изображение</li> 
                        <li>блок Model или Service</li>
                        <li>блок Function</li>
                        </ol>
                        Возвращает на выходе: маскированное изображение
                    """,
                    "MaskedImage": """
                        <b>Наложение маски по указанному классу</b> на изображение. Необходимо указать Id класса.
                        Необходимые <a href="https://google.com" target="_blank">связи</a> с другими блоками на входе: 
                        <ol>
                        <li>блок Model или Service c моделью сегментации изображений</li>
                        <li>блок Input исходных изображений или видео (по кадрам)</li>
                        </ol>
                        Возможные связи с другими блоками на выходе:
                        <ol>
                        <li>блок Output метод Видео или Изображение</li>
                        <li>блок Function Изменение размера данных</li>
                        </ol>
                        Возвращает на выходе: маскированное изображение
                    """,
                    "PlotMaskSegmentation": """
                        <b>Наложение маски по всем классам</b> на изображение
                        Необходимые <a href="https://google.com" target="_blank">связи</a> с другими блоками на входе:
                        <ol>
                        <li>блок Model или Service c моделью сегментации изображений</li> 
                        <li>блок Input исходных изображений или видео (по кадрам)</li>
                        </ol>
                        Возможные связи с другими блоками на выходе:
                        <ol>
                        <li>блок Output метод Видео или Изображение</li>
                        <li>блок Function Изменение размера данных</li>
                        </ol>
                        Возвращает на выходе: маскированное изображение
                    """,
                    "PutTag": """
                        <b>Наложение маски по всем классам</b> на изображение
                        Необходимые <a href="https://google.com" target="_blank">связи</a> с другими блоками на входе:
                        <ol>
                        <li>блок Model c моделью сегментации текста</li>
                        </ol>
                        Возможные связи с другими блоками на выходе:
                        <ol>
                        <li>блок Output метод Текст</li>
                        </ol>
                        Возвращает на выходе: размеченный тегами текст

                    """,
                    "PostprocessBoxes": """
                        <b>Постобработка</b> для моделей YOLOV3 и V4 
                        Необходимые <a href="https://google.com" target="_blank">связи</a> с другими блоками на входе:
                        <ol>
                        <li>блок Model c моделью YOLO</li>
                        <li>блок Input исходных изображений или видео</li>
                        </ol>
                        Возможные связи с другими блоками на выходе:
                        <ol> 
                        <li>блок Custom Трекер (Sort, DeepSort)</li>
                        <li>блок Function Наложение bbox на изображение</li>
                        </ol>
                        Возвращает на выходе: лучшие bbox по выставленным параметрам
                    """,
                    "PlotBBoxes": """
                        <b>Наложение bbox</b> на изображение YOLOV3 и V4 
                        Необходимые <a href="https://google.com" target="_blank">связи</a> с другими блоками на входе:
                        <ol>
                        <li>блок Function Постобработка yolo или блок Custom Трекер (Sort, DeepSort)</li>  
                        <li>блок Input исходных изображений или видео</li>
                        </ol>
                        Возможные связи с другими блоками на выходе:
                        <ol>
                        <li>блок Output</li>
                        </ol>
                        Возвращает на выходе: исходное изображение (фрейм) с наложенными bbox
                    """,
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
    BlockGroupChoice.Service: {
        "main": [
            {
                "type": "select",
                "name": "group",
                "label": "Группа",
                "parse": "parameters[main][group]",
                "value": BlockServiceGroupChoice.Tracking,
                "list": list(
                    map(
                        lambda item: {"value": item.name, "label": item.value},
                        list(BlockServiceGroupChoice),
                    )
                ),
                "fields": {
                    BlockServiceGroupChoice.Tracking: [
                        {
                            "type": "select",
                            "name": "type",
                            "label": "Выбор типа",
                            "parse": "parameters[main][type]",
                            "value": BlockServiceTypeChoice.Sort,
                            "list": list(
                                map(
                                    lambda item: {
                                        "value": item.name,
                                        "label": item.value,
                                    },
                                    list(BlockServiceTypeChoice),
                                )
                            ),
                        },
                    ]
                },
                "manual": {
                    "Sort": """
                        <b>Алгоритм трекера Sort</b> для моделей object_detection 
                        Необходимые <a href="https://google.com" target="_blank">связи</a> с другими блоками на входе:
                        <ol>
                        <li>блок Function Постобработка yolo</li>
                        </ol>  
                        Возможные <a href="https://google.com" target="_blank">связи</a> с другими блоками на выходе:
                        <ol>
                        <li>блок Function Наложение bbox на изображение</li>
                        </ol>
                        Возвращает на выходе: Возвращает аналогичный массив bbox, где последний столбец - это идентификатор объекта.
                    """,
                    "DeepSort": """
                        <b>Алгоритм трекера DeepSort</b> для моделей object_detection 
                        Необходимые <a href="https://google.com" target="_blank">связи</a> с другими блоками на входе:
                        <ol>
                        <li>блок Function Постобработка yolo</li>
                        </ol>  
                        Возможные <a href="https://google.com" target="_blank">связи</a> с другими блоками на выходе:
                        <ol> 
                        <li>блок Function Наложение bbox на изображение</li>
                        </ol>
                        Возвращает на выходе: Возвращает аналогичный массив bbox, где последний столбец - это идентификатор объекта.
                    """
                }
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
    }
}
