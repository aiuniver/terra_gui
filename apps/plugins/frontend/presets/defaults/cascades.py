from terra_ai.data.cascades.extra import BlockGroupChoice
from terra_ai.data.cascades.blocks.extra import (
    BlockServiceGroupChoice,
    BlockServiceTypeChoice,
    ChangeTypeAvailableChoice,
    PostprocessBoxesMethodAvailableChoice,
    BlockServiceBiTBasedTrackerMetricChoice,
    BlockServiceYoloV5VersionChoice,
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
    BlockFunctionTypeChoice.PutTag: [
        {
            "type": "number",
            "name": "alpha",
            "parse": "parameters[main][alpha]",
            "label": "Альфа - порог вероятности",
            "value": 0.5,
        },
    ],
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


ServiceTypesFields = {
    BlockServiceTypeChoice.Sort: [
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
    ],
    BlockServiceTypeChoice.BiTBasedTracker: [
        {
            "type": "number",
            "name": "max_age",
            "label": "Количество кадров для остановки слежения",
            "parse": "parameters[main][max_age]",
            "value": 4,
        },
        {
            "type": "number",
            "name": "distance_threshold",
            "label": "Порог сходства объектов",
            "parse": "parameters[main][distance_threshold]",
            "value": 0.4,
        },
        {
            "type": "select",
            "name": "metric",
            "parse": "parameters[main][metric]",
            "label": "Метрика сравнения сходства",
            "value": BlockServiceBiTBasedTrackerMetricChoice.euclidean,
            "list": list(
                map(
                    lambda item: {"value": item.name, "label": item.value},
                    list(BlockServiceBiTBasedTrackerMetricChoice),
                )
            ),
        },
    ],
    BlockServiceTypeChoice.YoloV5: [
        {
            "type": "select",
            "name": "version",
            "parse": "parameters[main][version]",
            "label": "Версия модели",
            "value": BlockServiceYoloV5VersionChoice.Small,
            "list": list(
                map(
                    lambda item: {"value": item.name, "label": item.value},
                    list(BlockServiceYoloV5VersionChoice),
                )
            ),
        },
        {
            "type": "checkbox",
            "name": "render_img",
            "label": "Выводить изображение",
            "parse": "parameters[main][render_img]",
            "value": True,
        },
    ]
}


ServiceGroupTypeRel = {
    BlockServiceGroupChoice.Tracking: [
        BlockServiceTypeChoice.Sort,
        BlockServiceTypeChoice.BiTBasedTracker,
    ],
    BlockServiceGroupChoice.ObjectDetection: [
        BlockServiceTypeChoice.YoloV5,
    ]
}


def get_type_field(type_name, group_type_rel, types_fields) -> dict:
    items = list(group_type_rel.get(type_name))
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
                map(lambda item: (item.name, types_fields.get(item, [])), items)
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
                "value": LayerInputTypeChoice.Video.name,
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
                            "label": "Разделить по кадрам",
                            "parse": "parameters[main][switch_on_frame]",
                        },
                    ]
                },
                "manual": {
                    "Video": """
                        <p>Видео - тип данных, который будет использоваться для обработки последующими блоками и каскадом в целом.
                        Включенный переключатель “Разделить по кадрам” разделяет видео на кадры, для дальнейшего использования с моделями обученными 
                        на изображениях (Object Detection,  Cегментации и т.п.)</p>
                        <p>Возможные связи с другими блоками на входе:<br />
                            <code>None</code>
                        </p>
                        <p>Возможные связи с другими блоками на выходе:</p>
                        <ol> 
                            <li>блок Model модель object detection или сегментации</li>
                            <li>блок Service модель object detection или сегментации</li>
                            <li>блок Function Наложение bbox на изображение</li>
                            <li>блок Function Постобработка yolo</li>
                        </ol>
                        <p>Возвращает на выходе:<br />
                            <code>Фреймы из видео</code>
                        </p>
                    """
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
                "value": LayerInputTypeChoice.Video.name,
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
                        <p>Сохранение результата в видео файл</p>
                        <p>Необходимые связи с другими блоками на входе:</p>
                        <ol>
                            <li>блок Function Наложение bbox на изображение или блок Service OD YoloV5 (активирован переключатель “Выводить изображение”)</li>
                        </ol>
                        <p>Возможные связи с другими блоками на выходе:<br />
                            <code>None</code>
                        </p>
                        <p>Возвращает на выходе:<br />
                            <code>Сохраняет переданные фреймы исходного видео в видеофайл с выставленными параметрами</code>
                        </p>
                    """,
                    "Image": """
                        <p>Сохранение результата в файл изображения</p>
                        <p>Необходимые связи с другими блоками на входе:</p>
                        <ol>
                            <li>блок Function Наложение bbox на изображение или</li>
                            <li>блок Function Наложение маски по классу на изображение или</li>
                            <li>блок Function Наложение маски всех классов по цветам.</li>
                        </ol>
                        <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgiIGhlaWdodD0iMjgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTIxLjQxOCA4LjA3Yy44MTUgMCAxLjQ4Mi0uNjY2IDEuNDgyLTEuNDgyIDAtLjgxNS0uNjY3LTEuNDgyLTEuNDgyLTEuNDgySDYuNTk0Yy0uODE1IDAtMS40ODIuNjY3LTEuNDgyIDEuNDgyIDAgLjgxNi42NjcgMS40ODMgMS40ODIgMS40ODNoMTQuODIzem0tOC44OTUgMTMuMzQyYzAgLjgxNS42NjggMS40ODIgMS40ODMgMS40ODIuODE1IDAgMS40ODItLjY2NyAxLjQ4Mi0xLjQ4MnYtOC42NDdjMC0uODE2LS42NjctMS40ODMtMS40ODItMS40ODMtLjgxNiAwLTEuNDgzLjY2Ny0xLjQ4MyAxLjQ4M3Y4LjY0N3oiIGZpbGw9IiM2NUI5RjQiLz48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZD0iTTI3LjE3Ny44MjRILjgyM3YyNi4zNTJoMjYuMzUyVi44MjV6TTAgMHYyOGgyOFYwSDB6IiBmaWxsPSIjNjVCOUY0Ii8+PC9zdmc+" width="100%" height="100px" alt="" />
                        <p>Возможные связи с другими блоками на выходе:<br />
                            <code>None</code>
                        </p>
                        <p>Возвращает на выходе:<br />
                            <code>Сохраняет переданные обработанные изображения</code>
                        </p>
                    """,
                    "Text": """
                        <p>Сохранение результата в текстовый файл</p>
                        <p>Необходимые связи с другими блоками на входе:</p>
                        <ol>
                            <li>блок Function Расстановка тэгов по вероятностям из модели или</li>
                            <li>блок Service  speech_to_text</li>
                        </ol>
                        <p>Возможные связи с другими блоками на выходе:<br />
                            <code>None</code>
                        </p>
                        <p>Возвращает на выходе:<br />
                            <code>Сохраняет обработанный текст</code>
                        </p>
                    """,
                    "Audio": """
                        <p>Сохранение результата в текстовый файл</p>
                        <p>Необходимые связи с другими блоками на входе:</p>
                        <ol>
                            <li>блок Service text_to_speech</li>
                        </ol>
                        <p>Возможные связи с другими блоками на выходе:<br />
                            <code>None</code>
                        </p>
                        <p>Возвращает на выходе:<br />
                            <code>Сохраняет переданное аудио</code>
                        </p>
                    """,
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
                "value": BlockFunctionGroupChoice.ObjectDetection,
                "list": list(
                    map(
                        lambda item: {"value": item.name, "label": item.value},
                        list(BlockFunctionGroupChoice),
                    )
                ),
                "fields": {
                    BlockFunctionGroupChoice.Image: get_type_field(
                        BlockFunctionGroupChoice.Image,
                        FunctionGroupTypeRel,
                        FunctionTypesFields,
                    ),
                    BlockFunctionGroupChoice.Text: get_type_field(
                        BlockFunctionGroupChoice.Text,
                        FunctionGroupTypeRel,
                        FunctionTypesFields,
                    ),
                    BlockFunctionGroupChoice.Audio: get_type_field(
                        BlockFunctionGroupChoice.Audio,
                        FunctionGroupTypeRel,
                        FunctionTypesFields,
                    ),
                    BlockFunctionGroupChoice.Video: get_type_field(
                        BlockFunctionGroupChoice.Video,
                        FunctionGroupTypeRel,
                        FunctionTypesFields,
                    ),
                    BlockFunctionGroupChoice.Array: get_type_field(
                        BlockFunctionGroupChoice.Array,
                        FunctionGroupTypeRel,
                        FunctionTypesFields,
                    ),
                    BlockFunctionGroupChoice.ObjectDetection: get_type_field(
                        BlockFunctionGroupChoice.ObjectDetection,
                        FunctionGroupTypeRel,
                        FunctionTypesFields,
                    ),
                    BlockFunctionGroupChoice.Segmentation: get_type_field(
                        BlockFunctionGroupChoice.Segmentation,
                        FunctionGroupTypeRel,
                        FunctionTypesFields,
                    ),
                    BlockFunctionGroupChoice.TextSegmentation: get_type_field(
                        BlockFunctionGroupChoice.TextSegmentation,
                        FunctionGroupTypeRel,
                        FunctionTypesFields,
                    ),
                },
                "manual": {
                    "ChangeType": """
                        <p>Изменение типа данных массива текстовых данных по указанным параметрам.</p>
                        <p>Необходимые связи с другими блоками на входе:</p>
                        <ol>
                            <li>любой блок (кроме Input с параметром Текст или Таблица)</li>
                        </ol>
                            <p>Возможные связи с другими блоками на выходе:</p> 
                        <ol>
                            <li>блок Output</li>
                            <li>блок Model или Service</li>
                            <li>блок Function</li>
                        </ol>
                        <p>Возвращает на выходе:<br />
                            <code>массивы данных указанного типа</code>
                        </p>
                    """,
                    "ChangeSize": """
                        <p>Изменение размера изображения по указанным параметрам.</p>
                        <p>Необходимые связи с другими блоками на входе:</p>
                        <ol>
                            <li>любой блок возвращающий изображение</li>
                        </ol>
                        <p>Возможные связи с другими блоками на выходе:</p>
                        <ol>
                            <li>блок Output метод Видео или Изображение</li> 
                            <li>блок Model или Service</li>
                            <li>блок Function</li>
                        </ol>
                        <p>Возвращает на выходе:<br />
                            <code>маскированное изображение</code>
                        </p>
                    """,
                    "MinMaxScale": """
                        <p>Нормализация массива данных по указанным параметрам.</p>
                        <p>Необходимые связи с другими блоками на входе:</p>
                        <ol>
                            <li>любой блок (кроме Input с параметром Текст или Таблица)</li>
                        </ol>
                        <p>Возможные связи с другими блоками на выходе:</p>
                        <ol>
                            <li>блок Output</li>
                            <li>блок Model или Service</li>
                            <li>блок Function</li>
                        </ol>
                        <p>Возвращает на выходе:<br />
                            <code>нормализованные массивы данных</code>
                        </p>
                    """,
                    "MaskedImage": """
                        <p>Наложение маски по указанному классу на изображение. Необходимо указать Id класса.</p>
                        <p>Необходимые связи с другими блоками на входе:</p>
                        <ol>
                            <li>блок Model или Service c моделью сегментации изображений</li>
                            <li>блок Input исходных изображений или видео (по кадрам)</li>
                        </ol>
                        <p>Возможные связи с другими блоками на выходе:</p>
                        <ol>
                            <li>блок Output метод Видео или Изображение</li>
                            <li>блок Function Изменение размера данных</li>
                        </ol>
                        <p>Возвращает на выходе:<br />
                            <code>маскированное изображение</code>
                        </p>
                    """,
                    "PlotMaskSegmentation": """
                        <p>Наложение маски по всем классам на изображение</p>
                        <p>Необходимые связи с другими блоками на входе:</p>
                        <ol>
                            <li>блок Model или Service c моделью сегментации изображений</li> 
                            <li>блок Input исходных изображений или видео (по кадрам)</li>
                        </ol>
                        <p>Возможные связи с другими блоками на выходе:</p>
                        <ol>
                            <li>блок Output метод Видео или Изображение</li>
                            <li>блок Function Изменение размера данных</li>
                        </ol>
                        <p>Возвращает на выходе:<br />
                            <code>маскированное изображение</code>
                        </p>
                    """,
                    "PutTag": """
                        <p>Наложение маски по всем классам на изображение</p>
                        <p>Необходимые связи с другими блоками на входе:</p>
                        <ol>
                            <li>блок Model c моделью сегментации текста</li>
                        </ol>
                        <p>Возможные связи с другими блоками на выходе:</p>
                        <ol>
                            <li>блок Output метод Текст</li>
                        </ol>
                        <p>Возвращает на выходе:<br />
                            <code>размеченный тегами текст</code>
                        </p>
                    """,
                    "PostprocessBoxes": """
                        <p>Постобработка</b> для моделей YOLOV3 и V4</p>
                        <p>Необходимые связи с другими блоками на входе:</p>
                        <ol>
                            <li>блок Model c моделью YOLO</li>
                            <li>блок Input исходных изображений или видео</li>
                        </ol>
                        <p>Возможные связи с другими блоками на выходе:</p>
                        <ol> 
                            <li>блок Service Tracking (Sort, BiTBasedTracker)</li>
                            <li>блок Function Наложение bbox на изображение</li>
                        </ol>
                        <p>Возвращает на выходе:<br />
                            <code>лучшие bbox по выставленным параметрам</code>
                        </p>
                    """,
                    "PlotBBoxes": """
                        <p>Наложение bbox</b> на изображение YOLOV3 и V4</p>
                        <p>Необходимые связи с другими блоками на входе:</p>
                        <ol>
                            <li>блок Function Постобработка yolo или блок Service Tracking (Sort, BiTBasedTracker)</li>  
                            <li>блок Input исходных изображений или видео</li>
                        </ol>
                        <p>Возможные связи с другими блоками на выходе:</p>
                        <ol>
                            <li>блок Output</li>
                        </ol>
                        <p>Возвращает на выходе:<br />
                            <code>исходное изображение (фрейм) с наложенными bbox</code>
                        </p>
                    """,
                },
            },
        ]
    },
    BlockGroupChoice.Custom: {"main": []},
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
                    BlockServiceGroupChoice.Tracking: get_type_field(
                        BlockServiceGroupChoice.Tracking,
                        ServiceGroupTypeRel,
                        ServiceTypesFields,
                    ),
                    BlockServiceGroupChoice.ObjectDetection: get_type_field(
                        BlockServiceGroupChoice.ObjectDetection,
                        ServiceGroupTypeRel,
                        ServiceTypesFields,
                    ),
                },
                "manual": {
                    "Sort": """
                        <p>Алгоритм трекера Sort</b> для моделей object_detection</p>
                        <p>Необходимые связи с другими блоками на входе:</p>
                        <ol>
                            <li>блок Function Постобработка yolo или блок Service YoloV5 (выключен переключатель “Выводить изображение”)</li>
                        </ol>  
                        <p>Возможные связи с другими блоками на выходе:</p>
                        <ol>
                            <li>блок Function Наложение bbox на изображение</li>
                        </ol>
                        <p>Возвращает на выходе:<br />
                            <code>возвращает аналогичный массив bbox, где последний столбец - это идентификатор объекта</code>
                        </p>
                    """,
                    "BiTBasedTracker": """
                        <p>Алгоритм трекера BiTBasedTracker</b> для моделей object_detection</p>
                        <p>Необходимые связи с другими блоками на входе:</p>
                        <ol>
                            <li>блок Function Постобработка yolo или блок Service YoloV5 (выключен переключатель “Выводить изображение”)</li>
                            <li>блок Input исходных изображений</li>
                        </ol>
                        <p>Возможные связи с другими блоками на выходе:</p>
                        <ol> 
                            <li>блок Function Наложение bbox на изображение</li>
                        </ol>
                        <p>Возвращает на выходе:<br />
                            <code>возвращает аналогичный массив bbox, где последний столбец - это идентификатор объекта</code>
                        </p>
                    """,
                    "YoloV5": f"""
                        <li>Предобученная YoloV5 на базе COCO:<li>
                        <p>“Версия модели” выбирается в зависимости от необходимой точности.</p>
                        <p>Переключатель “Выводить изображение” при включенном состоянии выводит исходное изображение с наложенными bbox, 
                        классами и вероятностями. В выключенном положении блок будет возвращать лучшие bbox по всем классам.</p>
                        <p>Необходимые связи с другими блоками на входе:</p>
                        <ol>
                        <li>блок Input исходных изображений (фреймов видео)</li>
                        </ol>
                        <p>Возможные связи с другими блоками на выходе:</p>
                        <ol>
                        <li>блок Function ObjectDetection Фильтрация классов Service YoloV5 (выключен переключатель “Выводить изображение”)</li>
                        <li>блок Output сохранение изображений или видео  (активирован переключатель “Выводить изображение”)</li>
                        <li>блок Service Tracking (Sort, DeepSort)(выключен переключатель “Выводить изображение”)</li>
                        </ol>
                        <p>Возвращает на выходе:<br />
                            <code>если активирован переключатель “Выводить изображение” то исходное изображение (фрейм) 
                            с наложенными bbox, иначе возвращает массив bbox, где последний столбец - это идентификатор объекта</code>
                        </p>
                    """,
                },
            },
        ]
    },
}
