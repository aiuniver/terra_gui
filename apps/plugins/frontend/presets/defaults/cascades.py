from terra_ai.data.cascades.extra import BlockGroupChoice


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
                lambda item: {"value": item.value, "label": item.name},
                list(BlockGroupChoice),
            )
        ),
        "visible": False,
    },
]


CascadesBlocksTypes = {
    BlockGroupChoice.InputData: {},
    BlockGroupChoice.OutputData: {},
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
    BlockGroupChoice.Custom: {
        "main": [
            {
                "type": "select",
                "name": "group",
                "label": "Группа",
                "parse": "parameters[main][group]",
                "list": [{"value": "Tracking", "label": "Tracking"}],
            },
            {
                "type": "select",
                "name": "type",
                "label": "Выбор типа",
                "parse": "parameters[main][type]",
                "list": [{"value": "Sort", "label": "Sort"}],
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
    BlockGroupChoice.Function: {
        "main": [
            {
                "type": "select",
                "name": "group",
                "label": "Группа",
                "parse": "parameters[main][group]",
                "list": [
                    {"value": "Image", "label": "Image"},
                    {"value": "Text", "label": "Text"},
                    {"value": "Audio", "label": "Audio"},
                    {"value": "Video", "label": "Video"},
                    {"value": "Array", "label": "Array"},
                    {"value": "Segmentation", "label": "Segmentation"},
                    {"value": "TextSegmentation", "label": "TextSegmentation"},
                    {"value": "ObjectDetection", "label": "ObjectDetection"},
                ],
            },
            {
                "type": "select",
                "name": "type",
                "label": "Выбор типа",
                "parse": "parameters[main][type]",
                "list": [
                    {"value": "ChangeType", "label": "Изменение типа данных"},
                    {"value": "ChangeSize", "label": "Изменение размера данных"},
                    {"value": "MinMaxScale", "label": "Нормализация (скелер)"},
                    {
                        "value": "MaskedImage",
                        "label": "Наложение маски по классу на изображение",
                    },
                    {
                        "value": "PlotMaskSegmentation",
                        "label": "Наложение маски всех классов по цветам",
                    },
                    {
                        "value": "PutTag",
                        "label": "Растановка тэгов по вероятностям из модели",
                    },
                    {"value": "PostprocessBoxes", "label": "Постобработка yolo"},
                    {"value": "PlotBBoxes", "label": "Наложение bbox на изображение"},
                ],
            },
            {
                "type": "select",
                "name": "change_type",
                "label": "Выбор типа",
                "parse": "parameters[main][change_type]",
                "value": "int",
                "list": [
                    {"value": "int", "label": "int"},
                    {"value": "int8", "label": "int8"},
                    {"value": "int32", "label": "int32"},
                    {"value": "int64", "label": "int64"},
                    {"value": "uint", "label": "uint"},
                    {"value": "uint8", "label": "uint8"},
                    {"value": "uint16", "label": "uint16"},
                    {"value": "uint32", "label": "uint32"},
                    {"value": "uint64", "label": "uint64"},
                    {"value": "float", "label": "float"},
                    {"value": "float16", "label": "float16"},
                    {"value": "float32", "label": "float32"},
                    {"value": "float64", "label": "float64"},
                    {"value": "complex", "label": "complex"},
                    {"value": "complex64", "label": "complex64"},
                    {"value": "complex128", "label": "complex128"},
                    {"value": "bool", "label": "bool"},
                ],
            },
            {
                "type": "text_array",
                "name": "shape",
                "label": "Размерность",
                "parse": "parameters[main][shape]",
            },
            {
                "type": "number",
                "name": "min_scale",
                "label": "Минимальное значение",
                "parse": "parameters[main][min_scale]",
                "value": 0,
            },
            {
                "type": "number",
                "name": "max_scale",
                "label": "Максимальное значение",
                "parse": "parameters[main][max_scale]",
                "value": 1,
            },
            {
                "type": "number",
                "name": "class_id",
                "label": "ID класса",
                "parse": "parameters[main][class_id]",
                "value": 0,
            },
            {
                "type": "text_array",
                "name": "classes_colors",
                "label": "Цвета классов",
                "parse": "parameters[main][classes_colors]",
            },
            {
                "type": "text_array",
                "name": "open_tag",
                "label": "Открывающие тэги",
                "parse": "parameters[main][open_tag]",
            },
            {
                "type": "text_array",
                "name": "close_tag",
                "label": "Закрывающие тэги",
                "parse": "parameters[main][close_tag]",
            },
            {
                "type": "number",
                "name": "alpha",
                "label": "Альфа - порог вероятности",
                "parse": "parameters[main][alpha]",
                "value": 0.5,
            },
            {
                "type": "number",
                "name": "score_threshold",
                "label": "Порог вероятности классов",
                "parse": "parameters[main][score_threshold]",
                "value": 0.3,
            },
            {
                "type": "number",
                "name": "iou_threshold",
                "label": "Порог пересечения",
                "parse": "parameters[main][iou_threshold]",
                "value": 0.45,
            },
            {
                "type": "select",
                "name": "method",
                "label": "Метод подавления немаксимумов",
                "parse": "parameters[main][method]",
                "list": [
                    {"value": "nms", "label": "nms"},
                    {"value": "soft_nms", "label": "soft_nms"},
                ],
            },
            {
                "type": "number",
                "name": "sigma",
                "label": "Коэффициент сглаживания",
                "parse": "parameters[main][sigma]",
                "value": 0.3,
            },
            {
                "type": "text_array",
                "name": "classes",
                "label": "Имена классов",
                "parse": "parameters[main][classes]",
            },
            {
                "type": "text_array",
                "name": "colors",
                "label": "Цвета классов",
                "parse": "parameters[main][colors]",
            },
            {
                "type": "number",
                "name": "line_thickness",
                "label": "Толщина линии рамки",
                "parse": "parameters[main][line_thickness]",
            },
        ]
    },
}
