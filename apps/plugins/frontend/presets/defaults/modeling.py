from terra_ai.data.modeling.extra import LayerTypeChoice


ModelingLayerForm = [
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
        "type": "select",
        "label": "Тип данных",
        "name": "datatype_input",
        "parse": "datatype_input",
        "list": [],
    },
    {
        "type": "select",
        "label": "Тип данных",
        "name": "datatype_output",
        "parse": "datatype_output",
        "list": [],
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
]

ModelingLayersTypes = {}
