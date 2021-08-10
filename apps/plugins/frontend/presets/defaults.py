from terra_ai.data.modeling.layers import Layer, types
from terra_ai.data.modeling.extra import LayerTypeChoice

from ..utils import prepare_pydantic_field
from ..choices import (
    LayerInputTypeChoice,
    LayerOutputTypeChoice,
    LayerNetChoice,
    LayerScalerChoice,
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
                                "label": "Высота",
                                "name": "width",
                                "parse": "width",
                            },
                            {
                                "type": "number",
                                "label": "Ширина",
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
                        ]
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
