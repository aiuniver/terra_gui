from terra_ai.data.datasets.extra import LayerInputTypeChoice, LayerOutputTypeChoice
from terra_ai.data.modeling.layers import Layer, types
from terra_ai.data.modeling.extra import LayerTypeChoice

from ..utils import prepare_pydantic_field


Defaults = {
    "datasets": {
        "creation": {
            "inputs": {
                "base": [
                    {
                        "type": "text",
                        "name": "name",
                        "label": "Название входа",
                        "parse": "name",
                    },
                    {
                        "type": "select",
                        "name": "type",
                        "label": "Тип данных",
                        "parse": "type",
                        "list": list(
                            map(
                                lambda item: {"value": item.value, "label": item.name},
                                list(LayerInputTypeChoice),
                            )
                        ),
                    },
                ],
                "type": {},
            },
            "outputs": {
                "base": [
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
                        "list": list(
                            map(
                                lambda item: {"value": item.value, "label": item.name},
                                list(LayerOutputTypeChoice),
                            )
                        ),
                    },
                ],
                "type": {},
            },
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
