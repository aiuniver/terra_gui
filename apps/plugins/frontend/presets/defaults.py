from terra_ai.data.modeling.layers import Layer
from terra_ai.data.modeling.layers import types
from terra_ai.data.modeling.extra import LayerTypeChoice

from ..utils import prepare_pydantic_field


Defaults = {
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
                        lambda item: {"value": item, "label": item},
                        LayerTypeChoice.values(),
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
            layer.name: {
                "main": __get_layer_type_params(params.ParametersMainData, "main"),
                "extra": __get_layer_type_params(params.ParametersExtraData, "extra"),
            }
        }
    )
