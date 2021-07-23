from terra_ai.data.modeling.layers import Layer
from terra_ai.data.modeling.layers import types

from ..utils import prepare_pydantic_field


Defaults = {
    "modeling": {"layers_types": {}},
}


def __get_params(data, group) -> list:
    output = []
    for name in data.__fields__:
        output.append(
            prepare_pydantic_field(
                data.__fields__[name], f"layers[%s][parameters][{group}][{name}]"
            )
        )
    return output


for layer in Layer:
    params = getattr(types, layer.name)
    Defaults["modeling"]["layers_types"].update(
        {
            layer.name: {
                "main": __get_params(params.ParametersMainData, "main"),
                "extra": __get_params(params.ParametersExtraData, "extra"),
            }
        }
    )
