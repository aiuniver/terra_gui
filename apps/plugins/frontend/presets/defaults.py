from terra_ai.data.modeling.layers import Layer, types
from terra_ai.data.modeling.extra import LayerTypeChoice
from terra_ai.data.training.extra import ArchitectureChoice
from defaults.training import Architectures
from defaults.datasets import DataSetsColumnProcessing, DataSetsInput, DataSetsOutput
from defaults.modeling import ModelingLayerForm, ModelingLayersTypes
from ..utils import prepare_pydantic_field


Defaults = {
    "datasets": {
        "creation": {
            "column_processing": DataSetsColumnProcessing,
            "input": DataSetsInput,
            "output": DataSetsOutput,
        },
    },
    "modeling": {
        "layer_form": ModelingLayerForm,
        "layers_types": ModelingLayersTypes,
    },
    "training": {"base": Architectures.get(ArchitectureChoice.Basic)},
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
