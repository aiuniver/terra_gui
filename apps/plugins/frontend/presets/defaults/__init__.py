from terra_ai.data.modeling.layers import Layer, types as layers_types
from terra_ai.data.modeling.extra import LayerTypeChoice
from terra_ai.data.training.extra import ArchitectureChoice

from ...utils import prepare_pydantic_field
from .datasets import DataSetsColumnProcessing, DataSetsInput, DataSetsOutput
from .modeling import ModelingLayerForm, ModelingLayersTypes
from .cascades import CascadesBlockForm, CascadesBlocksTypes
from .deploy import DeployTypeGroup, DeployServerGroup


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
    "training": {"architecture": ArchitectureChoice.Basic},
    "cascades": {
        "block_form": CascadesBlockForm,
        "blocks_types": CascadesBlocksTypes,
    },
    "deploy": {
        "type": DeployTypeGroup,
        "server": DeployServerGroup,
    },
}


def __get_group_type_params(data, group) -> list:
    output = []
    for name in data.__fields__:
        output.append(
            prepare_pydantic_field(
                data.__fields__[name], f"parameters[{group}][{name}]"
            )
        )
    return output


for layer in Layer:
    params = getattr(layers_types, layer.name)
    Defaults["modeling"]["layers_types"].update(
        {
            LayerTypeChoice[layer.name].value: {
                "main": __get_group_type_params(params.ParametersMainData, "main"),
                "extra": __get_group_type_params(params.ParametersExtraData, "extra"),
            }
        }
    )
