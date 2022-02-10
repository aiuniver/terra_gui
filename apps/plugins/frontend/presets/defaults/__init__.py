from terra_ai.data.modeling.layers import Layer, types as layers_types
from terra_ai.data.modeling.extra import LayerTypeChoice
from terra_ai.data.training.extra import ArchitectureChoice

from apps.plugins.frontend.utils import prepare_pydantic_field
from apps.plugins.frontend.presets.defaults.datasets import (
    DatasetsColumnProcessing,
    DatasetsInput,
    DatasetsOutput,
)
from apps.plugins.frontend.presets.defaults.modeling import (
    ModelingLayerForm,
    ModelingLayersTypes,
)
from apps.plugins.frontend.presets.defaults.cascades import (
    CascadesBlockForm,
    CascadesBlocksTypes,
)
from apps.plugins.frontend.presets.defaults.deploy import (
    DeployTypeGroup,
    DeployServerGroup,
)


Defaults = {
    "datasets": {
        "creation": {
            "column_processing": DatasetsColumnProcessing,
            "input": DatasetsInput,
            "output": DatasetsOutput,
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
