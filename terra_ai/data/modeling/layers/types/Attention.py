"""
## Тип слоя `Attention`
"""

from ....mixins import BaseMixinData


# LayerConfig = LayerConfigData(
#     num_uplinks=[1, "4+"],
#     input_dimension=2,
#     module=ModuleChoice.tensorflow_keras_layers,
#     module_type=ModuleTypeChoice.keras,
# )


class ParametersMainData(BaseMixinData):
    pass


class ParametersExtraData(BaseMixinData):
    use_scale: bool = False
