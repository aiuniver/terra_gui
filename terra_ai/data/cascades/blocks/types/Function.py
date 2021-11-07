from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.cascades.blocks.extra import BlockFunctionGroupChoice


class ParametersMainData(BaseMixinData):
    group: BlockFunctionGroupChoice = BlockFunctionGroupChoice.Image
