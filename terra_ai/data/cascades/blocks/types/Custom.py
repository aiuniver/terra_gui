from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.cascades.blocks.extra import BlockCustomGroupChoice


class ParametersMainData(BaseMixinData):
    group: BlockCustomGroupChoice = BlockCustomGroupChoice.Tracking
