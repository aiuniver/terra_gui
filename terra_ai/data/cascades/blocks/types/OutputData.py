from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.cascades.blocks.extra import BlockOutputDataSaveAsChoice


class ParametersMainData(BaseMixinData):
    save_as: BlockOutputDataSaveAsChoice = BlockOutputDataSaveAsChoice.source
