from typing import Any, Optional

from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.datasets.extra import LayerHandlerChoice


class BlockDataParameters(BaseMixinData):
    select_type: Any
    select_data: Any


class BlockHandlerParameters(BaseMixinData):
    type: Optional[LayerHandlerChoice]
    options: Any


class BlockLayerParameters(BaseMixinData):
    pass
