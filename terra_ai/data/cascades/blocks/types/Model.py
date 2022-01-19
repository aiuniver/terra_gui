from typing import Optional
from terra_ai.data.mixins import BaseMixinData


class ParametersMainData(BaseMixinData):
    path: Optional[str]
    postprocess: bool = True
