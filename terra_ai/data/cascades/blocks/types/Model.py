from terra_ai.data.mixins import BaseMixinData


class ParametersMainData(BaseMixinData):
    path: str
    postprocess: bool = True
