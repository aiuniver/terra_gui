from terra_ai.data.mixins import BaseMixinData


class MinMaxScalerData(BaseMixinData):
    min_scaler: int = 0
    max_scaler: int = 1
