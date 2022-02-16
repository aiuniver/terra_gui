from terra_ai.data.datasets.extra import LayerScalerDefaultChoice
from terra_ai.data.datasets.creations.blocks.extra import MinMaxScalerData


class OptionsData(MinMaxScalerData):
    """
    Обработчик типа задачи "регрессия".
    Inputs:
        scaler: str - тип скейлера. Варианты: 'standard_scaler', 'min_max_scaler'
    """

    scaler: LayerScalerDefaultChoice
