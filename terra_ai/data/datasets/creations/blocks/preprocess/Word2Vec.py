from terra_ai.data.datasets.creations.blocks.extra import BaseOptionsData
from terra_ai.data.datasets.extra import LayerPrepareMethodChoice


class OptionsData(BaseOptionsData):
    size: int
    window: int
    min_count: int
    workers: int
    iter: int
    # Внутренние параметры
    prepare_method: LayerPrepareMethodChoice = LayerPrepareMethodChoice.word_to_vec
