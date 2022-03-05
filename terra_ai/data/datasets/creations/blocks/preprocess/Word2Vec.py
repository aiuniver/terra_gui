from terra_ai.data.datasets.creations.blocks.extra import BaseOptionsData


class OptionsData(BaseOptionsData):
    size: int
    window: int
    min_count: int
    workers: int
    iter: int
