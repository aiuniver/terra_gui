from pydantic import PositiveInt

from terra_ai.data.datasets.creations.blocks.extra import BaseOptionsData


class OptionsData(BaseOptionsData):
    length: PositiveInt
    step: PositiveInt
    trend_limit: str

    # Внутренние параметры
    depth: PositiveInt = 1
