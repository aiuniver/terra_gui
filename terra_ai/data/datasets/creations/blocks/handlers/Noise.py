from typing import Optional

from terra_ai.data.datasets.creations.blocks.extra import BaseOptionsData


class OptionsData(BaseOptionsData):

    # Внутренние параметры
    shape: Optional[tuple]
    deploy: Optional[bool] = False
