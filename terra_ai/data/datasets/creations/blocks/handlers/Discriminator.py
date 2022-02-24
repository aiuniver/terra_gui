from typing import Optional

from terra_ai.data.datasets.creations.blocks.extra import BaseOptionsData


class OptionsData(BaseOptionsData):

    # Внутренние параметры
    shape: Optional[tuple] = (1,)
    deploy: Optional[bool] = False
