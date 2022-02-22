from typing import Optional

from terra_ai.data.mixins import BaseMixinData


class OptionsData(BaseMixinData):

    # Внутренние параметры
    shape: Optional[tuple]
    deploy: Optional[bool] = False
