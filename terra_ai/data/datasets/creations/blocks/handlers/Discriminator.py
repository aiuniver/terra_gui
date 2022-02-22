from typing import Optional

from terra_ai.data.mixins import BaseMixinData


class OptionsData(BaseMixinData):

    # Внутренние параметры
    shape: Optional[tuple] = (1,)
    deploy: Optional[bool] = False
