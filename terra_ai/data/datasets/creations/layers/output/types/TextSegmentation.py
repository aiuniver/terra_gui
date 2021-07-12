from typing import Optional

from ......mixins import BaseMixinData


class ParametersData(BaseMixinData):
    open_tags: Optional[str]
    close_tags: Optional[str]
