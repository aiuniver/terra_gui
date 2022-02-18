from typing import Dict, Optional

from terra_ai.logging import logger
from terra_ai.data.datasets.extra import LayerGroupChoice
from terra_ai.data.datasets.creation import (
    CreationBlockList,
    CreationValidateBlocksData,
)


class DatasetCreationValidate:
    _type: LayerGroupChoice
    _blocks: CreationBlockList

    def __init__(self, data: CreationValidateBlocksData):
        self._type = data.type
        self._blocks = data.items

        logger.info(self.__class__.__name__)
        logger.info(f"... Type: {self._type}")
        logger.info(f"... Items: {self._blocks.json(ensure_ascii=False)}")

    def validate(self) -> Dict[int, Optional[str]]:
        """
        Запуск валидации блоков

        :return:
            Dict[int, Optional[str]]:
                ключ - id блока
                значение - либо строка ошибки, либо None, если ошибки нет
        """
        return dict(map(lambda item: (item.id, None), self._blocks))
