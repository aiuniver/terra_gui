from typing import Dict, Optional

from terra_ai.logging import logger
from terra_ai.data.datasets.extra import LayerGroupChoice
from terra_ai.data.datasets.creation import CreationBlockList


class DatasetCreationValidate:
    _group: LayerGroupChoice
    _blocks: CreationBlockList

    def __init__(self, group: LayerGroupChoice, blocks: CreationBlockList):
        self._group = group
        self._blocks = blocks

        logger.info(self.__class__.__name__)
        logger.info(f"... Type: {self._group}")
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
