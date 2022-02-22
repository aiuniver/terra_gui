from typing import Dict, Optional

from terra_ai.logging import logger
from terra_ai.data.datasets.extra import LayerGroupChoice
from terra_ai.data.datasets.creation import CreationBlockList
from terra_ai.data.training.extra import ArchitectureChoice


class DatasetCreationValidate:
    _group: LayerGroupChoice
    _architecture: ArchitectureChoice
    _blocks: CreationBlockList

    def __init__(
        self,
        group: LayerGroupChoice,
        architecture: ArchitectureChoice,
        blocks: CreationBlockList,
    ):
        self._group = group
        self._architecture = architecture
        self._blocks = blocks

        logger.info(self.__class__.__name__)
        logger.info(f"... Type: {self._group}")
        logger.info(f"... Architecture: {self._architecture}")
        logger.info(f"... Items: {self._blocks.json(ensure_ascii=False)}")

    def validate(self) -> Dict[int, Optional[str]]:
        """
        Запуск валидации блоков

        :return:
            Dict[int, Optional[str]]:
                ключ - id блока
                значение - либо строка ошибки, либо None, если ошибки нет
        """
        errors = dict(map(lambda item: (item.id, None), self._blocks))
        if self._group == LayerGroupChoice.outputs:
            errors[1] = "Ошибка валидации данных: Data validation error"
        return errors
