from typing import Dict, Optional

from terra_ai.logging import logger
from terra_ai.data.datasets.extra import LayerGroupChoice
from terra_ai.data.datasets.creation import DatasetCreationArchitectureData
from terra_ai.data.training.extra import ArchitectureChoice


class DatasetCreationValidate:
    _group: LayerGroupChoice
    _architecture: ArchitectureChoice
    _blocks: DatasetCreationArchitectureData
    _errors: Dict[int, Optional[str]] = {}

    def __init__(
        self,
        group: LayerGroupChoice,
        architecture: ArchitectureChoice,
        blocks: DatasetCreationArchitectureData,
    ):
        self._group = group
        self._architecture = architecture
        self._blocks = blocks

        logger.info(self.__class__.__name__)
        logger.info(f"... Group: {self._group}")
        logger.info(f"... Architecture: {self._architecture}")
        logger.info(f"... Inputs: {self._blocks.inputs.json(ensure_ascii=False)}")
        logger.info(f"... Outputs: {self._blocks.outputs.json(ensure_ascii=False)}")

    def validate(self) -> Dict[int, Optional[str]]:
        """
        Запуск валидации блоков

        :return:
            Dict[int, Optional[str]]:
                ключ - id блока
                значение - либо строка ошибки, либо None, если ошибки нет
        """
        return self._errors
