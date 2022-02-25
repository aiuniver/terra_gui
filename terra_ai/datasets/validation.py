from typing import Dict, Optional

from terra_ai.logging import logger
from terra_ai.data.datasets.extra import LayerGroupChoice
from terra_ai.data.datasets.creation import CreationBlockList
from terra_ai.data.training.extra import ArchitectureChoice


class DatasetCreationValidate:
    _group: LayerGroupChoice
    _architecture: ArchitectureChoice
    _blocks: CreationBlockList
    _errors: Dict[int, Optional[str]] = {}

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
        logger.info(f"... Group: {self._group}")
        logger.info(f"... Architecture: {self._architecture}")
        logger.info(f"... Blocks: {self._blocks.json(ensure_ascii=False)}")

    def validate(self) -> Dict[int, Optional[str]]:
        """
        Запуск валидации блоков

        :return:
            Dict[int, Optional[str]]:
                ключ - id блока
                значение - либо строка ошибки, либо None, если ошибки нет
        """

        for put_data in self._blocks:
            if put_data.type == self._group[:-1]:
                self._errors[put_data.id] = getattr(self, f'check_{put_data.type}_block')(put_data)
                for put_id in put_data.bind.up:
                    block = self._blocks.get(put_id)
                    if block.type == 'handler':
                        self._errors[block.id] = self.check_handler_block(block)
                        for h_id in block.bind.up:
                            file_data = self._blocks.get(h_id)
                            self._errors[h_id] = self.check_data_block(file_data)
                    else:
                        self._errors[put_data.id] = 'К слою должен быть привязан обработчик.'

        return self._errors

    @staticmethod
    def check_data_block(block):
        if not block.parameters.data:
            return 'Выберите данные для обработки.'

    @staticmethod
    def check_handler_block(block):
        if not block.bind.up:
            return 'Соедините данные к обработчику.'

    @staticmethod
    def check_input_block(block):
        if not block.bind.up:
            return 'Соедините обработчик ко входу.'

    @staticmethod
    def check_output_block(block):
        if not block.bind.up:
            return 'Соедините обработчик к выходу.'

    # @staticmethod
    # def check_handler_data_count(block):
    #     if len(block.bind.up) > 1:
    #         return 'К обработчику может быть присоединен только один блок данных.'
