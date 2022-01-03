from enum import Enum

from terra_ai.cascades.input_blocks import Input
from terra_ai.cascades.services_blocks import Service
from terra_ai.data.cascades.extra import BlockGroupChoice


class BlockClasses:
    InputData = Input
    Service = Service

    @staticmethod
    def get(type_, data_type):
        block = BlockClasses().__getattribute__(type_)
        return block().get(input_type=data_type)


if __name__ == "__main__":
    block_generator = BlockClasses
    block_1 = block_generator.get(type_=BlockGroupChoice.InputData, data_type="video")
    block_2 = block_generator.get(type_=BlockGroupChoice.Service, data_type="speech2text")
    block_3 = block_generator.get(type_=BlockGroupChoice.Service, data_type="speech2text")
    print(type(block_1), "\n", type(block_2), "\n", type(block_2), "\n", block_2 == block_3)


