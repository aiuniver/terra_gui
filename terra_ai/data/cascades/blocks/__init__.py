import os
import shutil

from enum import Enum
from typing import Optional
from pathlib import Path
from pydantic.types import FilePath

from terra_ai import settings
from terra_ai.progress import utils as progress_utils
from terra_ai.data.mixins import BaseMixinData
from terra_ai.data.cascades.extra import BlockGroupChoice
from terra_ai.data.cascades.blocks import types
from terra_ai.data.cascades.blocks.extra import BlockServiceTypeChoice


class BlockBaseData(BaseMixinData):
    pass


class BlockInputDataData(BlockBaseData):
    main: types.InputData.ParametersMainData


class BlockOutputDataData(BlockBaseData):
    main: types.OutputData.ParametersMainData


class BlockModelData(BlockBaseData):
    main: types.Model.ParametersMainData


class BlockFunctionData(BlockBaseData):
    main: types.Function.ParametersMainData


class BlockCustomData(BlockBaseData):
    main: types.Custom.ParametersMainData


class BlockServiceData(BlockBaseData):
    main: types.Service.ParametersMainData
    model_path: Optional[FilePath]

    def dict(self, **kwargs):
        kwargs.update({"exclude": {"model_path"}})
        return super().dict(**kwargs)

    def model_load(self):
        os.makedirs(settings.WEIGHT_PATH, exist_ok=True)
        value = None
        if self.main.type == BlockServiceTypeChoice.DeepSort:
            weight_filename = "deepsort.t7"
            value = Path(settings.WEIGHT_PATH, weight_filename)
            if not value.is_file():
                filepath = progress_utils.download(
                    "weight_load",
                    "Загрузка весов `{weight_filename}`",
                    f"{settings.YANDEX_WEIGHT_STORAGE_URL}{weight_filename}",
                )
                shutil.move(filepath, value)
        self.model_path = value


Block = Enum(
    "Block",
    dict(
        map(lambda item: (item.name, f"Block{item.name}Data"), list(BlockGroupChoice))
    ),
    type=str,
)
