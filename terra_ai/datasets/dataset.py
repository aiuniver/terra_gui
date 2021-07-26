from pathlib import Path

from ..data.datasets.dataset import CustomDatasetConfigData, DatasetData
from .. import DATASET_EXT


class DTS:
    @staticmethod
    def get_dataset_keras_info(name: str) -> DatasetData:
        return DatasetData()

    @staticmethod
    def get_dataset_custom_info(name: str, path: Path) -> DatasetData:
        data = CustomDatasetConfigData(path=Path(path, f"{name}.{DATASET_EXT}"))
        return DatasetData(**data.config)
