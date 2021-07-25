from pathlib import Path

from ..data.datasets.dataset import CustomDataset, DatasetData


class DTS:
    @staticmethod
    def get_dataset_keras_info(name: str) -> DatasetData:
        return DatasetData()

    @staticmethod
    def get_dataset_custom_info(name: str, path: Path) -> DatasetData:
        data = CustomDataset(path=Path(path, f"{name}.trds"))
        return DatasetData(**data.config)
