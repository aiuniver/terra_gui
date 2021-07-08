import os

from pathlib import Path
from transliterate import slugify

from ..data.datasets.dataset import CustomDataset, DatasetsGroupsList
from ..data.datasets.creation import SourceData
from ..data.datasets.creation import FilePathSourcesList
from ..data.presets.datasets import DatasetsGroups
from . import exceptions
from . import temporary_methods
from . import progress


class Exchange:
    progress = progress.Progress()

    def __call__(self, method: str, *args, **kwargs) -> dict:
        # Получаем метод для вызова
        __method_name = f"_call_{method}"
        __method = getattr(self, __method_name, None)

        # Проверяем существует ли метод
        if __method is None:
            raise exceptions.CallMethodNotFoundException(self.__class__, __method_name)

        # Проверяем является ли методом вызываемым
        if not callable(__method):
            raise exceptions.MethodNotCallableException(self.__class__, __method_name)

        # Вызываем метод
        return __method(**kwargs)

    def _call_get_datasets_info(self, path: str) -> dict:
        """
        Получение данных для страницы датасетов: датасеты и теги
        """
        info = DatasetsGroupsList(DatasetsGroups)
        for dirname in os.listdir(path):
            try:
                dataset = CustomDataset(path=Path(path, dirname))
                alias = slugify(dataset.config.get("name"), language_code="ru")
                info.get("custom").datasets.append(
                    {
                        "alias": alias,
                        "name": dataset.config.get("name"),
                        "date": dataset.config.get("date"),
                        "size": {"value": dataset.config.get("size")},
                        "tags": dataset.tags.dict(),
                    }
                )
            except Exception:
                pass
        return info.dict()

    def _call_get_dataset_source(self, mode: str, value: str):
        """
        Загрузка исходников датасета
        """
        source = SourceData(mode=mode, value=value)
        temporary_methods.dataset_load(source, self.progress("dataset_load"))

    def _call_get_datasets_sources(self, path: str) -> list:
        """
        Получение списка исходников датасетов
        """
        files = FilePathSourcesList()
        for filename in os.listdir(path):
            filepath = Path(path, filename)
            try:
                files.append({"value": filepath})
            except Exception:
                pass
        return files.list()


agent_exchange = Exchange()
