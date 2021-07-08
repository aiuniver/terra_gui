from typing import Optional, Dict
from pydantic import BaseModel


class ProgressData(BaseModel):
    # percent:
    pass


class Progress:
    __pool: Dict[str, ProgressData] = {}

    def __call__(self, method: str, *args, **kwargs) -> Optional[callable]:
        # Получаем метод для вызова
        __method_name = f"_progress_{method}"
        __method = getattr(self, __method_name, None)

        # Проверяем существует ли метод
        if __method is None:
            return

        # Проверяем является ли методом вызываемым
        if not callable(__method):
            return

        # Возвращаем метод
        return __method

    @property
    def dataset_source_load(self):
        return {}

    def _progress_dataset_source_load(self, percent: float):
        print(percent)
