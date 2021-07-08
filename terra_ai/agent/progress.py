from typing import Optional


class Progress:
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

    def _progress_dataset_load(self, percent: float):
        print(percent)
