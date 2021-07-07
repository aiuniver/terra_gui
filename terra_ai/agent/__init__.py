from pathlib import PosixPath

from . import exceptions


class Exchange:
    def __call__(self, *args, **kwargs) -> dict:
        # Проверяем на количество переданных аргументов: должен быть 1
        if len(args) != 1:
            raise exceptions.NotOneArgumentException(self.__class__, len(args))

        # Получаем метод для вызова
        __method_name = f"_call_{args[0]}"
        __method = getattr(self, __method_name, None)

        # Проверяем существует ли метод
        if __method is None:
            raise exceptions.CallMethodNotFoundException(self.__class__, __method_name)

        # Проверяем является ли методом вызываемым
        if not callable(__method):
            raise exceptions.MethodNotCallableException(self.__class__, __method_name)

        # Вызываем метод
        return __method(**kwargs)

    def _call_get_datasets_sources(self, pathdir: str):
        print(pathdir)


agent_exchange = Exchange()
