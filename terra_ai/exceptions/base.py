"""
# Исключения

## Примеры создания исключения

### Базовое создание исключения без передачи параметров
```
In [1]: from enum import Enum

In [2]: from terra_ai.exceptions.base import TerraBaseException

In [3]: class MyMessage(str, Enum):
   ...:     Value = "Моя базовая ошибка без параметров"
   ...:

In [4]: class MyException(TerraBaseException):
   ...:     class Meta:
   ...:         message: str = MyMessage.Value
   ...:

In [5]: MyException()
Out[5]: __main__.MyException('Моя базовая ошибка без параметров')

In [6]: print(MyException())
Моя базовая ошибка без параметров
```

### Базовое создание исключения с передачей параметров
```
In [1]: from enum import Enum

In [2]: from terra_ai.exceptions.base import TerraBaseException

In [3]: class MyMessage(str, Enum):
   ...:     Value = "Моя базовая ошибка с параметром '%s'"
   ...:

In [4]: class MyException(TerraBaseException):
   ...:     class Meta:
   ...:         message: str = MyMessage.Value
   ...:
   ...:     def __init__(self, param:str, *args):
   ...:         super().__init__(self.Meta.message % str(param), *args)
   ...:

In [5]: MyException("Мой параметр")
Out[5]: __main__.MyException("Моя базовая ошибка с параметром 'Мой параметр'")

In [6]: print(MyException("Мой параметр"))
Моя базовая ошибка с параметром 'Мой параметр'
```

### Создание типового исключения без передачи параметров на примере датасетов
```
In [1]: from enum import Enum

In [2]: from terra_ai.exceptions.datasets import DatasetsException

In [3]: class MyMessage(str, Enum):
   ...:     Value = "Типовая ошибка без параметров на основе датасетов"
   ...:

In [4]: class MyException(DatasetsException):
   ...:     class Meta:
   ...:         message: str = MyMessage.Value
   ...:

In [5]: MyException()
Out[5]: __main__.MyException('Типовая ошибка без параметров на основе датасетов')

In [6]: print(MyException())
Типовая ошибка без параметров на основе датасетов
```

### Создание типового исключения с передачей параметров на примере датасетов
```
In [1]: from enum import Enum

In [2]: from terra_ai.exceptions.datasets import DatasetsException

In [3]: class MyMessage(str, Enum):
   ...:     Value = "Типовая ошибка с параметром '%s' на основе датасетов"
   ...:

In [4]: class MyException(DatasetsException):
   ...:     class Meta:
   ...:         message: str = MyMessage.Value
   ...:
   ...:     def __init__(self, param:str, *args):
   ...:         super().__init__(self.Meta.message % str(param), *args)
   ...:

In [5]: MyException("Мой параметр")
Out[5]: __main__.MyException("Типовая ошибка с параметром 'Мой параметр' на основе датасетов")

In [6]: print(MyException("Мой параметр"))
Типовая ошибка с параметром 'Мой параметр' на основе датасетов
```
"""

from enum import Enum

from .. import settings


class Messages(dict, Enum):
    Unknown = {"ru": "%s",
               "eng": "%s"}


class TerraBaseException(Exception):
    class Meta:
        message = Messages.Unknown

    def __init__(self, *args, lang: str = settings.LANGUAGE):
        error_msg = self.Meta.message.value.get(lang)

        if args:
            try:
                error_msg = error_msg % args
            except TypeError:
                pass

        super().__init__(error_msg)
