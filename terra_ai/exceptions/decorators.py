from . import terra_exception


def error_handler(obj):
    """
    Декоратор для класса и функции

    Если внутри функции или метода класса вызывается исключение, это исключение
    оборачивается в TerraBaseException (или в его потомка) и повторно вызывается
    """

    if isinstance(obj, type):
        callable_attributes = {k: v for k, v in obj.__dict__.items() if callable(v)}
        for name, func in callable_attributes.items():
            decorated = error_handler(func)
            setattr(obj, name, decorated)
        return obj

    def wrapper(*args, **kwargs):
        try:
            obj(*args, **kwargs)
        except Exception as error:
            raise terra_exception(error)

    return wrapper
