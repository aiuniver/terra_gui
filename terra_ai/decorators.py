from datetime import datetime


def execute_time(method):
    def wrapper(*args, **kwargs):
        _start = datetime.now()
        _result = method(*args, **kwargs)
        print(f"Execute time: \033[0;32m{datetime.now() - _start}\033[0m")
        return _result

    return wrapper
