import inspect


def get_functions(preprocess):
    filter_func = lambda x: not x.startswith('_') and inspect.isfunction(getattr(preprocess, x))
    # {NAME_PREPROCESS: FUNCTION}
    functions = {name: getattr(preprocess, name) for name in dir(preprocess) if filter_func(name)}

    return functions


def list2tuple(inp: dict):
    for key, value in inp.items():
        if isinstance(value, list):
            inp[key] = tuple(value)

    return inp
