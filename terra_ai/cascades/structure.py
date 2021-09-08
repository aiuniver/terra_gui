import cascades
import cascade_function

import types
import inspect
import cascade

standard_class = list(filter(lambda x: not x.startswith('_'), dir(cascade)))


def rec_dive_models(obj):
    attributes = list(filter(lambda x: not x.startswith('_'), dir(obj)))
    if not isinstance(getattr(obj, attributes[0]), types.ModuleType):
        for x in attributes:
            if x not in standard_class and inspect.isclass(getattr(obj, x)):
                return getattr(obj, x)

    out = dict()
    for name in attributes:
        out[name] = rec_dive_models(getattr(obj, name))

    return out


def rec_dive_function(obj):
    attributes = list(filter(lambda x: not x.startswith('_'), dir(obj)))

    if not isinstance(getattr(obj, attributes[0]), types.ModuleType):
        out = dict()
        for x in attributes:
            if inspect.isfunction(getattr(obj, x)):
                fun = getattr(obj, x)
                out[fun.__name__] = fun
        return out

    out = dict()
    for name in attributes:
        out[name] = rec_dive_function(getattr(obj, name))

    return out


def get_structure():
    file_struct = dict()

    file_struct['model'] = rec_dive_models(cascades)
    file_struct['function'] = rec_dive_function(cascade_function)
    return file_struct
