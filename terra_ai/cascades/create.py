from terra_ai.cascades import input, output
from terra_ai.cascades.cascade import CascadeInput, CascadeElement
from terra_ai.common import json2model_cascade
from terra_ai.utils import decamelize
from terra_ai import general_fucntions


def create_input(**params):
    iter = CascadeInput(getattr(input, params['type']))

    return iter


def create_output(**params):
    out = getattr(output, params['type'])

    return out


def create_model(**params):
    model = json2model_cascade(params["model"])

    return model


def create_function(**params):
    function = getattr(general_fucntions, decamelize(params['task']))
    function = CascadeElement(
        getattr(function, params['name'])(**params['params']),
        f"функция {params['name']}")
    return function
