from terra_ai.cascades import input, output
from terra_ai.cascades.cascade import CascadeElement, CascadeOutput, BuildModelCascade, CompleteCascade
from terra_ai.utils import decamelize
from terra_ai import general_fucntions
import json
import os
from tensorflow.keras.models import load_model
from pathlib import Path
from collections import OrderedDict
import sys


ROOT_PATH = str(Path(__file__).parent.parent.parent)


def make_preprocess(preprocess_list):
    def fun(*x):

        out = []

        for i, (prep, element) in enumerate(zip(preprocess_list, x)):
            out.append(prep(element))

        return out
    return fun


def json2model_cascade(path: str):
    with open(path) as cfg:
        config = json.load(cfg)

    path_model = os.path.join(ROOT_PATH, config['model'])
    model = load_model(path_model, compile=False, custom_objects=None)
    model.load_weights(config['weight'])

    if config['inputs']:
        preprocess = []

        for inp, param in config['inputs'].items():
            type_module = getattr(general_fucntions, decamelize(param['task']))
            preprocess.append(getattr(type_module, 'main')(**param))

        preprocess = make_preprocess(preprocess)
    else:
        preprocess = None

    if config['outputs']:  # пока так
        postprocessing = None
        pass
    else:
        postprocessing = None

    model = BuildModelCascade(preprocess, model, postprocessing)

    return model


def json2cascade(path: str):
    with open(path) as cfg:
        config = json.load(cfg)

    cascades = {}
    input_cascade = None

    for i, params in config['cascades'].items():
        if params['tag'] == 'input':
            input_cascade = getattr(sys.modules.get(__name__), "create_" + params['tag'])(**params)
        else:
            cascades[i] = getattr(sys.modules.get(__name__), "create_" + params['tag'])(**params)

    adjacency_map = OrderedDict()

    for i, inp in config['adjacency_map'].items():
        adjacency_map[cascades[i]] = [j if j in ["INPUT"] else cascades[j] for j in inp]

    main_block = CompleteCascade(input_cascade, adjacency_map)

    return main_block


def create_input(**params):
    iter = getattr(input, params['type'])

    return iter


def create_output(**params):
    out = CascadeOutput(getattr(output, params['type']),
                        params['params'] if 'params' in params.keys() else {})

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


