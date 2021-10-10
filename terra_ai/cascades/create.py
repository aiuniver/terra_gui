from . import cascade_input, cascade_output, general_fucntions
from .cascade import CascadeElement, CascadeOutput, BuildModelCascade, CompleteCascade, CascadeBlock

from .common import decamelize
import json
import os
from tensorflow.keras.models import load_model
from pathlib import Path
from collections import OrderedDict
import sys


# ROOT_PATH = str(Path(__file__).parent.parent.parent)


def make_processing(preprocess_list):
    def fun(*x):

        out = []

        for prep, element in zip(preprocess_list, x):
            if prep:
                out.append(prep(element))
            else:
                out.append(x)

        return out
    return fun


def json2model_cascade(path: str):
    weight = None
    model = None
    for i in os.listdir(path):
        if i[-3:] == '.h5' and 'best' in i:
            weight = i
        elif weight is None and i[-3:] == '.h5':
            weight = i
        elif i[-4:] == '.trm':
            model = i

    model = load_model(os.path.join(path, model), compile=False, custom_objects=None)
    model.load_weights(os.path.join(path, weight))

    dataset_path = os.path.join(path, "dataset", "config.json")
    with open(dataset_path) as cfg:
        config = json.load(cfg)

    if config['inputs']:
        preprocess = []

        for inp, param in config['inputs'].items():
            with open(os.path.join(
                    path, "dataset", "instructions", "parameters",
                    f"{inp}_{decamelize(param['task'])}.json")) as cfg:
                spec_config = json.load(cfg)
            param.update(spec_config)

            type_module = getattr(general_fucntions, decamelize(param['task']))
            preprocess.append(getattr(type_module, 'main')(
                **param, dataset_path=os.path.join(path, "dataset"), key=inp)
            )
        preprocess = make_processing(preprocess)
    else:
        preprocess = None

    if config['outputs']:
        postprocessing = []

        for inp, param in config['outputs'].items():
            with open(os.path.join(
                    path, "dataset", "instructions", "parameters",
                    f"{inp}_{decamelize(param['task'])}.json")) as cfg:
                spec_config = json.load(cfg)

            param.update(spec_config)

            try:
                type_module = getattr(general_fucntions, decamelize(param['task']))
                postprocessing.append(getattr(type_module, 'main')(**param))
            except:
                postprocessing.append(None)

        if any(postprocessing):
            postprocessing = make_processing(postprocessing)
        else:
            postprocessing = None
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
            if params['tag'] == 'model':
                params['model_path'] = os.path.split(path)[0]
            cascades[i] = getattr(sys.modules.get(__name__), "create_" + params['tag'])(**params)

    adjacency_map = OrderedDict()

    for i, inp in config['adjacency_map'].items():
        adjacency_map[cascades[i]] = [j if j in ["INPUT"] else cascades[j] for j in inp]

    if input_cascade is None:
        return CascadeBlock(adjacency_map)

    return CompleteCascade(input_cascade, adjacency_map)


def create_input(**params):
    iter = getattr(cascade_input, params['type'])

    return iter


def create_output(**params):
    out = CascadeOutput(getattr(cascade_output, params['type']),
                        params['params'] if 'params' in params.keys() else {})

    return out


def create_model(**params):
    model = json2model_cascade(os.path.join(params["model_path"], params["model"]))

    return model


def create_function(**params):
    function = getattr(general_fucntions, decamelize(params['task']))
    function = CascadeElement(
        getattr(function, params['name'])(**params['params']),
        f"функция {params['name']}")
    return function
