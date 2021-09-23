from terra_ai.cascades import cascade_input, cascade_output
from terra_ai.cascades.cascade import CascadeElement, CascadeOutput, BuildModelCascade, CompleteCascade
from terra_ai.utils import decamelize
from terra_ai.common import make_path
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


def make_postprocess(post_list):
    if post_list is None:
        return None

    def fun(*x):

        out = []

        for prep, element in zip(post_list, x):
            out.append(prep(element))

        return out
    return fun


def json2model_cascade(path: str):
    weight = None
    model = None

    for i in os.listdir(path):
        if i[-3:] == '.h5' and 'best' in i:
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

            with open(os.path.join(path, "dataset", "instructions", "parameters", f"{inp}_inputs.json")) as cfg:
                spec_config = json.load(cfg)

            param = param | spec_config

            type_module = getattr(general_fucntions, decamelize(param['task']))
            preprocess.append(getattr(type_module, 'main')(**param, dataset_path=os.path.join(path, "dataset")))

        preprocess = make_preprocess(preprocess)
    else:
        preprocess = None

    if config['outputs']:
        postprocessing = []

        for inp, param in config['outputs'].items():
            type_module = getattr(general_fucntions, decamelize(param['task']))
            try:
                postprocessing.append(getattr(type_module, 'main')(**param))
            except:
                postprocessing = None

        postprocessing = make_postprocess(postprocessing)
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
    iter = getattr(cascade_input, params['type'])

    return iter


def create_output(**params):
    out = CascadeOutput(getattr(cascade_output, params['type']),
                        params['params'] if 'params' in params.keys() else {})

    return out


def create_model(**params):
    model = json2model_cascade(make_path(params["model"]))

    return model


def create_function(**params):
    function = getattr(general_fucntions, decamelize(params['task']))
    function = CascadeElement(
        getattr(function, params['name'])(**params['params']),
        f"функция {params['name']}")
    return function


