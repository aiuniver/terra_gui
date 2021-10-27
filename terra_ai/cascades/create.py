from . import cascade_input, cascade_output, general_fucntions
from .cascade import CascadeElement, CascadeOutput, BuildModelCascade, CompleteCascade, CascadeBlock

from .common import decamelize, yolo_decode
import json
import os
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from collections import OrderedDict
import sys


def make_processing(preprocess_list):
    def fun(*x):
        inp = []
        for i in x:
            if isinstance(i, tuple):
                inp += i
            else:
                inp.append(i)

        out = []

        for prep, element in zip(preprocess_list, inp):
            if prep:
                out.append(prep(element))
            else:
                out.append(element)

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

    dataset_path = os.path.join(path, "dataset")
    with open(os.path.join(dataset_path, "config.json")) as cfg:
        config = json.load(cfg)

    object_detection = False

    if config['tags'][-1]['alias'] == 'object_detection':
        object_detection = True
        model = create_yolo(model, config)
    preprocess = []

    for inp in config['inputs'].keys():
        if config['inputs'][inp]['task'] != 'Dataframe':
            for inp, param in config['columns'][inp].items():
                with open(os.path.join(dataset_path, "instructions", "parameters", inp + '.json')) as cfg:
                    param.update(json.load(cfg))
            type_module = getattr(general_fucntions, decamelize(param['task']))
            preprocess.append(getattr(type_module, 'main')(
                **param, dataset_path=dataset_path, key=inp)
            )
        else:
            param = {}
            for key, cur_param in config['columns'][inp].items():
                param[key] = cur_param
                with open(os.path.join(dataset_path, "instructions", "parameters", key + '.json')) as cfg:
                    param[key].update(json.load(cfg))
            param = {'columns': param, 'dataset_path': dataset_path, 'shape': config['inputs'][inp]['shape']}
            type_module = getattr(general_fucntions, 'dataframe')
            preprocess.append(getattr(type_module, 'main')(**param))

    preprocess = make_processing(preprocess)

    postprocessing = []
    if object_detection:
        type_module = getattr(general_fucntions, "object_detection")
        postprocessing.append(getattr(type_module, 'main')(**config['outputs']))
    else:
        for inp in config['outputs'].keys():
            if config['outputs'][inp]['task'] not in ['Timeseries', 'TimeseriesTrend']:
                for inp, param in config['columns'][inp].items():
                    with open(os.path.join(dataset_path, "instructions", "parameters", inp + '.json')) as cfg:
                        spec_config = json.load(cfg)

                    param.update(spec_config)
                    try:
                        type_module = getattr(general_fucntions, decamelize(param['task']))
                        postprocessing.append(getattr(type_module, 'main')(**param, dataset_path=dataset_path, key=inp))
                    except:
                        postprocessing.append(None)
            else:
                param = {}
                for key, cur_param in config['columns'][inp].items():
                    param[key] = cur_param
                    with open(os.path.join(dataset_path, "instructions", "parameters", key + '.json')) as cfg:
                        param[key].update(json.load(cfg))
                param = {'columns': param, 'dataset_path': dataset_path, 'shape': config['outputs'][inp]['shape']}
                type_module = getattr(general_fucntions, decamelize(config['outputs'][inp]['task']))
                postprocessing.append(getattr(type_module, 'main')(**param))

    if any(postprocessing):
        postprocessing = make_processing(postprocessing)
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


def create_yolo(model, config):
    od = None
    img_conf = None
    for _, i in config['outputs'].items():
        if i['task'] == 'ObjectDetection':
            od = i
            break
    for _, i in config['inputs'].items():
        if i['task'] == 'Image':
            img_conf = i
            break
    classes = od['classes_names']

    if classes is None:
        classes = []
    num_class = len(classes)

    input_layer = Input(img_conf['shape'])
    conv_tensors = model(input_layer)
    if conv_tensors[0].shape[1] == 13:
        conv_tensors.reverse()
    output_tensors = []

    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = yolo_decode(conv_tensor, num_class, i)
        output_tensors.append(pred_tensor)
    yolo = Model(input_layer, output_tensors)

    return yolo
