from . import cascade_input, cascade_output, general_fucntions, service
from .cascade import CascadeElement, CascadeOutput, BuildModelCascade, CompleteCascade, CascadeBlock
from .common import decamelize, yolo_decode, type2str
import json
import os
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model, model_from_json
from collections import OrderedDict
import sys
from inspect import signature
import itertools
import importlib
import importlib.util


def make_processing(preprocess_list):
    def fun(*x):
        inp = []
        for i in x:
            if isinstance(i, tuple):
                inp += i
            else:
                inp.append(i)
        out = []
        if len(inp) == 1:
            # print('1')
            for prep, element in itertools.zip_longest(preprocess_list, inp, fillvalue=inp[0]):
                if prep:
                    out.append(prep(element))
                else:
                    out.append(element)
        else:
            # print('>1')
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
    custom_object = None
    model_tf_format = False
    for i in os.listdir(path):
        if i[-3:] == '.h5' and 'best' in i:
            weight = i
        elif 'model_best_weights.data' in i:
            weight = i.split('.')[0]
        if not weight:
            if i[-3:] == '.h5':
                weight = i
            elif 'model_weights.data' in i:
                weight = i.split('.')[0]
        if i == 'trained_model.trm':
            model = i
            model_tf_format = True
        elif i[-4:] == '.trm' and 'model_json' in i:
            model = i
        if i[-4:] == '.trm' and 'custom_obj_json' in i:
            custom_object = i

    def __get_json_data(path_model_json, path_custom_obj_json):
        with open(path_model_json) as json_file:
            data = json.load(json_file)

        with open(path_custom_obj_json) as json_file:
            custom_dict = json.load(json_file)

        return data, custom_dict

    def __set_custom_objects(custom_dict):
        custom_object = {}
        for k, v in custom_dict.items():
            try:
                package_ = "terra_ai.custom_objects"
                if not importlib.util.find_spec(v, package=package_):
                    package_ = "custom_objects"
                custom_object[k] = getattr(importlib.import_module(f".{v}", package=package_), k)
            except:
                continue
        return custom_object

    if model_tf_format:
        model = load_model(os.path.join(path, model), compile=False, custom_objects=None)
    else:
        model_data, custom_dict = __get_json_data(os.path.join(path, model), os.path.join(path, custom_object))
        custom_object = __set_custom_objects(custom_dict)

        model = model_from_json(model_data, custom_objects=custom_object)
    model.load_weights(os.path.join(path, weight))

    dataset_path = os.path.join(path, "dataset.json")
    dataset_data_path = path
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(path, "dataset", "config.json")
        dataset_data_path = os.path.join(path, "dataset")
    with open(dataset_path) as cfg:
        config = json.load(cfg)

    object_detection = False

    input_types, output_types = [], []

    if config['tags'][-1]['alias'] == 'object_detection':
        version = "v3"
        object_detection = True
        for inp in config['outputs'].keys():
            for inp, param in config['columns'][inp].items():
                with open(os.path.join(dataset_data_path, "instructions", "parameters", inp + '.json')) as cfg:
                    spec_config = json.load(cfg)
                if 'yolo' in spec_config.keys():
                    version = spec_config['yolo']
                    break
        model = make_yolo(model, config, version)
    preprocess = []

    for inp in config['inputs'].keys():
        if config['inputs'][inp]['task'] != 'Dataframe':
            param = {}  # for pycharm linter
            for inp, param in config['columns'][inp].items():
                with open(os.path.join(dataset_data_path, "instructions", "parameters", inp + '.json')) as cfg:
                    param.update(json.load(cfg))
            type_module = getattr(general_fucntions, decamelize(param['task']))
            preprocess.append(getattr(type_module, 'main')(
                **param, dataset_path=dataset_data_path, key=inp)
            )
        else:
            param = {}
            for key, cur_param in config['columns'][inp].items():
                param[key] = cur_param
                with open(os.path.join(dataset_data_path, "instructions", "parameters", key + '.json')) as cfg:
                    param[key].update(json.load(cfg))
            param = {'columns': param, 'dataset_path': dataset_data_path, 'shape': config['inputs'][inp]['shape']}
            type_module = getattr(general_fucntions, 'dataframe')
            preprocess.append(getattr(type_module, 'main')(**param))

    preprocess = make_processing(preprocess)
    postprocessing = []
    if object_detection:
        type_module = getattr(general_fucntions, "object_detection")
        postprocessing = getattr(type_module, 'main')()
        output_types.append(type2str(signature(postprocessing).return_annotation))

    else:
        for out in config['outputs'].keys():
            if config['outputs'][out]['task'] not in ['Timeseries', 'TimeseriesTrend']:
                for key, cur_param in config['columns'][out].items():
                    with open(os.path.join(dataset_data_path, "instructions", "parameters", key + '.json')) as cfg:
                        spec_config = json.load(cfg)

                    cur_param.update(spec_config)
                    try:
                        task = decamelize(cur_param['task'])
                        type_module = getattr(general_fucntions, task)
                        postprocessing.append(getattr(type_module, 'main')(**cur_param,
                                                                           dataset_path=dataset_data_path,
                                                                           key=key))
                        output_types.append(task)
                    except:
                        postprocessing.append(None)
            else:
                param = {}
                for key, cur_param in config['columns'][out].items():
                    param[key] = cur_param
                    with open(os.path.join(dataset_data_path, "instructions", "parameters", key + '.json')) as cfg:
                        param[key].update(json.load(cfg))
                param = {'columns': param, 'dataset_path': dataset_data_path,
                         'shape': config['outputs'][out]['shape']}
                task = decamelize(config['outputs'][out]['task'])
                type_module = getattr(general_fucntions, task)
                postprocessing.append(getattr(type_module, 'main')(**param))
                output_types.append(task)

        if any(postprocessing):
            postprocessing = make_processing(postprocessing)
        else:
            postprocessing = None

    model = BuildModelCascade(preprocess, model, postprocessing, output=output_types, input=input_types)

    return model


def json2cascade(path: str, cascade_config=None, mode="deploy"):
    if cascade_config:
        config = cascade_config
    else:
        with open(path) as cfg:
            config = json.load(cfg)
    cascades = {}
    input_cascade = None

    old_adjacency_map = {}
    not_in_adj = list(config['adjacency_map'].keys())

    def search_current(inp, name):
        if inp == ["INPUT"]:
            return name
        for i in inp:
            if i == "INPUT":
                continue
            if i not in old_adjacency_map.keys():
                current = search_current(config['adjacency_map'][i], i)
                return current
        return name

    while len(not_in_adj):
        block = not_in_adj[0]

        current = search_current(config['adjacency_map'][block], block)

        if current:
            old_adjacency_map[current] = config['adjacency_map'][current]
            not_in_adj.remove(current)

    for i, params in config['cascades'].items():
        if params['tag'] == 'input':
            input_cascade = getattr(sys.modules.get(__name__), "create_" + params['tag'])(**params)
        else:
            if params['tag'] == 'model':
                if mode == "run":
                    params["model"] = "model"
                    params['model_path'] = path
                else:
                    params['model_path'] = os.path.split(path)[0]
            cascades[i] = getattr(sys.modules.get(__name__), "create_" + params['tag'])(**params)

    adjacency_map = OrderedDict()

    for i, inp in old_adjacency_map.items():
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
    if "params" not in params.keys():
        params['params'] = {}
    function = getattr(general_fucntions, decamelize(params['task']))
    function = CascadeElement(
        getattr(function, params['name'])(**params['params']),
        f"функция {params['name']}")
    return function


def create_service(**params):
    function = getattr(service, decamelize(params['task']))

    if 'params' not in params.keys():
        params['params'] = {}

    function = CascadeElement(
        getattr(function, params['name'])(**params['params']),
        f"сервис {params['name']}"
    )

    return function


def make_yolo(model, config, version):
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
        pred_tensor = yolo_decode(conv_tensor, num_class, version, i)
        output_tensors.append(pred_tensor)
    yolo = Model(input_layer, output_tensors)

    return yolo
