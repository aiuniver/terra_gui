import importlib
import importlib.util
import json
import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras import Model as KerasModel
from tensorflow.keras.layers import Input

from .common import yolo_decode
from .main_blocks import BaseBlock, CascadeBlock

package_dataset = "terra_ai.datasets"
module = "arrays_create"
if not importlib.util.find_spec(f"{package_dataset}.{module}"):
    package_dataset = "cascades"

CreateArray = getattr(importlib.import_module(f".{module}", package=package_dataset), "CreateArray")


class BaseModel(BaseBlock):

    def __init__(self, **kwargs):
        super().__init__()

        self.path: str = kwargs.get("path")
        self.model_architecture = "base"
        self.yolo_version = "v3"
        self.model = None
        self.config = None
        self.model_architecture = None
        self.yolo_version = None

    def set_path(self, model_path: str):
        self.path = os.path.join(model_path, self.path, 'model')

    def set_architecture(self):
        with open(os.path.join(self.path, 'config.train')) as cfg:
            config = json.load(cfg)
        architecture = config.get('architecture', {}).get('type', '')[:-2].lower()
        version = config.get('architecture', {}).get('type', '')[-2:].lower()
        return architecture, version

    @staticmethod
    def __get_json_data(path_model_json, path_custom_obj_json):
        with open(path_model_json) as json_file:
            data = json.load(json_file)

        with open(path_custom_obj_json) as json_file:
            custom_dict = json.load(json_file)

        return data, custom_dict

    @staticmethod
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

    def __load_model(self):
        weight = None
        model = None
        custom_object = None
        model_tf_format = False
        for i in os.listdir(self.path):
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

        if model_tf_format:
            model = load_model(os.path.join(self.path, model), compile=False, custom_objects=None)
        else:
            model_data, custom_dict = self.__get_json_data(os.path.join(self.path, model),
                                                           os.path.join(self.path, custom_object))
            custom_object = self.__set_custom_objects(custom_dict)

            model = model_from_json(model_data, custom_objects=custom_object)
        model.load_weights(os.path.join(self.path, weight))

        return model

    @staticmethod
    def __make_yolo(model, config, version):
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
        yolo = KerasModel(input_layer, output_tensors)

        return yolo

    def __get_dataset_config(self):
        dataset_path = os.path.join(self.path, "dataset.json")
        dataset_data_path = self.path
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(self.path, "dataset", "config.json")
            dataset_data_path = os.path.join(self.path, "dataset")
        with open(dataset_path) as cfg:
            config = json.load(cfg)
        return config

    def execute(self):
        source = list(self.inputs.values())[0].execute()
        if not self.model:
            self.model = self.__load_model()
            self.config = self.__get_dataset_config()
            self.model_architecture, self.yolo_version = self.set_architecture()

            if self.model_architecture == 'yolo':
                self.model = self.__make_yolo(self.model, self.config, self.yolo_version)

        array = CreateArray().execute(array_class='image', dataset_path=self.path,
                                         sources=source)

        array = array.get("1")[np.newaxis, :]

        result = self.model.predict(x=array)

        if self.model_architecture == 'yolo':
            result = self.get_bboxes(result)

        for block, params in self.config.get('outputs', {}).items():
            if params.get('task', '').lower() == 'classification':
                classes = params.get('classes_names')

        # postprocessing = []
        # if object_detection:
        #     type_module = getattr(general_fucntions, "object_detection")
        #     postprocessing = getattr(type_module, 'main')()
        #     output_types.append(type2str(signature(postprocessing).return_annotation))
        #
        # else:
        #     for out in config['outputs'].keys():
        #         if config['outputs'][out]['task'] not in ['Timeseries', 'TimeseriesTrend']:
        #             for key, cur_param in config['columns'][out].items():
        #                 with open(os.path.join(dataset_data_path, "instructions", "parameters", key + '.json')) as cfg:
        #                     spec_config = json.load(cfg)
        #
        #                 cur_param.update(spec_config)
        #                 try:
        #                     task = decamelize(cur_param['task'])
        #                     type_module = getattr(general_fucntions, task)
        #                     postprocessing.append(getattr(type_module, 'main')(**cur_param,
        #                                                                        dataset_path=dataset_data_path,
        #                                                                        key=key))
        #                     output_types.append(task)
        #                 except:
        #                     postprocessing.append(None)
        #         else:
        #             param = {}
        #             for key, cur_param in config['columns'][out].items():
        #                 param[key] = cur_param
        #                 with open(os.path.join(dataset_data_path, "instructions", "parameters", key + '.json')) as cfg:
        #                     param[key].update(json.load(cfg))
        #             param = {'columns': param, 'dataset_path': dataset_data_path,
        #                      'shape': config['outputs'][out]['shape']}
        #             task = decamelize(config['outputs'][out]['task'])
        #             type_module = getattr(general_fucntions, task)
        #             postprocessing.append(getattr(type_module, 'main')(**param))
        #             output_types.append(task)
        #
        #     if any(postprocessing):
        #         postprocessing = make_processing(postprocessing)
        #     else:
        #         postprocessing = None
        #
        # model = BuildModelCascade(preprocess, model, postprocessing, output=output_types, input=input_types)

        return result, classes

    @staticmethod
    def get_bboxes(array):
        while len(array) == 1:
            array = array[0]
        bboxes = list(array)
        bboxes.pop()

        out_bbox = tf.concat([tf.reshape(x, (-1, tf.shape(x)[-1])) for x in bboxes], 0)
        return out_bbox


class Model(CascadeBlock):
    Model = BaseModel
