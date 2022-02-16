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
from .internal_out_blocks import ModelOutput
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
        self.save_path: str = ''
        self.model = None
        self.config = None
        self.architecture_type = None
        self.model_architecture = None
        self.outs = {}

    def set_path(self, model_path: str, save_path: str):
        self.path = os.path.join(model_path, self.path, 'model')
        self.save_path = save_path

    def get_outputs(self):
        if not self.model_architecture:
            self.__set_architecture()
        return list(self.outs.keys())

    def get_main_params(self):
        self.__set_architecture()
        self.__set_model()
        return self.model, self.__get_dataset_config()

    def __set_architecture(self):
        self.config = self.__get_dataset_config()
        with open(os.path.join(os.path.split(self.path)[0], 'config.json'), 'r', encoding='utf-8') as m_conf:
            train_config = json.load(m_conf).get('base')
        self.architecture_type = train_config.get('architecture', {}).get('type', '')
        self.model_architecture = self.architecture_type.lower()
        self.yolo_version = train_config.get('architecture', {}).get('type', '')[-2:].lower()
        self.outs = {out.data_type: out for out in ModelOutput().get(type_=self.architecture_type)}

    def __set_model(self):
        if not self.model_architecture:
            self.__set_architecture()
        self.model = self.__load_model()

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

    def __get_dataset_config(self):
        dataset_path = os.path.join(self.path, "dataset.json")
        dataset_data_path = self.path
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(self.path, "dataset", "config.json")
            dataset_data_path = os.path.join(self.path, "dataset")
        with open(dataset_path) as cfg:
            config = json.load(cfg)
        return config

    def __get_sources(self):
        source = {}
        step = 1
        for type_, input_ in self.inputs.items():
            if type_.lower() == 'cropimage':
                type_ = 'image'
            result = input_.execute()
            source.update({
                str(step): {
                    f"{step}_{type_.lower()}": [result] if isinstance(result, str) else result
                }
            })
            step += 1
        return source

    def execute(self):
        source = self.__get_sources()
        print(source)
        if not self.model:
            self.__set_model()
        array_class = [param_.get('task').lower() for param_ in self.config.get('columns').get('1').values()][0]

        array = CreateArray().execute(array_class=array_class, dataset_path=self.path,
                                      sources=source).get("1")  # [np.newaxis, :]
        print(array.shape[0])
        if array.shape[0] == 1:
            result = self.model.predict(x=array)
        else:
            result = []
            for array_ in array:
                print(array_.shape)
                result.append(self.model.predict(x=array_[np.newaxis, :]))

        for block, params in self.config.get('outputs', {}).items():
            if params.get('task', '').lower() == 'classification':
                classes = params.get('classes_names')

        data = {
            'source': source,
            'model_predict': result,
            'options': self.config,
            'save_path': self.save_path
        }

        return [out().execute(**data) for name, out in self.outs.items()]


class YoloModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_architecture = "base"
        self.yolo_version = "v3"

    def __make_yolo(self, model, classes, version):
        if classes is None:
            classes = []
        num_class = len(classes)
        conv_tensors = model.outputs
        if conv_tensors[0].shape[1] == 13:
            conv_tensors.reverse()
        output_tensors = []
        for i, conv_tensor in enumerate(conv_tensors):
            pred_tensor = self.__decode(conv_tensor, num_class, i, version)
            # output_tensors.append(conv_tensor)
            output_tensors.append(pred_tensor)
        yolo = tf.keras.Model(model.inputs, output_tensors)
        return yolo

    @staticmethod
    def __decode(conv_output, NUM_CLASS, i=0, YOLO_TYPE="v3", STRIDES=None):
        ANCHORS = []

        if STRIDES is None:
            STRIDES = [8, 16, 32]
        if (YOLO_TYPE == "v4") or (YOLO_TYPE == "v5"):
            ANCHORS = [[[12, 16], [19, 36], [40, 28]],
                       [[36, 75], [76, 55], [72, 146]],
                       [[142, 110], [192, 243], [459, 401]]]
        elif YOLO_TYPE == "v3":
            ANCHORS = [[[10, 13], [16, 30], [33, 23]],
                       [[30, 61], [62, 45], [59, 119]],
                       [[116, 90], [156, 198], [373, 326]]]
        # Train options
        # where i = 0, 1 or 2 to correspond to the three grid scales
        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))
        conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = \
            tf.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)

        xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
        xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
        xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        # Calculate the center position of the prediction box:
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
        # Calculate the length and width of the prediction box:
        pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]

        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        pred_conf = tf.sigmoid(conv_raw_conf)  # object box calculates the predicted confidence
        pred_prob = tf.sigmoid(conv_raw_prob)  # calculating the predicted probability category box object

        # calculating the predicted probability category box object
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    @staticmethod
    def __get_bboxes(array):
        while len(array) == 1:
            array = array[0]
        bboxes = list(array)
        bboxes.pop()

        out_bbox = tf.concat([tf.reshape(x, (-1, tf.shape(x)[-1])) for x in bboxes], 0)
        return out_bbox

    def __get_sources(self):
        source = {}
        step = 1
        for type_, input_ in self.inputs.items():
            source.update({
                str(step): {
                    f"{step}_{type_.lower()}": [
                        input_.execute()
                    ]
                }
            })
            step += 1
        return source

    def __get_classes(self):
        classes = []
        for id_, out_ in self.config.get('outputs', {}).items():
            classes.extend(out_.get('classes_names'))
        return sorted(list(set(classes)))

    def execute(self):
        source = self.__get_sources()
        if not self.model:
            self.model, self.config = self.get_main_params()
            self.model = self.__make_yolo(self.model, self.__get_classes(), self.yolo_version)

        array = CreateArray().execute(array_class='image', dataset_path=self.path,
                                      sources=source).get("1")
        result = self.__get_bboxes(self.model.predict(x=array))

        data = {
            'source': source,
            'array': array,
            'model_predict': result,
            'options': self.config,
            'save_path': self.save_path,
            'classes': self.__get_classes()
        }

        return {out.data_type: out().execute(**data) for name, out in self.outs.items()}


class Model(CascadeBlock):
    Model = BaseModel
    YoloModel = YoloModel

    def get(self, type_, **kwargs):
        model_type = 'YoloModel' if type_ == 'yolo' else type_

        if kwargs:
            return self.__getattribute__(model_type)(**kwargs)
        else:
            return self.__getattribute__(model_type)
