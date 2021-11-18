import gc
import importlib
import json
import math
import copy
import os
import threading
from pathlib import Path

import psutil
import time
import pynvml as N

from typing import Optional

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.python.keras.utils.np_utils import to_categorical

from config import settings
from terra_ai import progress
from terra_ai.callbacks.classification_callbacks import BaseClassificationCallback
from terra_ai.callbacks.interactive_callback import InteractiveCallback
# from terra_ai.training.customcallback import InteractiveCallback
from terra_ai.callbacks.utils import print_error, loss_metric_config, BASIC_ARCHITECTURE, CLASS_ARCHITECTURE, \
    YOLO_ARCHITECTURE, round_loss_metric, class_metric_list, CLASSIFICATION_ARCHITECTURE, get_dataset_length
from terra_ai.customLayers import terra_custom_layers
from terra_ai.data.datasets.dataset import DatasetData, DatasetOutputsData
from terra_ai.data.datasets.extra import LayerOutputTypeChoice, LayerInputTypeChoice, LayerEncodingChoice
from terra_ai.data.deploy.extra import DeployTypeChoice
from terra_ai.data.deploy.tasks import DeployData
from terra_ai.data.modeling.model import ModelDetailsData
from terra_ai.data.presets.training import Metric
from terra_ai.data.training.extra import CheckpointTypeChoice, ArchitectureChoice
from terra_ai.data.training.train import TrainingDetailsData

from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.deploy.create_deploy_package import CascadeCreator
from terra_ai.exceptions.deploy import MethodNotImplementedException
from terra_ai.modeling.validator import ModelValidator
from terra_ai.exceptions import training as exceptions
from terra_ai.training.yolo_utils import create_yolo, compute_loss, get_mAP
from terra_ai.utils import decamelize

__version__ = 0.02

interactive = InteractiveCallback()


class GUINN:

    def __init__(self) -> None:
        self.name = "GUINN"
        self.callbacks = None
        self.params: TrainingDetailsData
        self.nn_name: str = ''
        self.dataset: Optional[PrepareDataset] = None
        self.deploy_type = None
        self.model: Optional[Model] = None
        self.model_path: Path = Path("./")
        self.deploy_path: Path = Path("./")
        self.dataset_path: Path = Path("./")
        self.loss: dict = {}
        self.metrics: dict = {}
        self.yolo_pred = None
        self.batch_size = 128
        self.val_batch_size = 1
        self.epochs = 5
        self.sum_epoch = 0
        self.stop_training = False
        self.retrain_flag = False
        self.shuffle: bool = True
        self.model_is_trained: bool = False
        self.progress_name = "training"
        self.train_length, self.val_length = 0, 0

    def _set_training_params(self, dataset: DatasetData, params: TrainingDetailsData) -> None:
        method_name = '_set_training_params'
        try:
            print(method_name)
            self.params = params
            self.dataset = self._prepare_dataset(
                dataset=dataset, model_path=params.model_path, state=params.state.status)
            self.train_length, self.val_length = get_dataset_length(self.dataset)

            if not self.dataset.data.architecture or self.dataset.data.architecture == ArchitectureChoice.Basic:
                self.deploy_type = self._set_deploy_type(self.dataset)
            else:
                self.deploy_type = self.dataset.data.architecture

            self.nn_name = "trained_model"

            train_size = len(self.dataset.dataframe.get("train")) if self.dataset.data.use_generator \
                else len(self.dataset.dataset.get('train'))
            if params.base.batch > train_size:
                if params.state.status == "addtrain":
                    params.state.set("stopped")
                else:
                    params.state.set("no_train")
                raise exceptions.TooBigBatchSize(params.base.batch, train_size)

            if params.state.status == "addtrain":
                if params.logs.get("addtrain_epochs")[-1] >= self.sum_epoch:
                    self.sum_epoch += params.base.epochs
                if params.logs.get("addtrain_epochs")[-1] < self.sum_epoch:
                    self.epochs = self.sum_epoch - params.logs.get("addtrain_epochs")[-1]
                else:
                    self.epochs = params.base.epochs
            else:
                self.epochs = params.base.epochs

            self.batch_size = params.base.batch

            interactive.set_attributes(dataset=self.dataset, params=params)
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def _set_callbacks(self, dataset: PrepareDataset, train_details: TrainingDetailsData, initial_model=None) -> None:
        method_name = '_set_callbacks'
        try:
            print(method_name)
            progress.pool(self.progress_name, finished=False, message="Добавление колбэков...")
            retrain_epochs = self.sum_epoch if train_details.state.status == "addtrain" else self.epochs

            self.callback = FitCallback(dataset=dataset, retrain_epochs=retrain_epochs,
                                        training_details=train_details, model_name=self.nn_name,
                                        deploy_type=self.deploy_type.name, initialed_model=initial_model)
            progress.pool(self.progress_name, finished=False, message="Добавление колбэков выполнено")
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    @staticmethod
    def _set_deploy_type(dataset: PrepareDataset) -> str:
        method_name = '_set_deploy_type'
        try:
            print(method_name)
            data = dataset.data
            inp_tasks = []
            out_tasks = []
            for key, val in data.inputs.items():
                if val.task == LayerInputTypeChoice.Dataframe:
                    tmp = []
                    for value in data.columns[key].values():
                        tmp.append(value.get("task"))
                    unique_vals = list(set(tmp))
                    if len(unique_vals) == 1 and unique_vals[0] in LayerInputTypeChoice.__dict__.keys() and \
                            unique_vals[0] in [LayerInputTypeChoice.Image, LayerInputTypeChoice.Text,
                                               LayerInputTypeChoice.Audio, LayerInputTypeChoice.Video]:
                        inp_tasks.append(unique_vals[0])
                    else:
                        inp_tasks.append(val.task)
                else:
                    inp_tasks.append(val.task)
            for key, val in data.outputs.items():
                if val.task == LayerOutputTypeChoice.Dataframe:
                    tmp = []
                    for value in data.columns[key].values():
                        tmp.append(value.task)
                    unique_vals = list(set(tmp))
                    if len(unique_vals) == 1 and unique_vals[0] in LayerOutputTypeChoice.__dict__.keys():
                        out_tasks.append(unique_vals[0])
                    else:
                        out_tasks.append(val.task)
                else:
                    out_tasks.append(val.task)

            inp_task_name = list(set(inp_tasks))[0] if len(set(inp_tasks)) == 1 else LayerInputTypeChoice.Dataframe
            out_task_name = list(set(out_tasks))[0] if len(set(out_tasks)) == 1 else LayerOutputTypeChoice.Dataframe

            if inp_task_name + out_task_name in ArchitectureChoice.__dict__.keys():
                deploy_type = ArchitectureChoice.__dict__[inp_task_name + out_task_name]
            elif out_task_name in ArchitectureChoice.__dict__.keys():
                deploy_type = ArchitectureChoice.__dict__[out_task_name]
            elif out_task_name == LayerOutputTypeChoice.ObjectDetection:
                deploy_type = ArchitectureChoice.__dict__[dataset.instructions.get(2).parameters.model.title() +
                                                          dataset.instructions.get(2).parameters.yolo.title()]
            else:
                raise MethodNotImplementedException(__method=inp_task_name + out_task_name,
                                                    __class="ArchitectureChoice")
            return deploy_type
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    @staticmethod
    def _prepare_dataset(dataset: DatasetData, model_path: Path, state: str) -> PrepareDataset:
        method_name = '_prepare_dataset'
        try:
            print(method_name)
            prepared_dataset = PrepareDataset(data=dataset, datasets_path=dataset.path)
            prepared_dataset.prepare_dataset()
            if state != "addtrain":
                prepared_dataset.deploy_export(os.path.join(model_path, "dataset"))
            return prepared_dataset
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def _set_model(self, model: ModelDetailsData, train_details: TrainingDetailsData) -> Model:
        method_name = '_set_model'
        try:
            print(method_name)
            if train_details.state.status == "training":
                validator = ModelValidator(model)
                train_model = validator.get_keras_model()
            else:
                # model_file = f"{self.nn_name}.trm"
                # train_model = load_model(os.path.join(train_details.model_path, f"{model_file}"), compile=False)
                train_model = self.load_model_from_json()
                weight = None
                for i in os.listdir(train_details.model_path):
                    if i[-3:] == '.h5' and 'last' in i:
                        weight = i
                if weight:
                    train_model.load_weights(os.path.join(train_details.model_path, weight))
            return train_model
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    @staticmethod
    def _save_params_for_deploy(params: TrainingDetailsData):
        method_name = '_save_params_for_deploy'
        try:
            print(method_name)
            with open(os.path.join(params.model_path, "config.train"), "w", encoding="utf-8") as train_config:
                json.dump(params.base.native(), train_config)
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def save_model(self) -> None:
        method_name = 'save_model'
        try:
            """
            Saving last model on each epoch end
    
            Returns:
                None
            """
            print(method_name)

            model_json = f"{self.nn_name}_json.trm"
            file_path_model_json: str = os.path.join(self.params.model_path, f"{model_json}")
            with open(file_path_model_json, "w") as json_file:
                json_file.write(self.model.to_json())

            custom_obj_json = f"{self.nn_name}_custom_obj_json.trm"
            file_path_custom_obj_json = os.path.join(self.params.model_path, f"{custom_obj_json}")
            with open(file_path_custom_obj_json, "w") as json_file:
                json.dump(terra_custom_layers, json_file)

            model_weights = f"{self.nn_name}_weights.h5"
            file_path_model_weights: str = os.path.join(self.params.model_path, f"{model_weights}")
            self.model.save_weights(file_path_model_weights)
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def load_model_from_json(self):
        model_json = f"{self.nn_name}_json.trm"
        file_path_model_json: str = os.path.join(self.params.model_path, f"{model_json}")
        with open(file_path_model_json) as json_file:
            data = json.load(json_file)

        custom_obj_json = f"{self.nn_name}_custom_obj_json.trm"
        file_path_custom_obj_json = os.path.join(self.params.model_path, f"{custom_obj_json}")
        with open(file_path_custom_obj_json) as json_file:
            custom_dict = json.load(json_file)
        custom_object = {}
        for k, v in custom_dict.items():
            try:
                custom_object[k] = getattr(importlib.import_module(v), k)
            except:
                continue
        model = tf.keras.models.model_from_json(json.dumps(data), custom_objects=custom_object)

        # model_weights = f"{self.nn_name}_weights.h5"
        # file_path_model_weights: str = os.path.join(self.params.model_path, f"{model_weights}")
        # model.load_weights(file_path_model_weights)
        return model

    def _kill_last_training(self, state):
        method_name = '_kill_last_training'
        try:
            print(method_name)
            for one_thread in threading.enumerate():
                if one_thread.getName() == "current_train":
                    current_status = state.state.status
                    state.state.set("stopped")
                    progress.pool(self.progress_name,
                                  message="Найдено незавершенное обучение. Идет очистка. Подождите.")
                    one_thread.join()
                    state.state.set(current_status)
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def terra_fit(self, dataset: DatasetData, gui_model: ModelDetailsData, training: TrainingDetailsData) -> dict:
        method_name = 'model_fit'
        try:
            """
               This method created for using wth externally compiled models

               Args:
                   dataset: DatasetData
                   gui_model: Keras model for fit - ModelDetailsData
                   training: TrainingDetailsData

               Return:
                   dict
               """
            print(method_name)
            # self._kill_last_training(state=training)
            progress.pool.reset(self.progress_name)

            if training.state.status != "addtrain":
                self._save_params_for_deploy(params=training)

            self.nn_cleaner(retrain=True if training.state.status == "training" else False)

            self._set_training_params(dataset=dataset, params=training)

            self.model = self._set_model(model=gui_model, train_details=training)
            if training.state.status == "training":
                self.save_model()

            self.model_fit(params=training, dataset=self.dataset, model=self.model)
            return {"dataset": self.dataset}
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def nn_cleaner(self, retrain: bool = False) -> None:
        method_name = 'nn_cleaner'
        try:
            print(method_name)
            keras.backend.clear_session()
            self.dataset = None
            self.deploy_type = None
            self.model = None
            if retrain:
                self.sum_epoch = 0
                self.loss = {}
                self.metrics = {}
                self.callbacks = []
                interactive.clear_history()
            gc.collect()
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def get_nn(self):
        self.nn_cleaner(retrain=True)
        return self

    # @progress.threading
    def model_fit(self, params: TrainingDetailsData, model: Model, dataset: PrepareDataset) -> None:
        method_name = 'base_model_fit'
        try:
            print(method_name)
            yolo_arch = True if dataset.data.architecture in YOLO_ARCHITECTURE else False
            self._set_callbacks(dataset=dataset, train_details=params)
            # callback = FitCallback(dataset, params)
            threading.enumerate()[-1].setName("current_train")
            progress.pool(self.progress_name, finished=False, message="Компиляция модели ...")
            if yolo_arch:
                version = dataset.instructions.get(list(dataset.data.outputs.keys())[0]).get('2_object_detection').get(
                    'yolo')
                classes = dataset.data.outputs.get(list(dataset.data.outputs.keys())[0]).classes_names
                yolo = create_yolo(model, input_size=416, channels=3, training=True, classes=classes, version=version)
                self.train_yolo_model(yolo_model=yolo, params=params, dataset=dataset, callback=self.callback)
            else:
                self.train_base_model(params, dataset, model, self.callback)

            progress.pool(self.progress_name, finished=False, message="\n Компиляция модели выполнена")
            progress.pool(self.progress_name, finished=False, message="\n Начало обучения ...")
            if (params.state.status == "stopped" and self.callbacks[0].last_epoch < params.base.epochs) or \
                    (params.state.status == "trained" and self.callbacks[0].last_epoch - 1 == params.base.epochs):
                self.sum_epoch = params.base.epochs
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    @staticmethod
    def _prepare_loss_dict(params: TrainingDetailsData):
        method_name = '_prepare_loss_dict'
        try:
            print(method_name)
            loss_dict = {}
            for output_layer in params.base.architecture.parameters.outputs:
                loss_obj = getattr(
                    importlib.import_module(
                        loss_metric_config.get("loss").get(output_layer.loss.name, {}).get('module')),
                    output_layer.loss.name
                )()
                loss_dict.update({str(output_layer.id): loss_obj})
            return loss_dict
        except Exception as e:
            print_error(GUINN().name, method_name, e)
            return None

    @staticmethod
    def set_optimizer(params: TrainingDetailsData):
        method_name = 'set_optimizer'
        try:
            print(method_name)
            optimizer_object = getattr(keras.optimizers, params.base.optimizer.type)
            parameters = params.base.optimizer.parameters.main.native()
            parameters.update(params.base.optimizer.parameters.extra.native())
            return optimizer_object(**parameters)
        except Exception as e:
            print_error(GUINN().name, method_name, e)
            return None

    @progress.threading
    def train_yolo_model(self, yolo_model: Model, params: TrainingDetailsData, dataset: PrepareDataset, callback):
        method_name = 'train_yolo_model'
        try:
            yolo_iou_loss_thresh = params.base.architecture.parameters.yolo.yolo_iou_loss_thresh
            train_warmup_epochs = params.base.architecture.parameters.yolo.train_warmup_epochs
            train_lr_init = params.base.architecture.parameters.yolo.train_lr_init
            train_lr_end = params.base.architecture.parameters.yolo.train_lr_end
            num_class = dataset.data.outputs.get(list(dataset.data.outputs.keys())[0]).num_classes
            classes = dataset.data.outputs.get(list(dataset.data.outputs.keys())[0]).classes_names

            optimizer = self.set_optimizer(params=params)

            global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
            steps_per_epoch = int(len(dataset.dataframe['train']) // params.base.batch)
            warmup_steps = train_warmup_epochs * steps_per_epoch
            total_steps = params.base.epochs * steps_per_epoch

            @tf.function
            def train_step(image_array, conv_target, serv_target):
                with tf.GradientTape() as tape:
                    pred_result = yolo_model(image_array['1'], training=True)
                    giou_loss = conf_loss = prob_loss = 0
                    prob_loss_cls = {}
                    predict = []
                    for idx in range(num_class):
                        prob_loss_cls[classes[idx]] = 0
                    for n, elem in enumerate(conv_target.keys()):
                        conv, pred = pred_result[n * 2], pred_result[n * 2 + 1]
                        predict.append(pred)
                        loss_items = compute_loss(
                            pred=pred,
                            conv=conv,
                            label=conv_target[elem],
                            bboxes=serv_target[elem],
                            YOLO_IOU_LOSS_THRESH=yolo_iou_loss_thresh,
                            i=n, CLASSES=classes)
                        giou_loss += loss_items[0]
                        conf_loss += loss_items[1]
                        prob_loss += loss_items[2]
                        for idx in range(num_class):
                            prob_loss_cls[classes[idx]] += loss_items[3][classes[idx]]

                    total_loss = giou_loss + conf_loss + prob_loss
                    gradients = tape.gradient(total_loss, yolo_model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, yolo_model.trainable_variables))
                    global_steps.assign_add(1)
                    if global_steps < warmup_steps:
                        lr = global_steps / warmup_steps * train_lr_init
                    else:
                        lr = train_lr_end + 0.5 * (train_lr_init - train_lr_end) * (
                            (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
                    lr = tf.cast(lr, dtype='float32')
                    optimizer.lr.assign(lr)
                return global_steps, giou_loss, conf_loss, prob_loss, total_loss, prob_loss_cls, predict

            @tf.function
            def validate_step(image_array, conv_target, serv_target):
                pred_result = yolo_model(image_array['1'], training=False)
                print('pred_result', len(pred_result))
                giou_loss = conf_loss = prob_loss = tf.convert_to_tensor(0., dtype='float32')

                prob_loss_cls = {}
                for idx in range(num_class):
                    prob_loss_cls[classes[idx]] = tf.convert_to_tensor(0., dtype='float32')

                predict = []
                for n, elem in enumerate(conv_target.keys()):
                    conv, pred = pred_result[n * 2], pred_result[n * 2 + 1]
                    predict.append(pred)
                    loss_items = compute_loss(
                        pred=pred,
                        conv=conv,
                        label=conv_target[elem],
                        bboxes=serv_target[elem],
                        YOLO_IOU_LOSS_THRESH=yolo_iou_loss_thresh,
                        i=n, CLASSES=classes
                    )
                    giou_loss = tf.add(giou_loss, loss_items[0])
                    conf_loss = tf.add(conf_loss, loss_items[1])
                    prob_loss = tf.add(prob_loss, loss_items[2])
                    for idx in range(num_class):
                        prob_loss_cls[classes[idx]] = tf.add(prob_loss_cls[classes[idx]], loss_items[3][classes[idx]])

                total_loss = tf.add(giou_loss, conf_loss)
                total_loss = tf.add(total_loss, prob_loss)

                return giou_loss, conf_loss, prob_loss, total_loss, prob_loss_cls, predict

            current_epoch = 0
            train_pred, train_true, val_pred, val_true = [], [], [], []
            optimizer = self.set_optimizer(params=params)
            output_array = None
            for _, out, _ in self.dataset.dataset['train'].batch(1).take(1):
                output_array = out
            for array in output_array.values():
                train_target_shape, val_target_shape = [self.train_length], [self.val_length]
                train_target_shape.extend(list(array.shape[1:]))
                val_target_shape.extend(list(array.shape[1:]))
                train_pred.append(np.zeros(train_target_shape))
                train_true.append(np.zeros(train_target_shape))
                val_pred.append(np.zeros(val_target_shape))
                val_true.append(np.zeros(val_target_shape))

            train_data_idxs = np.arange(self.train_length).tolist()
            callback.on_train_begin()
            for epoch in range(current_epoch, params.base.epochs):
                callback.on_epoch_begin()
                st = time.time()
                print(f'\n New epoch {epoch + 1}, batch {params.base.batch}\n')
                current_logs = {"epochs": epoch + 1, 'loss': {}, "metrics": {}, 'class_loss': {}, 'class_metrics': {}}
                train_loss_cls = {}
                for cls in range(num_class):
                    train_loss_cls[classes[cls]] = 0.
                current_idx = 0
                cur_step, giou_train, conf_train, prob_train, total_train = 0, 0, 0, 0, 0
                for image_data, target1, target2 in dataset.dataset.get('train').batch(params.base.batch):
                    bt = time.time()
                    results = train_step(image_data, target1, target2)

                    giou_train += results[1].numpy()
                    conf_train += results[2].numpy()
                    prob_train += results[3].numpy()
                    total_train += results[4].numpy()
                    for cls in range(num_class):
                        train_loss_cls[classes[cls]] += results[5][classes[cls]].numpy()

                    true_array = list(target1.values())
                    length = results[6][0].shape[0]
                    for i in range(len(train_pred)):
                        train_pred[i][current_idx: current_idx + length] = results[6][i].numpy()
                        train_true[i][current_idx: current_idx + length] = true_array[i].numpy()
                    current_idx += length
                    cur_step += 1
                    if interactive.urgent_predict:
                        print('\nGUINN interactive.urgent_predict\n')
                        val_steps = 0
                        val_current_idx = 0
                        for val_image_data, val_target1, val_target2 in dataset.dataset.get('val').batch(params.base.batch):
                            results = validate_step(val_image_data, target1, target2)
                            val_true_array = list(val_target1.values())
                            length = val_true_array[0].shape[0]
                            for i in range(len(val_true_array)):
                                val_pred[i][val_current_idx: val_current_idx + length] = results[5][i].numpy()
                                val_true[i][val_current_idx: val_current_idx + length] = val_true_array[i].numpy()
                            val_current_idx += length
                            val_steps += 1
                        callback.on_train_batch_end(batch=cur_step, arrays={
                            "train_true": train_true, "val_true": val_true, "train_pred": train_pred,
                            "val_pred": val_pred}, train_data_idxs=train_data_idxs)
                    else:
                        callback.on_train_batch_end(batch=cur_step)
                    print(f' -- batch {cur_step} - {round(time.time() - bt, 3)} {current_idx, length}')
                    if callback.stop_training:
                        break
                print(f'\n train epoch time: {round(time.time() - st, 3)}\n')

                st = time.time()
                self.save_model()
                if callback.stop_training:
                    callback.on_train_end(yolo_model)
                    break
                print(f'\n save_model time: {round(time.time() - st, 3)},\n')

                current_logs['loss']['giou_loss'] = {'train': giou_train / cur_step}
                current_logs['loss']['conf_loss'] = {'train': conf_train / cur_step}
                current_logs['loss']['prob_loss'] = {'train': prob_train / cur_step}
                current_logs['loss']['total_loss'] = {'train': total_train / cur_step}
                current_logs['class_loss']['prob_loss'] = {}
                for cls in range(num_class):
                    current_logs['class_loss']['prob_loss'][str(classes[cls])] = \
                        {'train': train_loss_cls[str(classes[cls])] / cur_step}
                    train_loss_cls[str(classes[cls])] = train_loss_cls[str(classes[cls])] / cur_step
                print(
                    "\n epoch_time:{:7.2f} sec, giou_train_loss:{:7.2f}, conf_train_loss:{:7.2f}, prob_train_loss:{:7.2f}, total_train_loss:{:7.2f}\n".
                        format(time.time() - st, giou_train / cur_step, conf_train / cur_step, prob_train / cur_step,
                               total_train / cur_step))
                st = time.time()
                val_steps, giou_val, conf_val, prob_val, total_val = 0, 0, 0, 0, 0
                val_loss_cls = {}
                for cls in range(num_class):
                    val_loss_cls[classes[cls]] = 0.
                val_current_idx = 0
                for image_data, target1, target2 in dataset.dataset.get('val').batch(params.base.batch):
                    bt = time.time()
                    results = validate_step(image_data, target1, target2)
                    giou_val += results[0].numpy()
                    conf_val += results[1].numpy()
                    prob_val += results[2].numpy()
                    total_val += results[3].numpy()
                    for cls in range(num_class):
                        val_loss_cls[str(classes[cls])] += results[4][str(classes[cls])].numpy()

                    val_true_array = list(target1.values())
                    length = val_true_array[0].shape[0]
                    for i in range(len(val_true_array)):
                        val_pred[i][val_current_idx: val_current_idx + length] = results[5][i].numpy()
                        val_true[i][val_current_idx: val_current_idx + length] = val_true_array[i].numpy()
                    val_current_idx += length
                    val_steps += 1
                    print(f' -- val_batch {val_steps} - {round(time.time() - bt, 3)}', val_current_idx)
                print(f'\n val epoch time: {round(time.time() - st, 3)}\n')

                current_logs['loss']['giou_loss']["val"] = giou_val / val_steps
                current_logs['loss']['conf_loss']["val"] = conf_val / val_steps
                current_logs['loss']['prob_loss']["val"] = prob_val / val_steps
                current_logs['loss']['total_loss']["val"] = total_val / val_steps
                for cls in range(num_class):
                    current_logs['class_loss']['prob_loss'][str(classes[cls])]["val"] = \
                        val_loss_cls[str(classes[cls])] / val_steps
                print(
                    "\n epoch_time:{:7.2f} sec, giou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n".
                        format(time.time() - st, giou_val / val_steps, conf_val / val_steps, prob_val / val_steps, total_val / val_steps))
                st = time.time()
                map50 = get_mAP(yolo_model, dataset, score_threshold=0.05, iou_threshold=[0.50],
                                TRAIN_CLASSES=dataset.data.outputs.get(2).classes_names, dataset_path=dataset.data.path)
                current_logs['metrics']['mAP50'] = {"val": map50.get('val_mAP50')}
                current_logs['class_metrics']['mAP50'] = {}
                for cls in range(num_class):
                    try:
                        current_logs['class_metrics']['mAP50'][str(classes[cls])] = \
                            {"val": map50.get(f"val_mAP50_class_{classes[cls]}") * 100}
                    except:
                        current_logs['class_metrics']['mAP50'][str(classes[cls])] = {"val": None}
                print(f'\n get_mAP time: {round(time.time() - st, 3)} \n map50: {map50}\n')
                # print("\n\n epoch_time:{:7.2f} sec, mAP50:{}".format(time.time() - st, mAP50))
                st = time.time()
                callback.on_epoch_end(
                    epoch=epoch + 1,
                    arrays={"train_pred": train_pred, "val_pred": val_pred, "train_true": train_true,
                            "val_true": val_true},
                    train_data_idxs=train_data_idxs,
                    logs=current_logs
                )
                print(f'\n callback.on_epoch_end epoch time: {round(time.time() - st, 3)}')
                print(
                    f"\nEpoch {callback.current_logs.get('epochs')}:\n"
                    # f"\nlog_history: {callback.log_history.get('epochs')}, "
                    f"epoch_time={round(time.time() - callback._time_first_step, 3)}"
                    # f"\nloss={callback.log_history.get('output').get('loss')}"
                    # f"\nmetrics={callback.log_history.get('output').get('metrics')}"
                    # f"\nloss={callback.log_history.get('output').get('class_loss')}"
                    # f"\nmetrics={callback.log_history.get('output').get('class_metrics')}"
                    # f" \n loss={callback.current_logs.get('2').get('loss')}\n"
                    # f" \n metrics={callback.current_logs.get('2').get('metrics')}\n"
                    # f" \n class_loss={callback.current_logs.get('2').get('class_loss')}\n"
                    # f" \n class_metrics={callback.current_logs.get('2').get('class_metrics')}"
                )
                st = time.time()
                best_path = callback.save_best_weights()
                if best_path:
                    yolo_model.save_weights(best_path)
                    print(f"\nEpoch {epoch + 1}")
                    print(f"Best weights was saved in directory {best_path}")
                print(f'\n save_best_weights time: {round(time.time() - st, 3)}\n')
            callback.on_train_end(yolo_model)
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    @progress.threading
    def train_base_model(self, params: TrainingDetailsData, dataset: PrepareDataset, model: Model, callback):
        method_name = 'train_base_model'
        try:
            print(method_name)

            @tf.function
            def train_step(x_batch, y_batch, losses: dict, train_model: Model, set_optimizer):
                """
                losses = {'2': loss_fn}
                """
                with tf.GradientTape() as tape:
                    logits_ = train_model(x_batch, training=True)
                    y_true_ = list(y_batch.values())
                    if not isinstance(logits_, list):
                        loss_fn = losses.get(list(losses.keys())[0])
                        total_loss = loss_fn(y_true_[0], logits_)
                    else:
                        total_loss = tf.convert_to_tensor(0.)
                        for k, key in enumerate(losses.keys()):
                            loss_fn = losses[key]
                            total_loss = tf.add(loss_fn(y_true_[k], logits_[k]), total_loss)
                grads = tape.gradient(total_loss, model.trainable_weights)
                set_optimizer.apply_gradients(zip(grads, model.trainable_weights))
                return [logits_] if not isinstance(logits_, list) else logits_, y_true_

            @tf.function
            def test_step(train_model: Model, x_batch, y_batch):
                with tf.GradientTape() as tape:
                    test_logits = train_model(x_batch, training=False)
                    true_array = list(y_batch.values())
                    test_logits = test_logits if isinstance(test_logits, list) else [test_logits]
                return test_logits, true_array

            current_epoch = self.callback.last_epoch
            train_pred, train_true, val_pred, val_true = {}, {}, {}, {}
            optimizer = self.set_optimizer(params=params)
            loss = self._prepare_loss_dict(params=params)
            output_list = list(dataset.data.outputs.keys())
            for out in output_list:
                train_target_shape, val_target_shape = [self.train_length], [self.val_length]
                train_target_shape.extend(list(self.dataset.data.outputs.get(out).shape))
                val_target_shape.extend(list(self.dataset.data.outputs.get(out).shape))
                train_pred[f"{out}"] = np.zeros(train_target_shape)
                train_true[f"{out}"] = np.zeros(train_target_shape)
                val_pred[f"{out}"] = np.zeros(val_target_shape)
                val_true[f"{out}"] = np.zeros(val_target_shape)

            train_data_idxs = np.arange(self.train_length).tolist()
            callback.on_train_begin()
            for epoch in range(current_epoch, current_epoch + params.base.epochs):
                callback.on_epoch_begin()
                train_steps = 0
                st = time.time()
                current_idx = 0
                print(f'\n New epoch {epoch + 1}, batch {params.base.batch}\n')
                for x_batch_train, y_batch_train in dataset.dataset.get('train').batch(params.base.batch):
                    st1 = time.time()
                    logits, y_true = train_step(
                        x_batch=x_batch_train, y_batch=y_batch_train, train_model=model,
                        losses=loss, set_optimizer=optimizer
                    )
                    length = logits[0].shape[0]
                    for i, out in enumerate(output_list):
                        train_pred[f"{out}"][current_idx: current_idx + length] = logits[i].numpy()
                        train_true[f"{out}"][current_idx: current_idx + length] = y_true[i].numpy()
                    current_idx += length
                    # print('- train batch', train_steps, current_idx, current_idx + length,
                    #   train_pred[f"{out}"][length * train_steps], train_true[f"{out}"][length * train_steps])
                    train_steps += 1
                    if interactive.urgent_predict:
                        val_steps = 0
                        current_val_idx = 0
                        for x_batch_val, y_batch_val in dataset.dataset.get('val').batch(params.base.batch):
                            val_pred_array, val_true_array = test_step(
                                train_model=model, x_batch=x_batch_val, y_batch=y_batch_val)
                            length = val_true_array[0].shape[0]
                            for i, out in enumerate(output_list):
                                val_pred[f"{out}"][current_val_idx: current_val_idx + length] = \
                                    val_pred_array[i].numpy()
                                val_true[f"{out}"][current_val_idx: current_val_idx + length] = \
                                    val_true_array[i].numpy()
                            current_val_idx += length
                            val_steps += 1
                        callback.on_train_batch_end(batch=train_steps, arrays={
                            "train_true": train_true, "val_true": val_true, "train_pred": train_pred,
                            "val_pred": val_pred}, train_data_idxs=train_data_idxs)
                    else:
                        callback.on_train_batch_end(batch=train_steps)
                    print('- train batch', train_steps, round(time.time() - st1, 3))
                    if callback.stop_training:
                        break
                print(f'- train epoch time: {round(time.time() - st, 3)}\n')

                st = time.time()
                self.save_model()
                if callback.stop_training:
                    callback.on_train_end(model)
                    break
                print(f'\n- save_model time: {round(time.time() - st, 3)}\n')
                st = time.time()
                # Run a validation loop at the end of each epoch.
                print(f'\n Run a validation loop')
                val_steps = 0
                current_val_idx = 0
                for x_batch_val, y_batch_val in dataset.dataset.get('val').batch(params.base.batch):
                    # st1 = time.time()
                    val_pred_array, val_true_array = test_step(
                        train_model=model, x_batch=x_batch_val, y_batch=y_batch_val)
                    length = val_true_array[0].shape[0]
                    for i, out in enumerate(output_list):
                        val_pred[f"{out}"][current_val_idx: current_val_idx + length] = val_pred_array[i].numpy()
                        val_true[f"{out}"][current_val_idx: current_val_idx + length] = val_true_array[i].numpy()
                    current_val_idx += length
                    val_steps += 1
                    # print('- val batch', train_steps, round(time.time() - st1, 3))
                print(f'- val epoch time: {round(time.time() - st, 3)}\n')

                st = time.time()
                callback.on_epoch_end(
                    epoch=epoch + 1,
                    arrays={
                        "train_pred": train_pred,"val_pred": val_pred,"train_true": train_true,"val_true": val_true
                    },
                    train_data_idxs=train_data_idxs
                )
                print(f'\n callback.on_epoch_end epoch time: {round(time.time() - st, 2)}')
                print(
                    f"\nEpoch {callback.current_logs.get('epochs')}:"
                    f"\nlog_history: {callback.log_history}, "
                    f"epoch_time={round(time.time() - callback._time_first_step, 3)}"
                    # f"\nloss={callback.log_history.get('2').get('loss')}"
                    # f"\nmetrics={callback.log_history.get('2').get('metrics')}\n"
                    # f" \n loss={callback.current_logs.get('2').get('loss')}\n"
                    # f" \n metrics={callback.current_logs.get('2').get('metrics')}\n"
                    # f" \n class_loss={callback.current_logs.get('2').get('class_loss')}\n"
                    # f" \n class_metrics={callback.current_logs.get('2').get('class_metrics')}"
                )

                best_path = callback.save_best_weights()
                if best_path:
                    model.save_weights(best_path)
                    print(f"\nEpoch {epoch + 1}")
                    print(f"Best weights was saved in directory {best_path}")
            callback.on_train_end(model)
        except Exception as e:
            print_error(GUINN().name, method_name, e)


class MemoryUsage:
    def __init__(self, debug=False):
        self.debug = debug
        try:
            N.nvmlInit()
            self.gpu = settings.USE_GPU
        except:
            self.gpu = False

    def get_usage(self):
        usage_dict = {}
        if self.gpu:
            gpu_name = N.nvmlDeviceGetName(N.nvmlDeviceGetHandleByIndex(0))
            gpu_utilization = N.nvmlDeviceGetUtilizationRates(N.nvmlDeviceGetHandleByIndex(0))
            gpu_memory = N.nvmlDeviceGetMemoryInfo(N.nvmlDeviceGetHandleByIndex(0))
            usage_dict["GPU"] = {
                'gpu_name': gpu_name,
                'gpu_utilization': f'{gpu_utilization.gpu: .2f}',
                'gpu_memory_used': f'{gpu_memory.used / 1024 ** 3: .2f}GB',
                'gpu_memory_total': f'{gpu_memory.total / 1024 ** 3: .2f}GB'
            }
            if self.debug:
                print(f'GPU usage: {gpu_utilization.gpu: .2f} ({gpu_memory.used / 1024 ** 3: .2f}GB / '
                      f'{gpu_memory.total / 1024 ** 3: .2f}GB)')
        else:
            cpu_usage = psutil.cpu_percent(percpu=True)
            usage_dict["CPU"] = {
                'cpu_utilization': f'{sum(cpu_usage) / len(cpu_usage): .2f}',
            }
            if self.debug:
                print(f'Average CPU usage: {sum(cpu_usage) / len(cpu_usage): .2f}')
                print(f'Max CPU usage: {max(cpu_usage): .2f}')
        usage_dict["RAM"] = {
            'ram_utilization': f'{psutil.virtual_memory().percent: .2f}',
            'ram_memory_used': f'{psutil.virtual_memory().used / 1024 ** 3: .2f}GB',
            'ram_memory_total': f'{psutil.virtual_memory().total / 1024 ** 3: .2f}GB'
        }
        usage_dict["Disk"] = {
            'disk_utilization': f'{psutil.disk_usage("/").percent: .2f}',
            'disk_memory_used': f'{psutil.disk_usage("/").used / 1024 ** 3: .2f}GB',
            'disk_memory_total': f'{psutil.disk_usage("/").total / 1024 ** 3: .2f}GB'
        }
        if self.debug:
            print(f'RAM usage: {psutil.virtual_memory().percent: .2f} '
                  f'({psutil.virtual_memory().used / 1024 ** 3: .2f}GB / '
                  f'{psutil.virtual_memory().total / 1024 ** 3: .2f}GB)')
            print(f'Disk usage: {psutil.disk_usage("/").percent: .2f} '
                  f'({psutil.disk_usage("/").used / 1024 ** 3: .2f}GB / '
                  f'{psutil.disk_usage("/").total / 1024 ** 3: .2f}GB)')
        return usage_dict


class FitCallback:
    """CustomCallback for all task type"""

    def __init__(self, dataset: PrepareDataset, training_details: TrainingDetailsData, retrain_epochs: int = None,
                 model_name: str = "model", deploy_type: str = "", initialed_model=None):
        super().__init__()
        print('\n FitCallback')
        self.name = "FitCallback"
        self.current_logs = {}
        self.usage_info = MemoryUsage(debug=False)
        self.training_detail = training_details
        self.dataset = dataset
        self.dataset_path = dataset.data.path
        self.deploy_type = getattr(DeployTypeChoice, deploy_type)
        self.is_yolo = True if dataset.data.architecture in YOLO_ARCHITECTURE else False
        self.batch_size = training_details.base.batch
        self.epochs = training_details.base.epochs
        self.retrain_epochs = retrain_epochs
        self.still_epochs = training_details.base.epochs
        self.nn_name = model_name
        self.deploy_path = training_details.deploy_path
        self.model_path = training_details.model_path
        self.stop_training = False

        self.batch = 0
        self.num_batches = 0
        self.last_epoch = 0
        self._start_time = time.time()
        self._time_batch_step = time.time()
        self._time_first_step = time.time()
        self._sum_time = 0
        self._sum_epoch_time = 0
        self.progress_name = "training"
        self.result = {
            'info': None,
            "train_usage": {
                "hard_usage": self.usage_info.get_usage(),
                "timings": {
                    "estimated_time": 0,
                    "elapsed_time": 0,
                    "still_time": 0,
                    "avg_epoch_time": 0,
                    "elapsed_epoch_time": 0,
                    "still_epoch_time": 0,
                    "epoch": {
                        "current": 0,
                        "total": 0
                    },
                    "batch": {
                        "current": 0,
                        "total": 0
                    },
                }
            },
            'train_data': None,
            'states': {}
        }
        # аттрибуты для чекпоинта
        self.checkpoint_config = training_details.base.architecture.parameters.checkpoint
        self.checkpoint_mode = self._get_checkpoint_mode()  # min max
        self.num_outputs = len(self.dataset.data.outputs.keys())
        self.metric_checkpoint = self.checkpoint_config.metric_name  # "val_mAP50" if self.is_yolo else "loss"
        self.class_outputs = class_metric_list(self.dataset)
        if self.dataset.data.architecture in CLASSIFICATION_ARCHITECTURE:
            self.y_true, _ = BaseClassificationCallback().get_y_true(self.dataset)
            self.class_idx = BaseClassificationCallback().prepare_class_idx(self.y_true, self.dataset)

        self.log_history = self._load_logs()
        # self.log_history = self._prepare_log_history_template(self.dataset, self.training_detail)

        # yolo params
        self.model = initialed_model
        self.samples_train = []
        self.samples_val = []
        self.samples_target_train = []
        self.samples_target_val = []

    @staticmethod
    def _prepare_log_history_template(options: PrepareDataset, params: TrainingDetailsData):
        method_name = '_prepare_log_history_template'
        try:
            print(method_name)
            log_history = {"epochs": []}
            if options.data.architecture in BASIC_ARCHITECTURE:
                for output_layer in params.base.architecture.parameters.outputs:
                    out = f"{output_layer.id}"
                    log_history[out] = {
                        "loss": {}, "metrics": {},
                        "class_loss": {}, "class_metrics": {},
                        "progress_state": {"loss": {}, "metrics": {}}
                    }
                    log_history[out]["loss"][output_layer.loss.name] = {"train": [], "val": []}
                    log_history[out]["progress_state"]["loss"][output_layer.loss.name] = {
                        "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []
                    }
                    for metric in output_layer.metrics:
                        log_history[out]["metrics"][metric.name] = {"train": [], "val": []}
                        log_history[out]["progress_state"]["metrics"][metric.name] = {
                            "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []
                        }

                    if options.data.architecture in CLASS_ARCHITECTURE:
                        log_history[out]["class_loss"] = {}
                        log_history[out]["class_metrics"] = {}
                        for class_name in options.data.outputs.get(int(out)).classes_names:
                            log_history[out]["class_metrics"][class_name] = {}
                            log_history[out]["class_loss"][class_name] = \
                                {output_layer.loss.name: {"train": [], "val": []}}
                            for metric in output_layer.metrics:
                                log_history[out]["class_metrics"][class_name][metric.name] = {"train": [], "val": []}

            if options.data.architecture in YOLO_ARCHITECTURE:
                # log_history['learning_rate'] = []
                log_history['output'] = {
                    "loss": {
                        'giou_loss': {"train": [], "val": []},
                        'conf_loss': {"train": [], "val": []},
                        'prob_loss': {"train": [], "val": []},
                        'total_loss': {"train": [], "val": []}
                    },
                    "class_loss": {'prob_loss': {}},
                    "metrics": {'mAP50': {"train": [], "val": []}},
                    "class_metrics": {'mAP50': {}},
                    "progress_state": {
                        "loss": {
                            'giou_loss': {
                                "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []},
                            'conf_loss': {
                                "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []},
                            'prob_loss': {
                                "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []},
                            'total_loss': {
                                "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []}
                        },
                        "metrics": {
                            'mAP50': {"mean_log_history": [], "normal_state": [], "overfitting": []}
                        }
                    }
                }
                out = list(options.data.outputs.keys())[0]
                for class_name in options.data.outputs.get(out).classes_names:
                    log_history['output']["class_loss"]['prob_loss'][class_name] = {"train": [], "val": []}
                    log_history['output']["class_metrics"]['mAP50'][class_name] = {"train": [], "val": []}
            return log_history
        except Exception as e:
            print_error('FitCallback', method_name, e)

    @staticmethod
    def update_class_idx(dataset_class_idx, predict_idx):
        method_name = 'update_class_idx'
        try:
            print(method_name)
            update_idx = {'train': {}, "val": dataset_class_idx.get('val')}
            for out in dataset_class_idx['train'].keys():
                update_idx['train'][out] = {}
                for cls in dataset_class_idx['train'][out].keys():
                    shift = predict_idx[0]
                    update_idx['train'][out][cls] = list(np.array(dataset_class_idx['train'][out][cls]) - shift)
            return update_idx
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def current_basic_logs(self, epoch: int, arrays: dict, train_idx: list):
        method_name = 'current_basic_logs'
        try:
            print(method_name)
            self.current_logs = {"epochs": epoch}
            update_cls = {}
            if self.dataset.data.architecture in CLASSIFICATION_ARCHITECTURE:
                update_cls = self.update_class_idx(self.class_idx, train_idx)
            for output_layer in self.training_detail.base.architecture.parameters.outputs:
                out = f"{output_layer.id}"
                name_classes = self.dataset.data.outputs.get(output_layer.id).classes_names
                self.current_logs[out] = {"loss": {}, "metrics": {}, "class_loss": {}, "class_metrics": {}}

                # calculate loss
                loss_name = output_layer.loss.name
                loss_fn = getattr(
                    importlib.import_module(loss_metric_config.get("loss").get(loss_name, {}).get('module')), loss_name
                )
                train_loss = self._get_loss_calculation(
                    loss_obj=loss_fn, out=out, y_true=arrays.get("train_true").get(out),
                    y_pred=arrays.get("train_pred").get(out))
                val_loss = self._get_loss_calculation(
                    loss_obj=loss_fn, out=out, y_true=arrays.get("val_true").get(out),
                    y_pred=arrays.get("val_pred").get(out))
                print(train_loss, val_loss)
                self.current_logs[out]["loss"][output_layer.loss.name] = {"train": train_loss, "val": val_loss}
                if self.class_outputs.get(output_layer.id):
                    self.current_logs[out]["class_loss"][output_layer.loss.name] = {}
                    if self.dataset.data.architecture in CLASSIFICATION_ARCHITECTURE:
                        for i, cls in enumerate(name_classes):
                            train_class_loss = self._get_loss_calculation(
                                loss_obj=loss_fn, out=out,
                                y_true=arrays.get("train_true").get(out)[update_cls['train'][out][cls], ...],
                                y_pred=arrays.get("train_pred").get(out)[update_cls['train'][out][cls], ...])
                            val_class_loss = self._get_loss_calculation(
                                loss_obj=loss_fn, out=out,
                                y_true=arrays.get("val_true").get(out)[update_cls['val'][out][cls], ...],
                                y_pred=arrays.get("val_pred").get(out)[update_cls['val'][out][cls], ...])
                            self.current_logs[out]["class_loss"][output_layer.loss.name][cls] = \
                                {"train": train_class_loss, "val": val_class_loss}
                    else:
                        for i, cls in enumerate(name_classes):
                            train_class_loss = self._get_loss_calculation(
                                loss_obj=loss_fn, out=out, class_idx=i, show_class=True,
                                y_true=arrays.get("train_true").get(out), y_pred=arrays.get("train_pred").get(out))
                            val_class_loss = self._get_loss_calculation(
                                loss_obj=loss_fn, out=out, class_idx=i, show_class=True,
                                y_true=arrays.get("val_true").get(out), y_pred=arrays.get("val_pred").get(out))
                            self.current_logs[out]["class_loss"][output_layer.loss.name][cls] = \
                                {"train": train_class_loss, "val": val_class_loss}

                # calculate metrics
                for metric_name in output_layer.metrics:
                    metric_name = metric_name.name
                    metric_fn = getattr(
                        importlib.import_module(loss_metric_config.get("metric").get(metric_name, {}).get('module')),
                        metric_name
                    )
                    train_metric = self._get_metric_calculation(
                        metric_name, metric_fn, out,
                        arrays.get("train_true").get(out), arrays.get("train_pred").get(out))
                    val_metric = self._get_metric_calculation(
                        metric_name, metric_fn, out, arrays.get("val_true").get(out), arrays.get("val_pred").get(out))
                    self.current_logs[out]["metrics"][metric_name] = {"train": train_metric, "val": val_metric}
                    if self.class_outputs.get(output_layer.id):
                        self.current_logs[out]["class_metrics"][metric_name] = {}
                        if self.dataset.data.architecture in CLASSIFICATION_ARCHITECTURE and \
                                metric_name not in [Metric.BalancedRecall, Metric.BalancedPrecision,
                                                    Metric.BalancedFScore, Metric.FScore]:
                            for i, cls in enumerate(name_classes):
                                train_class_metric = self._get_metric_calculation(
                                    metric_name=metric_name, metric_obj=metric_fn, out=out,
                                    y_true=arrays.get("train_true").get(out)[update_cls['train'][out][cls], ...],
                                    y_pred=arrays.get("train_pred").get(out)[update_cls['train'][out][cls], ...])
                                val_class_metric = self._get_metric_calculation(
                                    metric_name=metric_name, metric_obj=metric_fn, out=out,
                                    y_true=arrays.get("val_true").get(out)[update_cls['val'][out][cls], ...],
                                    y_pred=arrays.get("val_pred").get(out)[update_cls['val'][out][cls], ...])
                                self.current_logs[out]["class_metrics"][metric_name][cls] = \
                                    {"train": train_class_metric, "val": val_class_metric}
                        else:
                            for i, cls in enumerate(name_classes):
                                train_class_metric = self._get_metric_calculation(
                                    metric_name=metric_name, metric_obj=metric_fn, out=out, show_class=True,
                                    y_true=arrays.get("train_true").get(out),
                                    y_pred=arrays.get("train_pred").get(out), class_idx=i)
                                val_class_metric = self._get_metric_calculation(
                                    metric_name=metric_name, metric_obj=metric_fn, out=out, show_class=True,
                                    y_true=arrays.get("val_true").get(out),
                                    y_pred=arrays.get("val_pred").get(out), class_idx=i)
                                self.current_logs[out]["class_metrics"][metric_name][cls] = \
                                    {"train": train_class_metric, "val": val_class_metric}
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _get_loss_calculation(self, loss_obj, out: str, y_true, y_pred, show_class=False, class_idx=0):
        method_name = '_get_loss_calculation'
        try:
            encoding = self.dataset.data.outputs.get(int(out)).encoding
            num_classes = self.dataset.data.outputs.get(int(out)).num_classes
            if show_class and (encoding == LayerEncodingChoice.ohe or encoding == LayerEncodingChoice.multi):
                true_array = y_true[..., class_idx:class_idx + 1]
                pred_array = y_pred[..., class_idx:class_idx + 1]
            elif show_class:
                true_array = to_categorical(y_true, num_classes)[..., class_idx:class_idx + 1]
                pred_array = y_pred[..., class_idx:class_idx + 1]
            else:
                true_array = y_true
                pred_array = y_pred
            loss_value = float(loss_obj()(true_array, pred_array).numpy())
            return loss_value if not math.isnan(loss_value) else None
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _get_metric_calculation(self, metric_name, metric_obj, out: str, y_true, y_pred, show_class=False, class_idx=0):
        method_name = '_get_metric_calculation'
        try:
            encoding = self.dataset.data.outputs.get(int(out)).encoding
            num_classes = self.dataset.data.outputs.get(int(out)).num_classes
            if metric_name == Metric.MeanIoU:
                m = metric_obj(num_classes)
            elif metric_name == Metric.BalancedDiceCoef:
                m = metric_obj(encoding=encoding.name)
            else:
                m = metric_obj()
            if show_class and (encoding == LayerEncodingChoice.ohe or encoding == LayerEncodingChoice.multi):
                if metric_name == Metric.Accuracy:
                    true_array = to_categorical(np.argmax(y_true, axis=-1), num_classes)[..., class_idx]
                    pred_array = to_categorical(np.argmax(y_pred, axis=-1), num_classes)[..., class_idx]
                    m.update_state(true_array, pred_array)
                elif metric_name in [Metric.BalancedRecall, Metric.BalancedPrecision, Metric.BalancedFScore,
                                     Metric.FScore, Metric.BalancedDiceCoef]:
                    m.update_state(y_true, y_pred, show_class=show_class, class_idx=class_idx)
                else:
                    m.update_state(y_true[..., class_idx:class_idx + 1], y_pred[..., class_idx:class_idx + 1])
            elif show_class:
                if metric_name == Metric.Accuracy:
                    true_array = y_true[..., class_idx]
                    pred_array = to_categorical(np.argmax(y_pred, axis=-1), num_classes)[..., class_idx]
                    m.update_state(true_array, pred_array)
                else:
                    true_array = to_categorical(y_true, num_classes)[..., class_idx:class_idx + 1]
                    pred_array = y_pred[..., class_idx:class_idx + 1]
                    m.update_state(true_array, pred_array)
            else:
                if metric_name in [Metric.UnscaledMAE, Metric.PercentMAE]:
                    m.update_state(y_true, y_pred, output=int(out), preprocess=self.dataset.preprocessing)
                else:
                    m.update_state(y_true, y_pred)
            metric_value = float(m.result().numpy())
            return metric_value if not math.isnan(metric_value) else None
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _update_log_history(self):
        method_name = '_update_log_history'
        try:
            print(method_name)
            if self.current_logs['epochs'] in self.log_history['epochs']:
                print(f"\nCurrent epoch {self.current_logs['epochs']} is already in log_history\n")
            self.log_history['epochs'].append(self.current_logs['epochs'])
            if self.dataset.data.architecture in BASIC_ARCHITECTURE:
                for output_layer in self.training_detail.base.architecture.parameters.outputs:
                    out = f"{output_layer.id}"
                    classes_names = self.dataset.data.outputs.get(output_layer.id).classes_names
                    loss_name = output_layer.loss.name
                    for data_type in ['train', 'val']:
                        self.log_history[out]['loss'][loss_name][data_type].append(
                            round_loss_metric(self.current_logs.get(out).get('loss').get(loss_name).get(data_type))
                        )
                    self.log_history[out]['progress_state']['loss'][loss_name]['mean_log_history'].append(
                        round_loss_metric(
                            self._get_mean_log(self.log_history.get(out).get('loss').get(loss_name).get('val')))
                    )
                    loss_underfitting = self._evaluate_underfitting(
                        metric_name=loss_name, train_log=self.log_history[out]['loss'][loss_name]['train'][-1],
                        val_log=self.log_history[out]['loss'][loss_name]['val'][-1], metric_type='loss'
                    )
                    loss_overfitting = self._evaluate_overfitting(
                        metric_name=loss_name, metric_type='loss',
                        mean_log=self.log_history[out]['progress_state']['loss'][loss_name]['mean_log_history']
                    )
                    if loss_underfitting or loss_overfitting:
                        normal_state = False
                    else:
                        normal_state = True

                    self.log_history[out]['progress_state']['loss'][loss_name][
                        'underfitting'].append(loss_underfitting)
                    self.log_history[out]['progress_state']['loss'][loss_name][
                        'overfitting'].append(loss_overfitting)
                    self.log_history[out]['progress_state']['loss'][loss_name][
                        'normal_state'].append(normal_state)

                    if self.current_logs.get(out).get("class_loss"):
                        for cls in classes_names:
                            self.log_history[out]['class_loss'][cls][loss_name]["train"].append(
                                round_loss_metric(self.current_logs[out]['class_loss'][loss_name][cls]["train"])
                            )
                            self.log_history[out]['class_loss'][cls][loss_name]["val"].append(
                                round_loss_metric(self.current_logs[out]['class_loss'][loss_name][cls]["val"])
                            )

                    for metric_name in output_layer.metrics:
                        metric_name = metric_name.name
                        for data_type in ['train', 'val']:
                            self.log_history[out]['metrics'][metric_name][data_type].append(
                                round_loss_metric(
                                    self.current_logs.get(out).get('metrics').get(metric_name).get(data_type)))
                        self.log_history[out]['progress_state']['metrics'][metric_name]['mean_log_history'].append(
                            round_loss_metric(self._get_mean_log(self.log_history[out]['metrics'][metric_name]['val']))
                        )
                        metric_underfittng = self._evaluate_underfitting(
                            metric_name=metric_name, metric_type='metric',
                            train_log=self.log_history[f"{out}"]['metrics'][metric_name]['train'][-1],
                            val_log=self.log_history[f"{out}"]['metrics'][metric_name]['val'][-1]
                        )
                        metric_overfitting = self._evaluate_overfitting(
                            metric_name=metric_name, metric_type='metric',
                            mean_log=self.log_history[out]['progress_state']['metrics']
                            [metric_name]['mean_log_history']
                        )
                        if metric_underfittng or metric_overfitting:
                            normal_state = False
                        else:
                            normal_state = True
                        self.log_history[out]['progress_state']['metrics'][metric_name][
                            'underfitting'].append(metric_underfittng)
                        self.log_history[out]['progress_state']['metrics'][metric_name][
                            'overfitting'].append(metric_overfitting)
                        self.log_history[out]['progress_state']['metrics'][metric_name][
                            'normal_state'].append(normal_state)
                        if self.current_logs.get(out).get("class_metrics"):
                            for cls in classes_names:
                                self.log_history[out]['class_metrics'][cls][metric_name]["train"].append(
                                    round_loss_metric(
                                        self.current_logs[out]['class_metrics'][metric_name][cls]["train"]))
                                self.log_history[out]['class_metrics'][cls][metric_name]["val"].append(
                                    round_loss_metric(self.current_logs[out]['class_metrics'][metric_name][cls]["val"]))

            if self.dataset.data.architecture in YOLO_ARCHITECTURE:
                # self.log_history['learning_rate'] = self.current_logs.get('learning_rate')
                out = list(self.dataset.data.outputs.keys())[0]
                classes_names = self.dataset.data.outputs.get(out).classes_names
                for key in self.log_history['output']["loss"].keys():
                    for data_type in ['train', 'val']:
                        self.log_history['output']["loss"][key][data_type].append(
                            round_loss_metric(self.current_logs.get('loss').get(key).get(data_type)))
                for key in self.log_history['output']["metrics"].keys():
                    self.log_history['output']["metrics"][key]["val"].append(
                        round_loss_metric(self.current_logs.get('metrics').get(key).get('val'))
                    )
                for name in classes_names:
                    self.log_history['output']["class_loss"]['prob_loss'][name]["val"].append(
                        round_loss_metric(self.current_logs.get('class_loss').get("prob_loss").get(name).get("val"))
                    )
                    self.log_history['output']["class_loss"]['prob_loss'][name]["train"].append(
                        round_loss_metric(self.current_logs.get('class_loss').get("prob_loss").get(name).get("train"))
                    )
                    self.log_history['output']["class_metrics"]['mAP50'][name]["val"].append(
                        round_loss_metric(self.current_logs.get('class_metrics').get("mAP50").get(name).get("val"))
                    )
                for loss_name in self.log_history['output']["loss"].keys():
                    # fill loss progress state
                    self.log_history['output']['progress_state']['loss'][loss_name]['mean_log_history'].append(
                        round_loss_metric(
                            self._get_mean_log(self.log_history.get('output').get('loss').get(loss_name).get('val')))
                    )
                    # get progress state data
                    loss_underfitting = self._evaluate_underfitting(
                        loss_name,
                        self.log_history['output']['loss'][loss_name]['train'][-1],
                        self.log_history['output']['loss'][loss_name]['val'][-1],
                        metric_type='loss'
                    )
                    loss_overfitting = self._evaluate_overfitting(
                        loss_name,
                        self.log_history['output']['progress_state']['loss'][loss_name]['mean_log_history'],
                        metric_type='loss'
                    )
                    if loss_underfitting or loss_overfitting:
                        normal_state = False
                    else:
                        normal_state = True
                    self.log_history['output']['progress_state']['loss'][loss_name]['underfitting'].append(
                        loss_underfitting)
                    self.log_history['output']['progress_state']['loss'][loss_name]['overfitting'].append(
                        loss_overfitting)
                    self.log_history['output']['progress_state']['loss'][loss_name]['normal_state'].append(
                        normal_state)
                for metric_name in self.log_history.get('output').get('metrics').keys():
                    self.log_history['output']['progress_state']['metrics'][metric_name]['mean_log_history'].append(
                        round_loss_metric(self._get_mean_log(self.log_history['output']['metrics'][metric_name]["val"]))
                    )
                    metric_overfitting = self._evaluate_overfitting(
                        metric_name,
                        self.log_history['output']['progress_state']['metrics'][metric_name]['mean_log_history'],
                        metric_type='metric'
                    )
                    if metric_overfitting:
                        normal_state = False
                    else:
                        normal_state = True
                    self.log_history['output']['progress_state']['metrics'][metric_name]['overfitting'].append(
                        metric_overfitting)
                    self.log_history['output']['progress_state']['metrics'][metric_name]['normal_state'].append(
                        normal_state)
        except Exception as e:
            print_error('FitCallback', method_name, e)

    @staticmethod
    def _get_mean_log(logs):
        method_name = '_get_mean_log'
        try:
            copy_logs = copy.deepcopy(logs)
            while None in copy_logs:
                copy_logs.pop(copy_logs.index(None))
            if len(copy_logs) < 5:
                return float(np.mean(copy_logs))
            else:
                return float(np.mean(copy_logs[-5:]))
        except Exception as e:
            print_error('FitCallback', method_name, e)
            return 0.

    @staticmethod
    def _evaluate_overfitting(metric_name: str, mean_log: list, metric_type: str):
        method_name = '_evaluate_overfitting'
        try:
            mode = loss_metric_config.get(metric_type).get(metric_name).get("mode")
            overfitting = False
            if mode == 'min':
                if min(mean_log) and mean_log[-1] and mean_log[-1] > min(mean_log) and \
                        (mean_log[-1] - min(mean_log)) * 100 / min(mean_log) > 2:
                    overfitting = True
            if mode == 'max':
                if max(mean_log) and mean_log[-1] and mean_log[-1] < max(mean_log) and \
                        (max(mean_log) - mean_log[-1]) * 100 / max(mean_log) > 2:
                    overfitting = True
            return overfitting
        except Exception as e:
            print_error('FitCallback', method_name, e)

    @staticmethod
    def _evaluate_underfitting(metric_name: str, train_log: float, val_log: float, metric_type: str):
        method_name = '_evaluate_underfitting'
        try:
            mode = loss_metric_config.get(metric_type).get(metric_name).get("mode")
            underfitting = False
            if mode == 'min' and train_log and val_log:
                if val_log < 1 and train_log < 1 and (val_log - train_log) > 0.05:
                    underfitting = True
                if (val_log >= 1 or train_log >= 1) and (val_log - train_log) / train_log * 100 > 5:
                    underfitting = True
            if mode == 'max' and train_log and val_log and (train_log - val_log) / train_log * 100 > 3:
                underfitting = True
            return underfitting
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _get_checkpoint_mode(self):
        method_name = '_get_checkpoint_mode'
        try:
            print(method_name, self.checkpoint_config)
            if self.checkpoint_config.type == CheckpointTypeChoice.Loss:
                return 'min'
            elif self.checkpoint_config.type == CheckpointTypeChoice.Metrics:
                metric_name = self.checkpoint_config.metric_name
                return loss_metric_config.get("metric").get(metric_name).get("mode")
            else:
                print('\nClass FitCallback method _get_checkpoint_mode: No checkpoint types are found\n')
                return None
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _save_logs(self):
        method_name = '_save_logs'
        try:
            print(method_name)
            logs = {
                "fit_log": self.log_history,
                "interactive_log": interactive.log_history,
                "progress_table": interactive.progress_table,
                "addtrain_epochs": interactive.addtrain_epochs
            }
            self.training_detail.logs = logs
            interactive_path = os.path.join(self.training_detail.model_path, "interactive.history")
            if not os.path.exists(interactive_path):
                os.mkdir(interactive_path)
            with open(os.path.join(self.training_detail.model_path, "log.history"), "w", encoding="utf-8") as history:
                json.dump(self.log_history, history)
            with open(os.path.join(interactive_path, "log.int"), "w", encoding="utf-8") as log:
                json.dump(interactive.log_history, log)
            with open(os.path.join(interactive_path, "table.int"), "w", encoding="utf-8") as table:
                json.dump(interactive.progress_table, table)
            with open(os.path.join(interactive_path, "addtraining.int"), "w", encoding="utf-8") as addtraining:
                json.dump({"addtrain_epochs": interactive.addtrain_epochs}, addtraining)
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _load_logs(self):
        method_name = '_load_logs'
        try:
            print(method_name)
            if self.training_detail.state.status == "addtrain":
                if self.training_detail.logs:
                    logs = self.training_detail.logs.get("fit_log")
                    interactive.log_history = self.training_detail.logs.get("interactive_log")
                    interactive.progress_table = self.training_detail.logs.get("progress_table")
                    interactive.addtrain_epochs = self.training_detail.logs.get("addtrain_epochs")
                else:
                    interactive_path = os.path.join(self.training_detail.model_path, "interactive.history")
                    with open(os.path.join(self.training_detail.model_path, "log.history"), "r",
                              encoding="utf-8") as history:
                        logs = json.load(history)
                    with open(os.path.join(interactive_path, "log.int"), "r", encoding="utf-8") as int_log:
                        interactive.log_history = json.load(int_log)
                    with open(os.path.join(interactive_path, "table.int"), "r", encoding="utf-8") as table_int:
                        interactive.progress_table = json.load(table_int)
                    with open(os.path.join(interactive_path, "addtraining.int"), "r",
                              encoding="utf-8") as addtraining_int:
                        interactive.addtrain_epochs = json.load(addtraining_int)["addtrain_epochs"]
                self.last_epoch = max(logs.get('epochs'))
                self.retrain_epochs = self.last_epoch + self.training_detail.base.epochs
                self.still_epochs = self.retrain_epochs - self.last_epoch
                return logs
            else:
                return self._prepare_log_history_template(self.dataset, self.training_detail)
        except Exception as e:
            print_error('FitCallback', method_name, e)

    @staticmethod
    def _logs_predict_extract(logs, prefix):
        pred_on_batch = []
        for key in logs.keys():
            if key.startswith(prefix):
                pred_on_batch.append(logs[key])
        return pred_on_batch

    def _best_epoch_monitoring(self):
        method_name = '_best_epoch_monitoring'
        try:
            print(method_name)
            # print('\n self.checkpoint_config', self.checkpoint_config)
            # print(self.log_history.get(f"{self.checkpoint_config.get('layer')}").keys())
            # print(self.log_history.get(f"{self.checkpoint_config.get('layer')}").get(
            #     self.checkpoint_config.get("type").lower()).keys())
            # print(self.log_history.get(f"{self.checkpoint_config.get('layer')}").get(
            #     self.checkpoint_config.get("type").lower()).get(self.checkpoint_config.get("metric_name")).keys())
            output = "output" if self.is_yolo else f"{self.checkpoint_config.layer}"
            checkpoint_list = self.log_history.get(output).get(self.checkpoint_config.type.name.lower()).get(
                self.checkpoint_config.metric_name.name).get(self.checkpoint_config.indicator.name.lower())
            if self.checkpoint_mode == 'min' and checkpoint_list[-1] == min(checkpoint_list):
                return True
            elif self.checkpoint_mode == "max" and checkpoint_list[-1] == max(checkpoint_list):
                return True
            else:
                return False
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _set_result_data(self, param: dict) -> None:
        method_name = '_set_result_data'
        try:
            for key in param.keys():
                if key in self.result.keys():
                    self.result[key] = param[key]
                elif key == "timings":
                    self.result["train_usage"]["timings"]["estimated_time"] = param[key][1] + param[key][2]
                    self.result["train_usage"]["timings"]["elapsed_time"] = param[key][1]
                    self.result["train_usage"]["timings"]["still_time"] = param[key][2]
                    self.result["train_usage"]["timings"]["avg_epoch_time"] = int(
                        self._sum_epoch_time / self.last_epoch)
                    self.result["train_usage"]["timings"]["elapsed_epoch_time"] = param[key][3]
                    self.result["train_usage"]["timings"]["still_epoch_time"] = param[key][4]
                    self.result["train_usage"]["timings"]["epoch"] = param[key][5]
                    self.result["train_usage"]["timings"]["batch"] = param[key][6]
            self.result["train_usage"]["hard_usage"] = self.usage_info.get_usage()
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _get_result_data(self):
        return self.result

    def _get_train_status(self) -> str:
        return self.training_detail.state.status

    def _get_predict(self, deploy_model=None):
        method_name = '_get_predict'
        try:
            print(method_name)
            current_model = deploy_model if deploy_model else self.model
            if self.is_yolo:
                current_predict = [np.concatenate(elem, axis=0) for elem in zip(*self.samples_val)]
                current_target = [np.concatenate(elem, axis=0) for elem in zip(*self.samples_target_val)]
            else:
                # TODO: настроить вывод массивов их обучения, выводить словарь
                #  {'train_true': train_true, 'train_pred': train_pred, 'val_true': val_true, 'val_pred': val_pred}
                if self.dataset.data.use_generator:
                    current_predict = current_model.predict(
                        self.dataset.dataset.get('val').batch(1), batch_size=1)
                else:
                    current_predict = current_model.predict(self.dataset.X.get('val'), batch_size=self.batch_size)
                # current_predict = None
                current_target = None
            return current_predict, current_target
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _deploy_predict(self, presets_predict):
        method_name = '_deploy_predict'
        try:
            print(method_name)
            result = CreateArray().postprocess_results(
                array=presets_predict, options=self.dataset, save_path=str(self.training_detail.model_path),
                dataset_path=str(self.dataset.data.path))
            deploy_presets = []
            if result:
                deploy_presets = list(result.values())[0]
            return deploy_presets
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _create_form_data_for_dataframe_deploy(self):
        method_name = '_create_form_data_for_dataframe_deploy'
        try:
            print(method_name)
            form_data = []
            with open(os.path.join(self.dataset.data.path, "config.json"), "r", encoding="utf-8") as dataset_conf:
                dataset_info = json.load(dataset_conf).get("columns", {})
            for inputs, input_data in dataset_info.items():
                if int(inputs) not in list(self.dataset.data.outputs.keys()):
                    for column, column_data in input_data.items():
                        label = column
                        available = column_data.get("classes_names") if column_data.get("classes_names") else None
                        widget = "select" if available else "input"
                        input_type = "text"
                        if widget == "select":
                            table_column_data = {
                                "label": label,
                                "widget": widget,
                                "available": available
                            }
                        else:
                            table_column_data = {
                                "label": label,
                                "widget": widget,
                                "type": input_type
                            }
                        form_data.append(table_column_data)
            with open(os.path.join(self.training_detail.deploy_path, "form.json"), "w", encoding="utf-8") as form_file:
                json.dump(form_data, form_file, ensure_ascii=False)
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _create_cascade(self, **data):
        method_name = '_create_cascade'
        try:
            print(method_name)
            if self.dataset.data.alias not in ["imdb", "boston_housing", "reuters"]:
                if "Dataframe" in self.deploy_type:
                    self._create_form_data_for_dataframe_deploy()
                if self.is_yolo:
                    func_name = "object_detection"
                else:
                    func_name = decamelize(self.deploy_type)
                config = CascadeCreator()
                config.create_config(
                    deploy_path=self.training_detail.deploy_path,
                    model_path=self.training_detail.model_path,
                    func_name=func_name
                )
                config.copy_package(
                    deploy_path=self.training_detail.deploy_path,
                    model_path=self.training_detail.model_path
                )
                config.copy_script(
                    deploy_path=self.training_detail.deploy_path,
                    function_name=func_name
                )
                if self.deploy_type == ArchitectureChoice.TextSegmentation:
                    with open(os.path.join(self.training_detail.deploy_path, "format.txt"),
                              "w", encoding="utf-8") as format_file:
                        format_file.write(str(data.get("tags_map", "")))
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _prepare_deploy(self, model):
        method_name = '_prepare_deploy'
        try:
            print(method_name, self.training_detail.deploy_path)
            weight = None
            cascade_data = {"deploy_path": self.training_detail.deploy_path}
            for i in os.listdir(self.training_detail.model_path):
                if i[-3:] == '.h5' and 'best' in i:
                    weight = i
            if weight:
                model.load_weights(os.path.join(self.training_detail.model_path, weight))
            deploy_predict, y_true = self._get_predict(deploy_model=model)
            deploy_presets_data = self._deploy_predict(deploy_predict)
            out_deploy_presets_data = {"data": deploy_presets_data}
            if self.deploy_type == ArchitectureChoice.TextSegmentation:
                cascade_data.update({"tags_map": deploy_presets_data.get("color_map")})
                out_deploy_presets_data = {
                    "data": deploy_presets_data.get("data", {}),
                    "color_map": deploy_presets_data.get("color_map")
                }
            elif "Dataframe" in self.deploy_type:
                columns = []
                predict_column = ""
                for inp, input_columns in self.dataset.data.columns.items():
                    for column_name in input_columns.keys():
                        columns.append(column_name[len(str(inp)) + 1:])
                        if input_columns[column_name].__class__ == DatasetOutputsData:
                            predict_column = column_name[len(str(inp)) + 1:]
                if self.deploy_type == ArchitectureChoice.DataframeRegression:
                    tmp_data = list(zip(deploy_presets_data.get("preset"), deploy_presets_data.get("label")))
                    tmp_deploy = [{"preset": elem[0], "label": elem[1]} for elem in tmp_data]
                    out_deploy_presets_data = {"data": tmp_deploy}
                out_deploy_presets_data["columns"] = columns
                out_deploy_presets_data[
                    "predict_column"] = predict_column if predict_column else "Предсказанные значения"

            out_deploy_data = dict([
                ("path", Path(self.training_detail.deploy_path)),
                ("path_model", Path(self.training_detail.model_path)),
                ("type", self.deploy_type),
                ("data", out_deploy_presets_data)
            ])
            print(self.deploy_type, type(self.deploy_type))
            self.training_detail.deploy = DeployData(**out_deploy_data)
            self._create_cascade(**cascade_data)
        except Exception as e:
            print_error('FitCallback', method_name, e)

    @staticmethod
    def _estimate_step(current, start, now):
        method_name = '_estimate_step'
        try:
            # print(method_name)
            if current:
                _time_per_unit = (now - start) / current
            else:
                _time_per_unit = (now - start)
            return _time_per_unit
        except Exception as e:
            print_error('FitCallback', method_name, e)

    @staticmethod
    def eta_format(eta):
        method_name = 'eta_format'
        try:
            print(method_name)
            if eta > 3600:
                eta_format = '%d ч %02d мин %02d сек' % (eta // 3600,
                                                         (eta % 3600) // 60, eta % 60)
            elif eta > 60:
                eta_format = '%d мин %02d сек' % (eta // 60, eta % 60)
            else:
                eta_format = '%d сек' % eta
            return ' %s' % eta_format
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def update_progress(self, target, current, start_time, finalize=False, stop_current=0, stop_flag=False):
        method_name = 'update_progress'
        try:
            # print(method_name)
            """
            Updates the progress bar.
            """
            if finalize:
                _now_time = time.time()
                eta = _now_time - start_time
            else:
                _now_time = time.time()
                if stop_flag:
                    time_per_unit = self._estimate_step(stop_current, start_time, _now_time)
                else:
                    time_per_unit = self._estimate_step(current, start_time, _now_time)
                eta = time_per_unit * (target - current)
            return int(eta)
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def on_train_begin(self):
        method_name = 'on_train_begin'
        try:
            print(method_name, self.dataset.dataframe.keys())
            status = self._get_train_status()
            self._start_time = time.time()
            if status != "addtrain":
                self.batch = 1
            if not self.dataset.data.use_generator:
                if len(list(self.dataset.X['train'].values())[0]) % self.batch_size:
                    self.num_batches = len(list(self.dataset.X['train'].values())[0]) // self.batch_size + 1
                else:
                    self.num_batches = len(list(self.dataset.X['train'].values())[0]) // self.batch_size
            else:
                if len(self.dataset.dataframe['train']) % self.batch_size:
                    self.num_batches = len(self.dataset.dataframe['train']) // self.batch_size + 1
                else:
                    self.num_batches = len(self.dataset.dataframe['train']) // self.batch_size
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def on_epoch_begin(self):
        print('on_epoch_begin')
        self.last_epoch += 1
        self._time_first_step = time.time()

    def on_train_batch_end(self, batch, arrays=None, train_data_idxs=None):
        method_name = 'on_train_batch_end'
        try:
            if self._get_train_status() == "stopped":
                print('_get_train_status() == "stopped"')
                self.stop_training = True
                progress.pool(
                    self.progress_name,
                    percent=self.last_epoch / (
                        self.retrain_epochs if self._get_train_status() == "addtrain" else self.epochs) * 100,
                    message="Обучение остановлено пользователем, ожидайте остановку...",
                    finished=False,
                )
            else:
                msg_batch = {"current": batch, "total": self.num_batches}
                msg_epoch = {"current": self.last_epoch,
                             "total": self.retrain_epochs if self._get_train_status() == "addtrain"
                             else self.epochs}
                still_epoch_time = self.update_progress(self.num_batches, batch, self._time_first_step)
                elapsed_epoch_time = time.time() - self._time_first_step
                elapsed_time = time.time() - self._start_time
                estimated_time = self.update_progress(
                    self.num_batches * self.still_epochs, self.batch, self._start_time, finalize=True)

                still_time = self.update_progress(
                    self.num_batches * self.still_epochs, self.batch, self._start_time)
                self.batch = batch
                if interactive.urgent_predict:
                    # print('\ninteractive.urgent_predict\n')
                    # if self.is_yolo:
                    # self.samples_train.append(self._logs_predict_extract(logs, prefix='pred'))
                    # self.samples_target_train.append(self._logs_predict_extract(logs, prefix='target'))

                    # y_pred, y_true = self._get_predict()
                    train_batch_data = interactive.update_state(arrays=arrays, train_idx=train_data_idxs)
                else:
                    train_batch_data = interactive.update_state(arrays=None, train_idx=None)
                    # print("train_batch_data", train_batch_data)
                if train_batch_data:
                    result_data = {
                        'timings': [estimated_time, elapsed_time, still_time,
                                    elapsed_epoch_time, still_epoch_time, msg_epoch, msg_batch],
                        'train_data': train_batch_data
                    }
                else:
                    result_data = {'timings': [estimated_time, elapsed_time, still_time,
                                               elapsed_epoch_time, still_epoch_time, msg_epoch, msg_batch]}
                # print('batch', self.batch, result_data)
                self._set_result_data(result_data)
                self.training_detail.result = self._get_result_data()
                progress.pool(
                    self.progress_name,
                    percent=self.last_epoch / (
                        self.retrain_epochs if self._get_train_status() == "addtrain" else self.epochs
                    ) * 100,
                    message=f"Обучение. Эпоха {self.last_epoch} из "
                            f"{self.retrain_epochs if self._get_train_status() in ['addtrain', 'stopped'] else self.epochs}",
                    finished=False,
                )
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def on_epoch_end(self, epoch, arrays=None, logs=None, train_data_idxs=None):
        method_name = 'on_epoch_end'
        try:
            print(method_name, epoch)
            # self.last_epoch = epoch
            # total_epochs = self.retrain_epochs if self._get_train_status() in ['addtrain', 'stopped'] else self.epochs
            if self.is_yolo:
                self.current_logs = logs
                # if self.last_epoch < total_epochs and not self.stop_training:
                #     self.samples_train = []
                #     self.samples_val = []
                #     self.samples_target_train = []
                #     self.samples_target_val = []
            else:
                self.current_basic_logs(
                    epoch=epoch, arrays=arrays, train_idx=train_data_idxs
                )
            print('\nFitCallback _update_log_history: start')
            t = time.time()
            self._update_log_history()
            print('\nFitCallback _update_log_history', round(time.time() - t, 3))
            if epoch == 1:
                interactive.log_history = self.log_history
            current_epoch_time = time.time() - self._time_first_step
            self._sum_epoch_time += current_epoch_time
            print('\nFitCallback interactive.update_state: start')
            t = time.time()
            train_epoch_data = interactive.update_state(
                fit_logs=self.log_history,
                arrays=arrays,
                current_epoch_time=current_epoch_time,
                on_epoch_end_flag=True,
                train_idx=train_data_idxs
            )
            print('\nFitCallback interactive.update_state', round(time.time() - t, 3))
            # print(method_name, 'train_epoch_data', train_epoch_data)
            self._set_result_data({'train_data': train_epoch_data})
            progress.pool(
                self.progress_name,
                percent=self.last_epoch / (
                    self.retrain_epochs
                    if self._get_train_status() == "addtrain" or self._get_train_status() == "stopped" else self.epochs
                ) * 100,
                message=f"Обучение. Эпоха {self.last_epoch} из "
                        f"{self.retrain_epochs if self._get_train_status() in ['addtrain', 'stopped'] else self.epochs}",
                finished=False,
            )
            # self.last_epoch += 1
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def save_best_weights(self):
        method_name = 'save_best_weights'
        try:
            print(method_name)
            if self.last_epoch > 1:
                if self._best_epoch_monitoring():
                    if not os.path.exists(self.model_path):
                        os.mkdir(self.model_path)
                    if not os.path.exists(os.path.join(self.training_detail.model_path, "deploy_presets")):
                        os.mkdir(os.path.join(self.training_detail.model_path, "deploy_presets"))
                    file_path_best: str = os.path.join(
                        self.training_detail.model_path, f"best_weights_{self.metric_checkpoint}.h5"
                    )
                    return file_path_best
        except Exception as e:
            print_error('FitCallback', method_name, e)
            print('\nself.model.save_weights failed', e)

    def on_train_end(self, model):
        method_name = 'on_train_end'
        try:
            print(method_name, self.last_epoch)
            interactive.addtrain_epochs.append(self.last_epoch)
            self._save_logs()

            if self.last_epoch > 1:
                file_path_last: str = os.path.join(
                    self.training_detail.model_path, f"last_weights_{self.metric_checkpoint}.h5"
                )
                model.save_weights(file_path_last)
            if not os.path.exists(os.path.join(self.training_detail.model_path, "deploy_presets")):
                os.mkdir(os.path.join(self.training_detail.model_path, "deploy_presets"))
            # self._prepare_deploy(model)

            time_end = self.update_progress(
                self.num_batches * self.epochs, self.batch, self._start_time, finalize=True)
            self._sum_time += time_end
            total_epochs = self.retrain_epochs \
                if self._get_train_status() in ['addtrain', 'trained'] else self.epochs
            if self.stop_training:
                progress.pool(
                    self.progress_name,
                    message=f"Обучение остановлено. Эпоха {self.last_epoch} из {total_epochs}. Модель сохранена.",
                    data=self._get_result_data(),
                    finished=True,
                )
            else:
                percent = self.last_epoch / (
                    self.retrain_epochs
                    if self._get_train_status() == "addtrain" or self._get_train_status() == "stopped"
                    else self.epochs
                ) * 100
                print('percent', percent, self.progress_name)

                self.training_detail.state.set("trained")
                self.training_detail.result = self._get_result_data()
                progress.pool(
                    self.progress_name,
                    percent=percent,
                    message=f"Обучение завершено. Эпоха {self.last_epoch} из {total_epochs}",
                    finished=True,
                )
        except Exception as e:
            print_error('FitCallback', method_name, e)
