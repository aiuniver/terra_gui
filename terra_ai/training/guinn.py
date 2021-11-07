import gc
import importlib
import json
import math
import re
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
from tensorflow.keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical

from terra_ai import progress
from terra_ai.callbacks.classification_callbacks import BaseClassificationCallback
from terra_ai.callbacks.interactive_callback import InteractiveCallback
from terra_ai.callbacks.utils import print_error, loss_metric_config, BASIC_ARCHITECTURE, CLASS_ARCHITECTURE, \
    YOLO_ARCHITECTURE, round_loss_metric, class_metric_list, CLASSIFICATION_ARCHITECTURE
from terra_ai.data.datasets.dataset import DatasetData, DatasetOutputsData
from terra_ai.data.datasets.extra import LayerOutputTypeChoice, LayerInputTypeChoice, LayerEncodingChoice
from terra_ai.data.deploy.tasks import DeployData
from terra_ai.data.modeling.model import ModelDetailsData, ModelData
from terra_ai.data.presets.training import Metric
from terra_ai.data.training.extra import CheckpointIndicatorChoice, CheckpointTypeChoice, MetricChoice, \
    ArchitectureChoice
from terra_ai.data.training.train import TrainData, TrainingDetailsData, StateData
from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.deploy.create_deploy_package import CascadeCreator
from terra_ai.exceptions.deploy import MethodNotImplementedException
from terra_ai.modeling.validator import ModelValidator
# from terra_ai.training.customcallback import InteractiveCallback
from terra_ai.training.customlosses import DiceCoef, UnscaledMAE, BalancedRecall, BalancedDiceCoef, \
    BalancedPrecision, BalancedFScore, FScore
from terra_ai.training.yolo_utils import create_yolo, CustomModelYolo, compute_loss, get_mAP
from terra_ai.exceptions import training as exceptions, terra_exception

__version__ = 0.02

from terra_ai.utils import camelize, decamelize

interactive = InteractiveCallback()


class GUINN:

    def __init__(self) -> None:
        self.name = "GUINN"
        self.callbacks = []
        self.params: dict = {}
        self.nn_name: str = ''
        self.dataset: Optional[PrepareDataset] = None
        self.deploy_type = None
        self.model: Optional[Model] = None
        self.training_path: str = ""
        # self.optimizer = None
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

    @staticmethod
    def _check_metrics(metrics: list, options: DatasetOutputsData, num_classes: int = 2, ) -> list:
        method_name = '_check_metrics'
        try:
            output = []
            for metric in metrics:
                if metric == MetricChoice.MeanIoU.value:
                    output.append(getattr(importlib.import_module("tensorflow.keras.metrics"), metric)(num_classes))
                elif metric == MetricChoice.DiceCoef:
                    output.append(DiceCoef())
                # elif metric == MetricChoice.RecallPercent:
                #     output.append(RecallPercent())
                elif metric == MetricChoice.BalancedPrecision:
                    output.append(BalancedPrecision())
                elif metric == MetricChoice.BalancedFScore:
                    output.append(BalancedFScore())
                elif metric == MetricChoice.FScore:
                    output.append(FScore())
                elif metric == MetricChoice.BalancedRecall:
                    output.append(BalancedRecall())
                elif metric == MetricChoice.BalancedDiceCoef:
                    output.append(BalancedDiceCoef(encoding=options.encoding.value))
                elif metric == MetricChoice.UnscaledMAE:
                    output.append(UnscaledMAE())
                elif metric == MetricChoice.mAP50 or metric == MetricChoice.mAP95:
                    pass
                else:
                    output.append(getattr(importlib.import_module("tensorflow.keras.metrics"), metric)())
            return output
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def _set_training_params(self, dataset: DatasetData, train_params: TrainingDetailsData,
                             training_path: str, dataset_path: str) -> None:
        method_name = '_set_training_params'
        try:
            params = train_params.base
            self.dataset = self._prepare_dataset(dataset, dataset_path, training_path, state=train_params.state.status)
            if not self.dataset.data.architecture or self.dataset.data.architecture == ArchitectureChoice.Basic:
                self.deploy_type = self._set_deploy_type(self.dataset)
            else:
                self.deploy_type = self.dataset.data.architecture
            self.training_path = training_path
            self.nn_name = str(training_path).split()[1] if not str(training_path).split()[1].startswith("__") \
                else "model"
            if self.dataset.data.use_generator:
                train_size = len(self.dataset.dataframe.get("train"))
            else:
                train_size = len(self.dataset.dataset.get('train'))
            if params.batch > train_size:
                if train_params.state.status == "addtrain":
                    train_params.state.set("stopped")
                else:
                    train_params.state.set("no_train")
                raise exceptions.TooBigBatchSize(params.batch, train_size)

            if train_params.state.status == "addtrain":
                if self.callbacks[0].last_epoch - 1 >= self.sum_epoch:
                    self.sum_epoch += params.epochs
                if (self.callbacks[0].last_epoch - 1) < self.sum_epoch:
                    self.epochs = self.sum_epoch - self.callbacks[0].last_epoch + 1
                else:
                    self.epochs = params.epochs
            else:
                self.epochs = params.epochs
            self.batch_size = params.batch
            # self.set_optimizer(params)

            # for output_layer in params.architecture.outputs_dict:
            #     self.metrics.update({
            #         str(output_layer["id"]):
            #             self._check_metrics(
            #                 metrics=output_layer.get("metrics", []),
            #                 num_classes=output_layer.get("classes_quantity"),
            #                 options=self.dataset.data.outputs.get(output_layer["id"])
            #             )
            #     })
            #     self.loss.update({str(output_layer["id"]): output_layer["loss"]})

            interactive.set_attributes(dataset=self.dataset, metrics=self.metrics, losses=self.loss,
                                       dataset_path=dataset_path, training_path=training_path,
                                       initial_config=train_params.interactive)
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def _set_callbacks(self, dataset: PrepareDataset,
                       batch_size: int, epochs: int, dataset_path: Path,
                       checkpoint: dict, save_model_path: Path,
                       state: StateData, deploy: DeployData, initial_model=None) -> None:
        method_name = '_set_callbacks'
        try:
            progress.pool(self.progress_name, finished=False, data={'status': 'Добавление колбэков...'})
            retrain_epochs = self.sum_epoch if state.status == "addtrain" else self.epochs

            callback = FitCallback(dataset=dataset, params=self.params,
                                   # batch_size=batch_size, epochs=epochs,
                                   retrain_epochs=retrain_epochs,
                                   training_path=save_model_path, model_name=self.nn_name,
                                   dataset_path=dataset_path, deploy_type=self.deploy_type,
                                   initialed_model=initial_model,
                                   state=state, deploy=deploy
                                   )
            self.callbacks = [callback]
            progress.pool(self.progress_name, finished=False, data={'status': 'Добавление колбэков выполнено'})
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    @staticmethod
    def _set_deploy_type(dataset: PrepareDataset) -> str:
        method_name = '_set_deploy_type'
        try:
            data = dataset.data
            inp_tasks = []
            out_tasks = []
            for key, val in data.inputs.items():
                if val.task == LayerInputTypeChoice.Dataframe:
                    tmp = []
                    for value in data.columns[key].values():
                        tmp.append(value.task)
                    unique_vals = list(set(tmp))
                    if len(unique_vals) == 1 and unique_vals[0] in LayerInputTypeChoice.__dict__.keys() \
                            and unique_vals[0] in [LayerInputTypeChoice.Image, LayerInputTypeChoice.Text,
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
                raise MethodNotImplementedException(
                    __method=inp_task_name + out_task_name, __class="ArchitectureChoice")
            return deploy_type
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    @staticmethod
    def _prepare_dataset(dataset: DatasetData, dataset_path: str, model_path: str, state: str) -> PrepareDataset:
        method_name = '_prepare_dataset'
        try:
            prepared_dataset = PrepareDataset(data=dataset, datasets_path=dataset_path)
            prepared_dataset.prepare_dataset()
            if state != "addtrain":
                prepared_dataset.deploy_export(os.path.join(model_path, "dataset"))
            return prepared_dataset
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def _set_model(self, model: ModelDetailsData) -> ModelData:
        method_name = '_set_model'
        try:
            if interactive.get_states().get("status") == "training":
                validator = ModelValidator(model)
                train_model = validator.get_keras_model()
            else:
                train_model = load_model(os.path.join(self.training_path, self.nn_name, f"{self.nn_name}.trm"),
                                         compile=False)
                weight = None
                for i in os.listdir(os.path.join(self.training_path, self.nn_name)):
                    if i[-3:] == '.h5' and 'last' in i:
                        weight = i
                if weight:
                    train_model.load_weights(os.path.join(self.training_path, self.nn_name, weight))
            return train_model
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    @staticmethod
    def _save_params_for_deploy(training_path: Path, params: TrainData):
        method_name = '_save_params_for_deploy'
        try:
            if not os.path.exists(training_path):
                os.mkdir(training_path)
            with open(os.path.join(training_path, "config.train"), "w", encoding="utf-8") as train_config:
                json.dump(params.native(), train_config)
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
            model_name = f"{self.nn_name}.trm"
            file_path_model: str = os.path.join(
                self.training_path, f"{model_name}"
            )
            self.model.save(file_path_model)
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def _kill_last_training(self, state):
        method_name = '_kill_last_training'
        try:
            for one_thread in threading.enumerate():
                if one_thread.getName() == "current_train":
                    current_status = state.status
                    state.set("stopped")
                    progress.pool(self.progress_name,
                                  message="Найдено незавершенное обучение. Идет очистка. Подождите.")
                    one_thread.join()
                    state.set(current_status)
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    @staticmethod
    def _get_val_batch_size(batch_size, len_val):
        method_name = '_get_val_batch_size'
        try:
            def issimple(a):
                lst = []
                for n in range(2, a):
                    if a % n == 0:
                        if not issimple(n):
                            lst.append(n)
                return lst

            min_step = 0
            for i in range(3):
                r = issimple(len_val - i)
                if len(r):
                    try:
                        min_step = min(r)
                        if len_val // min_step > batch_size:
                            for k in r:
                                if len_val // k <= batch_size:
                                    min_step = k
                                    break
                                else:
                                    min_step = len_val
                        break
                    except ValueError:
                        pass
                else:
                    min_step = len_val
            val_batch_size = len_val // min_step
            return val_batch_size
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def terra_fit(self, dataset: DatasetData, gui_model: ModelDetailsData, training: TrainingDetailsData,
                  training_path: Path = "", dataset_path: Path = "",  # training_params: TrainData = None,
                  # initial_config: InteractiveData = None
                  ) -> dict:
        method_name = 'terra_fit'
        try:
            """
            This method created for using wth externally compiled models
    
            Args:
                dataset: DatasetData
                gui_model: Keras model for fit - ModelDetailsData
                training: TrainingDetailsData
                training_path:
                dataset_path: str
    
            Return:
                dict
            """
            self._kill_last_training(state=training.state)
            progress.pool.reset(self.progress_name)

            if training.state.status != "addtrain":
                self._save_params_for_deploy(training_path=training_path, params=training.base)
            self.nn_cleaner(retrain=True if training.state.status == "training" else False)

            self._set_training_params(dataset=dataset, train_params=training,
                                      dataset_path=dataset_path, training_path=training_path)
            self.model = self._set_model(model=gui_model)

            if training.state.status == "training":
                self.save_model()

            self.base_model_fit(params=training, dataset=self.dataset, dataset_data=dataset,
                                verbose=0, save_model_path=training_path, dataset_path=dataset_path)
            return {"dataset": self.dataset, "metrics": self.metrics, "losses": self.loss}
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def nn_cleaner(self, retrain: bool = False) -> None:
        method_name = 'nn_cleaner'
        try:
            keras.backend.clear_session()
            self.dataset = None
            self.deploy_type = None
            self.model = None
            if retrain:
                self.sum_epoch = 0
                self.loss = {}
                self.metrics = {}
                self.callbacks = []
            gc.collect()
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    def get_nn(self):
        self.nn_cleaner(retrain=True)
        return self

    @progress.threading
    def base_model_fit(self, params: TrainingDetailsData, dataset: PrepareDataset, dataset_path: Path,
                       dataset_data: DatasetData, save_model_path: Path, verbose=0) -> None:
        method_name = 'base_model_fit'
        try:

            yolo_arch = True if self.deploy_type in YOLO_ARCHITECTURE else False
            model_yolo = None

            threading.enumerate()[-1].setName("current_train")
            progress.pool(self.progress_name, finished=False, data={'status': 'Компиляция модели ...'})
            if yolo_arch:
                warmup_epoch = params.base.architecture.parameters.yolo.train_warmup_epochs
                lr_init = params.base.architecture.parameters.yolo.train_lr_init
                lr_end = params.base.architecture.parameters.yolo.train_lr_end
                iou_thresh = params.base.architecture.parameters.yolo.yolo_iou_loss_thresh

                yolo = create_yolo(self.model, input_size=416, channels=3, training=True,
                                   classes=self.dataset.data.outputs.get(2).classes_names,
                                   version=self.dataset.instructions.get(2).get('2_object_detection').get('yolo'))
                model_yolo = CustomModelYolo(yolo, self.dataset, self.dataset.data.outputs.get(2).classes_names,
                                             self.epochs, self.batch_size, warmup_epoch=warmup_epoch,
                                             lr_init=lr_init, lr_end=lr_end, iou_thresh=iou_thresh)
                model_yolo.compile(optimizer=self.set_optimizer(self.params),
                                   loss=compute_loss)
            # else:
            #     self.model.compile(loss=self.loss,
            #                        optimizer=self.optimizer,
            #                        metrics=self.metrics
            #                        )
            progress.pool(self.progress_name, finished=False, data={'status': 'Компиляция модели выполнена'})
            self._set_callbacks(dataset=dataset, batch_size=params.base.batch,
                                epochs=params.base.epochs, save_model_path=save_model_path, dataset_path=dataset_path,
                                checkpoint=params.base.architecture.parameters.checkpoint.native(),
                                initial_model=self.model if yolo_arch else None, state=params.state,
                                deploy=params.deploy)
            progress.pool(self.progress_name, finished=False, data={'status': 'Начало обучения ...'})
            if self.dataset.data.use_generator:
                print('use generator')
                critical_val_size = len(self.dataset.dataframe.get("val"))
                buffer_size = 100
            else:
                print('dont use generator')
                critical_val_size = len(self.dataset.dataset.get('val'))
                buffer_size = 1000
            # print('critical_val_size', critical_val_size)
            if (critical_val_size == self.batch_size) or ((critical_val_size % self.batch_size) == 0):
                self.val_batch_size = self.batch_size
            elif critical_val_size < self.batch_size:
                self.val_batch_size = critical_val_size
            else:
                self.val_batch_size = self._get_val_batch_size(self.batch_size, critical_val_size)
            # print('self.batch_size', self.batch_size)
            # print('self.val_batch_size', self.val_batch_size)
            trained_model = model_yolo if model_yolo else self.model

            try:
                trained_model.fit(
                    self.dataset.dataset.get('train').shuffle(buffer_size).batch(
                        self.batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE).take(-1),
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    validation_data=self.dataset.dataset.get('val').batch(
                        self.val_batch_size,
                        drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE).take(-1),
                    validation_batch_size=self.val_batch_size,
                    epochs=self.epochs,
                    verbose=verbose,
                    callbacks=self.callbacks
                )
                if yolo_arch:
                    self.model.save_weights(os.path.join(self.training_path, 'yolo_last.h5'))
            except Exception as error:
                params.state.set("stopped") if params.state.status == "addtrain" else params.state.set("no_train")
                progress.pool(self.progress_name, error=terra_exception(error).__str__(), finished=True)

            if (params.state.status == "stopped" and self.callbacks[0].last_epoch < params.base.epochs) or \
                    (params.state.status == "trained" and self.callbacks[0].last_epoch - 1 == params.base.epochs):
                self.sum_epoch = params.base.epochs
        except Exception as e:
            print_error(GUINN().name, method_name, e)

    @staticmethod
    def _prepare_loss_dict(params: dict):
        method_name = '_prepare_loss_dict'
        try:
            loss_dict = {}
            for output_layer in params.get('architecture').get('parameters').get('outputs'):
                loss_obj = getattr(
                    importlib.import_module(loss_metric_config.get("loss").get(output_layer["loss"], {}).get('module')),
                    output_layer["loss"]
                )()
                loss_dict.update({str(output_layer["id"]): loss_obj})
            return loss_dict
        except Exception as e:
            print_error(GUINN().name, method_name, e)
            return None

    @staticmethod
    def set_optimizer(params: dict):
        method_name = 'set_optimizer'
        try:
            optimizer_object = getattr(keras.optimizers, params.get('optimizer').get('type'))
            parameters = params.get('optimizer').get('parameters').get('main')
            parameters.update(params.get('optimizer').get('parameters').get('extra'))
            return optimizer_object(**parameters)
        except Exception as e:
            print_error(GUINN().name, method_name, e)
            return None

    def train_base_model(self, params: dict, dataset: PrepareDataset, model: Model, callback):
        method_name = 'train_base_model'
        try:
            @tf.function
            def train_step(x_batch, y_batch, losses: dict, train_model: Model, set_optimizer):
                """
                losses = {'2': loss_fn}
                """
                with tf.GradientTape() as tape:
                    logits_ = train_model(x_batch, training=True)
                    y_true_ = list(y_batch.values())
                    # print(logits_.shape, y_true_[0].shape)
                    if not isinstance(logits_, list):
                        loss_fn = losses.get(list(losses.keys())[0])
                        total_loss = loss_fn(y_true_[0], logits_)
                    else:
                        total_loss = tf.convert_to_tensor(0.)
                        for i, key in enumerate(losses.keys()):
                            loss_fn = losses[key]
                            total_loss = tf.add(loss_fn(y_true_[i], logits_[i]), total_loss)
                grads = tape.gradient(total_loss, model.trainable_weights)
                set_optimizer.apply_gradients(zip(grads, model.trainable_weights))
                return [logits_] if not isinstance(logits_, list) else logits_, y_true_

            def test_step(options: PrepareDataset, parameters: dict, train_model: Model, outputs: list):
                test_pred = {}
                test_true = {}
                for x_batch_val, y_batch_val in options.dataset.get('val').batch(
                        parameters.get('batch'), drop_remainder=False):
                    test_logits = train_model(x_batch_val, training=False)
                    true_array = list(y_batch_val.values())
                    test_logits = test_logits if isinstance(test_logits, list) else [test_logits]
                    if not test_true:
                        for j, outp in enumerate(outputs):
                            test_pred[f"{outp}"] = test_logits[j].numpy().astype('float')
                            test_true[f"{outp}"] = true_array[j].numpy().astype('float')
                    else:
                        for j, outp in enumerate(outputs):
                            test_pred[f"{outp}"] = np.concatenate(
                                [test_pred[f"{outp}"], test_logits[j].numpy().astype('float')], axis=0)
                            test_true[f"{outp}"] = np.concatenate(
                                [test_true[f"{outp}"], true_array[j].numpy().astype('float')], axis=0)
                return test_pred, test_true

            current_epoch = 0
            train_pred = {}
            train_true = {}
            optimizer = self.set_optimizer(params=params)
            loss = self._prepare_loss_dict(params=params)
            first_epoch = True
            train_data_idxs = []
            urgent_predict = False
            for epoch in range(current_epoch, params.get('epochs')):
                callback._time_first_step = time.time()
                new_batch = True
                train_steps = 0
                for x_batch_train, y_batch_train in dataset.dataset.get('train').batch(
                        params.get('batch'), drop_remainder=False):
                    y_true, logits = train_step(
                        x_batch=x_batch_train, y_batch=y_batch_train, train_model=model,
                        losses=loss, set_optimizer=optimizer
                    )
                    if not train_true:
                        # print(11, dataset.data.outputs.keys())
                        for i, out in enumerate(dataset.data.outputs.keys()):
                            train_pred[f"{out}"] = logits[i].numpy().astype('float')
                            train_true[f"{out}"] = y_true[i].numpy().astype('float')
                        train_data_idxs = list(range(y_true[0].shape[0]))
                        # print('train_true is None: data_idxs', data_idxs[-1])
                    elif first_epoch:
                        for i, out in enumerate(dataset.data.outputs.keys()):
                            train_pred[f"{out}"] = np.concatenate(
                                [train_pred[f"{out}"], logits[i].numpy().astype('float')], axis=0)
                            train_true[f"{out}"] = np.concatenate(
                                [train_true[f"{out}"], y_true[i].numpy().astype('float')], axis=0)
                        train_data_idxs.extend(list(range(
                            train_data_idxs[-1] + 1, train_data_idxs[-1] + 1 + y_true[0].shape[0])))
                    else:
                        for i, out in enumerate(dataset.data.outputs.keys()):
                            train_pred[f"{out}"] = np.concatenate(
                                [train_pred[f"{out}"][logits[i].shape[0]:], logits[i].numpy().astype('float')], axis=0)
                            train_true[f"{out}"] = np.concatenate(
                                [train_true[f"{out}"][y_true[i].shape[0]:], y_true[i].numpy().astype('float')], axis=0)
                        if new_batch:
                            train_data_idxs = train_data_idxs[y_true[0].shape[0]:]
                            train_data_idxs.extend(list(range(y_true[0].shape[0])))
                        else:
                            train_data_idxs = train_data_idxs[y_true[0].shape[0]:]
                            train_data_idxs.extend(list(range(
                                train_data_idxs[-1] + 1, train_data_idxs[-1] + 1 + y_true[0].shape[0])))
                        if urgent_predict:
                            val_pred, val_true = test_step(
                                options=dataset, parameters=params, train_model=model,
                                outputs=list(dataset.data.outputs.keys())
                            )
                            # callback.current_basic_logs(
                            #     epoch=epoch + 1, train_y_true=train_true, train_y_pred=train_pred,
                            #     val_y_true=val_true, val_y_pred=val_pred, train_idx=train_data_idxs
                            # )
                    train_steps += 1
                    new_batch = False

                # Run a validation loop at the end of each epoch.
                val_pred, val_true = test_step(options=dataset, parameters=params, train_model=model,
                                               outputs=list(dataset.data.outputs.keys()))
                callback.on_epoch_end(
                    epoch=epoch + 1, train_true=train_true, train_pred=train_pred,
                    val_true=val_true, val_pred=val_pred, train_data_idxs=train_data_idxs
                )
                # callback._update_log_history()
                print(
                    # f"\nEpoch {callback.current_logs.get('epochs')}:\n"
                    f"\nlog_history: {callback.log_history.get('epochs')} "
                    f"\nloss={callback.log_history.get('2').get('loss')}"
                    f"\nmetrics={callback.log_history.get('2').get('metrics')}"
                    # f" \nloss={callback.current_logs.get('2').get('loss')}\n"
                    # f" \nmetrics={callback.current_logs.get('2').get('metrics')}\n"
                    # f" \nclass_loss={callback.current_logs.get('2').get('class_loss')}\n"
                    # f" \nclass_metrics={callback.current_logs.get('2').get('class_metrics')}"
                )
                # print(f"\nEpoch {epoch + 1}: train_acc={train_acc}, val_acc={val_acc}")
                best_path = callback.save_best_weights()
                if best_path:
                    model.save_weights(best_path)
                    print(f"\nEpoch {epoch + 1}")
                    print(f"Best weights was saved in directory {best_path}")
                first_epoch = False
        except Exception as e:
            print_error(GUINN().name, method_name, e)


class MemoryUsage:
    def __init__(self, debug=False):
        self.debug = debug
        try:
            N.nvmlInit()
            self.gpu = True
        except Exception:
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


class FitCallback(tf.keras.callbacks.Callback):
    """CustomCallback for all task type"""

    def __init__(self, dataset: PrepareDataset, params: dict,
                 # state: StateData, deploy: DeployData, batch_size: int = None, epochs: int = None,
                 dataset_path: Path = Path(""), retrain_epochs: int = None,
                 training_path: Path = Path("./"), model_name: str = "model",
                 deploy_type: str = "", initialed_model=None):
        """
        Для примера
        "checkpoint": {
                "layer": 2,
                "type": "Metrics",
                "indicator": "Val",
                "metric_name": "Accuracy",
                "save_best": True,
                "save_weights": False,
            },
        """

        super().__init__()
        print('\n FitCallback')
        self.name = "FitCallback"
        self.current_logs = {}
        self.usage_info = MemoryUsage(debug=False)
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.deploy_type = deploy_type
        self.is_yolo = True if self.deploy_type in YOLO_ARCHITECTURE else False
        self.batch_size = params.get('batch')
        self.epochs = params.get('epochs')
        self.batch = 0
        self.num_batches = 0
        self.last_epoch = 1
        self._start_time = time.time()
        self._time_batch_step = time.time()
        self._time_first_step = time.time()
        self._sum_time = 0
        self._sum_epoch_time = 0
        self.retrain_epochs = retrain_epochs
        self.still_epochs = params.get('epochs')
        self.training_path = training_path
        self.save_model_path = os.path.join(self.training_path, model_name)
        self.nn_name = model_name
        self.log_gap = 5
        self.progress_threashold = 3
        # self.state = state
        # self.deploy = deploy
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
        self.params = params
        self.checkpoint_config = params.get('architecture').get('parameters').get('checkpoint')
        self.checkpoint_mode = self._get_checkpoint_mode()  # min max
        self.num_outputs = len(self.dataset.data.outputs.keys())
        self.metric_checkpoint = self.checkpoint_config.get('metric_name')  # "val_mAP50" if self.is_yolo else "loss"
        self.class_outputs = class_metric_list(self.dataset)
        self.y_true, _ = BaseClassificationCallback().get_y_true(self.dataset)
        self.class_idx = BaseClassificationCallback().prepare_class_idx(self.y_true, self.dataset)
        print("self.class_outputs", self.class_outputs)
        # print(self.class_idx.get('train').get('2').keys())

        # self.log_history = self._load_logs()
        self.log_history = self._prepare_log_history_template(self.dataset, self.params)

        # yolo params
        self.yolo_model = initialed_model
        self.image_path = os.path.join(self.training_path, "deploy", 'chess_{}.jpg')
        self.samples_train = []
        self.samples_val = []
        self.samples_target_train = []
        self.samples_target_val = []

    @staticmethod
    def _prepare_log_history_template(options: PrepareDataset, params: dict):
        method_name = '_prepare_log_history_template'
        try:
            log_history = {"epochs": []}
            if options.data.architecture in BASIC_ARCHITECTURE:
                for output_layer in params.get('architecture').get('parameters').get("outputs"):
                    out = f"{output_layer['id']}"
                    log_history[out] = {
                        "loss": {}, "metrics": {},
                        "class_loss": {}, "class_metrics": {},
                        "progress_state": {"loss": {}, "metrics": {}}
                    }
                    log_history[out]["loss"][output_layer.get("loss")] = {"train": [], "val": []}
                    log_history[out]["progress_state"]["loss"][output_layer.get("loss")] = {
                        "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []
                    }
                    for metric in output_layer.get("metrics", []):
                        log_history[out]["metrics"][metric] = {"train": [], "val": []}
                        log_history[out]["progress_state"]["metrics"][metric] = {
                            "mean_log_history": [], "normal_state": [], "underfitting": [], "overfitting": []
                        }

                    if options.data.architecture in CLASS_ARCHITECTURE:
                        log_history[out]["class_loss"] = {}
                        log_history[out]["class_metrics"] = {}
                        for class_name in options.data.outputs.get(int(out)).classes_names:
                            log_history[out]["class_metrics"][class_name] = {}
                            log_history[out]["class_loss"][class_name] = \
                                {output_layer.get("loss"): {"train": [], "val": []}}
                            for metric in output_layer.get("metrics", []):
                                log_history[out]["class_metrics"][class_name][metric] = {"train": [], "val": []}

            if options.data.architecture in YOLO_ARCHITECTURE:
                log_history['learning_rate'] = []
                log_history['output'] = {
                    "loss": {
                        'giou_loss': {"train": [], "val": []},
                        'conf_loss': {"train": [], "val": []},
                        'prob_loss': {"train": [], "val": []},
                        'total_loss': {"train": [], "val": []}
                    },
                    "class_loss": {'prob_loss': {}},
                    "metrics": {'mAP50': []},
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
                    log_history['output']["class_loss"]['prob_loss'][class_name] = []
                    log_history['output']["class_metrics"]['mAP50'][class_name] = []
            return log_history
        except Exception as e:
            print_error('FitCallback', method_name, e)

    @staticmethod
    def update_class_idx(dataset_class_idx, predict_idx):
        method_name = 'update_class_idx'
        try:
            update_idx = {'train': {}, "val": dataset_class_idx.get('val')}
            for out in dataset_class_idx['train'].keys():
                update_idx['train'][out] = {}
                for cls in dataset_class_idx['train'][out].keys():
                    shift = predict_idx[0]
                    update_idx['train'][out][cls] = list(np.array(dataset_class_idx['train'][out][cls]) - shift)
            # print('shift', shift)
            return update_idx
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def current_basic_logs(self, epoch: int, train_y_true: dict, train_y_pred: dict,
                           val_y_true: dict, val_y_pred: dict, train_idx: list):
        method_name = 'current_basic_logs'
        try:
            self.current_logs = {"epochs": epoch}
            update_cls = {}
            if self.dataset.data.architecture in CLASSIFICATION_ARCHITECTURE:
                update_cls = self.update_class_idx(self.class_idx, train_idx)
                # print('\nupdate_cls', update_cls.get('train').get('2').get('0'))
            for output_layer in self.params.get('architecture').get('parameters').get("outputs"):
                out = f"{output_layer['id']}"
                name_classes = self.dataset.data.outputs.get(output_layer['id']).classes_names
                self.current_logs[out] = {"loss": {}, "metrics": {}, "class_loss": {}, "class_metrics": {}}

                # calculate loss
                loss_name = output_layer.get("loss")
                loss_fn = getattr(
                    importlib.import_module(loss_metric_config.get("loss").get(loss_name, {}).get('module')), loss_name
                )
                # print('loss_name', loss_name)
                train_loss = self._get_loss_calculation(loss_fn, out, train_y_true.get(out), train_y_pred.get(out))
                val_loss = self._get_loss_calculation(loss_fn, out, val_y_true.get(out), val_y_pred.get(out))
                self.current_logs[out]["loss"][output_layer.get("loss")] = {"train": train_loss, "val": val_loss}
                if self.class_outputs.get(output_layer['id']):
                    self.current_logs[out]["class_loss"][output_layer.get("loss")] = {}
                    if self.dataset.data.architecture in CLASSIFICATION_ARCHITECTURE:
                        # print()
                        for i, cls in enumerate(name_classes):
                            # print(cls, len(update_cls['val'][out][cls]), val_y_true.shape, val_y_pred.shape)
                            # print(train_y_true[update_cls['train'][out][cls], ...])
                            train_class_loss = self._get_loss_calculation(
                                loss_obj=loss_fn, out=out,
                                y_true=train_y_true.get(out)[update_cls['train'][out][cls], ...],
                                y_pred=train_y_pred.get(out)[update_cls['train'][out][cls], ...])
                            # print('train_class_loss', train_class_loss)
                            val_class_loss = self._get_loss_calculation(
                                loss_obj=loss_fn, out=out,
                                y_true=val_y_true.get(out)[update_cls['val'][out][cls], ...],
                                y_pred=val_y_pred.get(out)[update_cls['val'][out][cls], ...])
                            # print('val_class_loss', val_class_loss)
                            self.current_logs[out]["class_loss"][output_layer.get("loss")][cls] = \
                                {"train": train_class_loss, "val": val_class_loss}
                    else:
                        for i, cls in enumerate(name_classes):
                            train_class_loss = self._get_loss_calculation(
                                loss_obj=loss_fn, out=out, class_idx=i, show_class=True,
                                y_true=train_y_true.get(out), y_pred=train_y_pred.get(out))
                            val_class_loss = self._get_loss_calculation(
                                loss_obj=loss_fn, out=out, class_idx=i, show_class=True,
                                y_true=val_y_true.get(out), y_pred=val_y_pred.get(out))
                            self.current_logs[out]["class_loss"][output_layer.get("loss")][cls] = \
                                {"train": train_class_loss, "val": val_class_loss}

                # calculate metrics
                for metric_name in output_layer.get("metrics", []):
                    metric_fn = getattr(
                        importlib.import_module(loss_metric_config.get("metric").get(metric_name, {}).get('module')),
                        metric_name
                    )
                    train_metric = self._get_metric_calculation(
                        metric_name, metric_fn, out, train_y_true.get(out), train_y_pred.get(out))
                    val_metric = self._get_metric_calculation(
                        metric_name, metric_fn, out, val_y_true.get(out), val_y_pred.get(out))
                    self.current_logs[out]["metrics"][metric_name] = {"train": train_metric, "val": val_metric}
                    self.current_logs[out]["class_metrics"][metric_name] = {}
                    if self.class_outputs.get(output_layer['id']):
                        if self.dataset.data.architecture in CLASSIFICATION_ARCHITECTURE and \
                                metric_name not in [Metric.BalancedRecall, Metric.BalancedPrecision,
                                                    Metric.BalancedFScore, Metric.FScore]:
                            for i, cls in enumerate(name_classes):
                                train_class_metric = self._get_metric_calculation(
                                    metric_name=metric_name, metric_obj=metric_fn, out=out,
                                    y_true=train_y_true.get(out)[update_cls['train'][out][cls], ...],
                                    y_pred=train_y_pred.get(out)[update_cls['train'][out][cls], ...])
                                val_class_metric = self._get_metric_calculation(
                                    metric_name=metric_name, metric_obj=metric_fn, out=out,
                                    y_true=val_y_true.get(out)[update_cls['val'][out][cls], ...],
                                    y_pred=val_y_pred.get(out)[update_cls['val'][out][cls], ...])
                                self.current_logs[out]["class_metrics"][metric_name][cls] = \
                                    {"train": train_class_metric, "val": val_class_metric}
                        else:
                            for i, cls in enumerate(name_classes):
                                train_class_metric = self._get_metric_calculation(
                                    metric_name=metric_name, metric_obj=metric_fn, out=out, show_class=True,
                                    y_true=train_y_true.get(out), y_pred=train_y_pred.get(out), class_idx=i)
                                val_class_metric = self._get_metric_calculation(
                                    metric_name=metric_name, metric_obj=metric_fn, out=out, show_class=True,
                                    y_true=val_y_true.get(out), y_pred=val_y_pred.get(out), class_idx=i)
                                self.current_logs[out]["class_metrics"][metric_name][cls] = \
                                    {"train": train_class_metric, "val": val_class_metric}
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def current_yolo_logs(self, interactive_logs):
        method_name = 'current_yolo_logs'
        try:
            self.current_logs = None
            # # if self.options.data.architecture in self.yolo_architecture:
            # # # self._round_loss_metric(train_loss) if not math.isnan(float(train_loss)) else None
            # #                 interactive_log['learning_rate'] = self._round_loss_metric(logs.get('optimizer.lr'))
            # #                 interactive_log['output'] = {
            # #                     "train": {
            # #                         "loss": {
            # #                             'giou_loss': self._round_loss_metric(logs.get('giou_loss')),
            # #                             'conf_loss': self._round_loss_metric(logs.get('conf_loss')),
            # #                             'prob_loss': self._round_loss_metric(logs.get('prob_loss')),
            # #                             'total_loss': self._round_loss_metric(logs.get('total_loss'))
            # #                         },
            # #                         "metrics": {
            # #                             'mAP50': self._round_loss_metric(logs.get('mAP50')),
            # #                             # 'mAP95': logs.get('mAP95'),
            # #                         }
            # #                     },
            # #                     "val": {
            # #                         "loss": {
            # #                             'giou_loss': self._round_loss_metric(logs.get('val_giou_loss')),
            # #                             'conf_loss': self._round_loss_metric(logs.get('val_conf_loss')),
            # #                             'prob_loss': self._round_loss_metric(logs.get('val_prob_loss')),
            # #                             'total_loss': self._round_loss_metric(logs.get('val_total_loss'))
            # #                         },
            # #                         "class_loss": {
            # #                             'prob_loss': {},
            # #                         },
            # #                         "metrics": {
            # #                             'mAP50': self._round_loss_metric(logs.get('val_mAP50')),
            # #                             # 'mAP95': logs.get('val_mAP95'),
            # #                         },
            # #                         "class_metrics": {
            # #                             'mAP50': {},
            # #                             # 'mAP95': {},
            # #                         }
            # #                     }
            # #                 }
            # #                 for name in self.options.data.outputs.get(
            # list(self.options.data.outputs.keys())[0]).classes_names:
            # #                     interactive_log['output']['val']["class_loss"]['prob_loss'][name] =
            # self._round_loss_metric(
            # #                         logs.get(
            # #                             f'val_prob_loss_{name}'))
            # #                     interactive_log['output']['val']["class_metrics"]['mAP50'][name] =
            # self._round_loss_metric(logs.get(
            # #                         f'val_mAP50_class_{name}'))
            # #                     # interactive_log['output']['val']["class_metrics"]['mAP95'][name] =
            # logs.get(f'val_mAP95_class_{name}')
            # #
            # #             return interactive_log
            # self.current_logs = {"epochs": epoch}
            # for output_layer in self.params.architecture.outputs_dict:
            #     out = f"{output_layer['id']}"
            #     name_classes = self.dataset.data.outputs.get(output_layer['id']).classes_names
            #     self.current_logs[out] = {"loss": {}, "metrics": {}, "class_loss": {}, "class_metrics": {}}
            #
            #     # calculate loss
            #     loss_name = output_layer.get("loss")
            #     loss_fn = getattr(
            #         importlib.import_module(loss_metric_config.get("loss").get(loss_name, {}).get('module')),
            #         loss_name
            #     )
            #     train_loss = self._get_loss_calculation(loss_fn, out, train_y_true, train_y_pred)
            #     val_loss = self._get_loss_calculation(loss_fn, out, val_y_true, val_y_pred)
            #     self.current_logs[out]["loss"][output_layer.get("loss")] = {"train": train_loss, "val": val_loss}
            #     if self.class_outputs.get(output_layer['id']):
            #         for i, cls in enumerate(name_classes):
            #             train_class_loss = self._get_loss_calculation(
            #                 loss_obj=loss_fn, out=out,
            #                 y_true=train_y_true[..., i:i + 1], y_pred=train_y_pred[..., i:i + 1])
            #             val_class_loss = self._get_loss_calculation(
            #                 loss_obj=loss_fn, out=out,
            #                 y_true=val_y_true[..., i:i + 1], y_pred=val_y_pred[..., i:i + 1])
            #             self.current_logs[out]["class_metrics"][cls] = \
            #                 {output_layer.get("loss"): {"train": train_class_loss, "val": val_class_loss}}
            #
            #     # calculate metrics
            #     for metric_name in output_layer.get("metrics", []):
            #         metric_fn = getattr(
            #             importlib.import_module(loss_metric_config.get("metric").get(metric_name, {}).get('module')),
            #             metric_name
            #         )
            #         train_metric = self._get_metric_calculation(
            #         metric_name, metric_fn, out, train_y_true, train_y_pred)
            #         val_metric = self._get_metric_calculation(metric_name, metric_fn, out, val_y_true, val_y_pred)
            #         self.current_logs[out]["metrics"][metric_name] = {"train": train_metric, "val": val_metric}
            #
            #         if self.class_outputs.get(output_layer['id']):
            #             for i, cls in enumerate(name_classes):
            #                 train_class_metric = self._get_metric_calculation(
            #                     metric_name=metric_name, metric_obj=metric_fn, out=out, show_class=True,
            #                     y_true=train_y_true[..., i:i + 1], y_pred=train_y_pred[..., i:i + 1])
            #                 val_class_metric = self._get_metric_calculation(
            #                     metric_name=metric_name, metric_obj=metric_fn, out=out, show_class=True,
            #                     y_true=val_y_true[..., i:i + 1], y_pred=val_y_pred[..., i:i + 1])
            #                 self.current_logs[out]["class_metrics"][cls] = \
            #                     {metric_name: {"train": train_class_metric, "val": val_class_metric}}
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
            else:
                m = metric_obj()
            if show_class and (encoding == LayerEncodingChoice.ohe or encoding == LayerEncodingChoice.multi):
                if metric_name == Metric.Accuracy:
                    true_array = to_categorical(np.argmax(y_true, axis=-1), num_classes)[..., class_idx]
                    pred_array = to_categorical(np.argmax(y_pred, axis=-1), num_classes)[..., class_idx]
                    m.update_state(true_array, pred_array)
                elif metric_name in [Metric.BalancedRecall, Metric.BalancedPrecision, Metric.BalancedFScore,
                                     Metric.FScore]:
                    m.update_state(y_true, y_pred, show_class=show_class, class_idx=class_idx)
                # elif metric_name == Metric.BalancedDiceCoef:
                #     m.encoding = 'multi' if encoding == 'multi' else None
                #     m.update_state(y_true, y_pred)
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
                m.update_state(y_true, y_pred)
            metric_value = float(m.result().numpy())
            return metric_value if not math.isnan(metric_value) else None
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _update_log_history(self):
        method_name = '_update_log_history'
        try:
            if self.current_logs['epochs'] in self.log_history['epochs']:
                print(f"\nCurrent epoch {self.current_logs['epochs']} is already in log_history\n")
            self.log_history['epochs'].append(self.current_logs['epochs'])
            if self.dataset.data.architecture in BASIC_ARCHITECTURE:
                for output_layer in self.params.get('architecture').get('parameters').get("outputs"):
                    # print('\noutput_layer', output_layer)
                    out = f"{output_layer['id']}"
                    classes_names = self.dataset.data.outputs.get(output_layer['id']).classes_names
                    # print('\nclasses_names', classes_names)
                    loss_name = output_layer.get('loss')
                    for data_type in ['train', 'val']:
                        self.log_history[out]['loss'][loss_name][data_type].append(
                            round_loss_metric(self.current_logs.get(out).get('loss').get(loss_name).get(data_type))
                        )
                        # print(data_type, self.log_history[out]['loss'][loss_name][data_type])
                    self.log_history[out]['progress_state']['loss'][loss_name]['mean_log_history'].append(
                        self._get_mean_log(self.log_history.get(out).get('loss').get(loss_name).get('val'))
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

                    # print('progress_state', self.log_history[out]['progress_state']['loss'])
                    if self.current_logs.get(out).get("class_loss"):
                        for cls in classes_names:
                            # print('class_loss', cls, self.log_history[out]['class_loss'][cls])
                            self.log_history[out]['class_loss'][cls][loss_name]["train"].append(
                                round_loss_metric(self.current_logs[out]['class_loss'][loss_name][cls]["train"])
                            )
                            # print('class_loss', cls, self.log_history[out]['class_loss'])
                            self.log_history[out]['class_loss'][cls][loss_name]["val"].append(
                                round_loss_metric(self.current_logs[out]['class_loss'][loss_name][cls]["val"])
                            )

                    for metric_name in output_layer.get("metrics", []):
                        for data_type in ['train', 'val']:
                            self.log_history[out]['metrics'][metric_name][data_type].append(
                                round_loss_metric(
                                    self.current_logs.get(out).get('metrics').get(metric_name).get(data_type)))
                        self.log_history[out]['progress_state']['metrics'][metric_name]['mean_log_history'].append(
                            self._get_mean_log(self.log_history[out]['metrics'][metric_name]['val'])
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
                        # print('progress_state', self.log_history[out]['progress_state']['metrics'])
                        if self.current_logs.get(out).get("class_metrics"):
                            for cls in classes_names:
                                self.log_history[out]['class_metrics'][cls][metric_name]["train"].append(
                                    round_loss_metric(
                                        self.current_logs[out]['class_metrics'][metric_name][cls]["train"]))
                                self.log_history[out]['class_metrics'][cls][metric_name]["val"].append(
                                    round_loss_metric(self.current_logs[out]['class_metrics'][metric_name][cls]["val"]))

            if self.dataset.data.architecture in YOLO_ARCHITECTURE:
                self.log_history['learning_rate'] = self.current_logs.get('learning_rate')
                out = list(self.dataset.data.outputs.keys())[0]
                classes_names = self.dataset.data.outputs.get(out).classes_names
                for key in self.log_history['output']["loss"].keys():
                    for data_type in ['train', 'val']:
                        self.log_history['output']["loss"][key][data_type].append(
                            round_loss_metric(self.current_logs.get('output').get(data_type).get('loss').get(key)))
                for key in self.log_history['output']["metrics"].keys():
                    self.log_history['output']["metrics"][key].append(
                        round_loss_metric(self.current_logs.get('output').get(
                            'val').get('metrics').get(key))
                    )
                for name in classes_names:
                    self.log_history['output']["class_loss"]['prob_loss'][name].append(
                        round_loss_metric(self.current_logs.get('output').get("val").get(
                            'class_loss').get("prob_loss").get(name))
                    )
                    self.log_history['output']["class_metrics"]['mAP50'][name].append(
                        round_loss_metric(self.current_logs.get('output').get("val").get(
                            'class_metrics').get("mAP50").get(name))
                    )
                    # self.log_history['output']["class_metrics"]['mAP95'][name].append(
                    #     self._round_loss_metric(self.current_logs.get('output').get("val").get(
                    #         'class_metrics').get("mAP95").get(name))
                    # )
                for loss_name in self.log_history['output']["loss"].keys():
                    # fill loss progress state
                    self.log_history['output']['progress_state']['loss'][loss_name]['mean_log_history'].append(
                        self._get_mean_log(self.log_history.get('output').get('loss').get(loss_name).get('val'))
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
                    self.log_history['output']['progress_state']['metrics'][metric_name][
                        'mean_log_history'].append(
                        self._get_mean_log(self.log_history['output']['metrics'][metric_name])
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

    def _get_mean_log(self, logs):
        method_name = '_get_mean_log'
        try:
            copy_logs = copy.deepcopy(logs)
            while None in copy_logs:
                copy_logs.pop(copy_logs.index(None))
            if len(copy_logs) < self.log_gap:
                return float(np.mean(copy_logs))
            else:
                return float(np.mean(copy_logs[-self.log_gap:]))
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
            if self.checkpoint_config.get("type") == CheckpointTypeChoice.Loss:
                return 'min'
            elif self.checkpoint_config.get("type") == CheckpointTypeChoice.Metrics:
                metric_name = self.checkpoint_config.get("metric_name")
                return loss_metric_config.get("metric").get(metric_name).get("mode")
            else:
                print('\nClass FitCallback method _get_checkpoint_mode: No checkpoint types are found\n')
                return None
        except Exception as e:
            print_error('FitCallback', method_name, e)

    # def _get_metric_name_checkpoint(self, logs: dict):
    #     method_name = '_get_metric_name_checkpoint'
    #     try:
    #         """Поиск среди fit_logs нужного параметра"""
    #         for log in logs.keys():
    #             if self.checkpoint_config.get("type") == CheckpointTypeChoice.Loss and \
    #                     self.checkpoint_config.get("indicator") == CheckpointIndicatorChoice.Val and \
    #                     'val' in log and 'loss' in log:
    #                 if self.num_outputs == 1:
    #                     self.metric_checkpoint = log
    #                     break
    #                 else:
    #                     if f"{self.checkpoint_config.get('layer')}" in log:
    #                         self.metric_checkpoint = log
    #                         break
    #
    #             elif self.checkpoint_config.get("type") == CheckpointTypeChoice.Loss and \
    #                     self.checkpoint_config.get("indicator") == CheckpointIndicatorChoice.Train and \
    #                     'val' not in log and 'loss' in log:
    #                 if self.num_outputs == 1:
    #                     self.metric_checkpoint = log
    #                     break
    #                 else:
    #                     if f"{self.checkpoint_config.get('layer')}" in log:
    #                         self.metric_checkpoint = log
    #                         break
    #
    #             elif self.checkpoint_config.get("type") == CheckpointTypeChoice.Metrics and \
    #                     self.checkpoint_config.get("indicator") == CheckpointIndicatorChoice.Val and \
    #                     "val" in log:
    #                 camelize_log = self._clean_and_camelize_log_name(log)
    #                 if self.num_outputs == 1 and camelize_log == self.checkpoint_config.get("metric_name"):
    #                     self.metric_checkpoint = log
    #                     break
    #                 else:
    #                     if f"{self.checkpoint_config.get('layer')}" in log and \
    #                             camelize_log == self.checkpoint_config.get("metric_name"):
    #                         self.metric_checkpoint = log
    #                         break
    #
    #             elif self.checkpoint_config.get("type") == CheckpointTypeChoice.Metrics and \
    #                     self.checkpoint_config.get("indicator") == CheckpointIndicatorChoice.Train and \
    #                     'val' not in log:
    #                 camelize_log = self._clean_and_camelize_log_name(log)
    #                 if self.num_outputs == 1 and camelize_log == self.checkpoint_config.get("metric_name"):
    #                     self.metric_checkpoint = log
    #                     break
    #                 else:
    #                     if f"{self.checkpoint_config.get('layer')}" in log and \
    #                             camelize_log == self.checkpoint_config.get('metric_name'):
    #                         self.metric_checkpoint = log
    #                         break
    #             else:
    #                 pass
    #     except Exception as e:
    #         print_error('FitCallback', method_name, e)

    # @staticmethod
    # def _clean_and_camelize_log_name(fit_log_name):
    #     method_name = '_clean_and_camelize_log_name'
    #     try:
    #         """Камелизатор исключительно для fit_logs"""
    #         if re.search(r'_\d+$', fit_log_name):
    #             end = len(f"_{fit_log_name.split('_')[-1]}")
    #             fit_log_name = fit_log_name[:-end]
    #         if "val" in fit_log_name:
    #             fit_log_name = fit_log_name[4:]
    #         if re.search(r'^\d+_', fit_log_name):
    #             start = len(f"{fit_log_name.split('_')[0]}_")
    #             fit_log_name = fit_log_name[start:]
    #         return camelize(fit_log_name)
    #     except Exception as e:
    #         print_error('FitCallback', method_name, e)

    # def _fill_log_history(self, epoch, logs):
    #     method_name = '_fill_log_history'
    #     try:
    #         """Заполнение истории логов"""
    #         if epoch == 1:
    #             self.log_history = {
    #                 'epoch': [],
    #                 'logs': {}
    #             }
    #             for metric in logs:
    #                 self.log_history['logs'][metric] = []
    #             self._get_metric_name_checkpoint(logs)
    #             # print(f"Chosen {self.metric_checkpoint} for monitoring")
    #         # print("_fill_log_history", logs)
    #         self.log_history['epoch'].append(epoch)
    #         for metric in logs:
    #             # if logs.get(metric):
    #             self.log_history['logs'][metric].append(float(logs.get(metric)))
    #             # else:
    #             #     self.log_history['logs'][metric].append(0.)
    #     except Exception as e:
    #         print_error('FitCallback', method_name, e)

    def _save_logs(self):
        method_name = '_save_logs'
        try:
            interactive_path = os.path.join(self.save_model_path, "interactive.history")
            if not os.path.exists(interactive_path):
                os.mkdir(interactive_path)
            with open(os.path.join(self.save_model_path, "log.history"), "w", encoding="utf-8") as history:
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
            interactive_path = os.path.join(self.save_model_path, "interactive.history")
            if self.state.status == "addtrain":
                with open(os.path.join(self.save_model_path, "log.history"), "r", encoding="utf-8") as history:
                    logs = json.load(history)
                with open(os.path.join(interactive_path, "log.int"), "r", encoding="utf-8") as int_log:
                    interactive_logs = json.load(int_log)
                with open(os.path.join(interactive_path, "table.int"), "r", encoding="utf-8") as table_int:
                    interactive_table = json.load(table_int)
                with open(os.path.join(interactive_path, "addtraining.int"), "r", encoding="utf-8") as addtraining_int:
                    interactive.addtrain_epochs = json.load(addtraining_int)["addtrain_epochs"]
                self.last_epoch = max(logs.get('epoch')) + 1
                self.still_epochs = self.retrain_epochs - self.last_epoch + 1
                self._get_metric_name_checkpoint(logs.get('logs'))
                interactive.log_history = interactive_logs
                interactive.progress_table = interactive_table
                return logs
            else:
                return self._prepare_log_history_template(self.dataset, self.params)
        except Exception as e:
            print_error('FitCallback', method_name, e)

    @staticmethod
    def _logs_predict_extract(logs, prefix):
        method_name = '_logs_predict_extract'
        try:
            pred_on_batch = []
            for key in logs.keys():
                if key.startswith(prefix):
                    pred_on_batch.append(logs[key])
            return pred_on_batch
        except Exception as e:
            print_error('FitCallback', method_name, e)

    @staticmethod
    def _logs_losses_extract(logs, prefixes: list):
        method_name = '_logs_losses_extract'
        try:
            losses = {}
            for key in logs.keys():
                if key.find(prefixes[0]) != -1 or key.find(prefixes[1]) != -1:
                    pass
                else:
                    losses.update({key: logs[key]})
            return losses
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _best_epoch_monitoring(self):
        method_name = '_best_epoch_monitoring'
        try:
            # print('\nself.checkpoint_config', self.checkpoint_config)
            # print(self.log_history.get(f"{self.checkpoint_config.get('layer')}").keys())
            # print(self.log_history.get(f"{self.checkpoint_config.get('layer')}").get(
            #     self.checkpoint_config.get("type").lower()).keys())
            # print(self.log_history.get(f"{self.checkpoint_config.get('layer')}").get(
            #     self.checkpoint_config.get("type").lower()).get(self.checkpoint_config.get("metric_name")).keys())
            checkpoint_list = self.log_history.get(f"{self.checkpoint_config.get('layer')}").get(
                self.checkpoint_config.get("type").lower()).get(self.checkpoint_config.get("metric_name")).get(
                self.checkpoint_config.get("indicator").lower())
            # print('\ncheckpoint_list', checkpoint_list, self.checkpoint_mode, self.checkpoint_config.get("metric_name"))
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
                    self.result["train_usage"]["timings"]["avg_epoch_time"] = \
                        int(self._sum_epoch_time / self.last_epoch)
                    self.result["train_usage"]["timings"]["elapsed_epoch_time"] = param[key][3]
                    self.result["train_usage"]["timings"]["still_epoch_time"] = param[key][4]
                    self.result["train_usage"]["timings"]["epoch"] = param[key][5]
                    self.result["train_usage"]["timings"]["batch"] = param[key][6]
            self.result["train_usage"]["hard_usage"] = self.usage_info.get_usage()
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _get_result_data(self):
        self.result["states"] = self.state.native()
        return self.result

    def _get_train_status(self) -> str:
        return self.state.status

    def _get_predict(self, deploy_model=None):
        method_name = '_get_predict'
        try:
            current_model = deploy_model if deploy_model else self.model
            if self.is_yolo:
                current_predict = [np.concatenate(elem, axis=0) for elem in zip(*self.samples_val)]
                current_target = [np.concatenate(elem, axis=0) for elem in zip(*self.samples_target_val)]
            else:
                if self.dataset.data.use_generator:
                    current_predict = current_model.predict(self.dataset.dataset.get('val').batch(1),
                                                            batch_size=1)
                else:
                    current_predict = current_model.predict(self.dataset.X.get('val'), batch_size=self.batch_size)
                current_target = None
            return current_predict, current_target
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _deploy_predict(self, presets_predict):
        method_name = '_deploy_predict'
        try:
            result = CreateArray().postprocess_results(array=presets_predict,
                                                       options=self.dataset,
                                                       save_path=os.path.join(self.save_model_path,
                                                                              "deploy_presets"),
                                                       dataset_path=str(self.dataset_path))
            deploy_presets = []
            if result:
                deploy_presets = list(result.values())[0]
            return deploy_presets
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _create_form_data_for_dataframe_deploy(self, deploy_path):
        method_name = '_create_form_data_for_dataframe_deploy'
        try:
            form_data = []
            with open(os.path.join(self.dataset_path, "config.json"), "r", encoding="utf-8") as dataset_conf:
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
            with open(os.path.join(deploy_path, "form.json"), "w", encoding="utf-8") as form_file:
                json.dump(form_data, form_file, ensure_ascii=False)
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _create_cascade(self, **data):
        method_name = '_create_cascade'
        try:
            deploy_path = data.get("deploy_path")
            if self.dataset.data.alias not in ["imdb", "boston_housing", "reuters"]:
                if "Dataframe" in self.deploy_type:
                    self._create_form_data_for_dataframe_deploy(deploy_path=deploy_path)
                if self.is_yolo:
                    func_name = "object_detection"
                else:
                    func_name = decamelize(self.deploy_type)
                config = CascadeCreator()
                config.create_config(str(self.training_path), str(self.save_model_path), func_name=func_name)
                config.copy_package(str(self.training_path))
                config.copy_script(
                    training_path=str(self.training_path),
                    function_name=func_name
                )
                if self.deploy_type == ArchitectureChoice.TextSegmentation:
                    with open(os.path.join(deploy_path, "format.txt"), "w", encoding="utf-8") as format_file:
                        format_file.write(str(data.get("tags_map", "")))
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def _prepare_deploy(self):
        method_name = '_prepare_deploy'
        try:
            deploy_path = os.path.join(self.training_path, "deploy")
            weight = None
            cascade_data = {"deploy_path": deploy_path}
            for i in os.listdir(self.save_model_path):
                if i[-3:] == '.h5' and 'best' in i:
                    weight = i
            if weight:
                if self.yolo_model:
                    self.yolo_model.load_weights(os.path.join(self.save_model_path, weight))
                else:
                    self.model.load_weights(os.path.join(self.save_model_path, weight))
            deploy_predict, y_true = self._get_predict()
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
                out_deploy_presets_data["predict_column"] = predict_column if predict_column \
                    else "Предсказанные значения"
            # print(deploy_presets_data["predict"])
            self.deploy = DeployData(
                path=deploy_path,
                type=self.deploy_type,
                data=out_deploy_presets_data
            )
            # print(interactive.deploy_presets_data)
            self._create_cascade(**cascade_data)
        except Exception as e:
            print_error('FitCallback', method_name, e)

    @staticmethod
    def _estimate_step(current, start, now):
        method_name = '_estimate_step'
        try:
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

    def on_train_begin(self, logs=None):
        method_name = 'on_train_begin'
        try:
            status = self._get_train_status()
            self._start_time = time.time()
            if status != "addtrain":
                self.batch = 0

            if not self.dataset.data.use_generator:
                self.num_batches = len(list(self.dataset.X.get('train').values())[0]) // self.batch_size
            else:
                self.num_batches = len(self.dataset.dataframe['train']) // self.batch_size
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def on_epoch_begin(self, epoch, logs=None):
        self._time_first_step = time.time()

    def on_train_batch_end(self, batch, logs=None):
        method_name = 'on_train_batch_end'
        try:
            if self._get_train_status() == "stopped":
                self.model.stop_training = True
                msg = f'ожидайте остановку...'
                # self.batch += 1
                self._set_result_data({'info': f"'Обучение остановлено пользователем, '{msg}"})
            else:
                msg_batch = {"current": batch + 1, "total": self.num_batches}
                msg_epoch = {"current": self.last_epoch,
                             "total": self.retrain_epochs if interactive.get_states().get("status") == "addtrain"
                             else self.epochs}
                still_epoch_time = self.update_progress(self.num_batches, batch, self._time_first_step)
                elapsed_epoch_time = time.time() - self._time_first_step
                elapsed_time = time.time() - self._start_time
                estimated_time = self.update_progress(self.num_batches * self.still_epochs,
                                                      self.batch, self._start_time, finalize=True)

                still_time = self.update_progress(self.num_batches * self.still_epochs,
                                                  self.batch, self._start_time)
                self.batch += 1
                if interactive.urgent_predict:

                    if self.is_yolo:
                        self.samples_train.append(self._logs_predict_extract(logs, prefix='pred'))
                        self.samples_target_train.append(self._logs_predict_extract(logs, prefix='target'))

                    y_pred, y_true = self._get_predict()
                    train_batch_data = interactive.update_state(y_pred=y_pred, y_true=y_true)
                else:
                    train_batch_data = interactive.update_state(y_pred=None)
                if train_batch_data:
                    result_data = {
                        'timings': [estimated_time, elapsed_time, still_time,
                                    elapsed_epoch_time, still_epoch_time, msg_epoch, msg_batch],
                        'train_data': train_batch_data
                    }
                else:
                    result_data = {'timings': [estimated_time, elapsed_time, still_time,
                                               elapsed_epoch_time, still_epoch_time, msg_epoch, msg_batch]}
                self._set_result_data(result_data)
                # print("PROGRESS", [type(num) for num in self._get_result_data().get("train_data", {}).get(
                # "data_balance", {}).get("2", ["0"])[0].get("plot_data", ["0"])[0].get("values")])
                progress.pool(
                    self.progress_name,
                    percent=(self.last_epoch - 1) / (
                        self.retrain_epochs if interactive.get_states().get("status") == "addtrain" else self.epochs
                    ) * 100,
                    message=f"Обучение. Эпоха {self.last_epoch} из "
                            f"{self.retrain_epochs if interactive.get_states().get('status') in ['addtrain', 'stopped'] else self.epochs}",
                    data=self._get_result_data(),
                    finished=False,
                )
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def on_test_batch_end(self, batch, logs=None):
        method_name = 'on_test_batch_end'
        try:
            if self.is_yolo:
                self.samples_val.append(self._logs_predict_extract(logs, prefix='pred'))
                self.samples_target_val.append(self._logs_predict_extract(logs, prefix='target'))
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def on_epoch_end(self, epoch, train_true=None, train_pred=None, val_true=None, val_pred=None, logs=None,
                     train_data_idxs=None):
        method_name = 'on_epoch_end'
        try:
            self.last_epoch = epoch
            # y_pred, y_true = self._get_predict()
            # total_epochs = self.retrain_epochs if self._get_train_status() in ['addtrain',
            #                                                                    'stopped'] else self.epochs
            # if self.is_yolo:
            #     map50 = get_mAP(self.model, self.dataset, score_threshold=0.05, iou_threshold=[0.50],
            #                     TRAIN_CLASSES=self.dataset.data.outputs.get(2).classes_names,
            #                     dataset_path=self.dataset_path)
            #     interactive_logs = self._logs_losses_extract(logs, prefixes=['pred', 'target'])
            #     interactive_logs.update(map50)
            #     if self.last_epoch < total_epochs and not self.model.stop_training:
            #         self.samples_train = []
            #         self.samples_val = []
            #         self.samples_target_train = []
            #         self.samples_target_val = []
            # else:
            #     interactive_logs = copy.deepcopy(logs)
            self.current_basic_logs(
                epoch=epoch, train_y_true=train_true, train_y_pred=train_pred,
                val_y_true=val_true, val_y_pred=val_pred, train_idx=train_data_idxs
            )
            self._update_log_history()
            # interactive_logs['epoch'] = self.last_epoch
            # current_epoch_time = time.time() - self._time_first_step
            # self._sum_epoch_time += current_epoch_time

            train_epoch_data = interactive.update_state(
                fit_logs=self.log_history,
                arrays={'train_true': train_true, 'train_pred': train_pred, 'val_true': val_true, 'val_pred': val_pred},
                current_epoch_time=time.time() - self._time_first_step,
                on_epoch_end_flag=True,
                train_idx=train_data_idxs
            )

            # self._set_result_data({'train_data': train_epoch_data})
            # progress.pool(
            #     self.progress_name,
            #     percent=(self.last_epoch - 1) / (
            #         self.retrain_epochs if interactive.get_states().get("status") == "addtrain"
            #                                or interactive.get_states().get("status") == "stopped"
            #         else self.epochs
            #     ) * 100,
            #     message=f"Обучение. Эпоха {self.last_epoch} из "
            #             f"{self.retrain_epochs if interactive.get_states().get('status') in ['addtrain', 'stopped'] else self.epochs}",
            #     data=self._get_result_data(),
            #     finished=False,
            # )

            # сохранение лучших весов
            # if self.last_epoch > 1:
            #     try:
            #         if self._best_epoch_monitoring():
            #             if not os.path.exists(self.save_model_path):
            #                 os.mkdir(self.save_model_path)
            #             if not os.path.exists(os.path.join(self.save_model_path, "deploy_presets")):
            #                 os.mkdir(os.path.join(self.save_model_path, "deploy_presets"))
            #             file_path_best: str = os.path.join(
            #                 self.save_model_path, f"best_weights_{self.metric_checkpoint}.h5"
            #             )
            #             # if self.yolo_model:
            #             #     self.yolo_model.save_weights(file_path_best)
            #             # else:
            #             #     self.model.save_weights(file_path_best)
            #     except Exception as e:
            #         print('\nself.model.save_weights failed', e)
            # self._fill_log_history(self.last_epoch, interactive_logs)
            # self.last_epoch += 1
        except Exception as e:
            print_error('FitCallback', method_name, e)

    def save_best_weights(self):
        method_name = 'save_best_weights'
        try:
            if self.last_epoch > 1:
                if self._best_epoch_monitoring():
                    if not os.path.exists(self.save_model_path):
                        os.mkdir(self.save_model_path)
                    if not os.path.exists(os.path.join(self.save_model_path, "deploy_presets")):
                        os.mkdir(os.path.join(self.save_model_path, "deploy_presets"))
                    file_path_best: str = os.path.join(
                        self.save_model_path, f"best_weights_{self.metric_checkpoint}_epoch {self.last_epoch}.h5"
                    )
                    return file_path_best
        except Exception as e:
            print_error('FitCallback', method_name, e)
            print('\nself.model.save_weights failed', e)

    def on_train_end(self, logs=None):
        method_name = 'on_train_end'
        try:
            interactive.addtrain_epochs.append(self.last_epoch - 1)
            self._save_logs()

            if (self.last_epoch - 1) > 1:
                file_path_last: str = os.path.join(
                    self.save_model_path, f"last_weights_{self.metric_checkpoint}.h5"
                )
                if self.yolo_model:
                    self.yolo_model.save_weights(file_path_last)
                else:
                    self.model.save_weights(file_path_last)
            if not os.path.exists(os.path.join(self.save_model_path, "deploy_presets")):
                os.mkdir(os.path.join(self.save_model_path, "deploy_presets"))
            self._prepare_deploy()

            time_end = self.update_progress(self.num_batches * self.epochs + 1,
                                            self.batch, self._start_time, finalize=True)
            self._sum_time += time_end
            total_epochs = self.retrain_epochs if self._get_train_status() in ['addtrain',
                                                                               'trained'] else self.epochs
            if self.model.stop_training:
                self._set_result_data({'info': f"Обучение остановлено. Модель сохранена."})
                progress.pool(
                    self.progress_name,
                    message=f"Обучение остановлено. Эпоха {self.last_epoch - 1} из "
                            f"{total_epochs}",
                    data=self._get_result_data(),
                    finished=True,
                )
            else:
                if self._get_train_status() == "retrain":
                    msg = f'Затрачено времени на обучение: ' \
                          f'{self.eta_format(time_end)} '
                else:
                    msg = f'Затрачено времени на обучение: ' \
                          f'{self.eta_format(self._sum_time)} '
                self._set_result_data({'info': f"Обучение закончено. {msg}"})
                percent = (self.last_epoch - 1) / (
                    self.retrain_epochs if self._get_train_status() ==
                                           "addtrain" or self._get_train_status() == "stopped"
                    else self.epochs
                ) * 100
                self.state.set("trained")
                progress.pool(
                    self.progress_name,
                    percent=percent,
                    message=f"Обучение завершено. Эпоха {self.last_epoch - 1} из "
                            f"{total_epochs}",
                    data=self._get_result_data(),
                    finished=True,
                )
        except Exception as e:
            print_error('FitCallback', method_name, e)
