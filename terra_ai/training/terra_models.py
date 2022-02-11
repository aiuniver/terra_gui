import importlib
import json
import os
from typing import Optional

import numpy as np

from pathlib import Path

import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import preprocess_input, VGG19
from tensorflow.python.keras.models import Model

from terra_ai.callbacks import interactive
from terra_ai.callbacks.gan_callback import CGANCallback
from terra_ai.callbacks.utils import loss_metric_config, get_dataset_length, CLASSIFICATION_ARCHITECTURE
from terra_ai.custom_objects.custom_layers import terra_custom_layers
from terra_ai.data.training.extra import ArchitectureChoice
from terra_ai.data.training.train import TrainingDetailsData
from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.logging import logger
from terra_ai.training.yolo_utils import decode, compute_loss, get_mAP
import terra_ai.exceptions.callbacks as exception
from typing import Optional
from tensorflow.keras import backend as K


class BaseTerraModel:
    name = "BaseTerraModel"

    def __init__(self, model, model_name: str, model_path: Path):

        self.model_name = model_name
        self.model_json = f"model_json.trm"
        self.custom_obj_json = f"model_custom_obj_json.trm"
        self.model_weights = f"model_weights"
        self.model_best_weights = f"model_best_weights"

        self.saving_path = model_path
        self.file_path_model_json = os.path.join(self.saving_path, self.model_json)
        self.file_path_custom_obj_json = os.path.join(self.saving_path, self.custom_obj_json)
        self.file_path_model_weights = os.path.join(self.saving_path, self.model_weights)
        self.file_path_model_best_weights = os.path.join(self.saving_path, self.model_best_weights)

        if not isinstance(model, dict):
            self.base_model = model
            self.json_model = self.base_model.to_json() if model else None

        if not model:
            self.load()
            self.load_weights()

        self.callback = None
        self.optimizer = None

        self.train_length, self.val_length = 0, 0

    def save(self) -> None:
        method_name = 'save'
        try:
            self.__save_model_to_json()
            self.__save_custom_objects_to_json()
            self.save_weights()
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            raise exc

    def load(self) -> None:
        model_data, custom_dict = self.__get_json_data()
        custom_object = self.__set_custom_objects(custom_dict)
        self.base_model = tf.keras.models.model_from_json(model_data, custom_objects=custom_object)
        self.json_model = self.base_model.to_json()

    def save_weights(self, path_=None):
        if not path_:
            path_ = self.file_path_model_weights
        self.base_model.save_weights(path_)

    def load_weights(self):
        self.base_model.load_weights(self.file_path_model_weights)

    def set_callback(self, callback):
        self.callback = callback

    def set_optimizer(self, params: TrainingDetailsData):
        method_name = 'set_optimizer'
        try:
            optimizer_object = getattr(keras.optimizers, params.base.optimizer.type)
            parameters = params.base.optimizer.parameters.main.native()
            parameters.update(params.base.optimizer.parameters.extra.native())
            self.optimizer = optimizer_object(**parameters)
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            raise exc

    def __save_model_to_json(self):
        with open(self.file_path_model_json, "w", encoding="utf-8") as json_file:
            json.dump(self.json_model, json_file)

    def __save_custom_objects_to_json(self):
        with open(self.file_path_custom_obj_json, "w", encoding="utf-8") as json_file:
            json.dump(terra_custom_layers, json_file)

    @staticmethod
    def __set_custom_objects(custom_dict):
        custom_object = {}
        for k, v in custom_dict.items():
            try:
                custom_object[k] = getattr(importlib.import_module(f".{v}", package="terra_ai.custom_objects"), k)
            except:
                continue
        return custom_object


    def __get_json_data(self):
        with open(self.file_path_model_json) as json_file:
            data = json.load(json_file)

        with open(self.file_path_custom_obj_json) as json_file:
            custom_dict = json.load(json_file)

        return data, custom_dict

    @staticmethod
    def _prepare_loss_dict(params: TrainingDetailsData):
        method_name = '_prepare_loss_dict'
        try:
            loss_dict = {}
            for output_layer in params.base.architecture.parameters.outputs:
                loss_obj = getattr(
                    importlib.import_module(
                        loss_metric_config.get("loss").get(output_layer.loss.name, {}).get('module')),
                    output_layer.loss.name
                )()
                loss_dict.update({str(output_layer.id): loss_obj})
            return loss_dict
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            raise exc

    @staticmethod
    def _add_sample_weights(label, weights):
        class_weights = tf.constant(weights)
        class_weights = class_weights / tf.reduce_sum(class_weights)
        sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))
        return sample_weights

    @tf.function
    def __train_step(self, x_batch, y_batch, losses: dict, set_optimizer, sample_weights: dict):
        """
        losses = {'2': loss_fn}
        """
        with tf.GradientTape() as tape:
            logits_ = self.base_model(x_batch, training=True)
            y_true_ = list(y_batch.values())
            if not isinstance(logits_, list):
                out = list(losses.keys())[0]
                total_loss = losses[out](y_true_[0], logits_, sample_weights[out])
            else:
                total_loss = tf.convert_to_tensor(0.)
                for k, out in enumerate(losses.keys()):
                    loss_fn = losses[out]
                    total_loss = tf.add(loss_fn(y_true_[k], logits_[k], sample_weights[out]), total_loss)
        grads = tape.gradient(total_loss, self.base_model.trainable_weights)
        set_optimizer.apply_gradients(zip(grads, self.base_model.trainable_weights))
        return [logits_] if not isinstance(logits_, list) else logits_, y_true_

    @tf.function
    def __test_step(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            test_logits = self.base_model(x_batch)
            true_array = list(y_batch.values())
            test_logits = test_logits if isinstance(test_logits, list) else [test_logits]
        return test_logits, true_array

    def fit(self, params: TrainingDetailsData, dataset: PrepareDataset):
        method_name = 'fit'
        try:
            self.train_length, self.val_length = get_dataset_length(dataset)
            current_epoch = self.callback.last_epoch
            end_epoch = self.callback.total_epochs
            train_pred, train_true, val_pred, val_true = {}, {}, {}, {}
            self.set_optimizer(params=params)
            loss = self._prepare_loss_dict(params=params)
            output_list = list(dataset.data.outputs.keys())
            sample_weights = {}
            for out in output_list:
                train_target_shape, val_target_shape = [self.train_length], [self.val_length]
                train_target_shape.extend(list(dataset.data.outputs.get(out).shape))
                val_target_shape.extend(list(dataset.data.outputs.get(out).shape))
                train_pred[f"{out}"] = np.zeros(train_target_shape).astype('float32')
                train_true[f"{out}"] = np.zeros(train_target_shape).astype('float32')
                val_pred[f"{out}"] = np.zeros(val_target_shape).astype('float32')
                val_true[f"{out}"] = np.zeros(val_target_shape).astype('float32')
                sample_weights[f"{out}"] = None

            train_data_idxs = np.arange(self.train_length).tolist()
            first_epoch = True
            weight_dataset = True
            self.callback.on_train_begin()
            for epoch in range(current_epoch, end_epoch):
                logger.debug(f"Эпоха {epoch + 1}")
                self.callback.on_epoch_begin()
                train_steps = 0
                current_idx = 0
                logger.debug(f"Эпоха {epoch + 1}: обучение на тренировочной выборке...")
                for x_batch_train, y_batch_train in dataset.dataset.get('train').batch(params.base.batch):
                    length = list(y_batch_train.values())[0].shape[0]
                    batch_weights = {}
                    for out in sample_weights.keys():
                        if sample_weights[out] is None:
                            batch_weights[out] = None
                        else:
                            batch_weights[out] = sample_weights[out][current_idx: current_idx + length]
                    logits, y_true = self.__train_step(
                        x_batch=x_batch_train, y_batch=y_batch_train,
                        losses=loss, set_optimizer=self.optimizer, sample_weights=batch_weights
                    )
                    for i, out in enumerate(output_list):
                        train_pred[f"{out}"][current_idx: current_idx + length] = logits[i].numpy()
                        train_true[f"{out}"][current_idx: current_idx + length] = y_true[i].numpy()
                    current_idx += length
                    train_steps += 1

                    if interactive.urgent_predict:
                        logger.debug(f"Эпоха {epoch + 1}: urgent_predict, обработка проверочной выборки..")
                        val_steps = 0
                        current_val_idx = 0
                        for x_batch_val, y_batch_val in dataset.dataset.get('val').batch(params.base.batch):
                            val_pred_array, val_true_array = self.__test_step(x_batch=x_batch_val, y_batch=y_batch_val)
                            length = val_true_array[0].shape[0]
                            for i, out in enumerate(output_list):
                                val_pred[f"{out}"][current_val_idx: current_val_idx + length] = \
                                    val_pred_array[i].numpy()
                                val_true[f"{out}"][current_val_idx: current_val_idx + length] = \
                                    val_true_array[i].numpy()
                            current_val_idx += length
                            val_steps += 1
                        self.callback.on_train_batch_end(batch=train_steps, arrays={
                            "train_true": train_true, "val_true": val_true, "train_pred": train_pred,
                            "val_pred": val_pred}, train_data_idxs=train_data_idxs)
                    else:
                        self.callback.on_train_batch_end(batch=train_steps)

                    if self.callback.stop_training:
                        break
                self.save_weights()
                if self.callback.stop_training:
                    break

                logger.debug(f"Эпоха {epoch + 1}: обработка проверочной выборки...")
                val_steps = 0
                current_val_idx = 0
                for x_batch_val, y_batch_val in dataset.dataset.get('val').batch(params.base.batch):
                    val_pred_array, val_true_array = self.__test_step(x_batch=x_batch_val, y_batch=y_batch_val)
                    length = val_true_array[0].shape[0]
                    for i, out in enumerate(output_list):
                        val_pred[f"{out}"][current_val_idx: current_val_idx + length] = val_pred_array[i].numpy()
                        val_true[f"{out}"][current_val_idx: current_val_idx + length] = val_true_array[i].numpy()
                    current_val_idx += length
                    val_steps += 1

                if weight_dataset and first_epoch and (
                        dataset.data.architecture in CLASSIFICATION_ARCHITECTURE or
                        dataset.data.architecture in [ArchitectureChoice.TextSegmentation,
                                                      ArchitectureChoice.ImageSegmentation]
                ):
                    for out in output_list:
                        if dataset.data.outputs.get(int(out)).task == 'Classification':
                            classes = np.argmax(train_true[f"{out}"], axis=-1)
                            count = {}
                            for i in set(sorted(classes)):
                                count[i] = 0
                            for i in classes:
                                count[i] += 1
                            weighted_count = {}
                            for k, v in count.items():
                                weighted_count[k] = max(count.values()) / v
                            sample_weights[f"{out}"] = []
                            for i in classes:
                                sample_weights[f"{out}"].append(weighted_count[i])
                            sample_weights[f"{out}"] = tf.constant(sample_weights[f"{out}"])
                            logger.debug(f"weighted_count: {weighted_count}")
                        if dataset.data.outputs.get(int(out)).task in ['Segmentation', 'TextSegmentation'] and \
                                dataset.data.outputs.get(int(out)).encoding == 'ohe':
                            weights_dict = {}
                            for i in range(train_true[f"{out}"].shape[-1]):
                                weights_dict[i] = train_true[f"{out}"][..., i].sum()
                            weights = [max(weights_dict.values()) / weights_dict[i] for i in weights_dict.keys()]
                            sample_weights[f"{out}"] = self._add_sample_weights(
                                np.argmax(train_true[f"{out}"], axis=-1), weights)
                            logger.debug(f"weights: {weights}")
                    first_epoch = False

                self.callback.on_epoch_end(
                    epoch=epoch + 1,
                    arrays={
                        "train_pred": train_pred, "val_pred": val_pred, "train_true": train_true, "val_true": val_true
                    },
                    train_data_idxs=train_data_idxs
                )

                if self.callback.is_best():
                    self.save_weights(path_=self.file_path_model_best_weights)
            self.callback.on_train_end()
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            raise exc

    def predict(self, data_array, options: Optional[PrepareDataset] = None):
        return self.base_model(data_array)


class YoloTerraModel(BaseTerraModel):
    name = "YoloTerraModel"

    def __init__(self, model, model_name: str, model_path: Path, **options):
        super().__init__(model=model, model_name=model_name, model_path=model_path)
        self.yolo_model = self.__create_yolo(training=options.get("training"),
                                             classes=options.get("classes"),
                                             version=options.get("version"))

    def __create_yolo(self, training=False, classes=None, version='v3') -> tf.keras.Model:
        method_name = 'create_yolo'
        try:
            if classes is None:
                classes = []
            num_class = len(classes)
            conv_tensors = self.base_model.outputs
            if conv_tensors[0].shape[1] == 13:
                conv_tensors.reverse()
            output_tensors = []
            for i, conv_tensor in enumerate(conv_tensors):
                pred_tensor = decode(conv_tensor, num_class, i, version)
                if training:
                    output_tensors.append(conv_tensor)
                output_tensors.append(pred_tensor)
            yolo = tf.keras.Model(self.base_model.inputs, output_tensors)
            return yolo
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                YoloTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            raise exc

    @staticmethod
    def __create_yolo_parameters(params: TrainingDetailsData, dataset: PrepareDataset):
        yolo_iou_loss_thresh = params.base.architecture.parameters.yolo.yolo_iou_loss_thresh
        train_warmup_epochs = params.base.architecture.parameters.yolo.train_warmup_epochs
        train_lr_init = params.base.architecture.parameters.yolo.train_lr_init
        train_lr_end = params.base.architecture.parameters.yolo.train_lr_end
        num_class = dataset.data.outputs.get(list(dataset.data.outputs.keys())[0]).num_classes
        classes = dataset.data.outputs.get(list(dataset.data.outputs.keys())[0]).classes_names

        steps_per_epoch = int(len(dataset.dataframe['train']) // params.base.batch)
        warmup_steps = train_warmup_epochs * steps_per_epoch
        total_steps = params.base.epochs * steps_per_epoch

        out = {
            "parameters": {
                "yolo_iou_loss_thresh": yolo_iou_loss_thresh,
                "train_warmup_epochs": train_warmup_epochs,
                "train_lr_init": train_lr_init,
                "train_lr_end": train_lr_end,
                "num_class": num_class,
                "classes": classes
            },
            "steps": {
                "steps_per_epoch": steps_per_epoch,
                "warmup_steps": warmup_steps,
                "total_steps": total_steps
            }
        }
        return out

    @tf.function
    def __train_step(self, image_array, conv_target, serv_target, global_steps, **options):
        num_class = options.get("parameters").get("num_class")
        classes = options.get("parameters").get("classes")
        yolo_iou_loss_thresh = options.get("parameters").get("yolo_iou_loss_thresh")
        train_lr_init = options.get("parameters").get("train_lr_init")
        train_lr_end = options.get("parameters").get("train_lr_end")

        warmup_steps = options.get("steps").get("warmup_steps")
        total_steps = options.get("steps").get("total_steps")

        with tf.GradientTape() as tape:
            pred_result = self.yolo_model(image_array['1'], training=True)
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
            gradients = tape.gradient(total_loss, self.yolo_model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.yolo_model.trainable_variables))
            global_steps.assign_add(1)
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * train_lr_init
            else:
                lr = train_lr_end + 0.5 * (train_lr_init - train_lr_end) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
            lr = tf.cast(lr, dtype='float32')
            self.optimizer.lr.assign(lr)
        return global_steps, giou_loss, conf_loss, prob_loss, total_loss, prob_loss_cls, predict, lr

    @tf.function
    def __validate_step(self, image_array, conv_target, serv_target, **options):
        num_class = options.get("parameters").get("num_class")
        classes = options.get("parameters").get("classes")
        yolo_iou_loss_thresh = options.get("parameters").get("yolo_iou_loss_thresh")

        pred_result = self.yolo_model(image_array['1'], training=True)
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

    def fit(self, params: TrainingDetailsData, dataset: PrepareDataset):
        method_name = 'train_yolo_model'
        try:
            self.train_length, self.val_length = get_dataset_length(dataset)
            yolo_parameters = self.__create_yolo_parameters(params=params, dataset=dataset)
            num_class = yolo_parameters.get("parameters").get("num_class")
            classes = yolo_parameters.get("parameters").get("classes")
            global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)

            self.set_optimizer(params=params)

            current_epoch = self.callback.last_epoch
            end_epoch = self.callback.total_epochs
            train_pred, train_true, val_pred, val_true = [], [], [], []
            output_array = None
            for _, out, _ in dataset.dataset['train'].batch(1).take(1):
                output_array = out
            for array in output_array.values():
                train_target_shape, val_target_shape = [self.train_length], [self.val_length]
                val_target_shape.extend(list(array.shape[1:]))
                val_pred.append(np.zeros(val_target_shape))
                val_true.append(np.zeros(val_target_shape))

            train_data_idxs = np.arange(self.train_length).tolist()
            self.callback.on_train_begin()
            for epoch in range(current_epoch, end_epoch):
                logger.debug(f"Эпоха {epoch + 1}")
                self.callback.on_epoch_begin()
                current_logs = {"epochs": epoch + 1, 'loss': {}, "metrics": {}, 'class_loss': {}, 'class_metrics': {}}
                train_loss_cls = {}
                for cls in range(num_class):
                    train_loss_cls[classes[cls]] = 0.
                cur_step, giou_train, conf_train, prob_train, total_train = 0, 0, 0, 0, 0
                logger.debug(f"Эпоха {epoch + 1}: обучение на тренировочной выборке...")
                for image_data, target1, target2 in dataset.dataset.get('train').batch(params.base.batch):
                    results = self.__train_step(
                        image_data, target1, target2, global_steps=global_steps, **yolo_parameters)

                    giou_train += results[1].numpy()
                    conf_train += results[2].numpy()
                    prob_train += results[3].numpy()
                    total_train += results[4].numpy()
                    for cls in range(num_class):
                        train_loss_cls[classes[cls]] += results[5][classes[cls]].numpy()

                    cur_step += 1
                    if interactive.urgent_predict:
                        logger.debug(f"Эпоха {epoch + 1}: urgent_predict, обработка проверочной выборки...")
                        val_steps = 0
                        val_current_idx = 0
                        for val_image_data, val_target1, val_target2 in dataset.dataset.get('val').batch(
                                params.base.batch):
                            results = self.__validate_step(
                                val_image_data, val_target1, val_target2, **yolo_parameters)
                            val_true_array = list(val_target1.values())
                            length = val_true_array[0].shape[0]
                            for i in range(len(val_true_array)):
                                val_pred[i][val_current_idx: val_current_idx + length] = results[5][i].numpy()
                                val_true[i][val_current_idx: val_current_idx + length] = val_true_array[i].numpy()
                            val_current_idx += length
                            val_steps += 1
                        self.callback.on_train_batch_end(batch=cur_step, arrays={
                            "train_true": train_true, "val_true": val_true, "train_pred": train_pred,
                            "val_pred": val_pred}, train_data_idxs=train_data_idxs)
                    else:
                        self.callback.on_train_batch_end(batch=cur_step)
                    if self.callback.stop_training:
                        break
                self.save_weights()
                if self.callback.stop_training:
                    break

                current_logs['loss']['giou_loss'] = {'train': giou_train / cur_step}
                current_logs['loss']['conf_loss'] = {'train': conf_train / cur_step}
                current_logs['loss']['prob_loss'] = {'train': prob_train / cur_step}
                current_logs['loss']['total_loss'] = {'train': total_train / cur_step}
                current_logs['class_loss']['prob_loss'] = {}

                for cls in range(num_class):
                    current_logs['class_loss']['prob_loss'][str(classes[cls])] = \
                        {'train': train_loss_cls[str(classes[cls])] / cur_step}
                    train_loss_cls[str(classes[cls])] = train_loss_cls[str(classes[cls])] / cur_step

                logger.debug(f"Эпоха {epoch + 1}: обработка проверочной выборки...")
                val_steps, giou_val, conf_val, prob_val, total_val = 0, 0, 0, 0, 0
                val_loss_cls = {}
                for cls in range(num_class):
                    val_loss_cls[classes[cls]] = 0.
                val_current_idx = 0
                for image_data, target1, target2 in dataset.dataset.get('val').batch(params.base.batch):
                    results = self.__validate_step(image_data, target1, target2, **yolo_parameters)
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

                current_logs['loss']['giou_loss']["val"] = giou_val / val_steps
                current_logs['loss']['conf_loss']["val"] = conf_val / val_steps
                current_logs['loss']['prob_loss']["val"] = prob_val / val_steps
                current_logs['loss']['total_loss']["val"] = total_val / val_steps

                for cls in range(num_class):
                    current_logs['class_loss']['prob_loss'][str(classes[cls])]["val"] = \
                        val_loss_cls[str(classes[cls])] / val_steps

                logger.debug(f"Эпоха {epoch + 1}: расчет метрики map50...")
                map50 = get_mAP(self.yolo_model, dataset, score_threshold=0.05, iou_threshold=[0.50],
                                TRAIN_CLASSES=dataset.data.outputs.get(2).classes_names, dataset_path=dataset.data.path)
                current_logs['metrics']['mAP50'] = {"val": map50.get('val_mAP50')}
                current_logs['class_metrics']['mAP50'] = {}
                for cls in range(num_class):
                    try:
                        current_logs['class_metrics']['mAP50'][str(classes[cls])] = \
                            {"val": map50.get(f"val_mAP50_class_{classes[cls]}") * 100}
                    except:
                        current_logs['class_metrics']['mAP50'][str(classes[cls])] = {"val": None}
                self.callback.on_epoch_end(
                    epoch=epoch + 1,
                    arrays={"train_pred": train_pred, "val_pred": val_pred, "train_true": train_true,
                            "val_true": val_true},
                    train_data_idxs=train_data_idxs,
                    logs=current_logs
                )
                if self.callback.is_best():
                    self.save_weights(path_=self.file_path_model_best_weights)
            self.callback.on_train_end()
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                YoloTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            raise exc

    def predict(self, data_array, options: Optional[PrepareDataset] = None):
        return self.yolo_model(data_array)


class GANTerraModel:
    name = "GANTerraModel"

    def __init__(self, model: dict, model_name: str, model_path: Path, **options):
        self.saving_path = model_path
        self.model_name = model_name
        self.custom_obj_json = f"model_custom_obj_json.trm"
        self.file_path_gen_json = os.path.join(self.saving_path, "generator_json.trm")
        self.file_path_disc_json = os.path.join(self.saving_path, "discriminator_json.trm")
        self.file_path_custom_obj_json = os.path.join(self.saving_path, self.custom_obj_json)
        self.generator_weights = "generator_weights"
        self.file_path_gen_weights = os.path.join(self.saving_path, self.generator_weights)
        self.discriminator_weights = "discriminator_weights"
        self.file_path_disc_weights = os.path.join(self.saving_path, self.discriminator_weights)

        if not model:
            self.load()
            self.load_weights()
        else:
            self.generator: Model = model.get('generator')
            self.discriminator: Model = model.get('discriminator')

        self.vae_discriminator = self.detect_vae_discriminator()
        # if self.vae_discriminator:
        #     for layer in self.discriminator.layers:
        #         layer.training = True

        self.generator_json = self.generator.to_json()
        self.discriminator_json = self.discriminator.to_json()
        self.noise = self.generator.inputs[0].shape[1:]
        self.generator_loss_func = None
        self.discriminator_loss_func = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.seed = self.__prepare_seed(self.noise)
        self.callback = None
        self.train_length = 0
        self.custom_object = {}
        pass

    def detect_vae_discriminator(self):
        vae = False
        for layer in self.discriminator.layers:
            if "VAEDiscriminatorBlock" in layer.name:
                vae = True
                break
        return vae


    def save(self) -> None:
        method_name = 'save'
        try:
            self.__save_model_to_json()
            self.__save_custom_objects_to_json()
            self.save_weights()
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                GANTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            raise exc

    def load(self) -> None:
        gen_model_data, disc_model_data, custom_dict = self.__get_gan_json_data()
        logger.debug(f"custom_dict: {custom_dict}")
        self.custom_object = self.__set_custom_objects(custom_dict)
        logger.debug(f"self.custom_object: {self.custom_object}")
        self.generator = tf.keras.models.model_from_json(gen_model_data, custom_objects=self.custom_object)
        self.discriminator = tf.keras.models.model_from_json(disc_model_data, custom_objects=self.custom_object)
        self.generator_json = self.generator.to_json()
        self.discriminator_json = self.discriminator.to_json()

    def save_weights(self, gw_path_=None, dw_path_=None, save_type: str = "last"):
        if not gw_path_:
            gw_path_ = os.path.join(self.saving_path, self.generator_weights)
        self.generator.save_weights(gw_path_)
        logger.debug(f"self.generator.save_weights: {gw_path_}")
        if not dw_path_:
            dw_path_ = os.path.join(self.saving_path, self.discriminator_weights)
        self.discriminator.save_weights(dw_path_)
        logger.debug(f"self.discriminator.save_weights: {dw_path_}")

    def load_weights(self):
        logger.debug(f"self.file_path_gen_weights, {self.file_path_gen_weights}")
        logger.debug(f"self.file_path_disc_weights, {self.file_path_disc_weights}")
        logger.debug(f"self.discriminator: {self.discriminator.summary()}")
        self.generator.load_weights(self.file_path_gen_weights)
        self.discriminator.load_weights(self.file_path_disc_weights)

    def set_callback(self, callback):
        self.callback = callback

    @staticmethod
    def _prepare_loss_dict(params: TrainingDetailsData):
        method_name = '_prepare_loss_dict'
        try:
            loss_dict = {}
            for output_layer in params.base.architecture.parameters.outputs:
                logger.debug(f"_prepare_loss_dict {output_layer.task.name, output_layer.loss.name}")
                try:
                    loss_obj = getattr(
                        importlib.import_module(
                            loss_metric_config.get("loss").get(output_layer.loss.name).get('module')),
                        output_layer.loss.name
                    )(from_logits=True)
                except:
                    loss_obj = getattr(
                        importlib.import_module(
                            loss_metric_config.get("loss").get(output_layer.loss.name).get('module')),
                        output_layer.loss.name
                    )()
                loss_dict.update({str(output_layer.task.name).lower(): loss_obj})
            return loss_dict
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                GANTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            raise exc

    def __save_model_to_json(self):
        with open(self.file_path_gen_json, "w", encoding="utf-8") as json_file:
            json.dump(self.generator_json, json_file)

        with open(self.file_path_disc_json, "w", encoding="utf-8") as json_file:
            json.dump(self.discriminator_json, json_file)

    def __save_custom_objects_to_json(self):
        with open(self.file_path_custom_obj_json, "w", encoding="utf-8") as json_file:
            json.dump(terra_custom_layers, json_file)

    @staticmethod
    def __set_custom_objects(custom_dict):
        custom_object = {}
        for k, v in custom_dict.items():
            try:
                custom_object[k] = getattr(importlib.import_module(f".{v}", package="terra_ai.custom_objects"), k)
            except:
                continue
        return custom_object

    def __get_gan_json_data(self):
        self.file_path_gen_json = os.path.join(self.saving_path, "generator_json.trm")
        self.file_path_disc_json = os.path.join(self.saving_path, "discriminator_json.trm")
        with open(self.file_path_gen_json) as json_file:
            gen_data = json.load(json_file)

        with open(self.file_path_disc_json) as json_file:
            disc_data = json.load(json_file)

        with open(self.file_path_custom_obj_json) as json_file:
            custom_dict = json.load(json_file)

        return gen_data, disc_data, custom_dict

    @staticmethod
    def __prepare_seed(noise):
        shape = [50]
        shape.extend(list(noise))
        if None in shape:
            return None
        else:
            return tf.random.normal(shape=shape)

    @staticmethod
    def VDB_loss(loss_func, out, label, mean, logvar, beta):
        I_c = 0.5
        normal_D_loss = loss_func(out, label)
        kldiv_loss = - 0.5 * K.mean(1 + logvar - K.square(mean) - K.exp(logvar))
        kldiv_loss = kldiv_loss - I_c
        final_loss = normal_D_loss + beta * kldiv_loss
        return final_loss, kldiv_loss

    @staticmethod
    def discriminator_loss(loss_func, real_output, fake_output):
        NOISE_COEF = 0.02
        real_loss = loss_func(tf.ones_like(real_output) * (1 - NOISE_COEF), real_output)
        fake_loss = loss_func(tf.ones_like(fake_output) * NOISE_COEF, fake_output)
        total_loss = real_loss + fake_loss
        return total_loss, real_loss, fake_loss

    @staticmethod
    def vae_discriminator_loss(loss_func, real_output, fake_output, beta):
        alpha = 1e-5
        real_loss, loss_kldiv_real = GANTerraModel.VDB_loss(
            loss_func=loss_func,
            out=real_output[0][0],
            label=tf.ones_like(real_output[0][0]),
            mean=real_output[0][1],
            logvar=real_output[0][2],
            beta=beta
        )
        fake_loss, loss_kldiv_fake = GANTerraModel.VDB_loss(
            loss_func=loss_func,
            out=fake_output[0][0],
            label=tf.zeros_like(fake_output[0][0]),
            mean=fake_output[0][1],
            logvar=fake_output[0][2],
            beta=beta
        )
        total_loss = real_loss + fake_loss
        loss_kldiv = loss_kldiv_real + loss_kldiv_fake
        new_beta = tf.reduce_max([0., beta + alpha * loss_kldiv])
        return total_loss, real_loss, fake_loss, new_beta

    @staticmethod
    def generator_loss(loss_func, fake_output):
        return loss_func(tf.ones_like(fake_output), fake_output)

    @staticmethod
    def set_optimizer(params: TrainingDetailsData):
        method_name = 'set_optimizer'
        try:
            optimizer_object = getattr(keras.optimizers, params.base.optimizer.type)
            parameters = params.base.optimizer.parameters.main.native()
            parameters.update(params.base.optimizer.parameters.extra.native())
            return optimizer_object(**parameters)
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                GANTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            raise exc

    @staticmethod
    def add_noise_to_image(image):
        # image = tf.cast(image, dtype='float32')
        # return tf.add(image, tf.random.normal(image.shape) * 0.02)
        return image

    @tf.function
    def __train_step(self, images, gen_batch, dis_batch, beta, **options):
        # new_beta = tf.convert_to_tensor(0.)
        images = tf.cast(images, dtype='float32')
        noise_shape = [gen_batch]
        noise_shape.extend(list(self.noise))
        noise = tf.random.normal(noise_shape)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            # print('real_output', real_output[0])
            # print('fake_output', fake_output[0][0])

            if self.vae_discriminator:
                gen_loss = self.generator_loss(loss_func=self.generator_loss_func, fake_output=fake_output[0][0])
                disc_loss, disc_real_loss, disc_fake_loss, new_beta = self.vae_discriminator_loss(
                    loss_func=self.discriminator_loss_func,
                    real_output=real_output,
                    fake_output=fake_output,
                    beta=beta
                )
                # print('disc_loss, disc_real_loss, disc_fake_loss, new_beta', disc_loss, disc_real_loss, disc_fake_loss, new_beta)
            else:
                new_beta = tf.convert_to_tensor(0.)
                gen_loss = self.generator_loss(loss_func=self.generator_loss_func, fake_output=fake_output)
                disc_loss, disc_real_loss, disc_fake_loss = self.discriminator_loss(
                    loss_func=self.discriminator_loss_func, real_output=real_output, fake_output=fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss, disc_loss, disc_real_loss, disc_fake_loss, new_beta

    def fit(self, params: TrainingDetailsData, dataset: PrepareDataset):
        method_name = 'fit'
        try:
            self.train_length = len(dataset.dataframe.get('train'))
            self.generator_optimizer = self.set_optimizer(params=params)
            self.discriminator_optimizer = self.set_optimizer(params=params)
            loss_dict = self._prepare_loss_dict(params)
            self.generator_loss_func = loss_dict.get('generator')
            self.discriminator_loss_func = loss_dict.get('discriminator')
            self.set_optimizer(params=params)
            current_epoch = self.callback.last_epoch
            end_epoch = self.callback.total_epochs
            train_data_idxs = np.arange(self.train_length).tolist()
            self.callback.on_train_begin()
            beta = 0

            for epoch in range(current_epoch, end_epoch):
                self.callback.on_epoch_begin()
                current_logs = {"epochs": epoch + 1, 'loss': {}, "metrics": {}}
                cur_step, gen_loss, disc_loss, disc_real_loss, disc_fake_loss = 0, 0, 0, 0, 0
                logger.debug(f"Эпоха {epoch + 1}: обучение на тренировочной выборке...")
                for image_data, _ in dataset.dataset.get('train').shuffle(
                        buffer_size=params.base.batch).batch(params.base.batch):
                    cur_step += 1
                    image = self.add_noise_to_image(image_data.get(self.discriminator.inputs[0].name))
                    # print(tf.reduce_max(image))
                    results = self.__train_step(
                        images=image,
                        gen_batch=params.base.batch,
                        dis_batch=params.base.batch,
                        beta=beta
                    )
                    beta = results[-1].numpy()
                    # print('gen_loss, disc_loss, new_beta', results[0].numpy(), results[1].numpy(), results[-1].numpy())
                    gen_loss += results[0].numpy()
                    disc_loss += results[1].numpy()
                    disc_real_loss += results[2].numpy()
                    disc_fake_loss += results[3].numpy()

                    if interactive.urgent_predict:
                        self.callback.on_train_batch_end(
                            batch=cur_step,
                            arrays={
                                "train": self.generator(self.__prepare_seed(self.noise)).numpy(),
                                "seed": self.generator(self.seed).numpy()
                            }
                        )
                    else:
                        self.callback.on_train_batch_end(batch=cur_step)
                    if self.callback.stop_training:
                        break

                self.save_weights()

                if self.callback.stop_training:
                    break

                current_logs['loss']['gen_loss'] = {'train': gen_loss / cur_step}
                current_logs['loss']['disc_loss'] = {'train': disc_loss / cur_step}
                current_logs['loss']['disc_real_loss'] = {'train': disc_real_loss / cur_step}
                current_logs['loss']['disc_fake_loss'] = {'train': disc_fake_loss / cur_step}

                self.callback.on_epoch_end(
                    epoch=epoch + 1,
                    arrays={"train": self.generator(self.__prepare_seed(self.noise)).numpy(),
                            "seed": self.generator(self.seed)},
                    train_data_idxs=train_data_idxs,
                    logs=current_logs
                )
            self.callback.on_train_end()
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                GANTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            raise exc

    def predict(self, data_array, batch_size: Optional[int]):
        noise_shape = [128]
        noise_shape.extend(list(self.generator.inputs[0].shape[1:]))
        noise = tf.random.normal(noise_shape)
        return self.generator(noise)


class ConditionalGANTerraModel(GANTerraModel):
    name = "ConditionalGANTerraModel"

    def __init__(self, model: dict, model_name: str, model_path: Path, options: PrepareDataset):
        super().__init__(model=model, model_name=model_name, model_path=model_path)
        self.noise = self.get_noise(options)
        self.input_keys = self.get_input_keys(
            generator=self.generator, discriminator=self.discriminator, options=options)
        self.seed: dict = self.__prepare_cgan_seed(self.noise, options)
        pass

    @staticmethod
    def get_noise(options: PrepareDataset):
        method_name = '__get_noise'
        try:
            # logger.debug(f"{ConditionalGANTerraModel.name}, {ConditionalGANTerraModel.__get_noise.__name__}")
            for out in options.data.columns.keys():
                col_name = list(options.data.columns.get(out).keys())[0]
                if options.data.columns.get(out).get(col_name).get('task') == 'Noise':
                    return options.data.columns.get(out).get(col_name).get('shape')
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                ConditionalGANTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def __prepare_cgan_seed(noise, options: PrepareDataset):
        # logger.debug(f"{ConditionalGANTerraModel.name}, {ConditionalGANTerraModel.__prepare_seed.__name__}")
        class_names = []
        for out in options.data.columns:
            col_name = list(options.data.columns.get(out).keys())[0]
            if options.data.columns.get(out).get(col_name).get('task') == 'Classification':
                class_names = options.data.columns.get(out).get(col_name).get('classes_names')
                break
        seed = {}
        # random_idx = list(np.arange(len(class_names)))
        # random.shuffle(random_idx)
        for name in class_names:
            shape = [50]
            shape.extend(noise)
            seed[name] = tf.random.normal(shape=shape)
        # logger.debug(f"seed - {seed}")
        return seed

    @staticmethod
    def get_input_keys(generator, discriminator, options: PrepareDataset) -> dict:
        method_name = '__get_input_keys'
        try:
            keys = {}
            gen_inputs = [inp.name for inp in generator.inputs]
            disc_inputs = [inp.name for inp in discriminator.inputs]
            for out in options.data.columns.keys():
                col_name = list(options.data.columns.get(out).keys())[0]
                if options.data.columns.get(out).get(col_name).get('task') == 'Classification':
                    if f"{out}" in gen_inputs:
                        keys['gen_labels'] = f"{out}"
                    if f"{out}" in disc_inputs:
                        keys['disc_labels'] = f"{out}"
                if options.data.columns.get(out).get(col_name).get('task') == 'Image':
                    keys['image'] = f"{out}"
                if options.data.columns.get(out).get(col_name).get('task') == 'Noise':
                    keys['noise'] = f"{out}"
            logger.debug(f"__get_input_keys - keys - {keys}")
            return keys
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                ConditionalGANTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            raise exc

    @tf.function
    def __train_step(self, images, gen_labels, disc_labels, input_keys: dict, **options):
        # logger.debug(f"{GANTerraModel.name}, {GANTerraModel.__train_step.__name__}")
        images = tf.cast(images, dtype='float32')
        noise_shape = [gen_labels.shape[0]]
        noise_shape.extend(list(self.noise))
        noise = tf.random.normal(noise_shape)
        true_disc_input = {input_keys['image']: images, input_keys['disc_labels']: disc_labels}
        gen_input = {input_keys['noise']: noise, input_keys['gen_labels']: gen_labels}
        # logger.debug(f"gen_input: {gen_input}")
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(gen_input, training=True)
            fake_disc_input = {input_keys['image']: generated_images, input_keys['disc_labels']: disc_labels}

            real_output = self.discriminator(true_disc_input, training=True)
            fake_output = self.discriminator(fake_disc_input, training=True)

            gen_loss = GANTerraModel.generator_loss(
                loss_func=self.generator_loss_func, fake_output=fake_output)
            disc_loss, disc_real_loss, disc_fake_loss = GANTerraModel.discriminator_loss(
                loss_func=self.discriminator_loss_func, real_output=real_output, fake_output=fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss, disc_loss, disc_real_loss, disc_fake_loss

    def fit(self, params: TrainingDetailsData, dataset: PrepareDataset):
        method_name = 'fit'
        try:
            self.train_length = len(dataset.dataframe.get('train'))
            self.generator_optimizer = self.set_optimizer(params=params)
            self.discriminator_optimizer = self.set_optimizer(params=params)
            class_names = []
            for out in dataset.data.columns:
                col_name = list(dataset.data.columns.get(out).keys())[0]
                if dataset.data.columns.get(out).get(col_name).get('task') == 'Classification':
                    class_names = dataset.data.columns.get(out).get(col_name).get('classes_names')
                    break
            loss_dict = self._prepare_loss_dict(params)
            self.generator_loss_func = loss_dict.get('generator')
            self.discriminator_loss_func = loss_dict.get('discriminator')
            self.set_optimizer(params=params)
            current_epoch = self.callback.last_epoch
            end_epoch = self.callback.total_epochs
            train_data_idxs = np.arange(self.train_length).tolist()
            self.callback.on_train_begin()
            for epoch in range(current_epoch, end_epoch):
                self.callback.on_epoch_begin()
                current_logs = {"epochs": epoch + 1, 'loss': {}, "metrics": {}}
                cur_step, gen_loss, disc_loss, disc_real_loss, disc_fake_loss = 0, 0, 0, 0, 0
                for image_data, _ in dataset.dataset.get('train').shuffle(
                        buffer_size=params.base.batch).batch(params.base.batch):
                    cur_step += 1
                    image = GANTerraModel.add_noise_to_image(image_data.get(self.input_keys.get('image')))
                    # logger.debug(f"self.input_keys: {self.input_keys, image_data.keys()}")
                    results = self.__train_step(
                        images=image,
                        gen_labels=image_data.get(self.input_keys.get('gen_labels')),
                        disc_labels=image_data.get(self.input_keys.get('disc_labels')),
                        input_keys=self.input_keys
                    )
                    gen_loss += results[0].numpy()
                    disc_loss += results[1].numpy()
                    disc_real_loss += results[2].numpy()
                    disc_fake_loss += results[3].numpy()

                    if interactive.urgent_predict:
                        logger.debug(f"Эпоха {epoch + 1}: urgent_predict")
                        seed_predict = {}
                        random_predict = {}
                        for i, name in enumerate(class_names):
                            lbl = np.zeros(shape=(self.seed.get(name).shape[0], len(self.seed.keys())))
                            lbl[:, i] = 1
                            lbl = lbl.astype('float32')
                            seed_array_dict = {
                                self.input_keys['noise']: self.seed.get(name),
                                self.input_keys['gen_labels']: lbl
                            }
                            seed_predict[name] = self.generator(seed_array_dict).numpy()
                            random_array_dict = {
                                self.input_keys['noise']: tf.random.normal(shape=self.seed.get(name).shape),
                                self.input_keys['gen_labels']: lbl
                            }
                            random_predict[name] = self.generator(random_array_dict).numpy()
                        self.callback.on_train_batch_end(
                            batch=cur_step,
                            arrays={"train": random_predict, "seed": seed_predict}
                        )
                    else:
                        self.callback.on_train_batch_end(batch=cur_step)
                    if self.callback.stop_training:
                        break

                self.save_weights()

                if self.callback.stop_training:
                    break

                current_logs['loss']['gen_loss'] = {'train': gen_loss / cur_step}
                current_logs['loss']['disc_loss'] = {'train': disc_loss / cur_step}
                current_logs['loss']['disc_real_loss'] = {'train': disc_real_loss / cur_step}
                current_logs['loss']['disc_fake_loss'] = {'train': disc_fake_loss / cur_step}

                seed_predict = {}
                random_predict = {}
                for i, name in enumerate(class_names):
                    lbl = np.zeros(shape=(self.seed.get(name).shape[0], len(self.seed.keys())))
                    lbl[:, i] = 1
                    lbl = lbl.astype('float32')
                    seed_array_dict = {
                        self.input_keys['noise']: self.seed.get(name),
                        self.input_keys['gen_labels']: lbl
                    }
                    seed_predict[name] = self.generator(seed_array_dict).numpy()
                    random_array_dict = {
                        self.input_keys['noise']: tf.random.normal(shape=self.seed.get(name).shape),
                        self.input_keys['gen_labels']: lbl
                    }
                    random_predict[name] = self.generator(random_array_dict).numpy()

                self.callback.on_epoch_end(
                    epoch=epoch + 1,
                    arrays={"train": random_predict, "seed": seed_predict},
                    train_data_idxs=train_data_idxs,
                    logs=current_logs
                )
            self.callback.on_train_end()
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                ConditionalGANTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            raise exc

    def predict(self, data_array, batch_size: Optional[int]):
        bs = 16
        shape = [32 * bs]
        image_size = list(self.generator.outputs[0].shape[1:])
        shape.extend(image_size)
        predict = np.zeros(shape)
        cur_step = 0
        for image_data, _ in data_array.batch(bs).take(32):
            gen_labels = image_data.get(self.input_keys.get('gen_labels'))
            length = gen_labels.shape[0]
            noise_shape = [gen_labels.shape[0]]
            noise_shape.extend(self.noise)
            noise = tf.random.normal(noise_shape)
            gen_input = {self.input_keys['noise']: noise, self.input_keys['gen_labels']: gen_labels}
            predict[cur_step:cur_step + length] = self.generator.predict(gen_input)
            cur_step += length
        return predict


class TextToImageGANTerraModel(ConditionalGANTerraModel):
    name = "TextToImageGANTerraModel"

    def __init__(self, model: dict, model_name: str, model_path: Path, options: PrepareDataset):
        super().__init__(model=model, model_name=model_name, model_path=model_path, options=options)
        self.noise = ConditionalGANTerraModel.get_noise(options)
        self.input_keys = TextToImageGANTerraModel.get_input_keys(
            generator=self.generator, discriminator=self.discriminator, options=options)
        self.seed: dict = self.__prepare_tti_gan_seed(options=options, noise=self.noise)
        pass

    @staticmethod
    def get_input_keys(generator, discriminator, options: PrepareDataset) -> dict:
        method_name = '__get_input_keys'
        try:
            keys = {}
            gen_inputs = [inp.name for inp in generator.inputs]
            disc_inputs = [inp.name for inp in discriminator.inputs]
            for out in options.data.columns.keys():
                col_name = list(options.data.columns.get(out).keys())[0]
                if options.data.columns.get(out).get(col_name).get('task') == 'Text':
                    if f"{out}" in gen_inputs:
                        keys['gen_labels'] = f"{out}"
                    if f"{out}" in disc_inputs:
                        keys['disc_labels'] = f"{out}"
                if options.data.columns.get(out).get(col_name).get('task') == 'Image':
                    keys['image'] = f"{out}"
                if options.data.columns.get(out).get(col_name).get('task') == 'Noise':
                    keys['noise'] = f"{out}"
            logger.debug(f"__get_input_keys - keys - {keys}")
            return keys
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                ConditionalGANTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            raise exc

    @staticmethod
    def __prepare_tti_gan_seed(options: PrepareDataset, noise):
        method_name = '__prepare_tti_gan_seed'
        try:
            # logger.debug(f"{ConditionalGANTerraModel.name}, {ConditionalGANTerraModel.__prepare_seed.__name__}")
            random_idx = np.random.choice(len(options.dataframe.get('train')), 10).tolist()
            shape = [10, 3]
            shape.extend(noise)
            seed = {"noise": tf.random.normal(shape=shape), "indexes": random_idx}
            # logger.debug(f"seed - {seed}")
            return seed
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                TextToImageGANTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @tf.function
    def __train_step(self, images, gen_labels, disc_labels, input_keys: dict, **options):
        # logger.debug(f"{GANTerraModel.name}, {GANTerraModel.__train_step.__name__}")
        images = tf.cast(images, dtype='float32')
        noise_shape = [gen_labels.shape[0]]
        noise_shape.extend(list(self.noise))
        noise = tf.random.normal(noise_shape)
        true_disc_input = {input_keys['image']: images, input_keys['disc_labels']: disc_labels}
        gen_input = {input_keys['noise']: noise, input_keys['gen_labels']: gen_labels}
        # logger.debug(f"gen_input: {gen_input}")
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(gen_input, training=True)
            fake_disc_input = {input_keys['image']: generated_images, input_keys['disc_labels']: disc_labels}

            real_output = self.discriminator(true_disc_input, training=True)
            fake_output = self.discriminator(fake_disc_input, training=True)

            gen_loss = GANTerraModel.generator_loss(loss_func=self.generator_loss_func, fake_output=fake_output)
            disc_loss, disc_real_loss, disc_fake_loss = GANTerraModel.discriminator_loss(
                loss_func=self.discriminator_loss_func, real_output=real_output, fake_output=fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss, disc_loss, disc_real_loss, disc_fake_loss

    def fit(self, params: TrainingDetailsData, dataset: PrepareDataset):
        method_name = 'fit'
        try:
            inp = self.input_keys.get('gen_labels')
            # logger.debug(f"{inp, type(inp), dataset.data.columns.keys()}")
            column = list(dataset.data.columns.get(int(inp)).keys())[0]
            y_true_text = dataset.dataframe.get('train')[column].tolist()
            shape = [len(y_true_text)]
            shape.extend(dataset.data.inputs.get(int(inp)).shape)
            y_true_array = np.zeros(shape=shape)

            self.train_length = len(dataset.dataframe.get('train'))
            self.generator_optimizer = self.set_optimizer(params=params)
            self.discriminator_optimizer = self.set_optimizer(params=params)
            loss_dict = self._prepare_loss_dict(params)
            self.generator_loss_func = loss_dict.get('generator')
            self.discriminator_loss_func = loss_dict.get('discriminator')

            self.set_optimizer(params=params)
            current_epoch = self.callback.last_epoch
            end_epoch = self.callback.total_epochs
            train_data_idxs = np.arange(self.train_length).tolist()
            self.callback.on_train_begin()
            for epoch in range(current_epoch, end_epoch):
                self.callback.on_epoch_begin()
                current_logs = {"epochs": epoch + 1, 'loss': {}, "metrics": {}}
                cur_step, gen_loss, disc_loss, disc_real_loss, disc_fake_loss = 0, 0, 0, 0, 0
                for image_data, _ in dataset.dataset.get('train').batch(params.base.batch):
                    batch_length = image_data.get(self.input_keys.get('gen_labels')).shape[0]
                    y_true_array[cur_step * batch_length: (cur_step + 1) * batch_length] = \
                        image_data.get(self.input_keys.get('gen_labels')).numpy()

                    cur_step += 1
                    image = GANTerraModel.add_noise_to_image(image_data.get(self.input_keys.get('image')))
                    results = self.__train_step(
                        images=image,
                        gen_labels=image_data.get(self.input_keys.get('gen_labels')),
                        disc_labels=image_data.get(self.input_keys.get('disc_labels')),
                        input_keys=self.input_keys
                    )
                    gen_loss += results[0].numpy()
                    disc_loss += results[1].numpy()
                    disc_real_loss += results[2].numpy()
                    disc_fake_loss += results[3].numpy()

                    if interactive.urgent_predict:
                        logger.debug(f"Эпоха {epoch + 1}: urgent_predict")
                        seed_predict = {'text': [], 'predict': [], 'indexes': self.seed['indexes']}
                        random_predict = {'text': [], 'predict': [], 'indexes': []}
                        for i, idx in enumerate(self.seed['indexes']):
                            seed_array_dict = {
                                self.input_keys['noise']: self.seed.get('noise')[i],
                                self.input_keys['gen_labels']:
                                    np.concatenate([np.expand_dims(y_true_array[idx], axis=0)] * 3, axis=0)
                            }
                            seed_predict['predict'].append(self.generator(seed_array_dict).numpy())
                            seed_predict['text'].append(y_true_text[idx])
                            shape = [3]
                            shape.extend(self.noise)
                            random_idx = np.random.randint(len(dataset.dataframe.get('train')))
                            random_array_dict = {
                                self.input_keys['noise']: tf.random.normal(shape=shape),
                                self.input_keys['gen_labels']:
                                    np.concatenate([np.expand_dims(y_true_array[random_idx], axis=0)] * 3, axis=0)
                            }
                            random_predict['text'].append(y_true_text[random_idx])
                            random_predict['indexes'].append(random_idx)
                            random_predict['predict'].append(self.generator(random_array_dict).numpy())

                        self.callback.on_train_batch_end(
                            batch=cur_step, arrays={"train": random_predict, "seed": seed_predict}
                        )
                    else:
                        self.callback.on_train_batch_end(batch=cur_step)
                    if self.callback.stop_training:
                        break

                self.save_weights()

                if self.callback.stop_training:
                    break

                current_logs['loss']['gen_loss'] = {'train': gen_loss / cur_step}
                current_logs['loss']['disc_loss'] = {'train': disc_loss / cur_step}
                current_logs['loss']['disc_real_loss'] = {'train': disc_real_loss / cur_step}
                current_logs['loss']['disc_fake_loss'] = {'train': disc_fake_loss / cur_step}

                seed_predict = {'text': [], 'predict': [], 'indexes': self.seed['indexes']}
                random_predict = {'text': [], 'predict': [], 'indexes': []}
                for i, idx in enumerate(self.seed['indexes']):
                    seed_array_dict = {
                        self.input_keys['noise']: self.seed.get('noise')[i],
                        self.input_keys['gen_labels']:
                            np.concatenate([np.expand_dims(y_true_array[idx], axis=0)] * 3, axis=0)
                    }
                    seed_predict['predict'].append(self.generator(seed_array_dict).numpy())
                    seed_predict['text'].append(y_true_text[idx])

                    shape = [3]
                    shape.extend(self.noise)
                    random_idx = np.random.randint(len(dataset.dataframe.get('train')))
                    random_array_dict = {
                        self.input_keys['noise']: tf.random.normal(shape=shape),
                        self.input_keys['gen_labels']:
                            np.concatenate([np.expand_dims(y_true_array[random_idx], axis=0)] * 3, axis=0)
                    }
                    random_predict['text'].append(y_true_text[random_idx])
                    random_predict['indexes'].append(random_idx)
                    random_predict['predict'].append(self.generator(random_array_dict).numpy())

                self.callback.on_epoch_end(
                    epoch=epoch + 1,
                    arrays={"train": random_predict, "seed": seed_predict},
                    train_data_idxs=train_data_idxs,
                    logs=current_logs
                )
            self.callback.on_train_end()
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                TextToImageGANTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            raise exc

    def predict(self, data_array, batch_size: Optional[int]):
        gen_labels = data_array.get(self.input_keys.get('gen_labels'))
        noise_shape = [gen_labels.shape[0]]
        noise_shape.extend(self.noise)
        noise = tf.random.normal(noise_shape)
        gen_input = {self.input_keys['noise']: noise, self.input_keys['gen_labels']: gen_labels}
        return self.generator(gen_input)


class ImageToImageGANTerraModel(GANTerraModel):
    name = "ImageToImageGANTerraModel"

    def __init__(self, model: dict, model_name: str, model_path: Path, options: PrepareDataset):
        super().__init__(model=model, model_name=model_name, model_path=model_path, options=options)
        self.input_keys = self.__get_input_keys(options)
        self.seed: dict = self.__prepare_i2i_gan_seed(options=options)
        pass

    @staticmethod
    def __prepare_i2i_gan_seed(options: PrepareDataset):
        method_name = '__prepare_tti_gan_seed'
        try:
            random_idx = np.random.choice(len(options.dataframe.get('train')), 10).tolist()
            seed = {"indexes": random_idx}
            return seed
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                ImageToImageGANTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            raise exc

    def __get_input_keys(self, options: PrepareDataset) -> dict:
        method_name = '__get_input_keys'
        try:
            keys = {}
            gen_inputs = [inp.name for inp in self.generator.inputs]
            disc_inputs = [inp.name for inp in self.discriminator.inputs]
            for gen_inp in gen_inputs:
                col_name = list(options.data.columns.get(int(gen_inp)).keys())[0]
                if options.data.columns.get(int(gen_inp)).get(col_name).get('task') == 'Noise':
                    keys['noise'] = f"{gen_inp}"
                if options.data.columns.get(int(gen_inp)).get(col_name).get('task') == 'Image':
                    keys['gen_images'] = f"{gen_inp}"
            for disc_inp in disc_inputs:
                col_name = list(options.data.columns.get(int(disc_inp)).keys())[0]
                if options.data.columns.get(int(disc_inp)).get(col_name).get('task') == 'Image':
                    keys['disc_images'] = f"{disc_inp}"
            logger.debug(f"__get_input_keys - keys - {keys}")
            return keys
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                ImageToImageGANTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            raise exc

    @staticmethod
    def _generator_loss(loss_func, disc_generated_output, gen_output, target):
        LAMBDA = 100
        gan_loss = loss_func(tf.ones_like(disc_generated_output), disc_generated_output)

        # Mean absolute error
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

        total_gen_loss = gan_loss + (LAMBDA * l1_loss)
        # total_gen_loss = gan_loss * LAMBDA

        return total_gen_loss, gan_loss, l1_loss
        # return loss_func(tf.ones_like(fake_output), fake_output)

    @tf.function
    def __train_step(self, gen_array, disc_array, input_keys: dict, **options):
        gen_array = tf.cast(gen_array, dtype='float32')
        disc_array = tf.cast(disc_array, dtype='float32')
        true_disc_input = {input_keys['disc_images']: disc_array}
        gen_input = {input_keys['gen_images']: gen_array}
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(gen_input, training=True)
            fake_disc_input = {input_keys['disc_images']: generated_images}

            real_output = self.discriminator(true_disc_input, training=True)
            fake_output = self.discriminator(fake_disc_input, training=True)

            total_gen_loss, gen_gan_loss, gen_l1_loss = self._generator_loss(
                loss_func=self.generator_loss_func,
                disc_generated_output=fake_output,
                gen_output=generated_images,
                target=disc_array
            )
            # apply_gen_loss = tf.multiply(gen_loss, 100.)
            disc_loss, disc_real_loss, disc_fake_loss = GANTerraModel.discriminator_loss(
                loss_func=self.discriminator_loss_func, real_output=real_output, fake_output=fake_output)
        gradients_of_generator = gen_tape.gradient(total_gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return total_gen_loss, disc_loss, disc_real_loss, disc_fake_loss, generated_images

    def fit(self, params: TrainingDetailsData, dataset: PrepareDataset):
        method_name = 'fit'
        try:
            inp = self.input_keys.get('disc_images')
            self.train_length = len(dataset.dataframe.get('train'))
            shape = [10]
            shape.extend(dataset.data.inputs.get(int(inp)).shape)
            y_random_array = np.zeros(shape=shape)
            y_seed_array = np.zeros(shape=shape)

            self.generator_optimizer = self.set_optimizer(params=params)
            self.discriminator_optimizer = self.set_optimizer(params=params)
            loss_dict = self._prepare_loss_dict(params)
            self.generator_loss_func = loss_dict.get('generator')
            self.discriminator_loss_func = loss_dict.get('discriminator')

            self.set_optimizer(params=params)
            current_epoch = self.callback.last_epoch
            end_epoch = self.callback.total_epochs
            train_data_idxs = np.arange(self.train_length).tolist()
            self.callback.on_train_begin()
            for epoch in range(current_epoch, end_epoch):
                random_idx = list(np.random.randint(0, self.train_length, 10))
                self.callback.on_epoch_begin()
                current_logs = {"epochs": epoch + 1, 'loss': {}, "metrics": {}}
                cur_step, gen_loss, disc_loss, disc_real_loss, disc_fake_loss = 0, 0, 0, 0, 0
                cur_position = 0
                for image_data, _ in dataset.dataset.get('train').batch(params.base.batch):
                    batch_length = image_data.get(self.input_keys.get('gen_images')).shape[0]
                    cur_range = np.arange(cur_position, cur_position + batch_length).tolist()
                    # image = GANTerraModel.add_noise_to_image(image_data.get(self.input_keys.get('disc_images')))
                    results = self.__train_step(
                        # target_array=image_data.get(self.input_keys.get('disc_images')),
                        gen_array=image_data.get(self.input_keys.get('gen_images')),
                        disc_array=image_data.get(self.input_keys.get('disc_images')),
                        input_keys=self.input_keys
                    )
                    gen_loss += results[0].numpy()
                    disc_loss += results[1].numpy()
                    disc_real_loss += results[2].numpy()
                    disc_fake_loss += results[3].numpy()
                    for i, idx in enumerate(self.seed['indexes']):
                        if idx in cur_range:
                            idx_position = self.seed['indexes'].index(idx)
                            array_position = cur_range.index(idx)
                            y_seed_array[idx_position] = results[4][array_position].numpy()
                        if random_idx[i] in cur_range:
                            idx_position = random_idx.index(random_idx[i])
                            array_position = cur_range.index(random_idx[i])
                            y_random_array[idx_position] = results[4][array_position].numpy()

                    # y_pred_array[cur_step * batch_length: (cur_step + 1) * batch_length] = results[4].numpy()
                    cur_step += 1
                    cur_position += batch_length

                    if interactive.urgent_predict:
                        logger.debug(f"Эпоха {epoch + 1}: urgent_predict")
                        seed_predict = {'predict': y_seed_array, 'indexes': self.seed['indexes']}
                        random_predict = {'predict': y_random_array, 'indexes': random_idx}

                        self.callback.on_train_batch_end(
                            batch=cur_step,
                            arrays={"train": random_predict, "seed": seed_predict, 'inputs': self.input_keys}
                        )
                    else:
                        self.callback.on_train_batch_end(batch=cur_step)
                    if self.callback.stop_training:
                        break

                self.save_weights()

                if self.callback.stop_training:
                    break

                current_logs['loss']['gen_loss'] = {'train': gen_loss / cur_step}
                current_logs['loss']['disc_loss'] = {'train': disc_loss / cur_step}
                current_logs['loss']['disc_real_loss'] = {'train': disc_real_loss / cur_step}
                current_logs['loss']['disc_fake_loss'] = {'train': disc_fake_loss / cur_step}

                seed_predict = {'predict': y_seed_array, 'indexes': self.seed['indexes']}
                random_predict = {'predict': y_random_array, 'indexes': random_idx}
                self.callback.on_epoch_end(
                    epoch=epoch + 1,
                    arrays={"train": random_predict, "seed": seed_predict, 'inputs': self.input_keys},
                    train_data_idxs=train_data_idxs,
                    logs=current_logs
                )
            self.callback.on_train_end()
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                ImageToImageGANTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            raise exc

    def predict(self, data_array, batch_size: Optional[int]):
        gen_array = data_array.get(self.input_keys.get('gen_images'))
        gen_input = {self.input_keys['gen_images']: gen_array}
        return self.generator(gen_input)


class ImageSRGANTerraModel(GANTerraModel):
    name = "ImageSRGANTerraModel"

    def __init__(self, model: dict, model_name: str, model_path: Path, options: PrepareDataset):
        super().__init__(model=model, model_name=model_name, model_path=model_path, options=options)
        self.input_keys = self.__get_input_keys(options)
        self.full_lr_image_path = os.path.join(options.data.path, "LR")
        self.full_lr_image_list = []
        with os.scandir(self.full_lr_image_path) as files:
            for f in files:
                self.full_lr_image_list.append(os.path.join(self.full_lr_image_path, f.name))
        self.full_lr_image_list = sorted(self.full_lr_image_list)
        logger.debug(f"self.full_lr_image_list: {self.full_lr_image_list[:5]}")
        self.seed: dict = self.__prepare_srgan_seed(name_list=self.full_lr_image_list)
        logger.debug(f"self.seed: {self.seed['indexes']}")
        self.mean_squared_error = tf.keras.losses.MeanSquaredError()
        self.vgg = self._vgg()
        pass

    @staticmethod
    def _vgg():
        vgg = VGG19(input_shape=(None, None, 3), include_top=False)
        return Model(vgg.input, vgg.layers[20].output)

    @staticmethod
    def __prepare_srgan_seed(name_list: list):
        return {"indexes": np.random.choice(len(name_list), 10).tolist()}

    def __get_input_keys(self, options: PrepareDataset) -> dict:
        method_name = '__get_input_keys'
        try:
            keys = {}
            gen_inputs = [inp.name for inp in self.generator.inputs]
            disc_inputs = [inp.name for inp in self.discriminator.inputs]
            for gen_inp in gen_inputs:
                col_name = list(options.data.columns.get(int(gen_inp)).keys())[0]
                if options.data.columns.get(int(gen_inp)).get(col_name).get('task') == 'Noise':
                    keys['noise'] = f"{gen_inp}"
                if options.data.columns.get(int(gen_inp)).get(col_name).get('task') == 'Image':
                    keys['gen_images'] = f"{gen_inp}"
            for disc_inp in disc_inputs:
                col_name = list(options.data.columns.get(int(disc_inp)).keys())[0]
                if options.data.columns.get(int(disc_inp)).get(col_name).get('task') == 'Image':
                    keys['disc_images'] = f"{disc_inp}"
            logger.debug(f"__get_input_keys - keys - {keys}")
            return keys
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                ImageSRGANTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            raise exc

    @staticmethod
    def _pretrain_generator_loss(loss_func, image_true, image_pred):
        return loss_func(image_true, image_pred)

    @staticmethod
    def _generator_loss(loss_func, disc_output):
        return loss_func(tf.ones_like(disc_output), disc_output)

    @tf.function
    def _content_loss(self, true_image, gen_image):
        gen_image = preprocess_input(gen_image)
        true_image = preprocess_input(true_image)
        sr_features = self.vgg(gen_image) / 12.75
        hr_features = self.vgg(true_image) / 12.75
        return self.mean_squared_error(hr_features, sr_features)

    @tf.function
    def __pretrain_step(self, lr_array, hr_array):
        lr_array = tf.cast(lr_array, dtype='float32')
        hr_array = tf.cast(hr_array, dtype='float32')
        with tf.GradientTape() as gen_tape:
            gen_array = self.generator(lr_array, training=True)
            gen_loss = self._pretrain_generator_loss(
                loss_func=self.mean_squared_error,
                image_true=hr_array,
                image_pred=gen_array,
            )
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        return gen_loss

    @tf.function
    def __train_step(self, gen_array, disc_array, input_keys: dict, **options):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_array = tf.cast(gen_array, dtype='float32')
            disc_array = tf.cast(disc_array, dtype='float32')
            true_disc_input = {input_keys['disc_images']: disc_array}
            gen_input = {input_keys['gen_images']: gen_array}

            generated_images = self.generator(gen_input, training=True)
            fake_disc_input = {input_keys['disc_images']: generated_images}
            real_output = self.discriminator(true_disc_input, training=True)
            fake_output = self.discriminator(fake_disc_input, training=True)
            pretrain_loss = self._pretrain_generator_loss(self.mean_squared_error, disc_array, generated_images)

            content_loss = self._content_loss(disc_array, generated_images)
            gen_loss = self._generator_loss(self.generator_loss_func, fake_output)
            perception_loss = content_loss + 0.001 * gen_loss
            disc_loss, disc_real_loss, disc_fake_loss = GANTerraModel.discriminator_loss(
                loss_func=self.discriminator_loss_func, real_output=real_output, fake_output=fake_output)

        gradients_of_generator = gen_tape.gradient(perception_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return perception_loss, content_loss, gen_loss, disc_loss, disc_real_loss, disc_fake_loss, pretrain_loss

    def fit(self, params: TrainingDetailsData, dataset: PrepareDataset):
        method_name = 'fit'
        try:
            self.train_length = len(dataset.dataframe.get('train'))
            y_seed_array = {}
            self.generator_optimizer = self.set_optimizer(params=params)
            self.discriminator_optimizer = self.set_optimizer(params=params)
            current_logs = {
                'loss': {
                    'pretrain_loss': {'train': None},
                    'perception_loss': {'train': None},
                    'content_loss': {'train': None},
                    'gen_loss': {'train': None},
                    'disc_loss': {'train': None},
                    'disc_real_loss': {'train': None},
                    'disc_fake_loss': {'train': None}
                },
                "metrics": {}
            }
            if params.state.status != "addtrain":
                pretrain_epochs = 25
                # self.callback.total_epochs += pretrain_epochs
                self.callback.on_train_begin()
                for epoch in range(pretrain_epochs):
                    self.callback.on_epoch_begin()
                    current_logs["epochs"] = epoch + 1
                    # current_logs = {"epochs": epoch + 1, 'loss': {}, "metrics": {}}
                    cur_step, gen_loss, cur_position = 0, 0, 0
                    random_idx = np.random.choice(len(self.full_lr_image_list), 10).tolist()
                    y_random_array = {}
                    for image_data, _ in dataset.dataset.get('train').batch(params.base.batch):
                        batch_length = image_data.get(self.input_keys.get('gen_images')).shape[0]
                        results = self.__pretrain_step(
                            lr_array=image_data.get(self.input_keys.get('gen_images')),
                            hr_array=image_data.get(self.input_keys.get('disc_images')),
                        )
                        # logger.debug(f"Batch {cur_step + 1}, pretrain_loss={results.numpy()}")
                        gen_loss += results.numpy()
                        cur_step += 1
                        cur_position += batch_length

                        if interactive.urgent_predict:
                            for idx in self.seed['indexes']:
                                print(idx)
                                img = Image.open(self.full_lr_image_list[idx])
                                img = tf.keras.preprocessing.image.img_to_array(img)
                                if img.shape[-1] == 4:
                                    img = img[..., :-1]
                                y_seed_array[idx] = self.generator(tf.expand_dims(img, axis=0)).numpy()
                            for idx in random_idx:
                                img = Image.open(self.full_lr_image_list[idx])
                                img = tf.keras.preprocessing.image.img_to_array(img)
                                if img.shape[-1] == 4:
                                    img = img[..., :-1]
                                y_random_array[idx] = self.generator(tf.expand_dims(img, axis=0)).numpy()
                            seed_predict = {'predict': y_seed_array, 'indexes': self.seed['indexes']}
                            random_predict = {'predict': y_random_array, 'indexes': random_idx}
                            self.callback.on_train_batch_end(
                                batch=cur_step,
                                arrays={"train": random_predict, "seed": seed_predict, 'inputs': self.input_keys}
                            )
                        else:
                            self.callback.on_train_batch_end(batch=cur_step)
                        if self.callback.stop_training:
                            break
                    if self.callback.stop_training:
                        break
                    for idx in self.seed['indexes']:
                        img = Image.open(self.full_lr_image_list[idx])
                        img = tf.keras.preprocessing.image.img_to_array(img)
                        if img.shape[-1] == 4:
                            img = img[..., :-1]
                        y_seed_array[idx] = self.generator(tf.expand_dims(img, axis=0)).numpy()
                        print('predict', self.generator(tf.expand_dims(img, axis=0)).numpy()[0][0][0])
                    for idx in random_idx:
                        img = Image.open(self.full_lr_image_list[idx])
                        img = tf.keras.preprocessing.image.img_to_array(img)
                        if img.shape[-1] == 4:
                            img = img[..., :-1]
                        y_random_array[idx] = self.generator(tf.expand_dims(img, axis=0)).numpy()
                    self.save_weights()

                    current_logs['loss']['pretrain_loss']['train'] = gen_loss / cur_step
                    seed_predict = {'predict': y_seed_array, 'indexes': self.seed['indexes']}
                    random_predict = {'predict': y_random_array, 'indexes': random_idx}
                    self.callback.on_epoch_end(
                        epoch=epoch + 1,
                        arrays={"train": random_predict, "seed": seed_predict, 'inputs': self.input_keys},
                        logs=current_logs
                    )

            loss_dict = self._prepare_loss_dict(params)
            self.generator_loss_func = loss_dict.get('generator')
            self.discriminator_loss_func = loss_dict.get('discriminator')
            self.set_optimizer(params=params)
            current_epoch = self.callback.last_epoch
            end_epoch = self.callback.total_epochs
            self.callback.on_train_begin()
            for epoch in range(current_epoch, end_epoch):
                self.callback.on_epoch_begin()
                current_logs["epochs"] = epoch + 1
                # current_logs = {"epochs": epoch + 1, 'loss': {}, "metrics": {}}
                cur_step, perception_loss, content_loss, gen_loss, \
                    disc_loss, disc_real_loss, disc_fake_loss, pretrain_loss = 0, 0, 0, 0, 0, 0, 0, 0
                cur_position = 0
                random_idx = np.random.choice(len(self.full_lr_image_list), 10).tolist()
                y_random_array = {}
                for image_data, _ in dataset.dataset.get('train').batch(params.base.batch):
                    batch_length = image_data.get(self.input_keys.get('gen_images')).shape[0]
                    results = self.__train_step(
                        gen_array=image_data.get(self.input_keys.get('gen_images')),
                        disc_array=image_data.get(self.input_keys.get('disc_images')),
                        input_keys=self.input_keys
                    )

                    perception_loss += results[0].numpy()
                    content_loss += results[1].numpy()
                    gen_loss += results[2].numpy()
                    disc_loss += results[3].numpy()
                    disc_real_loss += results[4].numpy()
                    disc_fake_loss += results[5].numpy()
                    pretrain_loss += results[6].numpy()
                    cur_step += 1
                    cur_position += batch_length

                    if interactive.urgent_predict:
                        for idx in self.seed['indexes']:
                            img = Image.open(self.full_lr_image_list[idx])
                            img = tf.keras.preprocessing.image.img_to_array(img)
                            if img.shape[-1] == 4:
                                img = img[..., :-1]
                            y_seed_array[idx] = self.generator(tf.expand_dims(img, axis=0)).numpy()
                        for idx in random_idx:
                            img = Image.open(self.full_lr_image_list[idx])
                            img = tf.keras.preprocessing.image.img_to_array(img)
                            if img.shape[-1] == 4:
                                img = img[..., :-1]
                            y_random_array[idx] = self.generator(tf.expand_dims(img, axis=0)).numpy()
                        seed_predict = {'predict': y_seed_array, 'indexes': self.seed['indexes']}
                        random_predict = {'predict': y_random_array, 'indexes': random_idx}
                        self.callback.on_train_batch_end(
                            batch=cur_step,
                            arrays={"train": random_predict, "seed": seed_predict, 'inputs': self.input_keys}
                        )
                    else:
                        self.callback.on_train_batch_end(batch=cur_step)
                    if self.callback.stop_training:
                        break
                self.save_weights()
                if self.callback.stop_training:
                    break
                for idx in self.seed['indexes']:
                    img = Image.open(self.full_lr_image_list[idx])
                    img = tf.keras.preprocessing.image.img_to_array(img)
                    if img.shape[-1] == 4:
                        img = img[..., :-1]
                    y_seed_array[idx] = self.generator(tf.expand_dims(img, axis=0)).numpy()
                    print('predict', self.generator(tf.expand_dims(img, axis=0)).numpy()[0][0][0])
                for idx in random_idx:
                    img = Image.open(self.full_lr_image_list[idx])
                    img = tf.keras.preprocessing.image.img_to_array(img)
                    if img.shape[-1] == 4:
                        img = img[..., :-1]
                    y_random_array[idx] = self.generator(tf.expand_dims(img, axis=0)).numpy()

                current_logs['loss']['pretrain_loss']['train'] = pretrain_loss / cur_step
                current_logs['loss']['perception_loss']['train'] = perception_loss / cur_step
                current_logs['loss']['content_loss']['train'] = content_loss / cur_step
                current_logs['loss']['gen_loss']['train'] = gen_loss / cur_step
                current_logs['loss']['disc_loss']['train'] = disc_loss / cur_step
                current_logs['loss']['disc_real_loss']['train'] = disc_real_loss / cur_step
                current_logs['loss']['disc_fake_loss']['train'] = disc_fake_loss / cur_step

                seed_predict = {'predict': y_seed_array, 'indexes': self.seed['indexes']}
                random_predict = {'predict': y_random_array, 'indexes': random_idx}
                self.callback.on_epoch_end(
                    epoch=epoch + 1,
                    arrays={"train": random_predict, "seed": seed_predict, 'inputs': self.input_keys},
                    logs=current_logs
                )
            self.callback.on_train_end()
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                ImageSRGANTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            raise exc

    def predict(self, data_array, batch_size: Optional[int]):
        gen_array = data_array.get(self.input_keys.get('gen_images'))
        gen_input = {self.input_keys['gen_images']: gen_array}
        return self.generator(gen_input)
