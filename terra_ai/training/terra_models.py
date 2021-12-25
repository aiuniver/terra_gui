import importlib
import json
import os
import numpy as np

from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Model

from terra_ai.callbacks import interactive
from terra_ai.callbacks.utils import loss_metric_config, get_dataset_length
from terra_ai.custom_objects.customLayers import terra_custom_layers
from terra_ai.data.training.train import TrainingDetailsData
from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.logging import logger
from terra_ai.training.yolo_utils import decode, compute_loss, get_mAP
import terra_ai.exceptions.callbacks as exception


class BaseTerraModel:
    name = "BaseTerraModel"

    def __init__(self, model, model_name: str, model_path: Path):

        self.model_name = model_name
        self.model_json = f"{model_name}_json.trm"
        self.custom_obj_json = f"{model_name}_custom_obj_json.trm"
        self.model_weights = f"{model_name}_weights.h5"
        self.model_best_weights = f"{model_name}_best_weights.h5"

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

        """
        Saving last model on each epoch end

        Returns:
            None
        """
        try:
            self.__save_model_to_json()
            self.__save_custom_objects_to_json()
            self.save_weights()
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
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
            # logger.error(exc)
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
            # logger.error(exc)
            raise exc

    @tf.function
    def __train_step(self, x_batch, y_batch, losses: dict, set_optimizer):
        """
        losses = {'2': loss_fn}
        """
        with tf.GradientTape() as tape:
            logits_ = self.base_model(x_batch, training=True)
            y_true_ = list(y_batch.values())
            if not isinstance(logits_, list):
                loss_fn = losses.get(list(losses.keys())[0])
                total_loss = loss_fn(y_true_[0], logits_)
            else:
                total_loss = tf.convert_to_tensor(0.)
                for k, key in enumerate(losses.keys()):
                    loss_fn = losses[key]
                    total_loss = tf.add(loss_fn(y_true_[k], logits_[k]), total_loss)
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
            for out in output_list:
                train_target_shape, val_target_shape = [self.train_length], [self.val_length]
                train_target_shape.extend(list(dataset.data.outputs.get(out).shape))
                val_target_shape.extend(list(dataset.data.outputs.get(out).shape))
                train_pred[f"{out}"] = np.zeros(train_target_shape).astype('float32')
                train_true[f"{out}"] = np.zeros(train_target_shape).astype('float32')
                val_pred[f"{out}"] = np.zeros(val_target_shape).astype('float32')
                val_true[f"{out}"] = np.zeros(val_target_shape).astype('float32')

            train_data_idxs = np.arange(self.train_length).tolist()
            self.callback.on_train_begin()
            for epoch in range(current_epoch, end_epoch):
                logger.debug(f"Эпоха {epoch+1}")
                self.callback.on_epoch_begin()
                train_steps = 0
                current_idx = 0
                logger.debug(f"Эпоха {epoch + 1}: обучение на тренировочной выборке...")
                for x_batch_train, y_batch_train in dataset.dataset.get('train').batch(params.base.batch):
                    logits, y_true = self.__train_step(
                        x_batch=x_batch_train, y_batch=y_batch_train,
                        losses=loss, set_optimizer=self.optimizer
                    )
                    length = logits[0].shape[0]
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

                logger.debug(f"Эпоха {epoch + 1}: сохранение весов текущей эпохи...")
                self.save_weights()
                if self.callback.stop_training:
                    logger.info(f"Эпоха {epoch + 1}: остановка обучения", extra={"type": "info"})
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

                self.callback.on_epoch_end(
                    epoch=epoch + 1,
                    arrays={
                        "train_pred": train_pred, "val_pred": val_pred, "train_true": train_true, "val_true": val_true
                    },
                    train_data_idxs=train_data_idxs
                )

                if self.callback.is_best():
                    self.save_weights(path_=self.file_path_model_best_weights)
                    logger.info("Веса лучшей эпохи успешно сохранены", extra={"type": "info"})
            self.callback.on_train_end()
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc


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
            # logger.error(exc)
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
                train_target_shape.extend(list(array.shape[1:]))
                val_target_shape.extend(list(array.shape[1:]))
                train_pred.append(np.zeros(train_target_shape))
                train_true.append(np.zeros(train_target_shape))
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
                current_idx = 0
                cur_step, giou_train, conf_train, prob_train, total_train = 0, 0, 0, 0, 0
                logger.debug(f"Эпоха {epoch + 1}: обучение на тренировочной выборке...")
                for image_data, target1, target2 in dataset.dataset.get('train').batch(params.base.batch):
                    results = self.__train_step(image_data,
                                                target1,
                                                target2,
                                                global_steps=global_steps,
                                                **yolo_parameters)

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
                        logger.debug(f"Эпоха {epoch + 1}: urgent_predict, обработка проверочной выборки...")
                        val_steps = 0
                        val_current_idx = 0
                        for val_image_data, val_target1, val_target2 in dataset.dataset.get('val').batch(
                                params.base.batch):
                            results = self.__validate_step(val_image_data,
                                                           target1,
                                                           target2,
                                                           **yolo_parameters)
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

                logger.debug(f"Эпоха {epoch + 1}: сохраниеиние весов текущей эпохи...")
                self.save_weights()
                if self.callback.stop_training:
                    logger.info(f"Эпоха {epoch + 1}: остановка обучения", extra={"type": "success"})
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
                    results = self.__validate_step(image_data,
                                                   target1,
                                                   target2,
                                                   **yolo_parameters)
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
                    logger.info("Веса лучшей эпохи успешно сохранены", extra={"type": "success"})
            self.callback.on_train_end()
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                YoloTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc


class GANTerraModel(BaseTerraModel):
    name = "GANTerraModel"

    def __init__(self, model: dict, model_name: str, model_path: Path, **options):
        logger.debug(f"{GANTerraModel.name} is started")
        super().__init__(model=model, model_name=model_name, model_path=model_path)
        logger.debug(f'model: {model}')
        self.saving_path = model_path
        self.generator: Model = model.get('generator')
        self.discriminator: Model = model.get('discriminator')
        self.file_path_gen_json = os.path.join(self.saving_path, "generator_json.trm")
        self.file_path_disc_json = os.path.join(self.saving_path, "discriminator_json.trm")
        self.generator_json = self.generator.to_json()
        self.discriminator_json = self.discriminator.to_json()
        self.noise = self.generator.inputs[0].shape[1:]
        logger.debug(f'self.noise: {self.noise}')
        self.generator_loss_func = None
        self.discriminator_loss_func = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None
        self.seed = self.__prepare_seed(self.noise)

        self.generator_weights = "generator_weights.h5"
        self.file_path_gen_weights = os.path.join(self.saving_path, self.generator_weights)
        self.discriminator_weights = "discriminator_weights.h5"
        self.file_path_disc_weights = os.path.join(self.saving_path, self.discriminator_weights)
        pass

    def save(self) -> None:
        logger.debug(f"{GANTerraModel.name}, {GANTerraModel.save.__name__}")
        method_name = 'save'
        try:
            self.__save_model_to_json()
            self.__save_custom_objects_to_json()
            self.save_weights()
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                GANTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    def load(self) -> None:
        logger.debug(f"{GANTerraModel.name}, {GANTerraModel.load.__name__}")
        gen_model_data, disc_model_data, custom_dict = self.__get_json_data()
        custom_object = self.__set_custom_objects(custom_dict)
        self.generator = tf.keras.models.model_from_json(gen_model_data, custom_objects=custom_object)
        self.discriminator = tf.keras.models.model_from_json(gen_model_data, custom_objects=custom_object)
        self.generator_json = self.generator.to_json()
        self.discriminator_json = self.discriminator.to_json()

    def save_weights(self, gw_path_=None, dw_path_=None):
        logger.debug(f"{GANTerraModel.name}, {GANTerraModel.save_weights.__name__}")
        if not gw_path_:
            gw_path_ = os.path.join(self.saving_path, self.generator_weights)
        self.generator.save_weights(gw_path_)
        if not dw_path_:
            dw_path_ = os.path.join(self.saving_path, self.discriminator_weights)
        self.discriminator.save_weights(dw_path_)

    def load_weights(self):
        logger.debug(f"{GANTerraModel.name}, {GANTerraModel.load_weights.__name__}")
        self.generator.load_weights(self.file_path_gen_weights)
        self.discriminator.load_weights(self.file_path_disc_weights)

    @staticmethod
    def _prepare_loss_dict(params: TrainingDetailsData):
        method_name = '_prepare_loss_dict'
        try:
            loss_dict = {}
            logger.debug(f"params.base.architecture.parameters.outputs\n {params.base.architecture.parameters.outputs}")
            for output_layer in params.base.architecture.parameters.outputs:
                loss_obj = getattr(
                    importlib.import_module(
                        loss_metric_config.get("loss").get(output_layer.loss.name, {}).get('module')),
                    output_layer.loss.name
                )(from_logits=True)
                loss_dict.update({str(output_layer.task.name).lower(): loss_obj})
            return loss_dict
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                GANTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    def __save_model_to_json(self):
        logger.debug(f"{GANTerraModel.name}, {GANTerraModel.__save_model_to_json.__name__}")
        with open(self.file_path_gen_json, "w", encoding="utf-8") as json_file:
            json.dump(self.generator_json, json_file)

        with open(self.file_path_disc_json, "w", encoding="utf-8") as json_file:
            json.dump(self.discriminator_json, json_file)

    def __save_custom_objects_to_json(self):
        with open(self.file_path_custom_obj_json, "w", encoding="utf-8") as json_file:
            json.dump(terra_custom_layers, json_file)

    def __get_json_data(self):
        logger.debug(f"{GANTerraModel.name}, {GANTerraModel.__get_json_data.__name__}")
        with open(self.file_path_gen_json) as json_file:
            gen_data = json.load(json_file)

        with open(self.file_path_disc_json) as json_file:
            disc_data = json.load(json_file)

        with open(self.file_path_custom_obj_json) as json_file:
            custom_dict = json.load(json_file)

        return gen_data, disc_data, custom_dict

    @staticmethod
    def __prepare_seed(noise):
        logger.debug(f"{GANTerraModel.name}, {GANTerraModel.__prepare_seed.__name__}")
        shape = [50]
        shape.extend(list(noise))
        return tf.random.normal(shape=shape)

    @staticmethod
    def __discriminator_loss(loss_func, real_output, fake_output):
        # logger.debug(f"{GANTerraModel.name}, {GANTerraModel.__discriminator_loss.__name__}")
        real_loss = loss_func(tf.ones_like(real_output), real_output)
        fake_loss = loss_func(tf.zeros_like(fake_output), fake_output)
        total_loss = (real_loss + fake_loss) / 2
        return total_loss, real_loss, fake_loss

    @staticmethod
    def __generator_loss(loss_func, fake_output):
        # logger.debug(f'__generator_loss loss_func {loss_func}')
        # logger.debug(f"{GANTerraModel.name}, {GANTerraModel.__generator_loss.__name__}")
        return loss_func(tf.ones_like(fake_output), fake_output)

    @staticmethod
    def __gradient_penalty(batch_size, real_images, fake_images, discriminator):
        logger.debug(f"{GANTerraModel.name}, {GANTerraModel.__gradient_penalty.__name__}")
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        if real_images.shape[0] > fake_images.shape[0]:
            while real_images.shape[0] > fake_images.shape[0]:
                fake_images = tf.concat([fake_images, fake_images], axis=0)
        if real_images.shape[0] <= fake_images.shape[0]:
            fake_images = fake_images[:real_images.shape[0]]

        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = discriminator(interpolated)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def set_optimizer(self, params: TrainingDetailsData):
        logger.debug(f"{GANTerraModel.name}, {GANTerraModel.set_optimizer.__name__}")
        method_name = 'set_optimizer'
        try:
            optimizer_object = getattr(keras.optimizers, params.base.optimizer.type)
            parameters = params.base.optimizer.parameters.main.native()
            parameters.update(params.base.optimizer.parameters.extra.native())
            self.generator_optimizer = optimizer_object(**parameters)
            self.discriminator_optimizer = optimizer_object(**parameters)
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                GANTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @tf.function
    def __train_step(self, images, gen_batch, dis_batch, grad_penalty=False, gp_weight=1, **options):
        logger.debug(f"{GANTerraModel.name}, {GANTerraModel.__train_step.__name__}")
        images = tf.cast(images, dtype='float32')
        noise_shape = [gen_batch]
        noise_shape.extend(list(self.noise))
        noise = tf.random.normal(noise_shape)

        # gp_weight = tf.convert_to_tensor(gp_weight, dtype='float32')
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = self.__generator_loss(loss_func=self.generator_loss_func, fake_output=fake_output)
            disc_loss, disc_real_loss, disc_fake_loss = self.__discriminator_loss(
                loss_func=self.discriminator_loss_func, real_output=real_output, fake_output=fake_output)
            # if grad_penalty:
            #     gp = self.__gradient_penalty(
            #         batch_size=dis_batch, real_images=images, fake_images=generated_images,
            #         discriminator=self.discriminator)
            #     disc_loss = tf.add(disc_loss, gp * gp_weight)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        # disc_loss = tf.convert_to_tensor(disc_loss, dtype='float32')
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return generated_images, gen_loss, disc_loss, disc_real_loss, disc_fake_loss

    def fit(self, params: TrainingDetailsData, dataset: PrepareDataset):
        logger.debug(f"{GANTerraModel.name}, {GANTerraModel.fit.__name__}")
        method_name = 'fit'
        try:
            self.train_length = len(dataset.dataframe.get('train'))
            # yolo_parameters = self.__create_yolo_parameters(params=params, dataset=dataset)
            # num_class = yolo_parameters.get("parameters").get("num_class")
            # classes = yolo_parameters.get("parameters").get("classes")
            # global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
            loss_dict = self._prepare_loss_dict(params)
            self.generator_loss_func = loss_dict.get('generator')
            self.discriminator_loss_func = loss_dict.get('discriminator')
            logger.debug(f'loss_dict - {loss_dict}')

            self.set_optimizer(params=params)

            current_epoch = self.callback.last_epoch
            end_epoch = self.callback.total_epochs
            num_batches = self.train_length if self.train_length % params.base.batch == 0 \
                else (self.train_length // params.base.batch + 1) * params.base.batch
            target_shape = [num_batches]
            # target_shape, seed_shape = [params.base.batch * 10], [10]
            target_shape.extend(list(self.generator.outputs[0].shape[1:]))
            # seed_shape.extend(list(self.generator.outputs[0].shape[1:]))
            train_pred = np.zeros(target_shape).astype('float32')
            # seed_pred = np.zeros(seed_shape).astype('float32')

            train_data_idxs = np.arange(self.train_length).tolist()
            self.callback.on_train_begin()
            for epoch in range(current_epoch, end_epoch):
                # logger.debug(f"Эпоха {epoch + 1}")
                self.callback.on_epoch_begin()
                current_logs = {"epochs": epoch + 1, 'loss': {}, "metrics": {}}
                current_idx = 0
                cur_step, gen_loss, disc_loss, disc_real_loss, disc_fake_loss = 1, 0, 0, 0, 0
                logger.debug(f"Эпоха {epoch + 1}: обучение на тренировочной выборке...")
                for image_data, _ in dataset.dataset.get('train').batch(params.base.batch):
                    logger.debug(f"Batch {cur_step}: start...")
                    results = self.__train_step(images=image_data.get(self.discriminator.inputs[0].name),
                                                gen_batch=params.base.batch,
                                                dis_batch=params.base.batch)
                    # generated_images, gen_loss, disc_loss, disc_real_loss, disc_fake_loss
                    gen_loss += results[1].numpy()
                    disc_loss += results[2].numpy()
                    disc_real_loss += results[3].numpy()
                    disc_fake_loss += results[4].numpy()
                    logger.debug(f"Batch {cur_step}: "
                                 f"gen_loss={round(gen_loss / cur_step, 3)}, "
                                 f"disc_loss={round(disc_loss / cur_step, 3)}, "
                                 f"disc_real_loss={round(disc_real_loss / cur_step, 3)}, "
                                 f"disc_fake_loss={round(disc_fake_loss / cur_step, 3)}")

                    length = results[0].shape[0]
                    # for i in range(len(train_pred)):
                    train_pred[current_idx: current_idx + length] = results[0].numpy()
                    # logger.debug(f"Batch {cur_step}: finish add array")
                    current_idx += length
                    # if cur_step == 10:
                    #     break
                    cur_step += 1
                    if interactive.urgent_predict:
                        logger.debug(f"Эпоха {epoch + 1}: urgent_predict")
                        self.callback.on_train_batch_end(
                            batch=cur_step,
                            arrays={
                                "train": train_pred,
                                "seed": self.generator(self.seed).numpy()
                            }
                        )
                    else:
                        self.callback.on_train_batch_end(batch=cur_step - 1)
                    if self.callback.stop_training:
                        break

                logger.info(f"Эпоха {epoch + 1}: сохраниеиние весов текущей эпохи...", extra={"type": "info"})
                self.save_weights()
                if self.callback.stop_training:
                    logger.info(f"Эпоха {epoch + 1}: остановка обучения", extra={"type": "success"})
                    break

                current_logs['loss']['gen_loss'] = {'train': gen_loss / cur_step}
                current_logs['loss']['disc_loss'] = {'train': disc_loss / cur_step}
                current_logs['loss']['disc_real_loss'] = {'train': disc_real_loss / cur_step}
                current_logs['loss']['disc_fake_loss'] = {'train': disc_fake_loss / cur_step}
                # current_logs['class_loss']['prob_loss'] = {}

                self.callback.on_epoch_end(
                    epoch=epoch + 1,
                    arrays={"train": train_pred, "seed": self.generator(self.seed)},
                    train_data_idxs=train_data_idxs,
                    logs=current_logs
                )

                # if self.callback.is_best():
                #     self.save_weights(path_=self.file_path_model_best_weights)
                #     logger.info("Веса лучшей эпохи успешно сохранены", extra={"front_level": "success"})
            self.callback.on_train_end()
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                GANTerraModel.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc
