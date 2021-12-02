import importlib
import json
import os
import numpy as np

from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from terra_ai.callbacks import interactive
from terra_ai.callbacks.utils import print_error, loss_metric_config, get_dataset_length
from terra_ai.customLayers import terra_custom_layers
from terra_ai.data.training.train import TrainingDetailsData
from terra_ai.datasets.preparing import PrepareDataset
from terra_ai.training.yolo_utils import decode, compute_loss, get_mAP


class BaseTerraModel:

    def __init__(self, model, model_name: str, model_path: Path):
        self.base_model = model
        self.json_model = self.base_model.to_json() if model else None

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

        self.callback = None
        self.optimizer = None

        self.train_length, self.val_length = 0, 0

    def save(self) -> None:
        method_name = 'save_model'

        """
        Saving last model on each epoch end

        Returns:
            None
        """
        try:
            print(method_name)
            self.__save_model_to_json()
            self.__save_custom_objects_to_json()
            self.save_weights()
        except Exception as e:
            print_error(self.__class__.__name__, method_name, e)

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
        except Exception as e:
            print_error(self.__class__.__name__, method_name, e)
            return None

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
                custom_object[k] = getattr(importlib.import_module(v), k)
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
            print_error("BaseModel", method_name, e)
            return None

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
            test_logits = self.base_model(x_batch, training=False)
            true_array = list(y_batch.values())
            test_logits = test_logits if isinstance(test_logits, list) else [test_logits]
        return test_logits, true_array

    def fit(self, params: TrainingDetailsData, dataset: PrepareDataset):
        method_name = 'fit'
        try:
            print(method_name)
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
                train_pred[f"{out}"] = np.zeros(train_target_shape)
                train_true[f"{out}"] = np.zeros(train_target_shape)
                val_pred[f"{out}"] = np.zeros(val_target_shape)
                val_true[f"{out}"] = np.zeros(val_target_shape)

            train_data_idxs = np.arange(self.train_length).tolist()
            self.callback.on_train_begin()
            for epoch in range(current_epoch, end_epoch):
                self.callback.on_epoch_begin()
                train_steps = 0
                current_idx = 0

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
                    # val_steps = 0
                    # current_val_idx = 0
                    # for x_batch_val, y_batch_val in dataset.dataset.get('val').batch(params.base.batch):
                    #     val_pred_array, val_true_array = self.__test_step(x_batch=x_batch_val, y_batch=y_batch_val)
                    #     length = val_true_array[0].shape[0]
                    #     for i, out in enumerate(output_list):
                    #         val_pred[f"{out}"][current_val_idx: current_val_idx + length] = \
                    #             val_pred_array[i].numpy()
                    #         val_true[f"{out}"][current_val_idx: current_val_idx + length] = \
                    #             val_true_array[i].numpy()
                    #     current_val_idx += length
                    #     val_steps += 1
                    # self.callback.on_epoch_end(
                    #     epoch=epoch + 1,
                    #     arrays={
                    #         "train_pred": train_pred, "val_pred": val_pred, "train_true": train_true,
                    #         "val_true": val_true
                    #     },
                    #     train_data_idxs=train_data_idxs
                    # )
                    # self.callback.on_train_end()
                    break

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
                    print(f"Best weights was saved\n")
            self.callback.on_train_end()
        except Exception as e:
            print_error(self.__class__.__name__, method_name, e)


class YoloTerraModel(BaseTerraModel):

    def __init__(self, model, model_name: str, model_path: Path, **options):
        super().__init__(model=model, model_name=model_name, model_path=model_path)
        if not model:
            super().load()
        self.yolo_model = self.__create_yolo(training=options.get("training"),
                                             classes=options.get("classes"),
                                             version=options.get("version"))

    # def save_weights(self, path_=None):
    #     if not path_:
    #         path_ = self.file_path_model_weights
    #     self.base_model.save_weights(path_)
    #
    # def load_weights(self):
    #     self.base_model.load_weights(self.file_path_model_weights)

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
        except Exception as e:
            print_error("module yolo_utils", method_name, e)

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
        # if params.state.status != "addtrain":
        #     warmup_steps = train_warmup_epochs * steps_per_epoch
        # else:
        #     warmup_steps = 0
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

        pred_result = self.yolo_model(image_array['1'], training=False)
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
                self.callback.on_epoch_begin()
                current_logs = {"epochs": epoch + 1, 'loss': {}, "metrics": {}, 'class_loss': {}, 'class_metrics': {}}
                train_loss_cls = {}
                for cls in range(num_class):
                    train_loss_cls[classes[cls]] = 0.
                current_idx = 0
                cur_step, giou_train, conf_train, prob_train, total_train = 0, 0, 0, 0, 0
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

                self.save_weights()
                if self.callback.stop_training:
                    self.callback.on_train_end()
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
                    print(f"Best weights was saved\n")
            self.callback.on_train_end()
        except Exception as e:
            print_error(self.__class__.__name__, method_name, e)
