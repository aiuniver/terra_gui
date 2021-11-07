import os
import json
import os
from typing import Callable

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

from terra_ai.data.datasets.creation import CreationInputData
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.utils import decamelize


def count(bbox: np.ndarray) -> int:
    return bbox.shape[-2]


def head_cropping(width_object: int = 80, out_size: int = 64) -> Callable:
    size = (out_size, out_size)
    resize_img = lambda x: cv2.resize(x, size) / 255

    def fun(bbox: np.ndarray, img: np.ndarray) -> np.ndarray:
        heads = []

        for b in bbox:
            center = (b[3] + b[1]) // 2
            top = 0 if center < width_object else center - width_object
            bot = center + width_object

            center = (b[2] + b[0]) // 2
            left = 0 if center < width_object else center - width_object
            right = center + width_object

            heads.append(
                resize_img(img[top: bot, left: right])
            )

        return np.array(heads)

    return fun


def _preprocess_video(frame_size, frame):
    h, w, _ = frame.shape  # Получаем исходную высоту и ширину изображения
    scale = min(frame_size / w, frame_size / h)  # Получаем минимальное отношение между высотой и шириной
    nw, nh = int(scale * w), int(scale * h)  # Получаем новое значение высоты и ширины
    image_resized = cv2.resize(frame, (nw, nh))  # Изменяем размер изображения
    image_paded = np.full(shape=[frame_size, frame_size, 3], fill_value=128.0)  # Создаем пустой массив
    dw, dh = (frame_size - nw) // 2, (frame_size - nh) // 2  # Получаем центр изображения
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized  # Вставляем изображения
    image_paded = image_paded / 255.  # Нормируем изображение
    return image_paded[np.newaxis, ...].astype(np.float32)


def _bboxes_iou(boxes1, boxes2):
    """Get IoU between two boxes

    Parameters:
    boxes1 (np.array): four coordinates first box

    boxes2 (np.array): four coordinates second box

    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)
    return ious


def _(predict, original_image, frame_size, score_threshold=0.3, iou_threshold=0.45, soft_nms: bool = False,
      sigma=0.3):
    predict = tf.concat(
        [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in predict], axis=0
    )

    valid_scale = [0, np.inf]
    pred_bbox = np.array(predict)
    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]
    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)

    # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(frame_size / org_w, frame_size / org_h)
    dw = (frame_size - resize_ratio * org_w) / 2
    dh = (frame_size - resize_ratio * org_h) / 2
    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
    # 3. clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0
    # 4. discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))
    # 5. discard boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    if not len(coors):
        return None

    bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []
    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        # Process 1: Determine whether the number of bounding boxes is greater than 0
        while len(cls_bboxes) > 0:
            # Process 2: Select the bounding box with the highest score according to socre order A
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            # Process 3: Calculate this bounding box A and
            iou = _bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            if soft_nms:
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            else:
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return np.array(best_bboxes).astype(np.int32)


def _preprocess_model(**params):
    '''
    Exsample
        params.name: 'air2.trds'
        params.path_file: ['C:\PycharmProjects\yolov5x_helmet_demopanel_russian_label\Test_images\pos_54.jpg']
    '''
    MODEL_PATH = r'C:\Users\Anonim\Documents\terra_gui\TerraProjects'
    DATASET_PATH = 'C:\PycharmProjects/terra_gui/TerraAI/datasets'
    path_model = os.path.join(MODEL_PATH, params['model_name'])
    with open(os.path.join(path_model, 'config.json'), 'r') as cfg:
        dataset_data = DatasetData(**json.load(cfg))

    path_dataset = os.path.join(MODEL_PATH, dataset_data.alias)
    # print(path_dataset)
    createarray = CreateArray(DATASET_PATH)  # здесь лежат препроцессы

    def load_instructions(path, dataset_data):
        inputs = {}
        instructions = {}
        for instr_json in os.listdir(os.path.join(path, 'instructions', 'parameters')):
            with open(os.path.join(path_dataset, 'instructions', 'parameters', instr_json), 'r') as instr:
                instructions[int(os.path.splitext(instr_json)[0].split('_')[0])] = json.load(instr)

        creation = {}
        for key in dataset_data.inputs.keys():
            creation['id'] = key
            creation['name'] = dataset_data.inputs[key].name
            creation['type'] = dataset_data.inputs[key].task
            creation['parameters'] = instructions[key]
            creation['parameters']['sources_paths'] = []
            creation['parameters']['deploy'] = True
            inputs[key] = CreationInputData(**creation)
        # print(creation)
        return inputs, instructions, dataset_data

    def load_preprocess(path_dataset, inputs, createarray):

        createarray.load_preprocess(path_dataset, inputs.keys())

    def make_array(path, input, inputs, dataset_data, instructions, createarray):

        instr = {'instructions': {f'{input}_{decamelize(dataset_data.inputs[1].task)}': path},
                 'parameters': inputs[input]}
        arr = []
        # print(dataset_data)
        # print(dataset_data.inputs[1].task)
        # print(f'create_{decamelize(dataset_data.inputs[1].task)}')
        for elem in instr['instructions'][f'{input}_{decamelize(dataset_data.inputs[1].task)}']:
            arr.append(
                getattr(createarray, f'create_{decamelize(dataset_data.inputs[1].task)}')('', elem, **
                instructions[input]))
        # print(instructions[input])

        return np.array(arr)

    inputs, instructions, dataset_data = load_instructions(path_dataset, dataset_data)
    load_preprocess(path_dataset, inputs, createarray)

    return make_array(params['path_file'], 1, inputs, dataset_data, instructions, createarray)


def _postprocess_model(**params):
    """
    frame_size, score_threshold=0.3, iou_threshold=0.45, soft_nms: bool = False, sigma=0.3
    """
    MODEL_PATH = 'C:\PycharmProjects/terra_gui/TerraAI/training'
    path_model = os.path.join(MODEL_PATH, params['model_name'])
    with open(os.path.join(path_model, 'config.json'), 'r') as cfg:
        dataset_data = DatasetData(**json.load(cfg))
    # print(dataset_data)
    for key in dataset_data.outputs.keys():
        # print(dataset_data.outputs.get(key).task)
        if dataset_data.outputs.get(key).task == 'Segmentation':
            out = _plot_mask_segmentation(pred, dataset_data.num_classes.get(key),
                                          [x.as_rgb_tuple() for x in dataset_data.classes_colors.get(key)])
        elif dataset_data.outputs.get(key).task == 'Classification':
            # print(dataset_data.outputs.get(key).task)
            pass

    return out


def _plot_mask_segmentation(predict, num_classes, classes_colors):
    """
    Returns:
        mask_images
    """

    def _index2color(pix, num_cls, cls_colors):
        index = np.argmax(pix)
        color = []
        for i in range(num_cls):
            if index == i:
                color = cls_colors[i]
        return color

    def _get_colored_mask(mask, num_cls, cls_colors):
        """
        Transforms prediction mask to colored mask

        Parameters:
        mask : numpy array                 segmentation mask

        Returns:
        colored_mask : numpy array         mask with colors by classes
        """

        colored_mask = []
        shape_mask = mask.shape
        # print(shape_mask)
        mask = mask.reshape(-1, num_cls)
        # print(mask.shape)
        for pix in range(len(mask)):
            colored_mask.append(
                _index2color(mask[pix], num_cls, cls_colors)
            )
        colored_mask = np.array(colored_mask).astype(np.uint8)
        # print(colored_mask.shape)
        colored_mask = colored_mask.reshape((shape_mask[0], shape_mask[1], 3))
        return colored_mask

    image = np.squeeze(_get_colored_mask(predict[0], num_classes, classes_colors))

    return image


def _load_model(**params):
    MODEL_PATH = 'C:\PycharmProjects/terra_gui/TerraAI/training'
    path_model = os.path.join(MODEL_PATH, params['model_name'])

    model = load_model(path_model, compile=False,
                       custom_objects=None)
    model.load_weights(os.path.join(path_model, params['model_name'] + '_best.h5'))

    return model


if __name__ == "__main__":
    # Проверка препроцеса (для изображений база 'самолеты')
    params = {'model_name': 'airplanes_new',
              'path_file': [r"C:\Users\Anonim\Documents\terra_gui\TerraProjects\airplanes.trds\sources\1_image\Самолеты\0.jpg"]}
    x_input = _preprocess_model(**params)
    # print(x_input.shape)
    plt.imshow(x_input[0])
    plt.show()

    # Загрузка модели
    model = _load_model(**params)

    # Проверка постпроцесса
    pred = model.predict(x_input)
    mask = _postprocess_model(**params)
    plt.imshow(mask)
    plt.show()
