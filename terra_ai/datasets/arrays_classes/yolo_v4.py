import random
import numpy as np

from tensorflow import concat as tf_concat
from tensorflow import maximum as tf_maximum
from tensorflow import minimum as tf_minimum
from typing import Any

from terra_ai.datasets.utils import get_yolo_anchors, resize_bboxes, Yolo_terra
from terra_ai.data.datasets.extra import LayerODDatasetTypeChoice
from .base import Array


class YoloV4Array(Array):

    def prepare(self, sources, dataset_folder=None, **options):
        bounding_boxes = []
        annot_type = options['model_type']
        if annot_type == LayerODDatasetTypeChoice.Yolo_terra:
            for path in sources:
                with open(path, 'r') as coordinates:
                    coordinate = coordinates.read()
                bounding_boxes.append(' '.join([coord for coord in coordinate.split('\n') if coord]))

        else:
            model_type = eval(f'{annot_type}()')
            data, cls_hierarchy = model_type.parse(sources, options['classes_names'])
            yolo_terra = Yolo_terra(options['classes_names'], cls_hierarchy=cls_hierarchy)
            data = yolo_terra.generate(data)
            for key in data:
                bounding_boxes.append(data[key])

        instructions = {'instructions': bounding_boxes,
                        'parameters': {'num_classes': options['num_classes'],
                                       'classes_names': options['classes_names'],
                                       'put': options['put'],
                                       'cols_names': options['cols_names'],
                                       'frame_mode': options['frame_mode']}
                        }

        return instructions

    def create(self, source: Any, **options):
        """
                Args:
                    source: str
                        Координаты bounding box.
                    **options:
                        height: int ######!!!!!!
                            Высота изображения.
                        width: int ######!!!!!!
                            Ширина изображения.
                        num_classes: int
                            Количество классов.
                Returns:
                    array: np.ndarray
                        Массивы в трёх выходах.
                """

        if source:
            frame_mode = options['frame_mode'] if 'frame_mode' in options.keys() else 'stretch'  # Временное решение
            real_boxes = resize_bboxes(frame_mode, source, options['orig_x'], options['orig_y'])
        else:
            real_boxes = [[0, 0, 0, 0, 0]]

        num_classes: int = options['num_classes']
        zero_boxes_flag: bool = False
        strides = np.array([8, 16, 32])
        output_levels = len(strides)
        train_input_sizes = 416
        anchor_per_scale = 3

        yolo_anchors = get_yolo_anchors('v4')

        anchors = (np.array(yolo_anchors).T / strides).T
        max_bbox_per_scale = 100
        train_input_size = random.choice([train_input_sizes])
        train_output_sizes = train_input_size // strides

        label = [np.zeros((train_output_sizes[i], train_output_sizes[i], anchor_per_scale,
                           5 + num_classes)) for i in range(output_levels)]
        bboxes_xywh = [np.zeros((max_bbox_per_scale, 4)) for _ in range(output_levels)]
        bbox_count = np.zeros((output_levels,))

        for bbox in real_boxes:
            bbox_class_ind = int(bbox[4])
            bbox_coordinate = np.array(bbox[:4])
            one_hot = np.zeros(num_classes, dtype=np.float)
            one_hot[bbox_class_ind] = 0.0 if zero_boxes_flag else 1.0
            uniform_distribution = np.full(num_classes, 1.0 / num_classes)
            deta = 0.01
            smooth_one_hot = one_hot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coordinate[2:] + bbox_coordinate[:2]) * 0.5,
                                        bbox_coordinate[2:] - bbox_coordinate[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(output_levels):  # range(3):
                anchors_xywh = np.zeros((anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 0.0 if zero_boxes_flag else 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_one_hot

                    bbox_ind = int(bbox_count[i] % max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / anchor_per_scale)
                best_anchor = int(best_anchor_ind % anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 0.0 if zero_boxes_flag else 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_one_hot

                bbox_ind = int(bbox_count[best_detect] % max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        instructions = {'instructions': [np.array(label_sbbox, dtype='float32'), np.array(label_mbbox, dtype='float32'),
                                         np.array(label_lbbox, dtype='float32'), np.array(sbboxes, dtype='float32'),
                                         np.array(mbboxes, dtype='float32'), np.array(lbboxes, dtype='float32')],
                        'parameters': options}

        return instructions

    def preprocess(self, array: np.ndarray, **options):

        return array

    @staticmethod
    def bbox_iou(boxes1, boxes2):

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf_concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf_concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf_maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf_minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf_maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return 1.0 * inter_area / union_area
