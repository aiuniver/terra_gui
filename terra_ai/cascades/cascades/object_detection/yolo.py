from cascade import CascadeElement, CascadeBlock
import cv2
import numpy as np
import tensorflow as tf
import typing
from collections import OrderedDict


def postprocess(frame_size, score_threshold=0.3, iou_threshold=0.45, soft_nms: bool = False, sigma=0.3):
    def bboxes_iou(boxes1, boxes2):
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

    def fun(predict, original_image):
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
                iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
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

    return fun


def preprocess_video(frame_size):
    def fun(frame):
        h, w, _ = frame.shape  # Получаем исходную высоту и ширину изображения
        scale = min(frame_size / w, frame_size / h)  # Получаем минимальное отношение между высотой и шириной
        nw, nh = int(scale * w), int(scale * h)  # Получаем новое значение высоты и ширины
        image_resized = cv2.resize(frame, (nw, nh))  # Изменяем размер изображения
        image_paded = np.full(shape=[frame_size, frame_size, 3], fill_value=128.0)  # Создаем пустой массив
        dw, dh = (frame_size - nw) // 2, (frame_size - nh) // 2  # Получаем центр изображения
        image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized  # Вставляем изображения
        image_paded = image_paded / 255.  # Нормируем изображение
        return image_paded[np.newaxis, ...].astype(np.float32)

    return fun


class YoloCascade(CascadeBlock):

    def __init__(
            self, model: typing.Callable = None, frame_size: int = 416,
            score_threshold: float = .3,
            iou_threshold: float = .45, soft_nms=False, sigma: float = .31
    ):

        self.yolo = CascadeElement(model, "Yolo модель")

        self.preprocess_cascad = CascadeElement(
            preprocess_video(frame_size),
            "Препроцесс"
        )

        self.postprocess_cascad = CascadeElement(
            postprocess(frame_size, score_threshold, iou_threshold, soft_nms, sigma),
            "Постобработка"
        )

        adjacency_map = OrderedDict([
            (self.preprocess_cascad, ['ITER']),
            (self.yolo, [0]),
            (self.postprocess_cascad, [1, 'ITER'])
        ])

        super().__init__(adjacency_map)
