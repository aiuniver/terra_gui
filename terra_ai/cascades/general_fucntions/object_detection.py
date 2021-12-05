import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageDraw
from random import randrange


def _bboxes_iou():
    def fun(boxes1, boxes2):
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

    return fun


def postprocess_boxes(input_size=416, score_threshold=.3, iou_threshold=.45, method='nms', sigma=.3):
    bb = _bboxes_iou()

    def fun(pred_bbox, original_image):
        if len(original_image.shape) == 4:
            original_image = original_image[0]
        valid_scale = [0, np.inf]
        pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                    pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
        # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        org_h, org_w = np.squeeze(original_image).shape[:2]
        resize_ratio = min(input_size / org_w, input_size / org_h)

        dw = (input_size - resize_ratio * org_w) / 2
        dh = (input_size - resize_ratio * org_h) / 2

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
                # Process 3: Calculate this bounding box A and Remain all iou of the bounding box and remove those
                # bounding boxes whose iou value is higher than the threshold
                iou = bb(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0

                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return np.array(best_bboxes)

    return fun


def plot_b_boxes(classes, colors=None, line_thickness=None):
    if colors is None:
        colors = [tuple((randrange(1, 256) for _ in range(3)))
                  for _ in range(len(classes))]

    def fun(bboxes, img):
        img = np.squeeze(img)
        if bboxes.shape != (0, 5):
            brawl_color = True

            if bboxes.shape[-1] == 5:
                brawl_color = False

            # Plots one bounding box on image img
            tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
            if tl < 0:
                tl = 1
            for x in bboxes:
                cl = int(x[-1])
                if brawl_color:
                    color = colors[cl]
                else:
                    color = colors[0]
                c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                cv2.rectangle(img, c1, c2, color, int(tl), cv2.LINE_AA)

                t_size = (114, 16)
                c2 = int(c1[0] + t_size[0] + 5 if (c1[0] + t_size[0] + 5) > 0 else 0), \
                     int(c1[1] - t_size[1] - 5 if (c1[1] - t_size[1] - 5) > 0 else 0)
                c1 = int(c1[0] if (c1[0]) > 0 else 0), int(c1[1] - 3 if (c1[1] - 3) > 0 else 0)

                cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled

                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                d = ImageDraw.Draw(pil_img)

                if brawl_color:
                    d.text((c1[0] + 5, c1[1] - 17), classes[cl], fill=(225, 255, 255, 0))
                else:
                    d.text((c1[0] + 5, c1[1] - 17), str(cl), fill=(225, 255, 255, 0))

                img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        return img

    return fun


def filter_classes(filter_classes: list):
    def fun(bboxes: np.ndarray):
        g = bboxes[bboxes[:, -1] == filter_classes[0]]

        for i in filter_classes[1:]:
            g = np.concatenate((g, bboxes[bboxes[:, -1] == i]))

        return g

    return fun


def main(**params):
    def fun(*predict):
        while len(predict) == 1:
            predict = predict[0]
        predict = list(predict)
        predict.pop()

        pred_bbox = tf.concat([tf.reshape(x, (-1, tf.shape(x)[-1])) for x in predict], 0)

        return pred_bbox

    return fun
