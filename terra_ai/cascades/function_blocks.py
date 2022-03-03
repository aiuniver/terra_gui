import os
import uuid

import numpy as np
import tensorflow as tf
import cv2

from PIL import Image, ImageDraw, ImageFont
from random import randrange
from tensorflow.keras.utils import save_img

from .input_blocks import Input
from .internal_out_blocks import ModelOutput
from .main_blocks import CascadeBlock, BaseBlock
from terra_ai.data.cascades.blocks.extra import ObjectDetectionFilterClassesList


class BaseFunction(BaseBlock):

    def execute(self, **kwargs):
        pass


class ChangeType(BaseFunction):

    def __init__(self, format_type):
        super().__init__()
        self.format_type = format_type

    def execute(self):
        array_ = list(self.inputs.values())[0].execute()
        return array_.astype(self.format_type)


class ChangeSize(BaseFunction):

    def __init__(self, format_size):
        super().__init__()
        self.format_size = format_size if len(format_size) != 3 else format_size[:2]

    def execute(self, image_=None):
        if not image_:
            image_ = list(self.inputs.values())[0].execute()
        return tf.image.resize(image_, self.format_size).numpy()


class MinMaxScale(BaseFunction):

    def __init__(self, format_size):
        super().__init__()
        self.format_size = format_size if len(format_size) != 3 else format_size[:2]

    def execute(self):
        pass


class CropImage(BaseFunction):

    def __init__(self, **kwargs):
        super().__init__()
        fs = kwargs.get('format_size', [])
        # self.format_size = format_size if len(format_size) != 3 else format_size[:2]

    def execute(self, **kwargs):
        out = []
        result = list(self.inputs.values())[0].execute()
        pred_bb = result.get('bboxes')
        image = Image.open(result.get('source'))
        real_size = image.size
        # scale_w = real_size[0] / self.image_size[0]
        # scale_h = real_size[1] / self.image_size[1]
        # if len(pred_bb) > 0:
        #     pred_bb = self.__resize_bb(pred_bb, scale_w, scale_h)

        for i, box in enumerate(pred_bb[:, :4]):
            left, top, right, bottom = box
            top = max(0, np.floor(top + 0.5).astype('int'))
            left = max(0, np.floor(left + 0.5).astype('int'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int'))
            image_crop = image.crop((left, top, right, bottom))
            save_path = 'F:\\test_result\\croped\\'
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

            path_ = os.path.join(
                save_path, f"initial_{uuid.uuid4()}.webp"
            )
            image_crop.save(path_, "webp")
            out.append(path_)
        return out


class MaskedImage(BaseFunction):

    def __init__(self, format_size):
        super().__init__()
        self.format_size = format_size if len(format_size) != 3 else format_size[:2]

    def execute(self, **kwargs):
        pass


class PlotMaskSegmentation(BaseFunction):

    def __init__(self, format_size):
        super().__init__()
        self.format_size = format_size if len(format_size) != 3 else format_size[:2]

    def execute(self, **kwargs):
        pass


class PutTag(BaseFunction):

    def __init__(self, format_size):
        super().__init__()
        self.format_size = format_size if len(format_size) != 3 else format_size[:2]

    def execute(self, **kwargs):
        pass


class PostprocessBoxes(BaseFunction):

    def __init__(self, **kwargs):
        super().__init__()
        self.input_size = kwargs.get("input_size", 416)
        self.score_threshold = kwargs.get("score_threshold", 0.3)
        self.iou_threshold = kwargs.get("iou_threshold", 0.45)
        self.method = kwargs.get("method", 'nms')
        self.sigma = kwargs.get("sigma", 0.3)

    @staticmethod
    def __bboxes_iou(boxes1, boxes2):

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

    def __get_bboxes(self, original_image, pred_bbox, input_size=416):
        valid_scale = [0, np.inf]
        pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                    pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
        # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        org_h, org_w = original_image.size[1], original_image.size[0]
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
        score_mask = scores > self.score_threshold
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
                iou = self.__bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                if self.method == 'nms':
                    iou_mask = iou > self.iou_threshold
                    weight[iou_mask] = 0.0

                if self.method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / self.sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return np.array(best_bboxes)

    def execute(self, **kwargs):
        result = list(self.inputs.values())[0].execute()
        array = result.get('bboxes')[0]
        name_classes = result.get('bboxes')[1]
        img_path = result.get('initial_file')
        image = Image.open(img_path)

        channel_boxes = self.__get_bboxes(image, array)

        return {
            'source': result.get('initial_file'),
            'bboxes': channel_boxes,
            'classes': name_classes
        }


class PlotBboxes(BaseFunction):

    def __init__(self, **kwargs):
        super().__init__()
        self.classes = kwargs.get("classes", None)
        self.colors = kwargs.get("colors", [])
        self.line_thickness = kwargs.get("line_thickness", None)
        self.outs = {out.data_type: out for out in ModelOutput().get(type_=self.__class__.__name__)}
        self.image_size = (416, 416)
        self.save_path = ''

    def get_outputs(self):
        return list(self.outs.keys())

    def set_path(self, model_path: str, save_path: str, weight_path: str):
        self.save_path = save_path

    @staticmethod
    def __resize_bb(boxes, scale_width, scale_height):
        coord = boxes[:, :4].astype('float')
        resized_coord = np.concatenate(
            [coord[:, 0:1] * scale_height, coord[:, 1:2] * scale_width,
             coord[:, 2:3] * scale_height, coord[:, 3:4] * scale_width], axis=-1).astype('int')
        resized_coord = np.concatenate([resized_coord, boxes[:, 4:]], axis=-1)
        return resized_coord

    @staticmethod
    def __draw_box(image, font, draw, box, color, thickness, label=None, label_size=None,
                   dash_mode=False, show_label=False):
        left, top, right, bottom  = box
        top = max(0, np.floor(top + 0.5).astype('int'))
        left = max(0, np.floor(left + 0.5).astype('int'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int'))
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        if dash_mode:
            for cur_y in [top, bottom]:
                for x in range(left, right, 4):
                    draw.line([(x, cur_y), (x + thickness, cur_y)], fill=color, width=2)
            for cur_y in [left, right]:
                for x in range(top, bottom, 4):
                    draw.line([(cur_y, x), (cur_y + thickness, x)], fill=color, width=2)
        else:
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=color,
                )
        if show_label:
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=color
            )
            draw.text(tuple(text_origin), label, fill=(0, 0, 0), font=font)
        return draw

    def __set_classes_and_colors(self, input_data: dict):
        self.classes = input_data.get('classes', None)
        self.colors = input_data.get('colors', [])
        if self.classes is None:
            self.classes = ObjectDetectionFilterClassesList
        if not self.colors:
            self.colors = [tuple((randrange(1, 256) for _ in range(3))) for _ in range(len(self.classes))]

    def execute(self, **kwargs):

        image_pred = None
        result = list(self.inputs.values())[0].execute()
        self.__set_classes_and_colors(result)

        img_array = result.get('image_array')
        img_path = result.get('source')
        pred_bb = result.get('bboxes')

        image = Image.open(img_path) if img_path else Image.fromarray(img_array)

        real_size = image.size
        scale_w = real_size[0] / self.image_size[0] if img_path else 1
        scale_h = real_size[1] / self.image_size[1] if img_path else 1

        if len(pred_bb) > 0:
            pred_bb = self.__resize_bb(pred_bb, scale_w, scale_h)

            font = ImageFont.load_default()
            thickness = (image.size[0] + image.size[1]) // 300 if (image.size[0] + image.size[1]) > 800 else 2
            image_pred = image.copy()

            classes = pred_bb[:, 5].astype('int')
            for i, box in enumerate(pred_bb[:, :4]):
                draw = ImageDraw.Draw(image_pred)
                score = pred_bb[:, 4][i]  # pred_bb[:, 5:][i][classes[i]]
                predicted_class = self.classes[classes[i]] if isinstance(score, float) else classes[i]
                label = ' {} {:.2f} '.format(predicted_class, score)
                label_size = draw.textsize(label, font)
                draw = self.__draw_box(image_pred, font, draw, box, self.colors[classes[i]], thickness,
                                       label=label, label_size=label_size,
                                       dash_mode=True, show_label=True)
                del draw

        data = {
            'model_predict': image_pred if image_pred else image,
            'save_path': self.save_path
        }

        return {out.data_type: out().execute(**data) for name, out in self.outs.items()}


class FilterClasses(BaseFunction):

    def __init__(self, **kwargs):
        super().__init__()
        # self.filter_classes = kwargs.get("filter_classes", [])
        self.filter_classes = [ObjectDetectionFilterClassesList.index(x)
                               for x in kwargs.get("filter_classes", [])]

    def execute(self, **kwargs):
        result = list(self.inputs.values())[0].execute()
        bboxes = result.get('bboxes')
        if bboxes.shape[0]:
            filtered_bboxes = bboxes[bboxes[:, -1] == self.filter_classes[0]]

            for i in self.filter_classes[1:]:
                filtered_bboxes = np.concatenate((filtered_bboxes, bboxes[bboxes[:, -1] == i]))
        else:
            filtered_bboxes = bboxes

        return {'bboxes': filtered_bboxes, 'source': result.get('initial_file'),
                'image_array': result.get('image_array')}


class Function(CascadeBlock):
    ChangeType = ChangeType
    ChangeSize = ChangeSize
    MinMaxScale = MinMaxScale
    CropImage = CropImage
    MaskedImage = MaskedImage
    PlotMaskSegmentation = PlotMaskSegmentation
    PutTag = PutTag
    PostprocessBoxes = PostprocessBoxes
    PlotBboxes = PlotBboxes
    FilterClasses = FilterClasses
