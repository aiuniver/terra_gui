import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import random
import time
import colorsys
from tensorflow.keras.preprocessing.image import load_img
import os

from terra_ai.callbacks.utils import print_error
from terra_ai.datasets.utils import resize_bboxes

### DETECTION ###


def detect_image(Yolo, original_image, output_path, input_size=416, show=False, CLASSES=None,
                 score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', train=False):
    method_name = 'detect_image'
    try:
        # original_image = cv2.imread(image_path)
        # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        if CLASSES is None:
            CLASSES = []
        image_data = original_image[np.newaxis, ...].astype(np.float32)

        pred_bbox = Yolo.predict(image_data)
        # print(len(pred_bbox))
        # print(pred_bbox[0].shape)
        # print(pred_bbox[1].shape)
        # print(pred_bbox[2].shape)
        if train:
            pred_bbox = [pred_bbox[1], pred_bbox[3], pred_bbox[5]]

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        image = draw_bbox(original_image, bboxes, CLASSES=CLASSES, rectangle_colors=rectangle_colors)
        # CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))

        if output_path != '':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_path, image)
        if show:
            # Show the image
            cv2.imshow("predicted image", image)
            # Load and hold the image
            cv2.waitKey(0)
            # To close the window after the required kill value was provided
            cv2.destroyAllWindows()

        return image
    except Exception as e:
        print_error("module yolo_utils", method_name, e)


def draw_bbox(image, bboxes, CLASSES, show_label=True, show_confidence=True,
              Text_colors=(255, 255, 0), rectangle_colors='', tracking=False):
    method_name = 'draw_bbox'
    try:
        NUM_CLASS = CLASSES
        num_classes = len(NUM_CLASS)
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        # print("hsv_tuples", hsv_tuples)
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for i, bbox in enumerate(bboxes):
            coor = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            class_ind = int(bbox[5])
            bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 1000)
            if bbox_thick < 1: bbox_thick = 1
            fontScale = 0.75 * bbox_thick
            (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

            # put object rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick * 2)

            if show_label:
                # get text label
                score_str = " {:.2f}".format(score) if show_confidence else ""

                if tracking: score_str = " " + str(score)

                try:
                    label = "{}".format(NUM_CLASS[class_ind]) + score_str
                except KeyError:
                    print("You received KeyError, this might be that you are trying to use yolo original weights")
                    print(
                        "while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")

                # get text size
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                      fontScale, thickness=bbox_thick)
                # put filled text rectangle
                cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color,
                              thickness=cv2.FILLED)

                # put text above rectangle
                cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

        return image
    except Exception as e:
        print_error("module yolo_utils", method_name, e)


def bboxes_iou(boxes1, boxes2):
    method_name = 'bboxes_iou'
    try:
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
    except Exception as e:
        print_error("module yolo_utils", method_name, e)


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)
    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    method_name = 'nms'
    try:
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
                # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold
                iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                assert method in ['nms', 'soft-nms']

                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0

                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes
    except Exception as e:
        print_error("module yolo_utils", method_name, e)


def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    method_name = 'postprocess_boxes'
    try:
        valid_scale = [0, np.inf]
        pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                    pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
        # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        if isinstance(original_image, tuple) or (original_image == None):
            org_h, org_w = input_size, input_size
        else:
            org_h, org_w = original_image.shape[:2]
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

        return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
    except Exception as e:
        print_error("module yolo_utils", method_name, e)


### LOSSES ###

def bbox_iou(boxes1, boxes2):
    method_name = 'bbox_iou'
    try:
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return 1.0 * inter_area / union_area
    except Exception as e:
        print_error("module yolo_utils", method_name, e)


def bbox_giou(boxes1, boxes2):
    method_name = 'bbox_giou'
    try:
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        # Calculate the iou value between the two bounding boxes
        iou = inter_area / union_area

        # Calculate the coordinates of the upper left corner and the lower right corner of the smallest closed convex surface
        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

        # Calculate the area of the smallest closed convex surface C
        enclose_area = enclose[..., 0] * enclose[..., 1]

        # Calculate the GIoU value according to the GioU formula
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou
    except Exception as e:
        print_error("module yolo_utils", method_name, e)


# testing (should be better than giou)
def bbox_ciou(boxes1, boxes2):
    method_name = 'bbox_ciou'
    try:
        boxes1_coor = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2_coor = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left = tf.maximum(boxes1_coor[..., 0], boxes2_coor[..., 0])
        up = tf.maximum(boxes1_coor[..., 1], boxes2_coor[..., 1])
        right = tf.maximum(boxes1_coor[..., 2], boxes2_coor[..., 2])
        down = tf.maximum(boxes1_coor[..., 3], boxes2_coor[..., 3])

        c = (right - left) * (right - left) + (up - down) * (up - down)
        iou = bbox_iou(boxes1, boxes2)

        u = (boxes1[..., 0] - boxes2[..., 0]) * (boxes1[..., 0] - boxes2[..., 0]) + (
                    boxes1[..., 1] - boxes2[..., 1]) * (
                    boxes1[..., 1] - boxes2[..., 1])
        d = u / c

        ar_gt = boxes2[..., 2] / boxes2[..., 3]
        ar_pred = boxes1[..., 2] / boxes1[..., 3]

        ar_loss = 4 / (np.pi * np.pi) * (tf.atan(ar_gt) - tf.atan(ar_pred)) * (tf.atan(ar_gt) - tf.atan(ar_pred))
        alpha = ar_loss / (1 - iou + ar_loss + 0.000001)
        ciou_term = d + alpha * ar_loss

        return iou - ciou_term
    except Exception as e:
        print_error("module yolo_utils", method_name, e)


def compute_loss(pred, conv, label, bboxes, i=0, CLASSES=None, STRIDES=None, YOLO_IOU_LOSS_THRESH=0.5):
    method_name = 'compute_loss'
    try:
        if STRIDES is None:
            STRIDES = [8, 16, 32]
        if CLASSES is None:
            CLASSES = []
        NUM_CLASS = len(CLASSES)
        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = STRIDES[i] * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        # Find the value of IoU with the real box The largest prediction box
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        # If the largest iou is less than the threshold, it is considered that the prediction box contains no objects,
        # then the background box
        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < YOLO_IOU_LOSS_THRESH, tf.float32)

        conf_focal = tf.pow(respond_bbox - pred_conf, 2)

        # Calculate the loss of confidence we hope that if the grid contains objects, then the network output prediction
        # box has a confidence of 1 and 0 when there is no object.
        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))
        prob_loss_cls = {}
        for cls in range(NUM_CLASS):
            conv_raw_prob_cls = conv[:, :, :, :, 5 + cls:5 + cls + 1]
            label_prob_cls = label[:, :, :, :, 5 + cls:5 + cls + 1]
            prob_loss_cls[str(CLASSES[cls])] = tf.reduce_mean(
                tf.reduce_sum(
                    respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=label_prob_cls, logits=conv_raw_prob_cls), axis=[1, 2, 3, 4]))

        return giou_loss, conf_loss, prob_loss, prob_loss_cls
    except Exception as e:
        print_error("module yolo_utils", method_name, e)


### CREATE AND FIT MODEL ###

def decode(conv_output, NUM_CLASS, i=0, YOLO_TYPE="v3", STRIDES=None):
    method_name = 'decode'
    try:
        if STRIDES is None:
            STRIDES = [8, 16, 32]
        if (YOLO_TYPE == "v4") or (YOLO_TYPE == "v5"):
            ANCHORS = [[[12, 16], [19, 36], [40, 28]],
                       [[36, 75], [76, 55], [72, 146]],
                       [[142, 110], [192, 243], [459, 401]]]
        elif YOLO_TYPE == "v3":
            ANCHORS = [[[10, 13], [16, 30], [33, 23]],
                       [[30, 61], [62, 45], [59, 119]],
                       [[116, 90], [156, 198], [373, 326]]]
        # Train options
        # where i = 0, 1 or 2 to correspond to the three grid scales
        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

        # conv_raw_dxdy = conv_output[:, :, :, :, 0:2] # offset of center position
        # conv_raw_dwdh = conv_output[:, :, :, :, 2:4] # Prediction box length and width offset
        # conv_raw_conf = conv_output[:, :, :, :, 4:5] # confidence of the prediction box
        # conv_raw_prob = conv_output[:, :, :, :, 5: ] # category probability of the prediction box
        conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS),
                                                                              axis=-1)

        # next need Draw the grid. Where output_size is equal to 13, 26 or 52
        # y = tf.range(output_size, dtype=tf.int32)
        # y = tf.expand_dims(y, -1)
        # y = tf.tile(y, [1, output_size])
        # x = tf.range(output_size,dtype=tf.int32)
        # x = tf.expand_dims(x, 0)
        # x = tf.tile(x, [output_size, 1])
        xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
        xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
        xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        # xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        # xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
        # y_grid = tf.cast(xy_grid, tf.float32)

        # Calculate the center position of the prediction box:
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * STRIDES[i]
        # Calculate the length and width of the prediction box:
        pred_wh = (tf.exp(conv_raw_dwdh) * ANCHORS[i]) * STRIDES[i]

        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        pred_conf = tf.sigmoid(conv_raw_conf)  # object box calculates the predicted confidence
        pred_prob = tf.sigmoid(conv_raw_prob)  # calculating the predicted probability category box object

        # calculating the predicted probability category box object
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)
    except Exception as e:
        print_error("module yolo_utils", method_name, e)


# @tf.autograph.experimental.do_not_convert
def create_yolo(model, input_size=416, channels=3, training=False, classes=None, version='v3'):
    method_name = 'create_yolo'
    try:
        if classes is None:
            classes = []
        num_class = len(classes)
        input_layer = keras.layers.Input([input_size, input_size, channels])
        conv_tensors = model(input_layer)
        if conv_tensors[0].shape[1] == 13:
            conv_tensors.reverse()
        # print('conv_tensors', conv_tensors.reverse())
        output_tensors = []
        for i, conv_tensor in enumerate(conv_tensors):
            pred_tensor = decode(conv_tensor, num_class, i, version)
            if training: output_tensors.append(conv_tensor)
            output_tensors.append(pred_tensor)
        # print('output_tensors', output_tensors)
        yolo = tf.keras.Model(input_layer, output_tensors)
        return yolo
    except Exception as e:
        print_error("module yolo_utils", method_name, e)


class CustomModelYolo(keras.Model):

    def __init__(self, yolo, dataset, classes, train_epochs, train_batch, warmup_epoch=2, lr_init=1e-4,
                 lr_end=1e-6, iou_thresh=0.5):
        super().__init__()
        self.yolo = yolo
        self.dataset = dataset
        self.CLASSES = classes
        self.train_epochs = train_epochs
        self.train_batch = train_batch
        self.loss_fn = None
        self.optimizer = None
        self.TRAIN_WARMUP_EPOCHS = warmup_epoch
        self.TRAIN_LR_INIT = lr_init
        self.TRAIN_LR_END = lr_end
        self.YOLO_IOU_LOSS_THRESH = iou_thresh
        self.steps_per_epoch = int(len(self.dataset.dataframe.get("train")) // self.train_batch)
        self.global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        self.warmup_steps = self.TRAIN_WARMUP_EPOCHS * self.steps_per_epoch
        self.total_steps = self.train_epochs * self.steps_per_epoch

    def compile(self, optimizer, loss):
        method_name = 'compile'
        try:
            super(CustomModelYolo, self).compile()
            self.optimizer = optimizer
            self.loss_fn = loss
        except Exception as e:
            print_error("CustomModelYolo", method_name, e)

    @tf.function
    def change_lr(self):
        method_name = 'change_lr'
        try:
            if tf.less(self.global_steps.value(), self.warmup_steps):
                lr = self.global_steps.value() / self.warmup_steps * self.TRAIN_LR_INIT
            else:
                lr = self.TRAIN_LR_END + 0.5 * (self.TRAIN_LR_INIT - self.TRAIN_LR_END) * (
                    (1 + tf.cos((self.global_steps.value() - self.warmup_steps) /
                                (self.total_steps - self.warmup_steps) * np.pi)))
            return lr
        except Exception as e:
            print_error("CustomModelYolo", method_name, e)

    @tf.function
    def train_step(self, data):
        method_name = 'train_step'
        try:
            image_data, target, serv = data[0], data[1], data[2]
            input_key = [x for x in image_data.keys()]
            with tf.GradientTape() as tape:
                pred_result = self.yolo(image_data.get(input_key[0], '1'), training=True)
                giou_loss = conf_loss = prob_loss = 0
                prob_loss_cls = {}
                pred_out = {}
                target_out = {}

                # optimizing process
                grid = 3  # if not TRAIN_YOLO_TINY else 2
                for i, key in enumerate(target.keys()):
                    conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                    loss_items = self.loss_fn(pred, conv, *(target.get(key), serv.get(key)), i,
                                              CLASSES=self.CLASSES, YOLO_IOU_LOSS_THRESH=self.YOLO_IOU_LOSS_THRESH)
                    pred_out['pred_' + str(key)] = pred
                    target_out['target_' + str(key)] = target.get(key)
                    giou_loss += loss_items[0]
                    conf_loss += loss_items[1]
                    prob_loss += loss_items[2]

                    for cls_key in loss_items[3].keys():  # пробегаем по ключам словаря
                        try:
                            prob_loss_cls['prob_loss_' + str(cls_key)] += loss_items[3].get(
                                cls_key)  # складываем значения
                        except KeyError:  # если ключа еще нет - создаем
                            prob_loss_cls['prob_loss_' + str(cls_key)] = loss_items[3].get(cls_key)

                total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, self.yolo.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.yolo.trainable_variables))

            # update learning rate
            # about warmup: https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg

            self.global_steps.assign_add(1)
            lr = self.change_lr()
            self.optimizer.lr.assign(tf.cast(lr, tf.float32))

            out_info = {'global_steps': self.global_steps.value(), "optimizer.lr": self.optimizer.lr.value(),
                        "giou_loss": giou_loss, "conf_loss": conf_loss, "prob_loss": prob_loss,
                        "total_loss": total_loss}
            out_info.update(prob_loss_cls)
            out_info.update(pred_out)
            out_info.update(target_out)

            return out_info
        except Exception as e:
            print_error("CustomModelYolo", method_name, e)

    @tf.function
    def test_step(self, data):
        image_data, target, serv = data[0], data[1], data[2]
        input_key = [x for x in image_data.keys()]
        with tf.GradientTape() as tape:
            pred_result = self.yolo(image_data.get(input_key[0], '1'), training=False)
            giou_loss = conf_loss = prob_loss = 0
            prob_loss_cls = {}
            pred_out = {}
            target_out = {}

            # optimizing process
            grid = 3  # if not TRAIN_YOLO_TINY else 2

            for i, key in enumerate(target.keys()):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = self.loss_fn(pred, conv, *(target.get(key), serv.get(key)),
                                          i, CLASSES=self.CLASSES)
                pred_out['pred_' + str(key)] = pred
                target_out['target_' + str(key)] = target.get(key)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

                for cls_key in loss_items[3].keys():  # пробегаем по ключам словаря
                    try:
                        prob_loss_cls['prob_loss_' + cls_key] += loss_items[3].get(cls_key)  # складываем значения
                    except KeyError:  # если ключа еще нет - создаем
                        prob_loss_cls['prob_loss_' + cls_key] = loss_items[3].get(cls_key)

            total_loss = giou_loss + conf_loss + prob_loss

        out_info = {"giou_loss": giou_loss, "conf_loss": conf_loss, "prob_loss": prob_loss, "total_loss": total_loss}
        out_info.update(prob_loss_cls)
        out_info.update(pred_out)
        out_info.update(target_out)

        return out_info

    @tf.function
    def predict_step(self, data):
        return self.yolo(data, training=False)


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    method_name = 'voc_ap'
    try:
        rec.insert(0, 0.0)  # insert 0.0 at begining of list
        rec.append(1.0)  # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0)  # insert 0.0 at begining of list
        prec.append(0.0)  # insert 0.0 at end of list
        mpre = prec[:]
        """
         This part makes the precision monotonically decreasing
            (goes from the end to the beginning)
            matlab:  for i=numel(mpre)-1:-1:1
                                    mpre(i)=max(mpre(i),mpre(i+1));
        """
        # matlab indexes start in 1 but python in 0, so I have to do:
        #   range(start=(len(mpre) - 2), end=0, step=-1)
        # also the python function range excludes the end, resulting in:
        #   range(start=(len(mpre) - 2), end=-1, step=-1)
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        """
         This part creates a list of indexes where the recall changes
            matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
        """
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                i_list.append(i)  # if it was matlab would be i + 1
        """
         The Average Precision (AP) is the area under the curve
            (numerical integration)
            matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
        return ap, mrec, mpre
    except Exception as e:
        print_error("module yolo_utils", method_name, e)

# def resize_bboxes(coords, orig_x, orig_y, out_size = 416):
#     x_scale = orig_x / out_size
#     y_scale = orig_y / out_size
#     real_boxes = []
#     if x_scale == 1 and y_scale == 1:
#         for coord in coords.split(' '):
#             real_boxes.append([literal_eval(num) for num in coord.split(',')])
#     else:
#         for coord in coords.split(' '):
#             tmp = []
#             for i, num in enumerate(coord.split(',')):
#                 if i in [0, 2]:
#                     tmp_value = int(literal_eval(num) / x_scale) - 1
#                     scale_value = orig_x if tmp_value > orig_x else tmp_value
#                     tmp.append(scale_value)
#                 elif i in [1, 3]:
#                     tmp_value = int(literal_eval(num) / y_scale) - 1
#                     scale_value = orig_y if tmp_value > orig_y else tmp_value
#                     tmp.append(scale_value)
#                 else:
#                     tmp.append(literal_eval(num))
#             real_boxes.append(tmp)
#     return real_boxes

def get_mAP(Yolo, dataset, score_threshold=0.25, iou_threshold=None, TEST_INPUT_SIZE=416, TRAIN_CLASSES=None,
            pred=None, dataset_path=''):
    method_name = 'get_mAP'
    tt1 = time.time()
    try:
        if TRAIN_CLASSES is None:
            TRAIN_CLASSES = []
        if iou_threshold is None:
            iou_threshold = [0.50]
        MINOVERLAP = 0.5  # default value (defined in the PASCAL VOC2012 challenge)
        NUM_CLASS = TRAIN_CLASSES

        gt_counter_per_class = {}
        id_ground_truth = {}
        for index in range(len(dataset.dataframe.get("val"))):

            true_bbox = dataset.dataframe.get("val").iloc[index, 1] #.split(' ')
            tmp_im = load_img(os.path.join(dataset_path, dataset.dataframe.get("val").iloc[index, 0]))
            bbox_data_gt = np.array(resize_bboxes(true_bbox, tmp_im.width, tmp_im.height))
            # bbox_data_gt = np.array([list(map(int, box.split(','))) for box in y_true])

            if len(bbox_data_gt) == 0:
                bboxes_gt = []
                classes_gt = []
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]

            num_bbox_gt = len(bboxes_gt)

            bounding_boxes = []
            for i in range(num_bbox_gt):
                class_name = NUM_CLASS[classes_gt[i]]
                xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                bbox = xmin + " " + ymin + " " + xmax + " " + ymax
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})

                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    gt_counter_per_class[class_name] = 1
                bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'

            id_ground_truth[str(index)] = bounding_boxes

        # for cls in NUM_CLASS:
        #     if gt_counter_per_class.get(cls) is None:
        #         gt_counter_per_class[cls] = 1
        gt_classes = list(gt_counter_per_class.keys())

        # sort the classes alphabetically
        gt_classes = sorted(gt_classes)
        n_classes = len(gt_classes)

        times = []
        predict = []
        original_image_shape = []
        if pred is not None:
            for pred_bbox in pred:
                original_image_shape.append(None)
                t1 = time.time()
                pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
                pred_bbox = tf.concat(pred_bbox, axis=0)
                predict.append(pred_bbox)
                t2 = time.time()
                times.append(t2 - t1)
                ms = sum(times) / len(times) * 1000
                fps = 1000 / ms
        else:
            for inp, out, serv in dataset.dataset['val'].batch(1).take(-1):
                original_image = inp['1'].numpy()[0]
                image_data = inp['1'].numpy()
                original_image_shape.append(original_image.shape)
                t1 = time.time()

                pred_bbox = Yolo.predict(image_data)
                pred_bbox = [pred_bbox[1], pred_bbox[3], pred_bbox[5]]
                t2 = time.time()
                times.append(t2 - t1)

                pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
                pred_bbox = tf.concat(pred_bbox, axis=0)
                predict.append(pred_bbox)

                ms = sum(times) / len(times) * 1000
                fps = 1000 / ms

        ap_dictionary = {}
        for i_iou in iou_threshold:

            json_pred = [[] for i in range(n_classes)]
            class_predictions = {}
            len_bbox = 0
            for i_image, pred_bbox in enumerate(predict):

                bboxes = postprocess_boxes(pred_bbox, original_image_shape[i_image], TEST_INPUT_SIZE, score_threshold)
                bboxes = nms(bboxes, i_iou, method='nms')
                len_bbox += len(bboxes)
                for bbox in bboxes:
                    coor = np.array(bbox[:4], dtype=np.int32)
                    score = bbox[4]
                    class_ind = int(bbox[5])
                    class_name = NUM_CLASS[class_ind]
                    score = '%.4f' % score
                    xmin, ymin, xmax, ymax = list(map(str, coor))
                    bbox = xmin + " " + ymin + " " + xmax + " " + ymax
                    try:
                        json_pred[gt_classes.index(class_name)].append(
                            {"confidence": str(score), "file_id": str(i_image), "bbox": str(bbox)})
                    except Exception:
                        continue
            for class_name in gt_classes:
                json_pred[gt_classes.index(class_name)].sort(key=lambda x: float(x['confidence']), reverse=True)
                class_predictions[class_name] = json_pred[gt_classes.index(class_name)]

            # Calculate the AP for each class
            sum_AP = 0.0

            count_true_positives = {}
            for class_index, class_name in enumerate(gt_classes):
                count_true_positives[class_name] = 0
                # Load predictions of that class
                predictions_data = class_predictions.get(class_name)

                # Assign predictions to ground truth objects
                nd = len(predictions_data)
                tp = [0] * nd  # creates an array of zeros of size nd
                fp = [0] * nd
                for idx, prediction in enumerate(predictions_data):
                    file_id = prediction["file_id"]
                    # assign prediction to ground truth object if any
                    #   open ground-truth with that file_id
                    ground_truth_data = id_ground_truth.get(file_id)
                    ovmax = -1
                    gt_match = -1
                    # load prediction bounding-box
                    bb = [float(x) for x in prediction["bbox"].split()]  # bounding box of prediction
                    for obj in ground_truth_data:
                        # look for a class_name match
                        if obj["class_name"] == class_name:
                            bbgt = [float(x) for x in obj["bbox"].split()]  # bounding box of ground truth
                            bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                            iw = bi[2] - bi[0] + 1
                            ih = bi[3] - bi[1] + 1
                            if iw > 0 and ih > 0:
                                # compute overlap (IoU) = area of intersection / area of union
                                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                                  + 1) * (
                                                 bbgt[3] - bbgt[1] + 1) - iw * ih
                                ov = iw * ih / ua
                                if ov > ovmax:
                                    ovmax = ov
                                    gt_match = obj

                    # assign prediction as true positive/don't care/false positive
                    if ovmax >= MINOVERLAP:  # if ovmax > minimum overlap
                        if not bool(gt_match["used"]):
                            # true positive
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                        else:
                            # false positive (multiple detection)
                            fp[idx] = 1
                    else:
                        # false positive
                        fp[idx] = 1

                # compute precision/recall
                cumsum = 0
                for idx, val in enumerate(fp):
                    fp[idx] += cumsum
                    cumsum += val
                cumsum = 0
                for idx, val in enumerate(tp):
                    tp[idx] += cumsum
                    cumsum += val
                rec = tp[:]
                for idx, val in enumerate(tp):
                    rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
                prec = tp[:]
                for idx, val in enumerate(tp):
                    prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

                ap, mrec, mprec = voc_ap(rec, prec)
                sum_AP += ap
                ap_dictionary[f"val_mAP{int(i_iou * 100)}_class_{class_name}"] = ap
                mAP = sum_AP / n_classes

            ap_dictionary[f"val_mAP{int(i_iou * 100)}"] = mAP * 100
        ap_dictionary["val_fps"] = fps
        tt2 = time.time()
        # print('tt2-tt1', tt2-tt1)
        return ap_dictionary
    except Exception as e:
        print_error("module yolo_utils", method_name, e)

