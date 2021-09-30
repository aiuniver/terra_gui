import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import numpy as np
import cv2
import random
import time
import colorsys

### DETECTION ###

def draw_bbox(image, bboxes, CLASSES, show_label=True, show_confidence=True,
              Text_colors=(255, 255, 0), rectangle_colors='', tracking=False):
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
                print("while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")

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

def bbox_iou(boxes1, boxes2):
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

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)
    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
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

def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
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

### LOSSES ###

def bboxes_iou(boxes1, boxes2):
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


def bbox_giou(boxes1, boxes2):
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

# testing (should be better than giou)
def bbox_ciou(boxes1, boxes2):
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

    u = (boxes1[..., 0] - boxes2[..., 0]) * (boxes1[..., 0] - boxes2[..., 0]) + (boxes1[..., 1] - boxes2[..., 1]) * (boxes1[..., 1] - boxes2[..., 1])
    d = u / c

    ar_gt = boxes2[..., 2] / boxes2[..., 3]
    ar_pred = boxes1[..., 2] / boxes1[..., 3]

    ar_loss = 4 / (np.pi * np.pi) * (tf.atan(ar_gt) - tf.atan(ar_pred)) * (tf.atan(ar_gt) - tf.atan(ar_pred))
    alpha = ar_loss / (1 - iou + ar_loss + 0.000001)
    ciou_term = d + alpha * ar_loss

    return iou - ciou_term

# class YoloLoss:
#     def __init__(self):
#
#     def __call__(self, y_true, y_pred, sample_weight=None):

def compute_loss(pred, conv, label, bboxes, i=0, CLASSES=None, STRIDES=None, YOLO_IOU_LOSS_THRESH=0.5):
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

    # If the largest iou is less than the threshold, it is considered that the prediction box contains no objects, then the background box
    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < YOLO_IOU_LOSS_THRESH, tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    # Calculate the loss of confidence
    # we hope that if the grid contains objects, then the network output prediction box has a confidence of 1 and 0 when there is no object.
    conf_loss = conf_focal * (
            respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss


### CREATE AND FIT MODEL ###

def decode(conv_output, NUM_CLASS, i=0, YOLO_TYPE="v3", STRIDES=None):

    if STRIDES is None:
        STRIDES = [8, 16, 32]
    if YOLO_TYPE == "v4" or YOLO_TYPE == "v5":
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
    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)

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

#
# def create_yolo(
#         input_shape: Tuple[int, int, int],
#         num_anchor: int,
#         model: Model,
#         num_classes: int,
#     ) -> Model:
#     """
#         Функция создания полной модели
#             Входные параметры:
#               input_shape - размерность входного изображения для модели YOLO
#               num_anchors - общее количество анкоров
#               model - спроектированная модель
#               num_classes - количество классов
#     """

# @tf.autograph.experimental.do_not_convert
def create_yolo(model, input_size=416, channels=3, training=False, classes=[]):
    num_class = len(classes)
    input_layer = keras.layers.Input([input_size, input_size, channels])

    # if TRAIN_YOLO_TINY:
    #     if YOLO_TYPE == "yolov4":
    #         conv_tensors = YOLOv4_tiny(input_layer, NUM_CLASS)
    #     if YOLO_TYPE == "yolov3":
    #         conv_tensors = YOLOv3_tiny(input_layer, NUM_CLASS)
    # else:
    #     if YOLO_TYPE == "yolov4":
    #         conv_tensors = YOLOv4(input_layer, NUM_CLASS)
    #     if YOLO_TYPE == "yolov3":
    #         conv_tensors = YOLOv3(input_layer, NUM_CLASS)
    conv_tensors = model(input_layer)
    # print(conv_tensors)
    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, num_class, i)
        if training: output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)
    output_tensors.reverse()
    # print(output_tensors)
    yolo = tf.keras.Model(input_layer, output_tensors)
    return yolo


class CustomModelYolo(keras.Model):

    def __init__(self, yolo, dataset, classes, train_epochs):
        super().__init__()
        self.yolo = yolo
        self.dataset = dataset
        self.CLASSES = classes
        self.train_epochs = train_epochs
        self.loss_fn = None
        self.optimizer = None
    # global TRAIN_FROM_CHECKPOINT

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # print(f'GPUs {gpus}')
    # if len(gpus) > 0:
    #     try:
    #         tf.config.experimental.set_memory_growth(gpus[0], True)
    #     except RuntimeError:
    #         pass

    # if os.path.exists(TRAIN_LOGDIR): shutil.rmtree(TRAIN_LOGDIR)
    # writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
    #
    # trainset = dtts.dataset['train']
    # testset = dtts.dataset['val']

        self.steps_per_epoch = int(len(self.dataset.dataset['train']) // 2)
        self.global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
        self.warmup_steps = 2 * self.steps_per_epoch  #TRAIN_WARMUP_EPOCHS
        self.total_steps = self.train_epochs * self.steps_per_epoch
        self.TRAIN_LR_INIT = 1e-4
        self.TRAIN_LR_END = 1e-6

    def compile(self, optimizer, loss):
        super(CustomModelYolo, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss
    # if TRAIN_TRANSFER:
    #     Darknet = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
    #     load_yolo_weights(Darknet, Darknet_weights)  # use darknet weights
    #
    # yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, training=True, CLASSES=TRAIN_CLASSES)
    # if TRAIN_FROM_CHECKPOINT:
    #     try:
    #         yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}")
    #     except ValueError:
    #         print("Shapes are incompatible, transfering Darknet weights")
    #         TRAIN_FROM_CHECKPOINT = False
    #
    # if TRAIN_TRANSFER and not TRAIN_FROM_CHECKPOINT:
    #     for i, l in enumerate(Darknet.layers):
    #         layer_weights = l.get_weights()
    #         if layer_weights != []:
    #             try:
    #                 yolo.layers[i].set_weights(layer_weights)
    #             except:
    #                 print("skipping", yolo.layers[i].name)

    # optimizer = tf.keras.optimizers.Adam()

    def train_step(self, data):
        # print(data)
        image_data, target = data[0], data[1:]
        # print(image_data)
        # print(target)
        with tf.GradientTape() as tape:
            pred_result = self.yolo(image_data['1'], training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            grid = 3 #if not TRAIN_YOLO_TINY else 2
            for i, elem in enumerate([[2, 3], [4, 5], [6, 7]]):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = self.loss_fn(pred, conv, *(target[0].get(str(elem[0])), target[0].get(str(elem[1]))), i,
                                          CLASSES=self.CLASSES)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss
            # self.optimizer.minimize(total_loss, self.yolo.trainable_variables, tape=tape)

            gradients = tape.gradient(total_loss, self.yolo.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.yolo.trainable_variables))

            # grads = tape.gradient(g_loss, self.generator.trainable_weights)
            # self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

            # update learning rate
            # about warmup: https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg
            self.global_steps.assign_add(1)
            # print(self.global_steps.value())

            if tf.less(self.global_steps.value(), self.warmup_steps) is not None:  # and not TRAIN_TRANSFER:
                lr = self.global_steps.value() / self.warmup_steps * self.TRAIN_LR_INIT
            else:
                lr = self.TRAIN_LR_END + 0.5 * (self.TRAIN_LR_INIT - self.TRAIN_LR_END) * (
                    (1 + tf.cos((self.global_steps.value() - self.warmup_steps) /
                                (self.total_steps - self.warmup_steps) * np.pi)))
            self.optimizer.lr.assign(tf.cast(lr, tf.float32))

            # # writing summary data
            # with writer.as_default():
            #     tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            #     tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            #     tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            #     tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            #     tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            # writer.flush()

        return {'global_steps': self.global_steps.value(), "optimizer.lr": self.optimizer.lr.value(),
                "giou_loss": giou_loss, "conf_loss": conf_loss, "prob_loss": prob_loss, "total_loss": total_loss}

    # validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

    def validate_step(self, data):
        image_data, target = data[0], data[1:]
        with tf.GradientTape() as tape:
            pred_result = self.yolo(image_data['1'], training=False)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            grid = 3 #if not TRAIN_YOLO_TINY else 2
            for i, elem in enumerate([[2, 3], [4, 5], [6, 7]]):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = self.loss_fn(pred, conv, *(target[0].get(str(elem[0])), target[0].get(str(elem[1]))),
                                          i, CLASSES=self.CLASSES)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

        return {"giou_loss": giou_loss, "conf_loss": conf_loss, "prob_loss": prob_loss, "total_loss": total_loss}

    # mAP_model = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)  # create second model to measure mAP
    # test_set = 70
    # best_val_loss = 1000  # should be large at start
    # for epoch in range(TRAIN_EPOCHS):
    #     for image_data, target in trainset.batch(2):
    #         results = train_step(image_data, target)
    #         cur_step = results[0] % steps_per_epoch
    #         print(
    #             "epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, giou_loss:{:7.2f}, conf_loss:{:7.2f}, prob_loss:{:7.2f}, total_loss:{:7.2f}"
    #             .format(epoch, cur_step, steps_per_epoch, results[1], results[2], results[3], results[4], results[5]))
    #
    #     if test_set == 0:
    #         print("configure TEST options to validate model")
    #         yolo.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))
    #         continue
    #
    #     count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
    #     for image_data, target in testset.batch(2):
    #         results = validate_step(image_data, target)
    #         count += 1
    #         giou_val += results[0]
    #         conf_val += results[1]
    #         prob_val += results[2]
    #         total_val += results[3]
    #     # writing validate summary data
    #     with validate_writer.as_default():
    #         tf.summary.scalar("validate_loss/total_val", total_val / count, step=epoch)
    #         tf.summary.scalar("validate_loss/giou_val", giou_val / count, step=epoch)
    #         tf.summary.scalar("validate_loss/conf_val", conf_val / count, step=epoch)
    #         tf.summary.scalar("validate_loss/prob_val", prob_val / count, step=epoch)
    #     validate_writer.flush()
    #
    #     print("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".
    #           format(giou_val / count, conf_val / count, prob_val / count, total_val / count))
    #
    #     if TRAIN_SAVE_CHECKPOINT and not TRAIN_SAVE_BEST_ONLY:
    #         save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER,
    #                                       TRAIN_MODEL_NAME + "_val_loss_{:7.2f}.h5".format(total_val / count))
    #         yolo.save_weights(save_directory)
    #     if TRAIN_SAVE_BEST_ONLY and best_val_loss > total_val / count:
    #         save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
    #         yolo.save_weights(save_directory)
    #         best_val_loss = total_val / count
    #     if not TRAIN_SAVE_BEST_ONLY and not TRAIN_SAVE_CHECKPOINT:
    #         save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
    #         yolo.save_weights(save_directory)
