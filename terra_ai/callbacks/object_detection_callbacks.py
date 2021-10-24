import colorsys
import os
from typing import Optional

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from tensorflow.python.keras.utils.np_utils import to_categorical

from terra_ai.callbacks.utils import sort_dict
from terra_ai.data.training.extra import ExampleChoiceTypeChoice, BalanceSortedChoice
from terra_ai.settings import DEPLOY_PRESET_PERCENT


class YoloV3Callback:
    def __init__(self):
        pass

    @staticmethod
    def get_x_array():
        x_val = None
        inverse_x_val = None
        return x_val, inverse_x_val


class YoloV4Callback:
    def __init__(self):
        pass

    @staticmethod
    def get_x_array():
        x_val = None
        inverse_x_val = None
        return x_val, inverse_x_val


def get_yolo_y_true(options):
    y_true = {}
    bb = []
    for index in range(len(options.dataset['val'])):
        coord = options.dataframe.get('val').iloc[index, 1].split(' ')
        bbox_data_gt = np.array([list(map(int, box.split(','))) for box in coord])
        bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
        classes_gt = to_categorical(
            classes_gt, num_classes=len(options.data.outputs.get(2).classes_names)
        )
        bboxes_gt = np.concatenate(
            [bboxes_gt[:, 1:2], bboxes_gt[:, 0:1], bboxes_gt[:, 3:4], bboxes_gt[:, 2:3]], axis=-1)
        conf_gt = np.expand_dims(np.ones(len(bboxes_gt)), axis=-1)
        _bb = np.concatenate([bboxes_gt, conf_gt, classes_gt], axis=-1)
        bb.append(_bb)
    for channel in range(len(options.Y.get('val').keys())):
        y_true[channel] = bb
    return y_true


def get_yolo_y_pred(array, options, sensitivity: float = 0.15, threashold: float = 0.1):
    y_pred = {}
    name_classes = options.data.outputs.get(list(options.data.outputs.keys())[0]).classes_names
    for i, box_array in enumerate(array):
        channel_boxes = []
        for ex in box_array:
            boxes = get_predict_boxes(
                array=np.expand_dims(ex, axis=0),
                name_classes=name_classes,
                bb_size=i,
                sensitivity=sensitivity,
                threashold=threashold
            )
            channel_boxes.append(boxes)
        y_pred[i] = channel_boxes
    return y_pred


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


def non_max_suppression_fast(boxes: np.ndarray, scores: np.ndarray, sensitivity: float = 0.15):
    """
    :param boxes: list of unscaled bb coordinates
    :param scores: class probability in ohe
    :param sensitivity: float from 0 to 1
    """
    if len(boxes) == 0:
        return [], []

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    classes = np.argmax(scores, axis=-1)
    idxs = np.argsort(classes)[..., ::-1]

    mean_iou = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        mean_iou.append(overlap)
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > sensitivity)[0])))

    return pick, mean_iou


def get_predict_boxes(array, name_classes: list, bb_size: int = 1, sensitivity: float = 0.15,
                      threashold: float = 0.1):
    """
    Boxes for 1 example
    """
    num_classes = len(name_classes)
    anchors = np.array([[10, 13], [16, 30], [33, 23],
                        [30, 61], [62, 45], [59, 119],
                        [116, 90], [156, 198], [373, 326]])

    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    level_anchor = bb_size
    num_anchors = len(anchors[anchor_mask[level_anchor]])

    grid_shape = array.shape[1:3]

    feats = np.reshape(array, (-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5))

    xy_param = feats[..., :2]
    wh_param = feats[..., 2:4]
    conf_param = feats[..., 4:5]
    class_param = feats[..., 5:]

    box_yx = xy_param[..., ::-1].copy()
    box_hw = wh_param[..., ::-1].copy()

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    _boxes = np.concatenate([
        box_mins[..., 0:1],
        box_mins[..., 1:2],
        box_maxes[..., 0:1],
        box_maxes[..., 1:2]
    ], axis=-1)

    _boxes_reshape = np.reshape(_boxes, (-1, 4))
    _box_scores = conf_param * class_param
    _box_scores_reshape = np.reshape(_box_scores, (-1, num_classes))
    _class_param_reshape = np.reshape(class_param, (-1, num_classes))
    mask = _box_scores_reshape >= threashold
    _boxes_out = np.zeros_like(_boxes_reshape[0:1])
    _scores_out = np.zeros_like(_box_scores_reshape[0:1])
    _class_param_out = np.zeros_like(_class_param_reshape[0:1])
    for cl in range(num_classes):
        if np.sum(mask[:, cl]):
            _boxes_out = np.concatenate((_boxes_out, _boxes_reshape[mask[:, cl]]), axis=0)
            _scores_out = np.concatenate((_scores_out, _box_scores_reshape[mask[:, cl]]), axis=0)
            _class_param_out = np.concatenate((_class_param_out, _class_param_reshape[mask[:, cl]]), axis=0)
    _boxes_out = _boxes_out[1:].astype('int')
    _scores_out = _scores_out[1:]
    _class_param_out = _class_param_out[1:]
    _conf_param = (_scores_out / _class_param_out)[:, :1]
    pick, _ = non_max_suppression_fast(_boxes_out, _scores_out, sensitivity)
    return np.concatenate([_boxes_out[pick], _conf_param[pick], _scores_out[pick]], axis=-1)


def plot_boxes(true_bb, pred_bb, img_path, name_classes, colors, image_id, add_only_true=False, plot_true=True,
               image_size=(416, 416), save_path='', return_mode='deploy'):
    image = Image.open(img_path)
    image = image.resize(image_size, Image.BICUBIC)

    def draw_box(draw, box, color, thickness, label=None, label_size=None,
                 dash_mode=False, show_label=False):
        top, left, bottom, right = box
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
            for th in range(thickness):
                draw.rectangle(
                    [left + th, top + th, right - th, bottom - th],
                    outline=color,
                )

        if show_label:
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=color
            )
            draw.text(tuple(text_origin), label, fill=(255, 255, 255), font=font)
        return draw

    font = ImageFont.load_default()
    thickness = (image.size[0] + image.size[1]) // 300
    image_pred = image.copy()
    if plot_true:
        classes = np.argmax(true_bb[:, 5:], axis=-1)
        for i, box in enumerate(true_bb[:, :4]):
            draw = ImageDraw.Draw(image_pred)
            true_class = name_classes[classes[i]]
            label = '{}'.format(true_class)
            label_size = draw.textsize(label, font)
            draw = draw_box(draw, box, colors[classes[i]], thickness,
                            label=label, label_size=label_size,
                            dash_mode=False, show_label=True)
            del draw

    classes = np.argmax(pred_bb[:, 5:], axis=-1)
    for i, box in enumerate(pred_bb[:, :4]):
        draw = ImageDraw.Draw(image_pred)
        predicted_class = name_classes[classes[i]]
        score = pred_bb[:, 5:][i][classes[i]]  # * pred_bb[:, 4][i]
        label = '{} {:.2f}'.format(predicted_class, score)
        label_size = draw.textsize(label, font)
        draw = draw_box(draw, box, colors[classes[i]], thickness,
                        label=label, label_size=label_size,
                        dash_mode=True, show_label=True)
        del draw

    save_predict_path = os.path.join(
        save_path, f"{return_mode}_od_{'predict_true' if plot_true else 'predict'}_image_{image_id}.webp")
    image_pred.save(save_predict_path)

    save_true_path = ''
    if add_only_true:
        image_true = image.copy()
        classes = np.argmax(true_bb[:, 5:], axis=-1)
        for i, box in enumerate(true_bb[:, :4]):
            draw = ImageDraw.Draw(image_true)
            true_class = name_classes[classes[i]]
            label = '{}'.format(true_class)
            label_size = draw.textsize(label, font)
            draw = draw_box(draw, box, colors[classes[i]], thickness,
                            label=label, label_size=label_size,
                            dash_mode=False, show_label=True)
            del draw

        save_true_path = os.path.join(save_path, f"{return_mode}_od_true_image_{image_id}.webp")
        image_true.save(save_true_path)

    return save_predict_path, save_true_path


def get_yolo_example_statistic(true_bb, pred_bb, name_classes, sensitivity=0.25):
    compat = {
        'recognize': {
            "empty": [],
            'unrecognize': []
        },
        'class_stat': {},
        'total_stat': {}
    }
    for name in name_classes:
        compat['recognize'][name] = []

    predict = {}
    for i, k in enumerate(pred_bb[:, :4]):
        predict[i] = {
            'pred_class': name_classes[np.argmax(pred_bb[:, 5:][i])],
            'conf': pred_bb[:, 4][i].item(),
            'class_conf': pred_bb[:, 5:][i][np.argmax(pred_bb[:, 5:][i])],
        }

    count = 0
    total_conf = 0
    total_class = 0
    total_overlap = 0
    all_true = list(np.arange(len(true_bb)))
    for i, tr in enumerate(true_bb[:, :4]):
        for j, pr in enumerate(pred_bb[:, :4]):
            boxes = np.array([true_bb[:, :4][i], pred_bb[:, :4][j]])
            scores = np.array([true_bb[:, 5:][i], pred_bb[:, 5:][j]])
            pick, _ = non_max_suppression_fast(boxes, scores, sensitivity=sensitivity)
            if len(pick) == 1:
                mean_iou = bboxes_iou(boxes[0], boxes[1])
                compat['recognize'][name_classes[np.argmax(true_bb[:, 5:][i], axis=-1)]].append(
                    {
                        'pred_class': name_classes[np.argmax(pred_bb[:, 5:][j], axis=-1)],
                        'conf': pred_bb[:, 4][j].item(),
                        'class_conf': pred_bb[:, 5:][j][np.argmax(pred_bb[:, 5:][j], axis=-1)],
                        'class_result': True if np.argmax(true_bb[:, 5:][i], axis=-1) == np.argmax(
                            pred_bb[:, 5:][j], axis=-1) else False,
                        'overlap': mean_iou.item()
                    }
                )
                if np.argmax(true_bb[:, 5:][i], axis=-1) == np.argmax(pred_bb[:, 5:][j], axis=-1):
                    count += 1
                    total_conf += pred_bb[:, 4][j].item()
                    total_class += pred_bb[:, 5:][j][np.argmax(pred_bb[:, 5:][j], axis=-1)]
                    total_overlap += mean_iou.item()

                try:
                    predict.pop(j)
                    all_true.pop(all_true.index(i))
                except:
                    continue

    for val in predict.values():
        compat['recognize']['empty'].append(val)

    if all_true:
        for idx in all_true:
            compat['recognize']['unrecognize'].append(
                {
                    "class_name": name_classes[np.argmax(true_bb[idx, 5:], axis=-1)]
                }
            )

    for cl in compat['recognize'].keys():
        if cl != 'empty' and cl != 'unrecognize':
            mean_conf = 0
            mean_class = 0
            mean_overlap = 0
            for pr in compat['recognize'][cl]:
                if pr['class_result']:
                    mean_conf += pr['conf']
                    mean_class += pr['class_conf']
                    mean_overlap += pr['overlap']
            compat['class_stat'][cl] = {
                'mean_conf': mean_conf / len(compat['recognize'][cl]) if len(compat['recognize'][cl]) else None,
                'mean_class': mean_class / len(compat['recognize'][cl]) if len(compat['recognize'][cl]) else None,
                'mean_overlap': mean_overlap / len(compat['recognize'][cl]) if len(
                    compat['recognize'][cl]) else None
            }
    compat['total_stat'] = {
        'total_conf': total_conf / count if count else 0.,
        'total_class': total_class / count if count else 0.,
        'total_overlap': total_overlap / count if count else 0.,
        'total_metric': (total_conf + total_class + total_overlap) / 3 / count if count else 0.
    }
    return compat


def prepare_yolo_example_idx_to_show(array: dict, true_array: dict, name_classes: list, box_channel: Optional[int],
                                     count: int, choice_type: str = "best", seed_idx: list = None,
                                     sensitivity: float = 0.25, get_optimal_channel=False):
    if get_optimal_channel:
        channel_stat = []
        for channel in range(3):
            total_metric = 0
            for example in range(len(array.get(channel))):
                total_metric += get_yolo_example_statistic(
                    true_bb=true_array.get(channel)[example],
                    pred_bb=array.get(channel)[example],
                    name_classes=name_classes,
                    sensitivity=sensitivity
                )['total_stat']['total_metric']
            channel_stat.append(total_metric / len(array.get(channel)))
        box_channel = int(np.argmax(channel_stat, axis=-1))

    if choice_type == ExampleChoiceTypeChoice.best or choice_type == ExampleChoiceTypeChoice.worst:
        stat = []
        for example in range(len(array.get(box_channel))):
            stat.append(
                get_yolo_example_statistic(
                    true_bb=true_array.get(box_channel)[example],
                    pred_bb=array.get(box_channel)[example],
                    name_classes=name_classes,
                    sensitivity=sensitivity
                )['total_stat']['total_metric']
            )
        stat_dict = dict(zip(np.arange(0, len(stat)), stat))
        if choice_type == ExampleChoiceTypeChoice.best:
            example_idx, _ = sort_dict(stat_dict, mode=BalanceSortedChoice.descending)
            example_idx = example_idx[:count]
        else:
            example_idx, _ = sort_dict(stat_dict, mode=BalanceSortedChoice.ascending)
            example_idx = example_idx[:count]

    elif choice_type == ExampleChoiceTypeChoice.seed:
        example_idx = seed_idx[:count]

    elif choice_type == ExampleChoiceTypeChoice.random:
        example_idx = np.random.randint(0, len(true_array.get(box_channel)), count)
    else:
        example_idx = np.arange(count)
    return example_idx, box_channel


def postprocess_object_detection(predict_array, true_array, image_path: str, colors: list,
                                 sensitivity: float, image_id: int, save_path: str, show_stat: bool,
                                 name_classes: list, return_mode='deploy', image_size=(416, 416)):
    data = {
        "y_true": {},
        "y_pred": {},
        "stat": {}
    }
    if return_mode == 'deploy':
        pass

    if return_mode == 'callback':
        save_true_predict_path, _ = plot_boxes(
            true_bb=true_array,
            pred_bb=predict_array,
            img_path=image_path,
            name_classes=name_classes,
            colors=colors,
            image_id=image_id,
            add_only_true=False,
            plot_true=True,
            image_size=image_size,
            save_path=save_path,
            return_mode=return_mode
        )

        data["y_true"] = {
            "type": "image",
            "data": [
                {
                    "title": "Изображение",
                    "value": save_true_predict_path,
                    "color_mark": None
                }
            ]
        }

        save_predict_path, _ = plot_boxes(
            true_bb=true_array,
            pred_bb=predict_array,
            img_path=image_path,
            name_classes=name_classes,
            colors=colors,
            image_id=image_id,
            add_only_true=False,
            plot_true=False,
            image_size=image_size,
            save_path=save_path,
            return_mode=return_mode
        )

        data["y_pred"] = {
            "type": "image",
            "data": [
                {
                    "title": "Изображение",
                    "value": save_predict_path,
                    "color_mark": None
                }
            ]
        }
        if show_stat:
            box_stat = get_yolo_example_statistic(
                true_bb=true_array,
                pred_bb=predict_array,
                name_classes=name_classes,
                sensitivity=sensitivity
            )
            data["stat"]["Общая точность"] = [
                {
                    "title": "Среднее",
                    "value": f"{np.round(box_stat['total_stat']['total_metric'] * 100, 2)}%",
                    "color_mark": 'success' if box_stat['total_stat']['total_conf'] >= 0.7 else 'wrong'
                },
            ]
            data["stat"]['Средняя точность'] = [
                {
                    "title": "Перекрытие",
                    "value": f"{np.round(box_stat['total_stat']['total_overlap'] * 100, 2)}%",
                    "color_mark": 'success' if box_stat['total_stat']['total_overlap'] >= 0.7 else 'wrong'
                },
                {
                    "title": "Объект",
                    "value": f"{np.round(box_stat['total_stat']['total_conf'] * 100, 2)}%",
                    "color_mark": 'success' if box_stat['total_stat']['total_conf'] >= 0.7 else 'wrong'
                },
                {
                    "title": "Класс",
                    "value": f"{np.round(box_stat['total_stat']['total_overlap'] * 100, 2)}%",
                    "color_mark": 'success' if box_stat['total_stat']['total_overlap'] >= 0.7 else 'wrong'
                },
            ]

            for class_name in name_classes:
                mean_overlap = box_stat['class_stat'][class_name]['mean_overlap']
                mean_conf = box_stat['class_stat'][class_name]['mean_conf']
                mean_class = box_stat['class_stat'][class_name]['mean_class']
                data["stat"][class_name] = [
                    {
                        "title": "Перекрытие",
                        "value": "-" if mean_overlap is None else f"{np.round(mean_overlap * 100, 2)}%",
                        "color_mark": 'success' if mean_overlap and mean_overlap >= 0.7 else 'wrong'
                    },
                    {
                        "title": "Объект",
                        "value": "-" if mean_conf is None else f"{np.round(mean_conf * 100, 2)}%",
                        "color_mark": 'success' if mean_conf and mean_conf >= 0.7 else 'wrong'
                    },
                    {
                        "title": "Класс",
                        "value": "-" if mean_class is None else f"{np.round(mean_class * 100, 2)}%",
                        "color_mark": 'success' if mean_class and mean_class >= 0.7 else 'wrong'
                    },
                ]
        return data


def postprocess_deploy(array, options, save_path: str = "", dataset_path: str = "", sensitivity=0.15,
                       threashold=0.1) -> dict:
    return_data = {}
    y_true = get_yolo_y_true(options)
    y_pred = get_yolo_y_pred(array, options, sensitivity=sensitivity, threashold=threashold)
    name_classes = options.data.outputs.get(list(options.data.outputs.keys())[0]).classes_names
    hsv_tuples = [(x / len(name_classes), 1., 1.) for x in range(len(name_classes))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    image_size = options.data.inputs.get(list(options.data.inputs.keys())[0]).shape[:2]

    example_idx, bb = prepare_yolo_example_idx_to_show(
        array=y_pred,
        true_array=y_true,
        name_classes=options.data.outputs.get(list(options.data.outputs.keys())[0]).classes_names,
        box_channel=None,
        count=int(len(y_pred[0]) * DEPLOY_PRESET_PERCENT / 100),
        choice_type='best',
        sensitivity=sensitivity,
        get_optimal_channel=True
    )
    return_data[bb] = []
    for ex in example_idx:
        save_predict_path, _ = plot_boxes(
            true_bb=y_true[bb][ex],
            pred_bb=y_pred[bb][ex],
            img_path=os.path.join(dataset_path, options.dataframe['val'].iat[ex, 0]),
            name_classes=name_classes,
            colors=colors,
            image_id=ex,
            add_only_true=False,
            plot_true=True,
            image_size=image_size,
            save_path=save_path,
            return_mode='deploy'
        )
        return_data[bb].append(
            {
                'predict_img': save_predict_path
            }
        )
    return return_data
