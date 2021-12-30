import colorsys
import os
# import time

import matplotlib
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from tensorflow.python.keras.utils.np_utils import to_categorical

from terra_ai.callbacks.utils import sort_dict, round_loss_metric, fill_heatmap_front_structure, \
    fill_graph_front_structure, fill_graph_plot_data, set_preset_count
from terra_ai.data.training.extra import ExampleChoiceTypeChoice, BalanceSortedChoice
from terra_ai.settings import DEPLOY_PRESET_PERCENT
import terra_ai.exceptions.callbacks as exception


# noinspection PyTypeChecker,PyUnresolvedReferences
class BaseObjectDetectionCallback:
    name = 'BaseObjectDetectionCallback'

    def __init__(self):
        pass

    @staticmethod
    def get_x_array(options):
        x_val = None
        inverse_x_val = None
        return x_val, inverse_x_val

    @staticmethod
    def get_yolo_y_true(options, dataset_path):
        method_name = 'get_yolo_y_true'
        try:
            y_true = {}
            bb = []
            model_size = options.data.inputs.get(list(options.data.inputs.keys())[0]).shape[:2]
            for index in range(len(options.dataframe['val'])):
                image_path = os.path.join(
                    dataset_path, options.dataframe['val']['1_image'][index])
                img = Image.open(image_path)
                real_size = img.size
                scale_w = model_size[0] / real_size[0]
                scale_h = model_size[1] / real_size[1]
                coord = options.dataframe.get('val')['2_object_detection'][index].split(' ')
                bbox_data_gt = np.array([list(map(int, box.split(','))) for box in coord])
                bboxes_gt, classes_gt = bbox_data_gt[:, :4].astype('float'), bbox_data_gt[:, 4]
                classes_gt = to_categorical(
                    classes_gt, num_classes=len(options.data.outputs.get(2).classes_names)
                )
                bboxes_gt = np.concatenate(
                    [bboxes_gt[:, 1:2] * scale_h, bboxes_gt[:, 0:1] * scale_w,
                     bboxes_gt[:, 3:4] * scale_h, bboxes_gt[:, 2:3] * scale_w], axis=-1).astype('int')
                conf_gt = np.expand_dims(np.ones(len(bboxes_gt)), axis=-1)
                _bb = np.concatenate([bboxes_gt, conf_gt, classes_gt], axis=-1)
                bb.append(_bb)
            for channel in range(len(options.data.outputs.keys())):
                y_true[channel] = bb
            return y_true, None
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseObjectDetectionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def get_yolo_y_pred(array, options, sensitivity: float = 0.15, threashold: float = 0.1):
        method_name = 'get_yolo_y_pred'
        try:
            y_pred = {}
            name_classes = options.data.outputs.get(list(options.data.outputs.keys())[0]).classes_names
            for i, box_array in enumerate(array):
                channel_boxes = []
                for ex in box_array:
                    boxes = BaseObjectDetectionCallback().get_predict_boxes(
                        array=np.expand_dims(ex, axis=0),
                        name_classes=name_classes,
                        sensitivity=sensitivity,
                        threashold=threashold
                    )
                    channel_boxes.append(boxes)
                y_pred[i] = channel_boxes
            return y_pred
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseObjectDetectionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def get_inverse_array(array: dict, options, type="output"):
        inverse_array = {"train": {}, "val": {}}
        return inverse_array

    @staticmethod
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
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseObjectDetectionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def non_max_suppression_fast(boxes: np.ndarray, scores: np.ndarray, sensitivity: float = 0.15):
        method_name = 'non_max_suppression_fast'
        try:
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
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseObjectDetectionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def get_predict_boxes(array, name_classes: list, sensitivity: float = 0.15, threashold: float = 0.1):
        """
        Boxes for 1 example
        """
        method_name = 'get_predict_boxes'
        try:
            num_classes = len(name_classes)
            num_anchors = 3
            feats = np.reshape(array, (-1, array.shape[1], array.shape[2], num_anchors, num_classes + 5))
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
            pick, _ = BaseObjectDetectionCallback().non_max_suppression_fast(_boxes_out, _scores_out, sensitivity)
            return np.concatenate([_boxes_out[pick], _conf_param[pick], _scores_out[pick]], axis=-1)
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseObjectDetectionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def plot_boxes(true_bb, pred_bb, img_path, name_classes, colors, image_id, add_only_true=False, plot_true=True,
                   image_size=(416, 416), save_path='', return_mode='deploy'):
        method_name = 'plot_boxes'
        try:
            image = Image.open(img_path)
            real_size = image.size
            scale_w = real_size[0] / image_size[0]
            scale_h = real_size[1] / image_size[1]

            def resize_bb(boxes, scale_width, scale_height):
                coord = boxes[:, :4].astype('float')
                resized_coord = np.concatenate(
                    [coord[:, 0:1] * scale_height, coord[:, 1:2] * scale_width,
                     coord[:, 2:3] * scale_height, coord[:, 3:4] * scale_width], axis=-1).astype('int')
                resized_coord = np.concatenate([resized_coord, boxes[:, 4:]], axis=-1)
                return resized_coord

            true_bb = resize_bb(true_bb, scale_w, scale_h)
            pred_bb = resize_bb(pred_bb, scale_w, scale_h)

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

            font = ImageFont.load_default()
            thickness = (image.size[0] + image.size[1]) // 300 if (image.size[0] + image.size[1]) > 800 else 2
            image_pred = image.copy()
            if plot_true:
                classes = np.argmax(true_bb[:, 5:], axis=-1)
                for i, box in enumerate(true_bb[:, :4]):
                    draw = ImageDraw.Draw(image_pred)
                    true_class = name_classes[classes[i]]
                    label = ' {} '.format(true_class)
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
                label = ' {} {:.2f} '.format(predicted_class, score)
                label_size = draw.textsize(label, font)
                draw = draw_box(draw, box, colors[classes[i]], thickness,
                                label=label, label_size=label_size,
                                dash_mode=True, show_label=True)
                del draw

            save_predict_path = ''
            return_predict_path = ""
            if return_mode == 'deploy':
                save_predict_path = os.path.join(
                    save_path, "deploy_presets",
                    f"{return_mode}_od_{'predict_true' if plot_true else 'predict'}_image_{image_id}.webp")
                return_predict_path = os.path.join(
                    "deploy_presets",
                    f"{return_mode}_od_{'predict_true' if plot_true else 'predict'}_image_{image_id}.webp")
            if return_mode == 'callback':
                save_predict_path = os.path.join(
                    save_path, f"{return_mode}_od_{'predict_true' if plot_true else 'predict'}_image_{image_id}.webp")

            image_pred.save(save_predict_path)

            return_true_path = ''
            save_true_path = ''
            if add_only_true:
                image_true = image.copy()
                classes = np.argmax(true_bb[:, 5:], axis=-1)
                for i, box in enumerate(true_bb[:, :4]):
                    draw = ImageDraw.Draw(image_true)
                    true_class = name_classes[classes[i]]
                    label = ' {} '.format(true_class)
                    label_size = draw.textsize(label, font)
                    draw = draw_box(draw, box, colors[classes[i]], thickness,
                                    label=label, label_size=label_size,
                                    dash_mode=False, show_label=True)
                    del draw
                if return_mode == 'deploy':
                    save_true_path = os.path.join(
                        save_path, "deploy_presets", f"{return_mode}_od_true_image_{image_id}.webp")
                    return_true_path = os.path.join(
                        "deploy_presets", f"{return_mode}_od_true_image_{image_id}.webp")
                if return_mode == 'callback':
                    save_true_path = os.path.join(save_path, f"{return_mode}_od_true_image_{image_id}.webp")
                image_true.save(save_true_path)
            if return_mode == 'deploy':
                return return_predict_path, return_true_path
            if return_mode == 'callback':
                return save_predict_path, save_true_path
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseObjectDetectionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def get_yolo_example_statistic(true_bb, pred_bb, name_classes, sensitivity=0.25):
        method_name = 'get_yolo_example_statistic'
        try:
            compat = {
                'recognize': {"empty": [], 'unrecognize': []},
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
                    pick, _ = BaseObjectDetectionCallback().non_max_suppression_fast(
                        boxes, scores, sensitivity=sensitivity)
                    if len(pick) == 1:
                        mean_iou = BaseObjectDetectionCallback().bboxes_iou(boxes[0], boxes[1])
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
                        # count += 1
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
                        {"class_name": name_classes[np.argmax(true_bb[idx, 5:], axis=-1)]}
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
                        'mean_class': mean_class / len(compat['recognize'][cl]) if len(
                            compat['recognize'][cl]) else None,
                        'mean_overlap': mean_overlap / len(compat['recognize'][cl]) if len(
                            compat['recognize'][cl]) else None
                    }
            count += len(compat['recognize']['empty'])
            count = count + len(compat['recognize']['unrecognize'])
            compat['total_stat'] = {
                'total_conf': total_conf / count if count else 0.,
                'total_class': total_class / count if count else 0.,
                'total_overlap': total_overlap / count if count else 0.,
                'total_metric': (total_conf + total_class + total_overlap) / 3 / count if count else 0.
            }
            return compat
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseObjectDetectionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def prepare_example_idx_to_show(array: dict, true_array: dict, name_classes: list, box_channel,
                                    count: int, choice_type: str = "best", seed_idx: list = None,
                                    sensitivity: float = 0.25, get_optimal_channel=False):
        method_name = 'prepare_example_idx_to_show'
        try:
            if get_optimal_channel:
                channel_stat = []
                for channel in range(3):
                    total_metric = 0
                    for example in range(len(array.get(channel))):
                        total_metric += BaseObjectDetectionCallback().get_yolo_example_statistic(
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
                        BaseObjectDetectionCallback().get_yolo_example_statistic(
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
                true_false_dict = {'true': [], 'false': []}
                for example in range(len(array.get(box_channel))):
                    ex_stat = BaseObjectDetectionCallback().get_yolo_example_statistic(
                        true_bb=true_array.get(box_channel)[example],
                        pred_bb=array.get(box_channel)[example],
                        name_classes=name_classes,
                        sensitivity=sensitivity
                    )['total_stat']['total_metric']
                    if ex_stat > 0.7:
                        true_false_dict['true'].append(example)
                    else:
                        true_false_dict['false'].append(example)
                np.random.shuffle(true_false_dict['true'])
                np.random.shuffle(true_false_dict['false'])
                example_idx = []
                for _ in range(count):
                    if true_false_dict.get('true') and true_false_dict.get('false'):
                        key = np.random.choice(list(true_false_dict.keys()))
                    elif true_false_dict.get('true') and not true_false_dict.get('false'):
                        key = 'true'
                    else:
                        key = 'false'
                    example_idx.append(true_false_dict.get(key)[0])
                    true_false_dict.get(key).pop(0)
                np.random.shuffle(example_idx)
            else:
                example_idx = np.random.randint(0, len(true_array.get(box_channel)), count)
            return example_idx, box_channel
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseObjectDetectionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def postprocess_object_detection(predict_array, true_array, image_path: str, colors: list,
                                     sensitivity: float, image_id: int, save_path: str, show_stat: bool,
                                     name_classes: list, return_mode='deploy', image_size=(416, 416)):
        method_name = 'postprocess_object_detection'
        try:
            data = {"y_true": {}, "y_pred": {}, "stat": {}}
            if return_mode == 'deploy':
                pass

            if return_mode == 'callback':
                save_true_predict_path, _ = BaseObjectDetectionCallback().plot_boxes(
                    true_bb=true_array, pred_bb=predict_array, img_path=image_path, name_classes=name_classes,
                    colors=colors, image_id=image_id, add_only_true=False, plot_true=True, image_size=image_size,
                    save_path=save_path, return_mode=return_mode
                )

                data["y_true"] = {
                    "type": "image",
                    "data": [{"title": "Изображение", "value": save_true_predict_path,
                              "color_mark": None, "size": "large"}]
                }

                save_predict_path, _ = BaseObjectDetectionCallback().plot_boxes(
                    true_bb=true_array, pred_bb=predict_array, img_path=image_path, name_classes=name_classes,
                    colors=colors, image_id=image_id, add_only_true=False, plot_true=False, image_size=image_size,
                    save_path=save_path, return_mode=return_mode
                )

                data["y_pred"] = {
                    "type": "image",
                    "data": [{"title": "Изображение", "value": save_predict_path,
                              "color_mark": None, "size": "large"}]
                }
                if show_stat:
                    box_classes = []
                    for box in true_array:
                        cls = name_classes[np.argmax(box[5:], axis=-1)]
                        if cls not in box_classes:
                            box_classes.append(cls)
                    for box in predict_array:
                        cls = name_classes[np.argmax(box[5:], axis=-1)]
                        if cls not in box_classes:
                            box_classes.append(cls)
                    box_stat = BaseObjectDetectionCallback().get_yolo_example_statistic(
                        true_bb=true_array, pred_bb=predict_array, name_classes=name_classes,
                        sensitivity=sensitivity
                    )
                    data["stat"]["Общая точность"] = {
                        "type": "str",
                        "data": [{
                            "title": "Среднее",
                            "value": f"{np.round(box_stat['total_stat']['total_metric'] * 100, 2)}%",
                            "color_mark": 'success' if box_stat['total_stat']['total_metric'] >= 0.7 else 'wrong'
                        }]
                    }
                    data["stat"]['Средняя точность'] = {
                        "type": "str",
                        "data": [
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
                                "value": f"{np.round(box_stat['total_stat']['total_class'] * 100, 2)}%",
                                "color_mark": 'success' if box_stat['total_stat']['total_class'] >= 0.7 else 'wrong'
                            },
                        ]
                    }
                    for class_name in name_classes:
                        mean_overlap = box_stat['class_stat'][class_name]['mean_overlap'] \
                            if box_stat['class_stat'][class_name]['mean_overlap'] else 0.
                        mean_conf = box_stat['class_stat'][class_name]['mean_conf'] \
                            if box_stat['class_stat'][class_name]['mean_conf'] else 0.
                        mean_class = box_stat['class_stat'][class_name]['mean_class'] \
                            if box_stat['class_stat'][class_name]['mean_class'] else 0.
                        data["stat"][f'{class_name}'] = {
                            "type": "str",
                            "data": [
                                {
                                    "title": "Перекрытие",
                                    "value": "-" if class_name not in box_classes
                                    else f"{np.round(mean_overlap * 100, 2)}%",
                                    "color_mark": 'success' if mean_overlap and mean_overlap >= 0.7 else 'wrong'
                                },
                                {
                                    "title": "Объект",
                                    "value": "-" if class_name not in box_classes
                                    else f"{np.round(mean_conf * 100, 2)}%",
                                    "color_mark": 'success' if mean_conf and mean_conf >= 0.7 else 'wrong'
                                },
                                {
                                    "title": "Класс",
                                    "value": "-" if class_name not in box_classes
                                    else f"{np.round(mean_class * 100, 2)}%",
                                    "color_mark": 'success' if mean_class and mean_class >= 0.7 else 'wrong'
                                },
                            ]
                        }
                return data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseObjectDetectionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def postprocess_od_deploy(array, options, save_path: str = "", dataset_path: str = "", sensitivity=0.15,
                              threashold=0.1) -> dict:
        method_name = 'postprocess_od_deploy'
        try:
            return_data = {}
            y_true = BaseObjectDetectionCallback().get_yolo_y_true(options, dataset_path)[0]
            y_pred = BaseObjectDetectionCallback().get_yolo_y_pred(
                array, options, sensitivity=sensitivity, threashold=threashold)
            name_classes = options.data.outputs.get(list(options.data.outputs.keys())[0]).classes_names
            hsv_tuples = [(x / len(name_classes), 1., 1.) for x in range(len(name_classes))]
            colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
            image_size = options.data.inputs.get(list(options.data.inputs.keys())[0]).shape[:2]

            count = set_preset_count(len_array=len(y_pred[0]), preset_percent=DEPLOY_PRESET_PERCENT)
            example_idx, bb = BaseObjectDetectionCallback().prepare_example_idx_to_show(
                array=y_pred,
                true_array=y_true,
                name_classes=options.data.outputs.get(list(options.data.outputs.keys())[0]).classes_names,
                box_channel=None,
                count=count,
                choice_type='best',
                sensitivity=sensitivity,
                get_optimal_channel=True
            )
            return_data[bb] = []
            for ex in example_idx:
                img_path = os.path.join(dataset_path, options.dataframe['val']['1_image'][ex])
                img = Image.open(img_path)
                img = img.convert('RGB')
                source = os.path.join(save_path, "deploy_presets", f"deploy_od_initial_data_{ex}_box_{bb}.webp")
                return_source = os.path.join("deploy_presets", f"deploy_od_initial_data_{ex}_box_{bb}.webp")
                img.save(source, 'webp')
                save_predict_path, _ = BaseObjectDetectionCallback().plot_boxes(
                    true_bb=y_true[bb][ex], pred_bb=y_pred[bb][ex], img_path=img_path, name_classes=name_classes,
                    colors=colors, image_id=ex, add_only_true=False, plot_true=False, image_size=image_size,
                    save_path=save_path, return_mode='deploy'
                )
                return_data[bb].append({"source": return_source, "predict": save_predict_path})
            return return_data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseObjectDetectionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def get_box_square(bbs, imsize=(416, 416)):
        method_name = 'get_box_square'
        try:
            if len(bbs):
                square = 0
                for bb in bbs:
                    square += (bb[2] - bb[0]) * (bb[3] - bb[1])
                return square / len(bbs) / np.prod(imsize) * 100
            else:
                return 0.
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseObjectDetectionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def plot_bb_colormap(class_bb: dict, colors: list, name_classes: list, data_type: str,
                         save_path: str, imgsize=(416, 416)):
        method_name = 'plot_bb_colormap'
        try:
            template = np.zeros((imgsize[0], imgsize[1], 3))
            link_dict = {}
            total_len = 0
            for class_idx in class_bb[data_type].keys():
                total_len += len(class_bb[data_type][class_idx])
                class_template = np.zeros((imgsize[0], imgsize[1], 3))
                for box in class_bb[data_type][class_idx]:
                    template[box[0]:box[2], box[1]:box[3], :] += np.array(colors[class_idx])
                    class_template[box[0]:box[2], box[1]:box[3], :] += np.array(colors[class_idx])
                class_template = class_template / len(class_bb[data_type][class_idx])
                class_template = (class_template * 255 / class_template.max()).astype("uint8")
                img_save_path = os.path.join(
                    save_path, f"image_{data_type}_od_balance_colormap_class_{name_classes[class_idx]}.webp"
                )
                link_dict[name_classes[class_idx]] = img_save_path
                matplotlib.image.imsave(img_save_path, class_template)
            template = template / total_len
            template = (template * 255 / template.max()).astype('uint8')
            img_save_path = os.path.join(save_path, f"image_{data_type}_od_balance_colormap_all_classes.webp")
            link_dict['all_classes'] = img_save_path
            matplotlib.image.imsave(img_save_path, template)
            return link_dict
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseObjectDetectionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def resize_bb(boxes, scale_width, scale_height):
        coord = boxes[:, :4].astype('float')
        resized_coord = np.concatenate(
            [coord[:, 0:1] * scale_height, coord[:, 1:2] * scale_width,
             coord[:, 2:3] * scale_height, coord[:, 3:4] * scale_width], axis=-1).astype('int')
        resized_coord = np.concatenate([resized_coord, boxes[:, 4:]], axis=-1)
        return resized_coord

    @staticmethod
    def prepare_dataset_balance(options, class_colors, preset_path) -> dict:
        method_name = 'prepare_dataset_balance'
        try:
            dataset_balance = {}
            name_classes = options.data.outputs.get(list(options.data.outputs.keys())[0]).classes_names
            imsize = options.data.inputs.get(list(options.data.inputs.keys())[0]).shape
            class_bb = {}
            dataset_balance["output"] = {'class_count': {}, 'class_square': {}, 'colormap': {}}
            for data_type in ["train", "val"]:
                class_bb[data_type] = {}
                for cl in range(len(name_classes)):
                    class_bb[data_type][cl] = []
                for index in range(len(options.dataframe[data_type])):
                    y_true = options.dataframe.get(data_type)['2_object_detection'][index].split(' ')
                    img_path = os.path.join(
                        options.data.path, options.dataframe.get('val')['1_image'][index])
                    # img_path = options.dataframe.get(data_type)['1_image'][index]
                    bbox_data_gt = np.array([list(map(int, box.split(','))) for box in y_true])
                    bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                    bboxes_gt = np.concatenate(
                        [bboxes_gt[:, 1:2], bboxes_gt[:, 0:1], bboxes_gt[:, 3:4], bboxes_gt[:, 2:3]], axis=-1)
                    image = Image.open(img_path)
                    real_size = image.size
                    scale_w = real_size[0] / image_size[0]
                    scale_h = real_size[1] / image_size[1]
                    bboxes_gt = resize_bb(bboxes_gt, scale_w, scale_h)

                    for i, cl in enumerate(classes_gt):
                        class_bb[data_type][cl].append(bboxes_gt[i].tolist())

                dataset_balance["output"]['class_count'][data_type] = {}
                dataset_balance["output"]['class_square'][data_type] = {}
                for key, item in class_bb[data_type].items():
                    dataset_balance["output"]['class_count'][data_type][name_classes[key]] = len(item)
                    dataset_balance["output"]['class_square'][data_type][name_classes[key]] = \
                        round_loss_metric(
                            BaseObjectDetectionCallback().get_box_square(item, imsize=(imsize[0], imsize[1])))
                dataset_balance["output"]['colormap'][data_type] = BaseObjectDetectionCallback().plot_bb_colormap(
                    class_bb=class_bb, colors=class_colors, name_classes=name_classes, data_type=data_type,
                    save_path=preset_path, imgsize=(imsize[0], imsize[1])
                )
            return dataset_balance
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseObjectDetectionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def get_intermediate_result(options, yolo_interactive_config, y_pred, y_true, example_idx,
                                dataset_path, class_colors, preset_path) -> dict:
        method_name = 'get_intermediate_result'
        try:
            return_data = {}
            if yolo_interactive_config.intermediate_result.show_results:
                for idx in range(yolo_interactive_config.intermediate_result.num_examples):
                    return_data[f"{idx + 1}"] = {
                        'initial_data': {},
                        'true_value': {},
                        'predict_value': {},
                        'tags_color': {},
                        'statistic_values': {}
                    }
                    image_path = os.path.join(
                        dataset_path, options.dataframe.get('val')['1_image'][example_idx[idx]])
                    out = yolo_interactive_config.intermediate_result.box_channel
                    data = BaseObjectDetectionCallback().postprocess_object_detection(
                        predict_array=y_pred.get(out)[example_idx[idx]],
                        true_array=y_true.get(out)[example_idx[idx]],
                        image_path=image_path,
                        colors=class_colors,
                        sensitivity=yolo_interactive_config.intermediate_result.sensitivity,
                        image_id=idx,
                        image_size=options.data.inputs.get(list(options.data.inputs.keys())[0]).shape[:2],
                        name_classes=options.data.outputs.get(list(options.data.outputs.keys())[0]).classes_names,
                        save_path=preset_path,
                        return_mode='callback',
                        show_stat=yolo_interactive_config.intermediate_result.show_statistic
                    )
                    if data.get('y_true'):
                        return_data[f"{idx + 1}"]['true_value'][f"Выходной слой"] = data.get('y_true')
                    return_data[f"{idx + 1}"]['predict_value'][f"Выходной слой"] = data.get('y_pred')

                    if data.get('stat'):
                        return_data[f"{idx + 1}"]['statistic_values'] = data.get('stat')
                    else:
                        return_data[f"{idx + 1}"]['statistic_values'] = {}
            return return_data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseObjectDetectionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def get_statistic_data_request(yolo_interactive_config, options, y_true, y_pred) -> list:
        method_name = 'get_statistic_data_request'
        try:
            return_data = []
            box_channel = yolo_interactive_config.statistic_data.box_channel
            name_classes = options.data.outputs.get(list(options.data.outputs.keys())[0]).classes_names
            object_tt = 0
            object_tf = 0
            object_ft = 0
            line_names = []
            class_accuracy_hist = {}
            class_loss_hist = {}
            class_coord_accuracy = {}
            for class_name in name_classes:
                line_names.append(class_name)
                class_accuracy_hist[class_name] = []
                class_loss_hist[class_name] = []
                class_coord_accuracy[class_name] = []
            line_names.append('empty')

            class_matrix = np.zeros((len(line_names), len(line_names)))
            for i in range(len(y_pred.get(box_channel))):
                example_stat = BaseObjectDetectionCallback().get_yolo_example_statistic(
                    true_bb=y_true.get(box_channel)[i],
                    pred_bb=y_pred.get(box_channel)[i],
                    name_classes=name_classes,
                    sensitivity=yolo_interactive_config.statistic_data.sensitivity
                )
                object_ft += len(example_stat['recognize']['empty'])
                object_tf += len(example_stat['recognize']['unrecognize'])
                for class_name in line_names:
                    if class_name != 'empty':
                        object_tt += len(example_stat['recognize'][class_name])
                    for item in example_stat['recognize'][class_name]:
                        class_matrix[line_names.index(class_name)][line_names.index(item['pred_class'])] += 1
                        if class_name != 'empty':
                            if item['class_result']:
                                class_accuracy_hist[class_name].append(item['class_conf'])
                                class_coord_accuracy[class_name].append(item['overlap'])
                            else:
                                class_loss_hist[class_name].append(item['class_conf'])
                for item in example_stat['recognize']['unrecognize']:
                    class_matrix[line_names.index(item['class_name'])][-1] += 1
            for class_name in name_classes:
                class_accuracy_hist[class_name] = np.round(np.mean(class_accuracy_hist[class_name]) * 100, 2).item() \
                    if class_accuracy_hist[class_name] else 0.
                class_coord_accuracy[class_name] = np.round(np.mean(class_coord_accuracy[class_name]) * 100, 2).item() \
                    if class_coord_accuracy[class_name] else 0.
                class_loss_hist[class_name] = np.round(np.mean(class_loss_hist[class_name]) * 100, 2).item() if \
                    class_loss_hist[class_name] else 0.
            object_matrix = [[object_tt, object_tf], [object_ft, 0]]
            class_matrix_percent = []
            for i in class_matrix:
                class_matrix_percent.append(i * 100 / np.sum(i) if np.sum(i) else np.zeros_like(i))
            class_matrix_percent = np.round(class_matrix_percent, 2).tolist()
            class_matrix = class_matrix.astype('int').tolist()
            return_data.append(
                fill_heatmap_front_structure(
                    _id=1,
                    _type="heatmap",
                    graph_name=f"Бокс-канал «{box_channel}» - Матрица неточностей определения классов",
                    short_name=f"{box_channel} - Матрица классов",
                    x_label="Предсказание",
                    y_label="Истинное значение",
                    labels=line_names,
                    data_array=class_matrix,
                    data_percent_array=class_matrix_percent,
                )
            )
            return_data.append(
                fill_heatmap_front_structure(
                    _id=2,
                    _type="valheatmap",
                    graph_name=f"Бокс-канал «{box_channel}» - Матрица неточностей определения объектов",
                    short_name=f"{box_channel} - Матрица объектов",
                    x_label="Предсказание",
                    y_label="Истинное значение",
                    labels=['Объект', 'Отсутствие'],
                    data_array=object_matrix,
                    data_percent_array=None,
                )
            )
            return_data.append(
                fill_graph_front_structure(
                    _id=3,
                    _type='histogram',
                    graph_name=f'Бокс-канал «{box_channel}» - Средняя точность определеня  классов',
                    short_name=f"{box_channel} - точность классов",
                    x_label="Имя класса",
                    y_label="Средняя точность, %",
                    plot_data=[
                        fill_graph_plot_data(x=name_classes, y=[class_accuracy_hist[i] for i in name_classes])
                    ],
                )
            )
            return_data.append(
                fill_graph_front_structure(
                    _id=4,
                    _type='histogram',
                    graph_name=f'Бокс-канал «{box_channel}» - Средняя ошибка определеня  классов',
                    short_name=f"{box_channel} - ошибка классов",
                    x_label="Имя класса",
                    y_label="Средняя ошибка, %",
                    plot_data=[
                        fill_graph_plot_data(x=name_classes, y=[class_loss_hist[i] for i in name_classes])
                    ],
                )
            )
            return_data.append(
                fill_graph_front_structure(
                    _id=5,
                    _type='histogram',
                    graph_name=f'Бокс-канал «{box_channel}» - '
                               f'Средняя точность определения  координат объекта класса (MeanIoU)',
                    short_name=f"{box_channel} - координаты классов",
                    x_label="Имя класса",
                    y_label="Средняя точность, %",
                    plot_data=[
                        fill_graph_plot_data(x=name_classes, y=[class_coord_accuracy[i] for i in name_classes])
                    ],
                )
            )
            return return_data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseObjectDetectionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def get_balance_data_request(options, dataset_balance, interactive_config) -> list:
        method_name = 'get_balance_data_request'
        try:
            return_data = []
            _id = 0
            for class_type in dataset_balance.get("output").keys():
                preset = {}
                if class_type in ["class_count", "class_square"]:
                    for data_type in ['train', 'val']:
                        names, count = sort_dict(
                            dict_to_sort=dataset_balance.get("output").get(class_type).get(data_type),
                            mode=interactive_config.data_balance.sorted.name
                        )
                        type_name = 'Тренировочная' if data_type == 'train' else 'Проверочная'
                        cls_name = 'баланс присутсвия' if class_type == 'class_count' else 'процент пространства'
                        preset[data_type] = fill_graph_front_structure(
                            _id=_id,
                            _type='histogram',
                            type_data=data_type,
                            graph_name=f"{type_name} выборка - {cls_name}",
                            short_name=f"{type_name} - "
                                       f"{'присутсвие' if class_type == 'class_count' else 'пространство'}",
                            x_label="Название класса",
                            y_label="Значение",
                            plot_data=[fill_graph_plot_data(x=names, y=count)],
                        )
                        _id += 1
                    return_data.append(preset)

                if class_type == "colormap":
                    classes_name = sorted(
                        list(dataset_balance.get("output").get('colormap').get('train').keys()))
                    for class_name in classes_name:
                        preset = {}
                        for data_type in ['train', 'val']:
                            _dict = dataset_balance.get("output").get('colormap').get(data_type)
                            preset[data_type] = fill_graph_front_structure(
                                _id=_id,
                                _type='colormap',
                                type_data=data_type,
                                graph_name=f"{'Тренировочная' if data_type == 'train' else 'Проверочная'} выборка "
                                           f"- Цветовая карта "
                                           f"{'всех классов' if class_name == 'all_classes' else 'класса'} "
                                           f"{'' if class_name == 'all_classes' else class_name}",
                                short_name="",
                                x_label="",
                                y_label="",
                                plot_data=_dict.get(class_name),
                            )
                            _id += 1
                        return_data.append(preset)
            return return_data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                BaseObjectDetectionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc


class YoloV3Callback(BaseObjectDetectionCallback):
    name = 'YoloV3Callback'

    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def get_y_true(options, dataset_path):
        return YoloV3Callback().get_yolo_y_true(options=options, dataset_path=dataset_path)

    @staticmethod
    def get_y_pred(y_pred, options, sensitivity: float = 0.15, threashold: float = 0.1):
        return YoloV3Callback().get_yolo_y_pred(
            array=y_pred, options=options, sensitivity=sensitivity, threashold=threashold)

    @staticmethod
    def postprocess_deploy(array, options, save_path: str = "", dataset_path: str = "", sensitivity=0.15,
                           threashold=0.1):
        return YoloV3Callback().postprocess_od_deploy(
            array=array, options=options, save_path=save_path, dataset_path=dataset_path,
            sensitivity=sensitivity, threashold=threashold
        )

    @staticmethod
    def dataset_balance(options, y_true, preset_path: str, class_colors) -> dict:
        return YoloV3Callback().prepare_dataset_balance(
            options=options, class_colors=class_colors, preset_path=preset_path)

    @staticmethod
    def intermediate_result_request(options, interactive_config, example_idx, dataset_path,
                                    preset_path, x_val, inverse_x_val, y_pred, inverse_y_pred,
                                    y_true, inverse_y_true, class_colors):
        return YoloV3Callback().get_intermediate_result(
            options=options, yolo_interactive_config=interactive_config, y_true=y_true, y_pred=y_pred,
            example_idx=example_idx, dataset_path=dataset_path, class_colors=class_colors, preset_path=preset_path)

    @staticmethod
    def statistic_data_request(interactive_config, inverse_y_true, y_pred, inverse_y_pred, options=None,
                               y_true=None) -> list:
        return YoloV3Callback().get_statistic_data_request(
            yolo_interactive_config=interactive_config, options=options, y_true=y_true, y_pred=y_pred)

    @staticmethod
    def balance_data_request(options, dataset_balance, interactive_config) -> list:
        return YoloV3Callback().get_balance_data_request(
            options=options, dataset_balance=dataset_balance, interactive_config=interactive_config)


class YoloV4Callback(BaseObjectDetectionCallback):
    name = 'YoloV4Callback'

    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def get_y_true(options, dataset_path):
        return YoloV4Callback().get_yolo_y_true(options=options, dataset_path=dataset_path)

    @staticmethod
    def get_y_pred(y_pred, options, sensitivity: float = 0.15, threashold: float = 0.1):
        return YoloV4Callback().get_yolo_y_pred(
            array=y_pred, options=options, sensitivity=sensitivity, threashold=threashold)

    @staticmethod
    def postprocess_deploy(array, options, save_path: str = "", dataset_path: str = "", sensitivity=0.15,
                           threashold=0.1):
        return YoloV4Callback().postprocess_od_deploy(
            array=array, options=options, save_path=save_path, dataset_path=dataset_path,
            sensitivity=sensitivity, threashold=threashold
        )

    @staticmethod
    def dataset_balance(options, y_true, preset_path: str, class_colors) -> dict:
        return YoloV4Callback().prepare_dataset_balance(
            options=options, class_colors=class_colors, preset_path=preset_path)

    @staticmethod
    def intermediate_result_request(options, interactive_config, example_idx, dataset_path,
                                    preset_path, x_val, inverse_x_val, y_pred, inverse_y_pred,
                                    y_true, inverse_y_true, class_colors):
        return YoloV4Callback().get_intermediate_result(
            options=options, yolo_interactive_config=interactive_config, y_true=y_true, y_pred=y_pred,
            example_idx=example_idx, dataset_path=dataset_path, class_colors=class_colors, preset_path=preset_path)

    @staticmethod
    def statistic_data_request(interactive_config, inverse_y_true, y_pred, inverse_y_pred, options=None,
                               y_true=None) -> list:
        return YoloV4Callback().get_statistic_data_request(
            yolo_interactive_config=interactive_config, options=options, y_true=y_true, y_pred=y_pred)

    @staticmethod
    def balance_data_request(options, dataset_balance, interactive_config) -> list:
        return YoloV4Callback().get_balance_data_request(
            options=options, dataset_balance=dataset_balance, interactive_config=interactive_config)
