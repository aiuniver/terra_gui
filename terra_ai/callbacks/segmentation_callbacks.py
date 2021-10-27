import colorsys
import os
from typing import Optional

import matplotlib
import numpy as np
from PIL import Image
from pandas import DataFrame
from tensorflow.python.keras.utils.np_utils import to_categorical

from terra_ai.callbacks.utils import dice_coef, sort_dict, get_y_true
from terra_ai.data.datasets.dataset import DatasetOutputsData
from terra_ai.data.datasets.extra import LayerInputTypeChoice, LayerEncodingChoice
from terra_ai.data.training.extra import ExampleChoiceTypeChoice, BalanceSortedChoice
from terra_ai.settings import CALLBACK_CLASSIFICATION_TREASHOLD_VALUE, DEPLOY_PRESET_PERCENT


class SegmentationCallback:
    def __init__(self):
        pass

    @staticmethod
    def get_x_array():
        x_val = None
        inverse_x_val = None
        return x_val, inverse_x_val

    @staticmethod
    def postprocess_initial_source(options, input_id: int, example_id: int, dataset_path: str, preset_path: str,
                                   save_id: int = None, return_mode='deploy'):
        column_idx = []
        for inp in options.data.inputs.keys():
            for column_name in options.dataframe.get('val').columns:
                if column_name.split('_')[0] == f"{inp}":
                    column_idx.append(options.dataframe.get('val').columns.tolist().index(column_name))
        initial_file_path = os.path.join(
            dataset_path, options.dataframe.get('val').iat[example_id, column_idx[0]]
        )
        if not save_id:
            return str(os.path.abspath(initial_file_path))

        img = Image.open(initial_file_path)
        img = img.resize(options.data.inputs.get(input_id).shape[0:2][::-1], Image.ANTIALIAS)
        img = img.convert('RGB')
        source = os.path.join(preset_path, f"initial_data_image_{save_id}_input_{input_id}.webp")
        img.save(source, 'webp')
        if return_mode == 'deploy':
            return source
        if return_mode == 'callback':
            data_type = LayerInputTypeChoice.Image.name
            data = [
                {
                    "title": "Изображение",
                    "value": source,
                    "color_mark": None
                }
            ]
            return data, data_type.lower()

    @staticmethod
    def postprocess_deploy(array, options, save_path: str = "", dataset_path: str = "") -> dict:
        return_data = {}
        for i, output_id in enumerate(options.data.outputs.keys()):
            true_array = get_y_true(options, output_id)
            if len(options.data.outputs.keys()) > 1:
                postprocess_array = array[i]
            else:
                postprocess_array = array
            example_idx = prepare_example_idx_to_show(
                array=postprocess_array,
                true_array=true_array,
                options=options,
                output=output_id,
                count=int(len(true_array) * DEPLOY_PRESET_PERCENT / 100)
            )
            return_data[output_id] = []
            data = []
            for j, cls in enumerate(options.data.outputs.get(output_id).classes_names):
                data.append((cls, options.data.outputs.get(output_id).classes_colors[j].as_rgb_tuple()))
            for idx in example_idx:
                input_id = list(options.data.inputs.keys())[0]
                return_data[output_id].append(
                    {
                        "source": SegmentationCallback.postprocess_initial_source(
                            options=options,
                            input_id=input_id,
                            save_id=idx,
                            example_id=idx,
                            dataset_path=dataset_path,
                            preset_path=save_path,
                            return_mode='deploy'
                        ),
                        "segment": postprocess_segmentation(
                            predict_array=array[idx],
                            true_array=None,
                            options=options.data.outputs.get(output_id),
                            output_id=output_id,
                            image_id=idx,
                            save_path=save_path,
                            return_mode='deploy'
                        ),
                        "data": data
                    }
                )
        return return_data


class TextSegmentationCallback:
    def __init__(self):
        pass

    @staticmethod
    def get_x_array():
        x_val = None
        inverse_x_val = None
        return x_val, inverse_x_val

    @staticmethod
    def postprocess_initial_source(options, example_id: int, return_mode='deploy'):
        column_idx = []
        for inp in options.data.inputs.keys():
            for column_name in options.dataframe.get('val').columns:
                if column_name.split('_')[0] == f"{inp}":
                    column_idx.append(options.dataframe.get('val').columns.tolist().index(column_name))
        data = []
        data_type = ""
        source = ""
        for column in column_idx:
            source = options.dataframe.get('val').iat[example_id, column]
            if return_mode == 'deploy':
                break
            if return_mode == 'callback':
                data_type = LayerInputTypeChoice.Text.name
                title = "Текст"
                data = [
                    {
                        "title": title,
                        "value": source,
                        "color_mark": None
                    }
                ]
        if return_mode == 'deploy':
            return source
        if return_mode == 'callback':
            return data, data_type.lower()

    @staticmethod
    def postprocess_deploy(array, options) -> dict:
        return_data = {}
        for i, output_id in enumerate(options.data.outputs.keys()):
            true_array = get_y_true(options, output_id)
            if len(options.data.outputs.keys()) > 1:
                postprocess_array = array[i]
            else:
                postprocess_array = array
            example_idx = prepare_example_idx_to_show(
                array=postprocess_array,
                true_array=true_array,
                options=options,
                output=output_id,
                count=int(len(true_array) * DEPLOY_PRESET_PERCENT / 100)
            )
            return_data[output_id] = {"color_map": None, "data": []}
            output_column = list(options.instructions.get(output_id).keys())[0]
            for idx in example_idx:
                source, segment, colors = postprocess_text_segmentation(
                    pred_array=postprocess_array[idx],
                    options=options.data.outputs[output_id],
                    dataframe=options.dataframe.get("val"),
                    example_id=idx,
                    dataset_params=options.instructions.get(output_id).get(output_column),
                    return_mode='deploy'
                )
                return_data[output_id]["data"].append(
                    {"source": source, "format": segment}
                )
            return_data[output_id]["color_map"] = colors
        return return_data


def prepare_example_idx_to_show(array: np.ndarray, true_array: np.ndarray, options, output: int, count: int,
                                choice_type: str = "best", seed_idx: list = None) -> dict:
    example_idx = []
    encoding = options.data.outputs.get(output).encoding
    if choice_type == ExampleChoiceTypeChoice.best or choice_type == ExampleChoiceTypeChoice.worst:
        if encoding == LayerEncodingChoice.ohe:
            array = to_categorical(
                np.argmax(array, axis=-1),
                num_classes=options.data.outputs.get(output).num_classes
            )
        if encoding == LayerEncodingChoice.multi:
            array = np.where(array >= CALLBACK_CLASSIFICATION_TREASHOLD_VALUE / 100, 1, 0)
        dice_val = dice_coef(true_array, array, batch_mode=True)
        dice_dict = dict(zip(np.arange(0, len(dice_val)), dice_val))
        if choice_type == ExampleChoiceTypeChoice.best:
            example_idx, _ = sort_dict(dice_dict, mode=BalanceSortedChoice.descending)
            example_idx = example_idx[:count]
        if choice_type == ExampleChoiceTypeChoice.worst:
            example_idx, _ = sort_dict(dice_dict, mode=BalanceSortedChoice.ascending)
            example_idx = example_idx[:count]

    elif choice_type == ExampleChoiceTypeChoice.seed and len(seed_idx):
        example_idx = seed_idx[:count]

    elif choice_type == ExampleChoiceTypeChoice.random:
        if encoding == LayerEncodingChoice.ohe:
            array = to_categorical(
                np.argmax(array, axis=-1),
                num_classes=options.data.outputs.get(output).num_classes
            )
        if encoding == LayerEncodingChoice.multi:
            array = np.where(array >= CALLBACK_CLASSIFICATION_TREASHOLD_VALUE / 100, 1, 0)
        dice_val = dice_coef(true_array, array, batch_mode=True)

        true_id = []
        false_id = []
        for i, ex in enumerate(dice_val):
            if ex >= CALLBACK_CLASSIFICATION_TREASHOLD_VALUE / 100:
                true_id.append(i)
            else:
                false_id.append(i)
        np.random.shuffle(true_id)
        np.random.shuffle(false_id)
        true_false_dict = {'true': true_id, 'false': false_id}

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
        example_idx = np.random.randint(0, len(true_array), count)

    return example_idx


def postprocess_segmentation(predict_array: np.ndarray, true_array: Optional[np.ndarray], options: DatasetOutputsData,
                             output_id: int, image_id: int, save_path: str, colors: list = None, return_mode='deploy',
                             show_stat: bool = False):
    data = {
        "y_true": {},
        "y_pred": {},
        "stat": {}
    }
    if return_mode == 'deploy':
        array = np.expand_dims(np.argmax(predict_array, axis=-1), axis=-1) * 512
        for i, color in enumerate(options.classes_colors):
            array = np.where(
                array == i * 512,
                np.array(color.as_rgb_tuple()),
                array
            )
        array = array.astype("uint8")
        img_save_path = os.path.join(
            save_path,
            f"image_segmentation_postprocessing_{image_id}_output_{output_id}.webp"
        )
        matplotlib.image.imsave(img_save_path, array)
        return img_save_path

    if return_mode == 'callback':
        y_true = np.expand_dims(np.argmax(true_array, axis=-1), axis=-1) * 512
        for i, color in enumerate(colors):
            y_true = np.where(y_true == i * 512, np.array(color), y_true)
        y_true = y_true.astype("uint8")
        y_true_save_path = os.path.join(
            save_path,
            f"true_segmentation_data_image_{image_id}_output_{output_id}.webp"
        )
        matplotlib.image.imsave(y_true_save_path, y_true)

        data["y_true"] = {
            "type": "image",
            "data": [
                {
                    "title": "Изображение",
                    "value": y_true_save_path,
                    "color_mark": None
                }
            ]
        }

        y_pred = np.expand_dims(np.argmax(predict_array, axis=-1), axis=-1) * 512
        for i, color in enumerate(colors):
            y_pred = np.where(y_pred == i * 512, np.array(color), y_pred)
        y_pred = y_pred.astype("uint8")
        y_pred_save_path = os.path.join(
            save_path,
            f"predict_segmentation_data_image_{image_id}_output_{output_id}.webp"
        )
        matplotlib.image.imsave(y_pred_save_path, y_pred)
        data["y_pred"] = {
            "type": "image",
            "data": [
                {
                    "title": "Изображение",
                    "value": y_pred_save_path,
                    "color_mark": None
                }
            ]
        }
        if show_stat:
            data["stat"] = {
                "type": "str",
                "data": []
            }
            y_true = np.array(true_array).astype('int')
            y_pred = to_categorical(np.argmax(predict_array, axis=-1), options.num_classes).astype('int')
            count = 0
            mean_val = 0
            for idx, cls in enumerate(options.classes_names):
                dice_val = np.round(
                    dice_coef(y_true[:, :, idx], y_pred[:, :, idx], batch_mode=False) * 100, 1)
                count += 1
                mean_val += dice_val
                data["stat"]["data"].append(
                    {
                        'title': cls,
                        'value': f"{dice_val}%",
                        'color_mark': 'success' if dice_val >= 90 else 'wrong'
                    }
                )
            data["stat"]["data"].insert(
                0,
                {
                    'title': "Средняя точность",
                    'value': f"{round(mean_val / count, 2)}%",
                    'color_mark': 'success' if mean_val / count >= 90 else 'wrong'
                }
            )
        return data


def postprocess_text_segmentation(pred_array: np.ndarray, options: DatasetOutputsData, dataframe: DataFrame,
                                  dataset_params: dict, example_id: int, return_mode='deploy',
                                  class_colors: list = None, show_stat: bool = False, true_array: np.ndarray = None):
    def add_tags_to_word(word: str, tag_: str):
        if tag_:
            for t in tag_:
                word = f"<{t[1:-1]}>{word}</{t[1:-1]}>"
            return word
        else:
            return f"<p1>{word}</p1>"

    def reformat_tags(y_array: np.ndarray, tag_list: list, sensitivity: float = 0.9):
        norm_array = np.where(y_array >= sensitivity, 1, 0).astype('int')
        reformat_list = []
        for word_tag in norm_array:
            if np.sum(word_tag) == 0:
                reformat_list.append(None)
            else:
                mix_tag = []
                for wt, wtag in enumerate(word_tag):
                    if wtag == 1:
                        mix_tag.append(tag_list[wt])
                reformat_list.append(mix_tag)
        return reformat_list

    def text_colorization(text: str, label_array: np.ndarray, tag_list: list):
        text = text.split(" ")
        labels = reformat_tags(label_array, tag_list)
        colored_text = []
        for w, word in enumerate(text):
            colored_text.append(add_tags_to_word(word, labels[w]))
        return ' '.join(colored_text)

    # TODO: пока исходим что для сегментации текста есть только один вход с текстом, если будут сложные модели
    #  на сегментацию текста на несколько входов то придется искать решения

    classes_names = {}
    dataset_tags = dataset_params.get("open_tags").split()
    colors = {}
    if options.classes_colors and return_mode == 'deploy':
        for i, name in enumerate(dataset_tags):
            colors[name] = options.classes_colors[i].as_rgb_tuple()
            classes_names[name] = options.classes_names[i]
    elif class_colors and return_mode == 'callback':
        for i, name in enumerate(dataset_tags):
            colors[name] = class_colors[i]
            classes_names[name] = options.classes_names[i]
    else:
        hsv_tuples = [(x / len(dataset_tags), 1., 1.) for x in range(len(dataset_tags))]
        gen_colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        gen_colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), gen_colors))
        for i, name in enumerate(dataset_tags):
            colors[name] = gen_colors[i]
            classes_names[name] = options.classes_names[i]

    if return_mode == 'deploy':
        initinal_text = dataframe.iat[example_id, 0]
        text_segmentation = text_colorization(
            text=initinal_text, label_array=pred_array, tag_list=dataset_tags
        )

        data = [('<p1>', '<p1>', (200, 200, 200))]
        for tag in colors.keys():
            data.append(
                (tag, classes_names[tag], colors[tag])
            )
        return initinal_text, text_segmentation, data

    if return_mode == 'callback':
        data = {"y_true": {}, "y_pred": {}, "tags_color": {}, "stat": {}}
        text_for_preparation = dataframe.iat[example_id, 0]
        true_text_segmentation = text_colorization(
            text=text_for_preparation, label_array=true_array, tag_list=dataset_tags,
        )

        data["y_true"] = {
            "type": "segmented_text",
            "data": [{"title": "Текст", "value": true_text_segmentation, "color_mark": None}]
        }
        pred_text_segmentation = text_colorization(
            text=text_for_preparation, label_array=pred_array, tag_list=dataset_tags,
        )
        data["y_pred"] = {
            "type": "segmented_text",
            "data": [{"title": "Текст", "value": pred_text_segmentation, "color_mark": None}]
        }
        colors_ = {}
        for key, val in colors.items():
            colors_[key[1:-1]] = val
        data["tags_color"] = colors_

        if show_stat:
            data["stat"] = {
                "type": "str",
                "data": []
            }
            y_true = np.array(true_array).astype('int')
            y_pred = np.where(pred_array >= 0.9, 1., 0.)
            count = 0
            mean_val = 0
            for idx, cls in enumerate(options.classes_names):
                if np.sum(y_true[:, idx]) == 0 and np.sum(y_pred[:, idx]) == 0:
                    data["stat"]["data"].append({'title': cls, 'value': "-", 'color_mark': None})
                elif np.sum(y_true[:, idx]) == 0:
                    data["stat"]["data"].append({'title': cls, 'value': "0.0%", 'color_mark': 'wrong'})
                    count += 1
                else:
                    class_recall = np.sum(y_true[:, idx] * y_pred[:, idx]) * 100 / np.sum(y_true[:, idx])
                    data["stat"]["data"].append(
                        {
                            'title': cls,
                            'value': f"{np.round(class_recall, 1)} %",
                            'color_mark': 'success' if class_recall >= 90 else 'wrong'
                        }
                    )
                    count += 1
                    mean_val += class_recall
            if count and mean_val / count >= 90:
                mean_color_mark = "success"
                mean_stat = f"{round(mean_val / count, 1)}%"
            elif count and mean_val / count < 90:
                mean_color_mark = "wrong"
                mean_stat = f"{round(mean_val / count, 2)}%"
            else:
                mean_color_mark = None
                mean_stat = '-'
            data["stat"]["data"].insert(
                0, {'title': "Средняя точность", 'value': mean_stat, 'color_mark': mean_color_mark}
            )
        return data

