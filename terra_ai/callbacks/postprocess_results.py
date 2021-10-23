import colorsys
import copy
from typing import Optional

import matplotlib
from PIL import Image, ImageFont, ImageDraw
from pandas import DataFrame
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.utils.np_utils import to_categorical

from terra_ai.callbacks.utils import round_list, dice_coef, sort_dict
from terra_ai.data.training.extra import ExampleChoiceTypeChoice, BalanceSortedChoice, ArchitectureChoice
from terra_ai.data.datasets.dataset import DatasetOutputsData, DatasetData
from terra_ai.data.datasets.extra import LayerOutputTypeChoice, DatasetGroupChoice, LayerInputTypeChoice, \
    LayerEncodingChoice

import os
import numpy as np
from pydub import AudioSegment
import moviepy.editor as moviepy_editor

from terra_ai.settings import DEPLOY_PRESET_PERCENT, CALLBACK_CLASSIFICATION_TREASHOLD_VALUE, \
    CALLBACK_REGRESSION_TREASHOLD_VALUE


class PostprocessResults:

    @staticmethod
    def get_y_true(options, output_id):
        if not options.data.use_generator:
            y_true = options.Y.get('val').get(f"{output_id}")
        else:
            y_true = []
            for _, y_val in options.dataset['val'].batch(1):
                y_true.extend(y_val.get(f'{output_id}').numpy())
            y_true = np.array(y_true)
        return y_true

    @staticmethod
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

    @staticmethod
    def get_yolo_y_pred(array, options, sensitivity: float = 0.15, threashold: float = 0.1):
        y_pred = {}
        name_classes = options.data.outputs.get(list(options.data.outputs.keys())[0]).classes_names
        for i, box_array in enumerate(array):
            channel_boxes = []
            for ex in box_array:
                boxes = PostprocessResults().get_predict_boxes(
                    array=np.expand_dims(ex, axis=0),
                    name_classes=name_classes,
                    bb_size=i,
                    sensitivity=sensitivity,
                    threashold=threashold
                )
                channel_boxes.append(boxes)
            y_pred[i] = channel_boxes
        return y_pred

    @staticmethod
    def get_x_array(options):
        x_val = None
        inverse_x_val = None
        if options.data.architecture in [ArchitectureChoice.Basic, ArchitectureChoice.ImageClassification,
                                         ArchitectureChoice.ImageSegmentation, ArchitectureChoice.TextSegmentation,
                                         ArchitectureChoice.TextClassification, ArchitectureChoice.AudioClassification,
                                         ArchitectureChoice.VideoClassification,
                                         ArchitectureChoice.DataframeClassification,
                                         ArchitectureChoice.DataframeRegression, ArchitectureChoice.Timeseries,
                                         ArchitectureChoice.TimeseriesTrend]:
            if options.data.group == DatasetGroupChoice.keras:
                x_val = options.X.get("val")
            dataframe = False
            for inp in options.data.inputs.keys():
                if options.data.inputs.get(inp).task == LayerInputTypeChoice.Dataframe:
                    dataframe = True
                    break
            ts = False
            for out in options.data.outputs.keys():
                if options.data.outputs.get(out).task == LayerOutputTypeChoice.Timeseries or \
                        options.data.outputs.get(out).task == LayerOutputTypeChoice.TimeseriesTrend:
                    ts = True
                    break
            if dataframe and not options.data.use_generator:
                x_val = options.X.get("val")

            elif dataframe and options.data.use_generator:
                x_val = {}
                for inp in options.dataset['val'].keys():
                    x_val[inp] = []
                    for x_val_, _ in options.dataset['val'].batch(1):
                        x_val[inp].extend(x_val_.get(f'{inp}').numpy())
                    x_val[inp] = np.array(x_val[inp])
            else:
                pass

            if ts:
                inverse_x_val = {}
                for inp in x_val.keys():
                    preprocess_dict = options.preprocessing.preprocessing.get(int(inp))
                    inverse_x = np.zeros_like(x_val.get(inp)[:, :, 0:1])
                    for i, column in enumerate(preprocess_dict.keys()):
                        if type(preprocess_dict.get(column)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                            _options = {
                                int(inp): {
                                    column: x_val.get(inp)[:, :, i]
                                }
                            }
                            inverse_col = np.expand_dims(
                                options.preprocessing.inverse_data(_options).get(int(inp)).get(column), axis=-1)
                        else:
                            inverse_col = x_val.get(inp)[:, :, i:i + 1]
                        inverse_x = np.concatenate([inverse_x, inverse_col], axis=-1)
                    inverse_x_val[inp] = inverse_x[:, :, 1:]
        return x_val, inverse_x_val

    @staticmethod
    def postprocess_results(array, options, save_path: str = "", dataset_path: str = "", sensitivity=0.15,
                            threashold=0.1) -> dict:
        x_array, inverse_x_array = PostprocessResults().get_x_array(options)
        return_data = {}

        if options.data.architecture in [ArchitectureChoice.Basic, ArchitectureChoice.ImageClassification,
                                         ArchitectureChoice.ImageSegmentation, ArchitectureChoice.TextSegmentation,
                                         ArchitectureChoice.TextClassification, ArchitectureChoice.AudioClassification,
                                         ArchitectureChoice.VideoClassification,
                                         ArchitectureChoice.DataframeClassification,
                                         ArchitectureChoice.DataframeRegression, ArchitectureChoice.Timeseries,
                                         ArchitectureChoice.TimeseriesTrend]:
            for i, output_id in enumerate(options.data.outputs.keys()):
                true_array = PostprocessResults().get_y_true(options, output_id)
                if len(options.data.outputs.keys()) > 1:
                    postprocess_array = array[i]
                else:
                    postprocess_array = array
                example_idx = PostprocessResults().prepare_example_idx_to_show(
                    array=postprocess_array,
                    true_array=true_array,
                    options=options,
                    output=output_id,
                    count=int(len(true_array) * DEPLOY_PRESET_PERCENT / 100)
                )
                if options.data.outputs[output_id].task == LayerOutputTypeChoice.Classification:
                    return_data[output_id] = []
                    _id = 1
                    for idx in example_idx:
                        input_id = list(options.data.inputs.keys())[0]
                        source = PostprocessResults().postprocess_initial_source(
                            options=options,
                            input_id=input_id,
                            save_id=_id,
                            example_id=idx,
                            dataset_path=dataset_path,
                            preset_path=save_path,
                            x_array=None if not x_array else x_array.get(f"{input_id}"),
                            inverse_x_array=None if not inverse_x_array else inverse_x_array.get(f"{input_id}"),
                            return_mode='deploy'
                        )
                        actual_value, predict_values = PostprocessResults().postprocess_classification(
                            predict_array=np.expand_dims(postprocess_array[idx], axis=0),
                            true_array=true_array[idx],
                            options=options.data.outputs[output_id],
                            return_mode='deploy'
                        )

                        return_data[output_id].append(
                            {
                                "source": source,
                                "actual": actual_value,
                                "data": predict_values[0]
                            }
                        )
                        _id += 1

                elif options.data.outputs[output_id].task == LayerOutputTypeChoice.TimeseriesTrend:
                    return_data[output_id] = {}
                    # TODO: считаетм что инпут один
                    input_id = list(options.data.inputs.keys())[0]
                    inp_col_id = []
                    for j, out_col in enumerate(options.data.columns.get(output_id).keys()):
                        for k, inp_col in enumerate(options.data.columns.get(input_id).keys()):
                            if out_col.split('_', 1)[-1] == inp_col.split('_', 1)[-1]:
                                inp_col_id.append((k, inp_col, j, out_col))
                                break
                    preprocess = options.preprocessing.preprocessing.get(output_id)
                    for channel in inp_col_id:
                        return_data[output_id][channel[3]] = []
                        for idx in example_idx:
                            if type(preprocess.get(channel[3])).__name__ in ['StandardScaler', 'MinMaxScaler']:
                                inp_options = {int(output_id): {
                                    channel[3]: options.X.get('val').get(f"{input_id}")[idx, channel[0]:channel[0] + 1]}
                                }
                                inverse_true = options.preprocessing.inverse_data(inp_options).get(output_id).get(
                                    channel[3])
                                inverse_true = inverse_true.squeeze().astype('float').tolist()
                            else:
                                inverse_true = options.X.get('val').get(f"{input_id}")[
                                               idx, channel[0]:channel[0] + 1].squeeze().astype('float').tolist()
                            actual_value, predict_values = PostprocessResults().postprocess_classification(
                                predict_array=np.expand_dims(postprocess_array[idx], axis=0),
                                true_array=true_array[idx],
                                options=options.data.outputs[output_id],
                                return_mode='deploy'
                            )
                            button_save_path = os.path.join(
                                save_path, f"ts_trend_button_channel_{channel[2]}_image_{idx}.jpg")
                            return_data[output_id][channel[3]].append(
                                {
                                    "button_link": button_save_path,
                                    "data": [inverse_true, predict_values]
                                }
                            )

                elif options.data.outputs[output_id].task == LayerOutputTypeChoice.Segmentation:
                    return_data[output_id] = []
                    data = []
                    for j, cls in enumerate(options.data.outputs.get(output_id).classes_names):
                        data.append((cls, options.data.outputs.get(output_id).classes_colors[j].as_rgb_tuple()))
                    for idx in example_idx:
                        input_id = list(options.data.inputs.keys())[0]
                        colors = [color.as_rgb_tuple() for color in options.data.outputs.get(output_id).classes_colors]
                        return_data[output_id].append(
                            {
                                "source": PostprocessResults().postprocess_initial_source(
                                    options=options,
                                    input_id=input_id,
                                    save_id=idx,
                                    example_id=idx,
                                    dataset_path=dataset_path,
                                    preset_path=save_path,
                                    x_array=None if not x_array else x_array.get(f"{input_id}"),
                                    inverse_x_array=None if not inverse_x_array else inverse_x_array.get(f"{input_id}"),
                                    return_mode='deploy'
                                ),
                                "segment": PostprocessResults().postprocess_segmentation(
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

                elif options.data.outputs[output_id].task == LayerOutputTypeChoice.TextSegmentation:
                    return_data[output_id] = {
                        "color_map": None,
                        "data": []
                    }
                    output_column = list(options.instructions.get(output_id).keys())[0]
                    for idx in example_idx:
                        source, segment, colors = PostprocessResults().postprocess_text_segmentation(
                            pred_array=postprocess_array[idx],
                            options=options.data.outputs[output_id],
                            dataframe=options.dataframe.get("val"),
                            example_id=idx,
                            dataset_params=options.instructions.get(output_id).get(output_column),
                            return_mode='deploy'
                        )
                        return_data[output_id]["data"].append(
                            {
                                "source": source,
                                "format": segment,
                            }
                        )
                    return_data[output_id]["color_map"] = colors

                elif options.data.outputs[output_id].task == LayerOutputTypeChoice.Timeseries:
                    return_data[output_id] = []
                    preprocess = options.preprocessing.preprocessing.get(output_id)
                    for idx in example_idx:
                        data = {
                            'source': {},
                            'predict': {}
                        }
                        for inp in options.data.inputs.keys():
                            for k, inp_col in enumerate(options.data.columns.get(inp).keys()):
                                data['source'][inp_col.split('_', 1)[-1]] = \
                                    round_list(list(inverse_x_array[f"{inp}"][idx][:, k]))

                        for ch, channel in enumerate(options.data.columns.get(output_id).keys()):
                            if type(preprocess.get(channel)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                                inp_options = {output_id: {
                                    channel: options.Y.get('val').get(f"{output_id}")[idx, :, ch:ch + 1]}
                                }
                                inverse_true = options.preprocessing.inverse_data(inp_options).get(output_id).get(
                                    channel)
                                inverse_true = round_list(
                                    inverse_true.squeeze().astype('float').tolist())
                                out_options = {int(output_id): {
                                    channel: array[idx, :, ch:ch + 1].reshape(-1, 1)}
                                }
                                inverse_pred = options.preprocessing.inverse_data(out_options).get(output_id).get(
                                    channel)
                                inverse_pred = round_list(inverse_pred.squeeze().astype('float').tolist())
                            else:
                                inverse_true = options.Y.get('val').get(f"{output_id}")[
                                               idx, :, ch:ch + 1].squeeze().astype('float').tolist()
                                inverse_pred = array[idx, :, ch:ch + 1].squeeze().astype('float').tolist()
                            data['predict'][channel.split('_', 1)[-1]] = [inverse_true, inverse_pred]
                        return_data[output_id].append(data)

                elif options.data.outputs[output_id].task == LayerOutputTypeChoice.Regression:
                    return_data[output_id] = {
                        'preset': [],
                        'label': []
                    }
                    source_col = []
                    for inp in options.data.inputs.keys():
                        source_col.extend(list(options.data.columns.get(inp).keys()))
                    preprocess = options.preprocessing.preprocessing.get(output_id)
                    for idx in example_idx:
                        row_list = []
                        for inp_col in source_col:
                            row_list.append(f"{options.dataframe.get('val')[inp_col][idx]}")
                        return_data[output_id]['preset'].append(row_list)
                        for ch, col in enumerate(list(options.data.columns.get(output_id).keys())):
                            if type(preprocess.get(col)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                                _options = {int(output_id): {col: array[idx, ch:ch + 1].reshape(-1, 1)}}
                                inverse_col = options.preprocessing.inverse_data(_options).get(output_id).get(col)
                                inverse_col = inverse_col.squeeze().astype('float').tolist()
                            else:
                                inverse_col = array[idx, ch:ch + 1].astype('float').tolist()
                        return_data[output_id]['label'].append([str(inverse_col)])

                else:
                    return_data[output_id] = []

        elif options.data.architecture in [ArchitectureChoice.YoloV3, ArchitectureChoice.YoloV4]:
            y_true = PostprocessResults().get_yolo_y_true(options)
            y_pred = PostprocessResults().get_yolo_y_pred(array, options, sensitivity=sensitivity,
                                                          threashold=threashold)
            name_classes = options.data.outputs.get(list(options.data.outputs.keys())[0]).classes_names
            hsv_tuples = [(x / len(name_classes), 1., 1.) for x in range(len(name_classes))]
            colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
            image_size = options.data.inputs.get(list(options.data.inputs.keys())[0]).shape[:2]

            example_idx, bb = PostprocessResults().prepare_yolo_example_idx_to_show(
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
                save_predict_path, _ = PostprocessResults().plot_boxes(
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

        else:
            return_data = {}

        return return_data

    @staticmethod
    def postprocess_initial_source(
            options,
            input_id: int,
            example_id: int,
            dataset_path: str,
            preset_path: str,
            save_id: int = None,
            x_array=None,
            inverse_x_array=None,
            return_mode='deploy',
            max_length: int = 50,
            templates: list = None
    ):
        column_idx = []
        input_task = options.data.inputs.get(input_id).task
        if options.data.group != DatasetGroupChoice.keras:
            for inp in options.data.inputs.keys():
                if options.data.inputs.get(inp).task == LayerInputTypeChoice.Dataframe:
                    input_task = LayerInputTypeChoice.Dataframe
                for column_name in options.dataframe.get('val').columns:
                    if column_name.split('_')[0] == f"{inp}":
                        column_idx.append(options.dataframe.get('val').columns.tolist().index(column_name))
            if input_task == LayerInputTypeChoice.Text or input_task == LayerInputTypeChoice.Dataframe:
                initial_file_path = ""
            else:
                initial_file_path = os.path.join(dataset_path,
                                                 options.dataframe.get('val').iat[example_id, column_idx[0]])
            if not save_id:
                return str(os.path.abspath(initial_file_path))
        else:
            initial_file_path = ""

        data = []
        data_type = ""
        source = ""

        if input_task == LayerInputTypeChoice.Image:
            if options.data.group != DatasetGroupChoice.keras:
                img = Image.open(initial_file_path)
                img = img.resize(
                    options.data.inputs.get(input_id).shape[0:2][::-1],
                    Image.ANTIALIAS
                )
            else:
                img = image.array_to_img(x_array[example_id])
            img = img.convert('RGB')
            source = os.path.join(preset_path, f"initial_data_image_{save_id}_input_{input_id}.webp")
            img.save(source, 'webp')
            if return_mode == 'callback':
                data_type = LayerInputTypeChoice.Image.name
                data = [
                    {
                        "title": "Изображение",
                        "value": source,
                        "color_mark": None
                    }
                ]

        elif input_task == LayerInputTypeChoice.Text:
            regression_task = False
            for out in options.data.outputs.keys():
                if options.data.outputs.get(out).task == LayerOutputTypeChoice.Regression:
                    regression_task = True
            for column in column_idx:
                source = options.dataframe.get('val').iat[example_id, column]
                if return_mode == 'deploy':
                    break
                if return_mode == 'callback':
                    data_type = LayerInputTypeChoice.Text.name
                    title = "Текст"
                    if regression_task:
                        title = list(options.dataframe.get('val').columns)[column].split("_", 1)[-1]
                    data = [
                        {
                            "title": title,
                            "value": source,
                            "color_mark": None
                        }
                    ]

        elif input_task == LayerInputTypeChoice.Video:
            clip = moviepy_editor.VideoFileClip(initial_file_path)
            source = os.path.join(preset_path, f"initial_data_video_{save_id}_input_{input_id}.webm")
            clip.write_videofile(source)
            if return_mode == 'callback':
                data_type = LayerInputTypeChoice.Video.name
                data = [
                    {
                        "title": "Видео",
                        "value": source,
                        "color_mark": None
                    }
                ]

        elif input_task == LayerInputTypeChoice.Audio:
            source = os.path.join(preset_path, f"initial_data_audio_{save_id}_input_{input_id}.webm")
            AudioSegment.from_file(initial_file_path).export(source, format="webm")
            if return_mode == 'callback':
                data_type = LayerInputTypeChoice.Audio.name
                data = [
                    {
                        "title": "Аудио",
                        "value": source,
                        "color_mark": None
                    }
                ]

        elif input_task == LayerInputTypeChoice.Dataframe:
            time_series_choice = False
            for out in options.data.outputs.keys():
                if options.data.outputs.get(out).task == LayerOutputTypeChoice.Timeseries or \
                        options.data.outputs.get(out).task == LayerOutputTypeChoice.TimeseriesTrend:
                    time_series_choice = True
                    break
            if time_series_choice:
                graphics_data = []
                names = ""
                multi = False
                if return_mode == 'callback':
                    for i, channel in enumerate(options.data.columns.get(input_id).keys()):
                        multi = True if i > 0 else False
                        names += f"«{channel.split('_', 1)[-1]}», "
                        length = len(inverse_x_array) if len(inverse_x_array) < max_length else max_length
                        graphics_data.append(
                            templates[1](
                                _id=i + 1,
                                _type='graphic',
                                graph_name=f"График канала «{channel.split('_', 1)[-1]}»",
                                short_name=f"«{channel.split('_', 1)[-1]}»",
                                x_label="Время",
                                y_label="Значение",
                                plot_data=[
                                    templates[0](
                                        label="Исходное значение",
                                        x=np.arange(inverse_x_array[example_id].shape[-2]).astype('int').tolist()[
                                          -length:],
                                        y=inverse_x_array[example_id][:, i].astype('float').tolist()[-length:]
                                    )
                                ],
                            )
                        )
                    data_type = "graphic"
                    data = [
                        {
                            "title": f"График{'и' if multi else ''} по канал{'ам' if multi else 'у'} {names[:-2]}",
                            "value": graphics_data,
                            "color_mark": None
                        }
                    ]
            else:
                data_type = "str"
                source = []
                # for inp in options.data.inputs.keys():
                for col_name in options.data.columns.get(input_id).keys():
                    value = options.dataframe.get('val')[col_name].to_list()[example_id]
                    # source.append((col_name, value))
                    if return_mode == 'deploy':
                        source.append(value)
                    if return_mode == 'callback':
                        data.append(
                            {
                                "title": col_name.split("_", 1)[-1],
                                "value": value,
                                "color_mark": None
                            }
                        )

        else:
            pass

        if return_mode == 'deploy':
            return source
        if return_mode == 'callback':
            return data, data_type.lower()

    @staticmethod
    def prepare_example_idx_to_show(array: np.ndarray, true_array: np.ndarray, options, output: int, count: int,
                                    choice_type: str = "best", seed_idx: list = None) -> dict:
        example_idx = []
        encoding = options.data.outputs.get(output).encoding
        task = options.data.outputs.get(output).task
        if choice_type == ExampleChoiceTypeChoice.best or choice_type == ExampleChoiceTypeChoice.worst:
            if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.TimeseriesTrend:
                if array.shape[-1] == true_array.shape[-1] and encoding == \
                        LayerEncodingChoice.ohe and true_array.shape[-1] > 1:
                    classes = np.argmax(true_array, axis=-1)
                elif len(true_array.shape) == 1 and not encoding == LayerEncodingChoice.ohe and array.shape[-1] > 1:
                    classes = copy.deepcopy(true_array)
                elif len(true_array.shape) == 1 and not encoding == LayerEncodingChoice.ohe and array.shape[-1] == 1:
                    classes = copy.deepcopy(true_array)
                else:
                    classes = copy.deepcopy(true_array)
                class_idx = {}
                for _id in range(options.data.outputs.get(output).num_classes):
                    class_idx[_id] = {}
                for i, pred in enumerate(array):
                    class_idx[classes[i]][i] = pred[classes[i]]
                for _id in range(options.data.outputs.get(output).num_classes):
                    class_idx[_id] = list(sort_dict(class_idx[_id], mode=BalanceSortedChoice.ascending)[0])

                num_ex = copy.deepcopy(count)
                while num_ex:
                    key = np.random.choice(list(class_idx.keys()))
                    if choice_type == ExampleChoiceTypeChoice.best:
                        example_idx.append(class_idx[key][-1])
                        class_idx[key].pop(-1)
                    if choice_type == ExampleChoiceTypeChoice.worst:
                        example_idx.append(class_idx[key][0])
                        class_idx[key].pop(0)
                    num_ex -= 1

            elif task == LayerOutputTypeChoice.Segmentation or task == LayerOutputTypeChoice.TextSegmentation:
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

            elif task == LayerOutputTypeChoice.Timeseries or task == LayerOutputTypeChoice.Regression:
                delta = np.abs(true_array - array) * 100 / true_array
                while len(delta.shape) != 1:
                    delta = np.mean(delta, axis=-1)
                delta_dict = dict(zip(np.arange(0, len(delta)), delta))
                if choice_type == ExampleChoiceTypeChoice.best:
                    example_idx, _ = sort_dict(delta_dict, mode=BalanceSortedChoice.ascending)
                    example_idx = example_idx[:count]
                if choice_type == ExampleChoiceTypeChoice.worst:
                    example_idx, _ = sort_dict(delta_dict, mode=BalanceSortedChoice.descending)
                    example_idx = example_idx[:count]

            else:
                pass

        elif choice_type == ExampleChoiceTypeChoice.seed and len(seed_idx):
            example_idx = seed_idx[:count]

        elif choice_type == ExampleChoiceTypeChoice.random:
            if task == LayerOutputTypeChoice.Classification or task == LayerOutputTypeChoice.TimeseriesTrend:
                true_id = []
                false_id = []
                for i, ex in enumerate(true_array):
                    if np.argmax(ex, axis=-1) == np.argmax(array[i], axis=-1):
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

            elif task == LayerOutputTypeChoice.Segmentation or task == LayerOutputTypeChoice.TextSegmentation:
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

            elif task == LayerOutputTypeChoice.Timeseries or task == LayerOutputTypeChoice.Regression:
                delta = np.abs(true_array - array) * 100 / true_array
                while len(delta.shape) != 1:
                    delta = np.mean(delta, axis=-1)
                true_id = []
                false_id = []
                for i, ex in enumerate(delta):
                    if ex >= CALLBACK_REGRESSION_TREASHOLD_VALUE:
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

        else:
            pass
        return example_idx

    @staticmethod
    def prepare_yolo_example_idx_to_show(array: dict, true_array: dict, name_classes: list, box_channel: Optional[int],
                                         count: int, choice_type: str = "best", seed_idx: list = None,
                                         sensitivity: float = 0.25, get_optimal_channel=False):
        if get_optimal_channel:
            channel_stat = []
            for channel in range(3):
                total_metric = 0
                for example in range(len(array.get(channel))):
                    total_metric += PostprocessResults().get_yolo_example_statistic(
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
                    PostprocessResults().get_yolo_example_statistic(
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

    @staticmethod
    def postprocess_classification(
            predict_array: np.ndarray,
            true_array: np.ndarray,
            options: DatasetOutputsData,
            show_stat: bool = False,
            return_mode='deploy'
    ):
        labels = options.classes_names
        ohe = True if options.encoding == LayerEncodingChoice.ohe else False
        actual_value = np.argmax(true_array, axis=-1) if ohe else true_array
        data = {
            "y_true": {},
            "y_pred": {},
            "stat": {}
        }
        if return_mode == 'deploy':
            labels_from_array = []
            for class_idx in predict_array:
                class_dist = sorted(class_idx, reverse=True)
                labels_dist = []
                for j in class_dist:
                    labels_dist.append((labels[list(class_idx).index(j)], round(float(j) * 100, 1)))
                labels_from_array.append(labels_dist)
            return labels[actual_value], labels_from_array

        elif return_mode == 'callback':
            data["y_true"] = {
                "type": "str",
                "data": [
                    {
                        "title": "Класс",
                        "value": labels[actual_value],
                        "color_mark": None
                    }
                ]
            }
            if labels[actual_value] == labels[np.argmax(predict_array)]:
                color_mark = 'success'
            else:
                color_mark = 'wrong'
            data["y_pred"] = {
                "type": "str",
                "data": [
                    {
                        "title": "Класс",
                        "value": labels[np.argmax(predict_array)],
                        "color_mark": color_mark
                    }
                ]
            }
            if show_stat:
                data["stat"] = {
                    "type": "str",
                    "data": []
                }
                for i, val in enumerate(predict_array):
                    if val == max(predict_array) and labels[i] == labels[actual_value]:
                        class_color_mark = "success"
                    elif val == max(predict_array) and labels[i] != labels[actual_value]:
                        class_color_mark = "wrong"
                    else:
                        class_color_mark = None
                    data["stat"]["data"].append(
                        {
                            'title': labels[i],
                            'value': f"{round(val * 100, 1)}%",
                            'color_mark': class_color_mark
                        }
                    )
            return data

    @staticmethod
    def postprocess_segmentation(
            predict_array: np.ndarray,
            true_array: Optional[np.ndarray],
            options: DatasetOutputsData,
            output_id: int,
            image_id: int,
            save_path: str,
            colors: list = None,
            return_mode='deploy',
            show_stat: bool = False
    ):
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

    @staticmethod
    def postprocess_text_segmentation(
            pred_array: np.ndarray,
            options: DatasetOutputsData,
            dataframe: DataFrame,
            dataset_params: dict,
            example_id: int,
            return_mode='deploy',
            class_colors: list = None,
            show_stat: bool = False,
            true_array: np.ndarray = None,
    ):

        def add_tags_to_word(word: str, tag: str):
            if tag:
                for t in tag:
                    word = f"<{t[1:-1]}>{word}</{t[1:-1]}>"
                return word
            else:
                return f"<p1>{word}</p1>"

        def reformat_tags(y_array: np.ndarray, tag_list: list,  # classes_names: dict, colors: dict,
                          sensitivity: float = 0.9):
            norm_array = np.where(y_array >= sensitivity, 1, 0).astype('int')
            reformat_tags = []
            for word_tag in norm_array:
                if np.sum(word_tag) == 0:
                    reformat_tags.append(None)
                else:
                    mix_tag = []
                    for i, tag in enumerate(word_tag):
                        if tag == 1:
                            mix_tag.append(tag_list[i])
                    reformat_tags.append(mix_tag)
            return reformat_tags

        def text_colorization(text: str, label_array: np.ndarray, tag_list: list, class_names: dict, colors: dict):
            text = text.split(" ")
            labels = reformat_tags(label_array, tag_list)
            colored_text = []
            for i, word in enumerate(text):
                colored_text.append(add_tags_to_word(word, labels[i]))
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
            initial_text = dataframe.iat[example_id, 0]
            text_segmentation = text_colorization(
                text=initial_text,
                label_array=pred_array,
                tag_list=dataset_tags,
                class_names=classes_names,
                colors=colors
            )

            data = [('<p1>', '<p1>', (200, 200, 200))]
            for tag in colors.keys():
                data.append(
                    (tag, classes_names[tag], colors[tag])
                )
            return initial_text, text_segmentation, data

        if return_mode == 'callback':
            data = {
                "y_true": {},
                "y_pred": {},
                "tags_color": {},
                "stat": {}
            }
            text_for_preparation = dataframe.iat[example_id, 0]
            true_text_segmentation = text_colorization(
                text=text_for_preparation,
                label_array=true_array,
                tag_list=dataset_tags,
                class_names=classes_names,
                colors=colors
            )

            data["y_true"] = {
                "type": "segmented_text",
                "data": [
                    {
                        "title": "Текст",
                        "value": true_text_segmentation,
                        "color_mark": None
                    }
                ]
            }
            pred_text_segmentation = text_colorization(
                text=text_for_preparation,
                label_array=pred_array,
                tag_list=dataset_tags,
                class_names=classes_names,
                colors=colors
            )
            data["y_pred"] = {
                "type": "segmented_text",
                "data": [
                    {
                        "title": "Текст",
                        "value": pred_text_segmentation,
                        "color_mark": None
                    }
                ]
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
                        data["stat"]["data"].append(
                            {
                                'title': cls,
                                'value': "-",
                                'color_mark': None
                            }
                        )
                    else:
                        dice_val = np.round(
                            dice_coef(y_true[:, idx], y_pred[:, idx], batch_mode=False) * 100, 1)
                        data["stat"]["data"].append(
                            {
                                'title': cls,
                                'value': f"{dice_val} %",
                                'color_mark': 'success' if dice_val >= 90 else 'wrong'
                            }
                        )
                        count += 1
                        mean_val += dice_val
                if count and mean_val / count >= 90:
                    mean_color_mark = "success"
                elif count and mean_val / count < 90:
                    mean_color_mark = "wrong"
                else:
                    mean_color_mark = None
                data["stat"]["data"].insert(
                    0,
                    {
                        'title': "Средняя точность",
                        'value': f"{round(mean_val / count, 2)}%" if count else "-",
                        'color_mark': mean_color_mark
                    }
                )
            return data

    @staticmethod
    def postprocess_regression(
            column_names: list,
            inverse_y_true: np.ndarray,
            inverse_y_pred: np.ndarray,
            show_stat: bool = False,
            return_mode='deploy',
    ):
        data = {"y_true": {
            "type": "str",
            "data": []
        }}
        if return_mode == 'deploy':
            source = []
            return source
        else:
            for i, name in enumerate(column_names):
                data["y_true"]["data"].append(
                    {
                        "title": name.split('_', 1)[-1],
                        "value": f"{inverse_y_true[i]: .2f}",
                        "color_mark": None
                    }
                )
            deviation = np.abs((inverse_y_pred - inverse_y_true) * 100 / inverse_y_true)
            data["y_pred"] = {
                "type": "str",
                "data": []
            }
            for i, name in enumerate(column_names):
                color_mark = 'success' if deviation[i] < 2 else "wrong"
                data["y_pred"]["data"].append(
                    {
                        "title": name.split('_', 1)[-1],
                        "value": f"{inverse_y_pred[i]: .2f}",
                        "color_mark": color_mark
                    }
                )
            if show_stat:
                data["stat"] = {
                    "type": "str",
                    "data": []
                }
                for i, name in enumerate(column_names):
                    color_mark = 'success' if deviation[i] < 2 else "wrong"
                    data["stat"]["data"].append(
                        {
                            'title': f"Отклонение - «{name.split('_', 1)[-1]}»",
                            'value': f"{np.round(deviation[i], 2)} %",
                            'color_mark': color_mark
                        }
                    )
            return data

    @staticmethod
    def postprocess_time_series(
            options: DatasetData,
            real_x: np.ndarray,
            inverse_y_true: np.ndarray,
            inverse_y_pred: np.ndarray,
            output_id: int,
            depth: int,
            show_stat: bool = False,
            templates: list = None,
            max_length: int = 50
    ):

        """
        real_x = self.inverse_x_val.get(f"{input}")[example_idx]
        inverse_y_true = self.inverse_y_true.get("val").get(output_id)[example_idx]
        inverse_y_pred = self.inverse_y_pred.get(output_id)[example_idx]
        depth = self.inverse_y_true.get("val").get(output_id)[example_idx].shape[-1]
        templates = [self._fill_graph_plot_data, self._fill_graph_front_structure]
        """
        # try:
        data = {
            "y_true": {},
            "y_pred": {},
            "stat": {}
        }
        graphics = []
        _id = 1
        for i, channel in enumerate(options.columns.get(output_id).keys()):
            for inp in options.inputs.keys():
                for input_column in options.columns.get(inp).keys():
                    if channel.split("_", 1)[-1] == input_column.split("_", 1)[-1]:
                        init_column = list(options.columns.get(inp).keys()).index(input_column)
                        length = len(real_x) if len(real_x) < max_length else max_length
                        x_tr = round_list(real_x[:, init_column])
                        y_tr = round_list(inverse_y_true[:, i])
                        y_tr.insert(0, x_tr[-1])
                        y_pr = round_list(inverse_y_pred[:, i])
                        y_pr.insert(0, x_tr[-1])
                        graphics.append(
                            templates[1](
                                _id=_id,
                                _type='graphic',
                                graph_name=f'График канала «{channel.split("_", 1)[-1]}»',
                                short_name=f"«{channel.split('_', 1)[-1]}»",
                                x_label="Время",
                                y_label="Значение",
                                plot_data=[
                                    templates[0](
                                        label="Исходное значение",
                                        x=np.arange(real_x.shape[-2]).astype('int').tolist()[-length:],
                                        y=x_tr[-length:]
                                    ),
                                    templates[0](
                                        label="Истинное значение",
                                        x=np.arange(real_x.shape[-2] - 1, real_x.shape[-2] + depth).astype(
                                            'int').tolist(),
                                        y=y_tr
                                    ),
                                    templates[0](
                                        label="Предсказанное значение",
                                        x=np.arange(real_x.shape[-2] - 1, real_x.shape[-2] + depth).astype(
                                            'int').tolist(),
                                        y=y_pr
                                    ),
                                ],
                            )
                        )
                        _id += 1
                        break
        data["y_pred"] = {
            "type": "graphic",
            "data": [
                {
                    "title": "Графики",
                    "value": graphics,
                    "color_mark": None
                }
            ]
        }
        if show_stat:
            data["stat"] = {
                "type": "table",
                "data": []
            }
            for i, channel in enumerate(options.columns.get(output_id).keys()):
                data["stat"]["data"].append({'title': channel.split("_", 1)[-1], 'value': [], 'color_mark': None})
                for step in range(inverse_y_true.shape[-2]):
                    deviation = (inverse_y_pred[step, i] - inverse_y_true[step, i]) * 100 / inverse_y_true[step, i]
                    data["stat"]["data"][-1]["value"].append(
                        {
                            "Шаг": f"{step + 1}",
                            "Истина": f"{inverse_y_true[step, i].astype('float'): .2f}",
                            "Предсказание": f"{inverse_y_pred[step, i].astype('float'): .2f}",
                            "Отклонение": {
                                "value": f"{deviation: .2f} %",
                                "color_mark": "success" if abs(deviation) < CALLBACK_REGRESSION_TREASHOLD_VALUE
                                else "wrong"
                            }
                        }
                    )
        return data

    @staticmethod
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
            save_true_predict_path, _ = PostprocessResults().plot_boxes(
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

            save_predict_path, _ = PostprocessResults().plot_boxes(
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
                box_stat = PostprocessResults().get_yolo_example_statistic(
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
        pick, _ = PostprocessResults().non_max_suppression_fast(_boxes_out, _scores_out, sensitivity)
        return np.concatenate([_boxes_out[pick], _conf_param[pick], _scores_out[pick]], axis=-1)

    @staticmethod
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

    @staticmethod
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
                pick, _ = PostprocessResults().non_max_suppression_fast(boxes, scores, sensitivity=sensitivity)
                if len(pick) == 1:
                    mean_iou = PostprocessResults().bboxes_iou(boxes[0], boxes[1])
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
