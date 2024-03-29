import numpy as np
import pandas as pd

from terra_ai.callbacks.utils import sort_dict, get_y_true, get_distribution_histogram, get_correlation_matrix, \
    get_scatter, fill_graph_front_structure, fill_graph_plot_data, fill_heatmap_front_structure, round_loss_metric, \
    set_preset_count
from terra_ai.data.datasets.extra import LayerInputTypeChoice
from terra_ai.data.training.extra import ExampleChoiceTypeChoice, BalanceSortedChoice
from terra_ai.logging import logger
from terra_ai.settings import CALLBACK_REGRESSION_TREASHOLD_VALUE, DEPLOY_PRESET_PERCENT
import terra_ai.exceptions.callbacks as exception


class DataframeRegressionCallback:
    name = 'DataframeRegressionCallback'

    def __init__(self):
        pass

    @staticmethod
    def get_x_array(options):
        method_name = 'get_x_array'
        try:
            inverse_x_val = None
            if not options.data.use_generator:
                x_val = options.X.get("val")
            else:
                x_val = {}
                for inp in options.data.inputs.keys():
                    x_val[inp] = []
                    for x_val_, _ in options.dataset['val'].batch(1):
                        x_val[inp].extend(x_val_.get(f'{inp}').numpy())
                    x_val[inp] = np.array(x_val[inp])
            return x_val, inverse_x_val
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                DataframeRegressionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def get_y_true(options, dataset_path):
        method_name = 'get_y_true'
        try:
            y_true = {"train": {}, "val": {}}
            inverse_y_true = {"train": {}, "val": {}}
            for data_type in y_true.keys():
                for out in options.data.outputs.keys():
                    if not options.data.use_generator:
                        y_true[data_type][f"{out}"] = options.Y.get(data_type).get(f"{out}")
                    else:
                        y_true[data_type][f"{out}"] = []
                        for _, y_val in options.dataset[data_type].batch(1):
                            y_true[data_type][f"{out}"].extend(y_val.get(f'{out}').numpy())
                        y_true[data_type][f"{out}"] = np.array(y_true[data_type][f"{out}"])

                    preprocess_dict = options.preprocessing.preprocessing.get(out)
                    inverse_y = np.zeros_like(y_true.get(data_type).get(f"{out}")[:, 0:1])
                    for i, column in enumerate(preprocess_dict.keys()):
                        if type(preprocess_dict.get(column)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                            _options = {int(out): {column: y_true.get(data_type).get(f"{out}")[:, i:i + 1]}}
                            inverse_col = options.preprocessing.inverse_data(_options).get(out).get(column)
                        else:
                            inverse_col = y_true.get(data_type).get(f"{out}")[:, i:i + 1]
                        inverse_y = np.concatenate([inverse_y, inverse_col], axis=-1)
                    inverse_y_true[data_type][f"{out}"] = inverse_y[:, 1:]
            return y_true, inverse_y_true
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                DataframeRegressionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc


    @staticmethod
    def get_inverse_array(array: dict, options):
        method_name = 'get_inverse_array'
        try:
            inverse_array = {}
            for data_type in array.keys():
                inverse_array[data_type] = {}
                for idx, out in enumerate(array.get(data_type).keys()):
                    preprocess_dict = options.preprocessing.preprocessing.get(int(out))
                    inverse_y = np.zeros_like(array.get(data_type).get(out)[:, 0:1])
                    for i, column in enumerate(preprocess_dict.keys()):
                        if type(preprocess_dict.get(column)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                            _options = {int(out): {column: array.get(data_type).get(out)[:, i:i + 1]}}
                            inverse_col = options.preprocessing.inverse_data(_options).get(int(out)).get(column)
                        else:
                            inverse_col = array.get(data_type).get(out)[:, i:i + 1]
                        inverse_y = np.concatenate([inverse_y, inverse_col], axis=-1)
                    inverse_array[data_type][out] = inverse_y[:, 1:]
            return inverse_array
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                DataframeRegressionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def postprocess_initial_source(options, input_id: int, example_id: int, return_mode='deploy', data_type='val'):
        method_name = 'postprocess_initial_source'
        try:
            data = []
            source = []
            for col_name in options.data.columns.get(input_id).keys():
                value = options.dataframe.get(data_type)[col_name].to_list()[example_id]
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
            if return_mode == 'deploy':
                return source
            if return_mode == 'callback':
                return data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                DataframeRegressionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def postprocess_deploy(array, options, save_path: str = "", dataset_path: str = "") -> dict:
        method_name = 'postprocess_deploy'
        try:
            return_data = {}
            for i, output_id in enumerate(options.data.outputs.keys()):
                true_array = get_y_true(options, output_id)
                if len(options.data.outputs.keys()) > 1:
                    postprocess_array = array[i]
                else:
                    postprocess_array = array
                count = set_preset_count(len_array=len(postprocess_array), preset_percent=DEPLOY_PRESET_PERCENT)
                example_idx = DataframeRegressionCallback().prepare_example_idx_to_show(
                    array=postprocess_array[:len(array)],
                    true_array=true_array[:len(array)],
                    count=count
                )
                return_data[output_id] = {'preset': [], 'label': []}
                source_col = []
                for inp in options.data.inputs.keys():
                    source_col.extend(list(options.data.columns.get(inp).keys()))
                preprocess = options.preprocessing.preprocessing.get(output_id)
                for idx in example_idx:
                    row_list = []
                    for inp_col in source_col:
                        row_list.append(f"{options.dataframe.get('val')[inp_col][idx]}")
                    return_data[output_id]['preset'].append(row_list)
                    channel_inverse_col = []
                    for ch, col in enumerate(list(options.data.columns.get(output_id).keys())):
                        channel_inverse_col = []
                        if type(preprocess.get(col)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                            _options = {int(output_id): {col: array[idx, ch:ch + 1].reshape(-1, 1)}}
                            inverse_col = options.preprocessing.inverse_data(_options).get(output_id).get(col)
                            inverse_col = inverse_col.squeeze().astype('float').tolist()
                        else:
                            inverse_col = array[idx, ch:ch + 1].astype('float').tolist()
                        channel_inverse_col.append(round_loss_metric(inverse_col))
                    return_data[output_id]['label'].append(channel_inverse_col)  # [0]
            return return_data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                DataframeRegressionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def dataset_balance(options, y_true, preset_path: str, class_colors) -> dict:
        method_name = 'dataset_balance'
        try:
            dataset_balance = {}
            for out in options.data.outputs.keys():
                dataset_balance[f"{out}"] = {'histogram': {}, 'correlation': {}}
                for data_type in ['train', 'val']:
                    dataset_balance[f"{out}"]['histogram'][data_type] = {}
                    for column in list(options.dataframe.get('train').columns):
                        column_id = int(column.split("_")[0])
                        column_task = options.data.columns.get(column_id).get(column).get('task')
                        column_data = list(options.dataframe.get(data_type)[column])
                        if column_task == LayerInputTypeChoice.Text:
                            continue
                        elif column_task == LayerInputTypeChoice.Classification:
                            x, y = get_distribution_histogram(column_data, categorical=True)
                            hist_type = "histogram"
                        else:
                            x, y = get_distribution_histogram(column_data, categorical=False)
                            hist_type = "bar"
                        dataset_balance[f"{out}"]['histogram'][data_type][column] = {
                            "name": column.split("_", 1)[-1],
                            "type": hist_type,
                            "x": x,
                            "y": y
                        }
                for data_type in ['train', 'val']:
                    labels, matrix = get_correlation_matrix(pd.DataFrame(options.dataframe.get(data_type)))
                    dataset_balance[f"{out}"]['correlation'][data_type] = {
                        "labels": labels,
                        "matrix": matrix
                    }
            return dataset_balance
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                DataframeRegressionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def intermediate_result_request(options, interactive_config, example_idx, dataset_path,
                                    preset_path, x_val, inverse_x_val, y_pred, inverse_y_pred,
                                    y_true, inverse_y_true, class_colors, raw_y_pred=None) -> dict:
        method_name = 'intermediate_result_request'
        try:
            return_data = {}
            if interactive_config.intermediate_result.show_results:
                data_type = interactive_config.intermediate_result.data_type.name
                count = interactive_config.intermediate_result.num_examples \
                    if len(options.dataframe.get(data_type)) > interactive_config.intermediate_result.num_examples \
                    else len(options.dataframe.get(data_type))
                for idx in range(count):
                    return_data[f"{idx + 1}"] = {
                        'initial_data': {},
                        'true_value': {},
                        'predict_value': {},
                        'tags_color': {},
                        'statistic_values': {}
                    }
                    # if not len(options.data.outputs.keys()) == 1:
                    for inp in options.data.inputs.keys():
                        data = DataframeRegressionCallback.postprocess_initial_source(
                            options=options,
                            input_id=inp,
                            example_id=example_idx[idx],
                            return_mode='callback',
                            data_type=data_type
                        )
                        return_data[f"{idx + 1}"]['initial_data'][f"Входной слой «{inp}»"] = {
                            'type': 'str', 'data': data,
                        }

                    for out in options.data.outputs.keys():
                        data = DataframeRegressionCallback().postprocess_regression(
                            column_names=list(options.data.columns.get(out).keys()),
                            inverse_y_true=inverse_y_true.get(data_type).get(f"{out}")[example_idx[idx]],
                            inverse_y_pred=inverse_y_pred.get(data_type).get(f"{out}")[example_idx[idx]],
                            show_stat=interactive_config.intermediate_result.show_statistic,
                            return_mode='callback'
                        )
                        if data.get('y_true'):
                            return_data[f"{idx + 1}"]['true_value'][f"Выходной слой «{out}»"] = data.get('y_true')
                        return_data[f"{idx + 1}"]['predict_value'][f"Выходной слой «{out}»"] = data.get('y_pred')
                        return_data[f"{idx + 1}"]['tags_color'] = None
                        if data.get('stat'):
                            return_data[f"{idx + 1}"]['statistic_values'][f"Выходной слой «{out}»"] = data.get('stat')
                        else:
                            return_data[f"{idx + 1}"]['statistic_values'] = {}
            return return_data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                DataframeRegressionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def statistic_data_request(interactive_config, inverse_y_true, y_pred, inverse_y_pred, options=None,
                               y_true=None, raw_y_pred=None) -> list:
        method_name = 'statistic_data_request'
        try:
            return_data = []
            _id = 1
            for out in interactive_config.statistic_data.output_id:
                for data_type in inverse_y_true.keys():
                    type_name = "Тренировочная" if data_type == 'train' else "Проверочная"
                    y_true = inverse_y_true.get(data_type).get(f'{out}').squeeze()
                    y_pred = inverse_y_pred.get(data_type).get(f'{out}').squeeze()
                    x_scatter, y_scatter = get_scatter(y_true, y_pred)
                    return_data.append(
                        fill_graph_front_structure(
                            _id=_id,
                            _type='scatter',
                            graph_name=f"Выход «{out}» - Скаттер - {type_name} выборка",
                            short_name=f"{out} - Скаттер - {type_name}",
                            x_label="Истинные значения",
                            y_label="Предсказанные значения",
                            plot_data=[fill_graph_plot_data(x=x_scatter, y=y_scatter)],
                        )
                    )
                    _id += 1

                for data_type in inverse_y_true.keys():
                    type_name = "Тренировочная" if data_type == 'train' else "Проверочная"
                    y_true = inverse_y_true.get(data_type).get(f'{out}').squeeze()
                    y_pred = inverse_y_pred.get(data_type).get(f'{out}').squeeze()
                    deviation = (y_pred - y_true) * 100 / y_true
                    x_mae, y_mae = get_distribution_histogram(np.abs(deviation), categorical=False)
                    return_data.append(
                        fill_graph_front_structure(
                            _id=_id,
                            _type='bar',
                            graph_name=f'Выход «{out}» - Распределение абсолютной ошибки - {type_name} выборка',
                            short_name=f"{out} - Распределение MAE - {type_name}",
                            x_label="Абсолютная ошибка",
                            y_label="Значение",
                            plot_data=[fill_graph_plot_data(x=x_mae, y=y_mae)],
                        )
                    )
                    _id += 1

                for data_type in inverse_y_true.keys():
                    type_name = "Тренировочная" if data_type == 'train' else "Проверочная"
                    y_true = inverse_y_true.get(data_type).get(f'{out}').squeeze()
                    y_pred = inverse_y_pred.get(data_type).get(f'{out}').squeeze()
                    deviation = (y_pred - y_true) * 100 / y_true
                    x_me, y_me = get_distribution_histogram(deviation, categorical=False)
                    return_data.append(
                        fill_graph_front_structure(
                            _id=_id,
                            _type='bar',
                            graph_name=f'Выход «{out}» - Распределение ошибки - {type_name} выборка',
                            short_name=f"{out} - Распределение ME - {type_name}",
                            x_label="Ошибка",
                            y_label="Значение",
                            plot_data=[fill_graph_plot_data(x=x_me, y=y_me)],
                        )
                    )
                    _id += 1
            return return_data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                DataframeRegressionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def balance_data_request(options, dataset_balance, interactive_config) -> list:
        method_name = 'balance_data_request'
        try:
            return_data = []
            _id = 0
            for out in options.data.outputs.keys():
                for class_type in dataset_balance[f"{out}"].keys():
                    if class_type == 'histogram':
                        for column in dataset_balance[f"{out}"][class_type]["train"].keys():
                            preset = {}
                            for data_type in ["train", "val"]:
                                histogram = dataset_balance[f"{out}"][class_type][data_type][column]
                                if histogram.get("type") == 'histogram':
                                    dict_to_sort = dict(zip(histogram.get("x"), histogram.get("y")))
                                    x, y = sort_dict(
                                        dict_to_sort=dict_to_sort,
                                        mode=interactive_config.data_balance.sorted.name
                                    )
                                else:
                                    x = histogram.get("x")
                                    y = histogram.get("y")
                                data_type_name = "Тренировочная" if data_type == "train" else "Проверочная"
                                preset[data_type] = fill_graph_front_structure(
                                    _id=_id,
                                    _type=histogram.get("type"),
                                    type_data=data_type,
                                    graph_name=f"Выход {out} - {data_type_name} выборка - "
                                               f"Гистограмма распределения колонки «{histogram['name']}»",
                                    short_name=f"{data_type_name} - {histogram['name']}",
                                    x_label="Значение",
                                    y_label="Количество",
                                    plot_data=[
                                        fill_graph_plot_data(x=x, y=y)],
                                )
                                _id += 1
                            return_data.append(preset)
                    if class_type == 'correlation':
                        preset = {}
                        for data_type in ["train", "val"]:
                            data_type_name = "Тренировочная" if data_type == "train" else "Проверочная"
                            preset[data_type] = fill_heatmap_front_structure(
                                _id=_id,
                                _type="corheatmap",
                                type_data=data_type,
                                graph_name=f"Выход {out} - {data_type_name} выборка - Матрица корреляций",
                                short_name=f"Матрица корреляций",
                                x_label="Колонка",
                                y_label="Колонка",
                                labels=dataset_balance[f"{out}"]['correlation'][data_type]["labels"],
                                data_array=dataset_balance[f"{out}"]['correlation'][data_type]["matrix"],
                            )
                            _id += 1
                        return_data.append(preset)
            return return_data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                DataframeRegressionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def prepare_example_idx_to_show(array: np.ndarray, true_array: np.ndarray, count: int, options=None, output=None,
                                    choice_type: ExampleChoiceTypeChoice = ExampleChoiceTypeChoice.best,
                                    seed_idx: list = None) -> dict:
        method_name = 'prepare_example_idx_to_show'
        try:
            example_idx = []
            if choice_type == ExampleChoiceTypeChoice.best or choice_type == ExampleChoiceTypeChoice.worst:
                delta = np.abs(true_array - array + 0.000001) * 100 / (true_array + 0.000001)
                while len(delta.shape) != 1:
                    delta = np.mean(delta, axis=-1)
                delta_dict = dict(zip(np.arange(0, len(delta)), delta))
                if choice_type == ExampleChoiceTypeChoice.best:
                    example_idx, _ = sort_dict(delta_dict, mode=BalanceSortedChoice.ascending)
                    example_idx = example_idx[:count]
                if choice_type == ExampleChoiceTypeChoice.worst:
                    example_idx, _ = sort_dict(delta_dict, mode=BalanceSortedChoice.descending)
                    example_idx = example_idx[:count]

            elif choice_type == ExampleChoiceTypeChoice.seed and len(seed_idx):
                example_idx = seed_idx[:count]

            elif choice_type == ExampleChoiceTypeChoice.random:
                delta = np.abs(true_array - array + 0.000001) * 100 / (true_array + 0.000001)
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
            return example_idx
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                DataframeRegressionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc

    @staticmethod
    def postprocess_regression(column_names: list, inverse_y_true: np.ndarray, inverse_y_pred: np.ndarray,
                               show_stat: bool = False, return_mode='deploy'):
        method_name = 'postprocess_regression'
        try:
            data = {"y_true": {"type": "str", "data": []}}
            if return_mode == 'deploy':
                source = []
                return source
            else:
                for i, name in enumerate(column_names):
                    data["y_true"]["data"].append(
                        {"title": name.split('_', 1)[-1], "value": f"{inverse_y_true[i]: .2f}", "color_mark": None}
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
                    data["stat"] = {"type": "str", "data": []}
                    for i, name in enumerate(column_names):
                        color_mark = 'success' if deviation[i] < 2 else "wrong"
                        data["stat"]["data"].append(
                            {
                                'title': f"Отклонение - «{name.split('_', 1)[-1]}»",
                                'value': f"{round(deviation[i].item(), 2)} %",
                                'color_mark': color_mark
                            }
                        )
                return data
        except Exception as error:
            exc = exception.ErrorInClassInMethodException(
                DataframeRegressionCallback.name, method_name, str(error)).with_traceback(error.__traceback__)
            # logger.error(exc)
            raise exc
