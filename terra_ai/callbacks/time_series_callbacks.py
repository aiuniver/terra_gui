import numpy as np

from terra_ai.callbacks.utils import fill_graph_front_structure, fill_graph_plot_data, sort_dict, round_list, get_y_true
from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.training.extra import ExampleChoiceTypeChoice, BalanceSortedChoice
from terra_ai.settings import MAX_GRAPH_LENGTH, CALLBACK_REGRESSION_TREASHOLD_VALUE, DEPLOY_PRESET_PERCENT


class TimeseriesCallback:
    def __init__(self):
        pass

    @staticmethod
    def get_x_array(options):
        inverse_x_val = {}
        if not options.data.use_generator:
            x_val = options.X.get("val")
        else:
            x_val = {}
            for inp in options.dataset['val'].keys():
                x_val[inp] = []
                for x_val_, _ in options.dataset['val'].batch(1):
                    x_val[inp].extend(x_val_.get(f'{inp}').numpy())
                x_val[inp] = np.array(x_val[inp])
        for inp in x_val.keys():
            preprocess_dict = options.preprocessing.preprocessing.get(int(inp))
            inverse_x = np.zeros_like(x_val.get(inp)[:, :, 0:1])
            for i, column in enumerate(preprocess_dict.keys()):
                if type(preprocess_dict.get(column)).__name__ in ['StandardScaler', 'MinMaxScaler']:
                    _options = {int(inp): {column: x_val.get(inp)[:, :, i]}}
                    inverse_col = np.expand_dims(
                        options.preprocessing.inverse_data(_options).get(int(inp)).get(column), axis=-1
                    )
                else:
                    inverse_col = x_val.get(inp)[:, :, i:i + 1]
                inverse_x = np.concatenate([inverse_x, inverse_col], axis=-1)
            inverse_x_val[inp] = inverse_x[:, :, 1:]
        return x_val, inverse_x_val

    @staticmethod
    def postprocess_initial_source(options, input_id: int, example_id: int, inverse_x_array=None,
                                   return_mode='deploy'):
        column_idx = []
        for inp in options.data.inputs.keys():
            for column_name in options.dataframe.get('val').columns:
                if column_name.split('_')[0] == f"{inp}":
                    column_idx.append(options.dataframe.get('val').columns.tolist().index(column_name))

        source = ""
        graphics_data = []
        names = ""
        multi = False
        if return_mode == 'callback':
            for i, channel in enumerate(options.data.columns.get(input_id).keys()):
                multi = True if i > 0 else False
                names += f"«{channel.split('_', 1)[-1]}», "
                length = len(inverse_x_array) if len(inverse_x_array) < MAX_GRAPH_LENGTH else MAX_GRAPH_LENGTH
                graphics_data.append(
                    fill_graph_front_structure(
                        _id=i + 1,
                        _type='graphic',
                        graph_name=f"График канала «{channel.split('_', 1)[-1]}»",
                        short_name=f"«{channel.split('_', 1)[-1]}»",
                        x_label="Время",
                        y_label="Значение",
                        plot_data=[
                            fill_graph_plot_data(
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
            return data, data_type.lower()

        if return_mode == 'deploy':
            return source

    @staticmethod
    def postprocess_results(array, options) -> dict:
        x_array, inverse_x_array = TimeseriesCallback.get_x_array(options)
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
                count=int(len(true_array) * DEPLOY_PRESET_PERCENT / 100)
            )
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
        return return_data


def prepare_example_idx_to_show(array: np.ndarray, true_array: np.ndarray, count: int,
                                choice_type: str = "best", seed_idx: list = None) -> dict:
    example_idx = []
    if choice_type == ExampleChoiceTypeChoice.best or choice_type == ExampleChoiceTypeChoice.worst:
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

    elif choice_type == ExampleChoiceTypeChoice.seed and len(seed_idx):
        example_idx = seed_idx[:count]

    elif choice_type == ExampleChoiceTypeChoice.random:
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
    return example_idx


def postprocess_time_series(options: DatasetData, real_x: np.ndarray, inverse_y_true: np.ndarray,
                            inverse_y_pred: np.ndarray, output_id: int, depth: int, show_stat: bool = False,
                            templates: list = None, max_length: int = 50):
    """
        real_x = self.inverse_x_val.get(f"{input}")[example_idx]
        inverse_y_true = self.inverse_y_true.get("val").get(output_id)[example_idx]
        inverse_y_pred = self.inverse_y_pred.get(output_id)[example_idx]
        depth = self.inverse_y_true.get("val").get(output_id)[example_idx].shape[-1]
        templates = [self._fill_graph_plot_data, self._fill_graph_front_structure]
        """
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
