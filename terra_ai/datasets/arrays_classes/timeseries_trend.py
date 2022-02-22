import numpy as np
from tensorflow.keras import utils
from typing import Any
from .base import Array


class TimeseriesTrendArray(Array):

    def prepare(self, sources, dataset_folder=None, **options):

        classes = []
        trend_dict = {0: "Не изменился",
                      1: "Вверх",
                      2: "Вниз"}
        tmp = []
        depth = 1
        step = options['step']
        length = options['length']
        trend_lim = options["trend_limit"]
        trend_limit = float(trend_lim[: trend_lim.find("%")]) if "%" in trend_lim else float(trend_lim)
        for i in range(0, len(sources) - length - depth, step):
            first_value = sources[i]
            second_value = sources[i + length]
            if "%" in trend_lim:
                if abs((second_value - first_value) / first_value) * 100 <= trend_limit:
                    tmp.append(0)
                elif second_value > first_value:
                    tmp.append(1)
                else:
                    tmp.append(2)
            else:
                if abs(second_value - first_value) <= trend_limit:
                    tmp.append(0)
                elif second_value > first_value:
                    tmp.append(1)
                else:
                    tmp.append(2)

        # for i in set(tmp):
        #     classes.append(trend_dict[i])
        options['classes_names'] = ["Не изменился", "Вверх", "Вниз"]
        options['num_classes'] = 3  # len(classes)

        instructions = {'instructions': sources,
                        'parameters': options}

        return instructions

    def create(self, source: Any, **options):

        # if options["trend"]:
        trend_dict = {0: "Не изменился",
                      1: "Вверх",
                      2: "Вниз"}
        first_value = source[0]
        second_value = source[1]

        trend_limit = options["trend_limit"]
        if "%" in trend_limit:
            trend_limit = float(trend_limit[: trend_limit.find("%")])
            if abs((second_value - first_value) / first_value) * 100 <= trend_limit:
                array = 0
            elif second_value > first_value:
                array = 1
            else:
                array = 2
        else:
            trend_limit = float(trend_limit)
            if abs(second_value - first_value) <= trend_limit:
                array = 0
            elif second_value > first_value:
                array = 1
            else:
                array = 2
        idx = options['classes_names'].index(trend_dict[array])
        array = utils.to_categorical(idx, num_classes=options['num_classes'], dtype='uint8')
        # else:
        #     array = source

        instructions = {'instructions': np.array(array),
                        'parameters': options}

        return instructions

    def preprocess(self, array: np.ndarray, **options):

        if options['scaler'] not in ['no_scaler', None]:
            orig_shape = array.shape
            array = options['preprocess'].transform(array.reshape(-1, 1))
            array = array.reshape(orig_shape)

        return array