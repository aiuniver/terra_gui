import os

import pandas as pd
import numpy as np

from typing import Any
from tensorflow.keras import utils

from .base import Array


class ClassificationArray(Array):

    def prepare(self, sources, dataset_folder=None, **options):
        length = options['length'] if 'length' in options.keys() else None
        depth = options['depth'] if 'depth' in options.keys() else None
        step = options['step'] if 'step' in options.keys() else None

        type_processing = options['type_processing']

        if 'sources_paths' in options.keys():
            classes_names = sorted([os.path.basename(elem) for elem in options['sources_paths']])
        else:
            if type_processing == "categorical":
                classes_names = list(dict.fromkeys(sources))
            else:
                if len(options["ranges"].split(" ")) == 1:
                    border = max(sources) / int(options["ranges"])
                    classes_names = np.linspace(border, max(sources), int(options["ranges"])).tolist()
                else:
                    classes_names = options["ranges"].split(" ")

        instructions = {'instructions': sources,
                        'parameters': {'classes_names': classes_names,
                                       'encoding': 'ohe',
                                       'num_classes': len(classes_names),
                                       'cols_names': options['cols_names'],
                                       'put': options['put'],
                                       'type_processing': type_processing,
                                       'length': length,
                                       'step': step,
                                       'depth': depth
                                       }
                        }

        return instructions

    def create(self, source: Any, **options):

        class_name = source.to_list() if isinstance(source, pd.Series) else source
        class_name = class_name if isinstance(class_name, list) else [class_name]
        if options['type_processing'] == 'categorical':
            if len(class_name) == 1:
                index = [options['classes_names'].index(class_name[0])]
            else:
                index = []
                for i in range(len(class_name)):
                    index.append(options['classes_names'].index(class_name[i]))
        else:
            index = []
            for i in range(len(class_name)):
                for j, cl_name in enumerate(options['classes_names']):
                    if class_name[i] <= float(cl_name):
                        index.append(j)
                        break
        if len(class_name) == 1:
            index = utils.to_categorical(index[0], num_classes=options['num_classes'], dtype='uint8')
        else:
            index = utils.to_categorical(index, num_classes=options['num_classes'], dtype='uint8')

        index = np.array(index)

        instructions = {'instructions': index,
                        'parameters': options}

        return instructions

    def preprocess(self, array: np.ndarray, **options):

        return array
