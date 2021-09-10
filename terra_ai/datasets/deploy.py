from terra_ai.data.datasets.dataset import DatasetData, DatasetPathsData
from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.datasets.preprocessing import CreatePreprocessing
from terra_ai.utils import decamelize

import tempfile
import json
import os
import numpy as np


class DeployDataset(object):

    def __init__(self, dataset_path: str):

        self.data = None
        self.paths: DatasetPathsData = DatasetPathsData(basepath=dataset_path)
        self.temp_directory = tempfile.mkdtemp()

        self.instructions: dict = {}

        for instr_json in os.listdir(os.path.join(self.paths.instructions, 'parameters')):
            idx, name = os.path.splitext(instr_json)[0].split('_')
            if name == 'inputs':
                with open(os.path.join(self.paths.instructions, 'parameters', instr_json), 'r') as instr:
                    self.instructions[int(idx)] = json.load(instr)
        with open(os.path.join(self.paths.basepath, 'config.json'), 'r') as cfg:
            self.data = DatasetData(**json.load(cfg))

        self.preprocessing: CreatePreprocessing = CreatePreprocessing(dataset_path)
        self.preprocessing.load_preprocesses(self.instructions.keys())

    def make_array(self, paths_dict: dict):

        paths_array = paths_dict.copy()

        for inp in self.data.inputs.keys():
            instr = getattr(CreateArray(),
                            f'instructions_{decamelize(self.data.inputs[inp].task)}')(paths_dict[inp],
                                                                                      **self.instructions[inp])
            cut = getattr(CreateArray(),
                          f'cut_{decamelize(self.data.inputs[inp].task)}')(instr['instructions'],
                                                                           tmp_folder=self.temp_directory,
                                                                           **instr['parameters'],
                                                                           **self.preprocessing.preprocessing[inp])
            array = []
            for elem in cut['instructions']:
                array.append(getattr(CreateArray(),
                                     f'create_{decamelize(self.data.inputs[inp].task)}')(elem, **cut['parameters']))
            paths_array[inp] = np.array(array)

        return paths_array
