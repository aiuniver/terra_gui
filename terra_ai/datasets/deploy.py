from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.datasets.preprocessing import CreatePreprocessing

import tempfile
import json
import os
import numpy as np
import shutil


class DeployDataset(object):

    def __init__(self, dataset_path: str):

        self.data = None
        self.dataset_path = dataset_path
        self.instructions: dict = {}

        with open(os.path.join(self.dataset_path, 'config.json'), 'r') as cfg:
            self.data = DatasetData(**json.load(cfg))

        for put_id in self.data.inputs.keys():
            self.instructions[put_id] = {}
            for instr_json in os.listdir(os.path.join(self.dataset_path, 'instructions', 'parameters')):
                idx, *name = os.path.splitext(instr_json)[0].split('_')
                name = '_'.join(name)
                if put_id == int(idx):
                    with open(os.path.join(self.dataset_path, 'instructions', 'parameters', instr_json), 'r') as instr:
                        self.instructions[put_id].update([(f'{idx}_{name}', json.load(instr))])

        self.preprocessing: CreatePreprocessing = CreatePreprocessing(dataset_path)
        self.preprocessing.load_preprocesses(self.data.columns)

    def make_array(self, paths_dict: dict):

        temp_directory = tempfile.mkdtemp()
        out_array = {}
        temp_array = {}
        for put_id, cols_names in self.instructions.items():
            temp_array[put_id] = {}
            for col_name, data in cols_names.items():
                instr = getattr(CreateArray(), f'instructions_{data["put_type"]}')(paths_dict[put_id][col_name], **data)
                cut = getattr(CreateArray(), f'cut_{data["put_type"]}')(instr['instructions'],
                                                                        dataset_folder=temp_directory,
                                                                        **self.instructions[put_id][col_name])
                temp_array[put_id][col_name] = []
                for elem in cut['instructions']:
                    arr = getattr(CreateArray(), f'create_{data["put_type"]}')(elem, **{
                        'preprocess': self.preprocessing.preprocessing[put_id][col_name]},
                                                                               **self.instructions[put_id][col_name])
                    arr = getattr(CreateArray(), f'preprocess_{data["put_type"]}')(arr['instructions'],
                                                                                   **arr['parameters'])
                    temp_array[put_id][col_name].append(arr)

            concat_list = []
            for col_name in temp_array[put_id].keys():
                concat_list.append(temp_array[put_id][col_name])

            out_array[put_id] = np.concatenate(concat_list, axis=1)

        shutil.rmtree(temp_directory, ignore_errors=True)

        return out_array
