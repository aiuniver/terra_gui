import os
import shutil
import pandas as pd
from itertools import product


class MarkUp:

    def __init__(self, inp_dict: dict):
        self.task_type = inp_dict['task_type']
        self.classes_names = inp_dict['classes_names']
        self.alias = inp_dict['alias']

        self.BASEPATH = '/content/drive/MyDrive/TerraAI/annotations/sources'
        self.dst_path = '/content/drive/MyDrive/TerraAI/datasets/sources'
        ANNOT_EXT = 'annot'
        temp_directory = '/content/drive/MyDrive/TerraAI/annotations'
        self.tmp_path = os.path.join(temp_directory, f"{self.alias}.{ANNOT_EXT}")
        self.raw_path = os.path.join(self.tmp_path, 'raw_data')

        self.make_folders(inp_dict)
        self.make_none_dataframe()

    def make_folders(self, inp_dict: dict):
        os.makedirs(self.raw_path, exist_ok=True)
        shutil.unpack_archive(os.path.join(self.BASEPATH, inp_dict['name']), self.raw_path)
        if self.task_type != 'tracker':
            for cl_name in self.classes_names:
                os.makedirs(os.path.join(self.tmp_path, cl_name), exist_ok=True)

    def make_none_dataframe(self):
        if self.task_type == 'tracker':
            file_1 = []
            file_2 = []
            for i in range(1):  # range(len(sorted(os.listdir(self.raw_path)))-1):
                for crop_1, crop_2 in product(
                        sorted(os.listdir(os.path.join(self.raw_path, sorted(os.listdir(self.raw_path))[i]))),
                        sorted(os.listdir(os.path.join(self.raw_path, sorted(os.listdir(self.raw_path))[i + 1])))):
                    file_1.append(os.path.join(sorted(os.listdir(self.raw_path))[i], crop_1))
                    file_2.append(os.path.join(sorted(os.listdir(self.raw_path))[i + 1], crop_2))
            dataframe = pd.DataFrame({'file_1': file_1,
                                      'file_2': file_2,
                                      'class': [None for i in range(len(file_1))]})
            dataframe.to_csv(os.path.join(self.raw_path, 'dataframe.csv'), index=False)
        else:
            dataframe = pd.DataFrame({'file': [x for x in sorted(os.listdir(self.raw_path))],
                                      'class': [None for i in os.listdir(self.raw_path)]})
            dataframe.to_csv(os.path.join(self.tmp_path, 'dataframe.csv'), index=False)

    def fill_dataframe(self, idx_lbl: dict):
        path = self.tmp_path if self.task_type != 'tracker' else self.raw_path

        df = pd.read_csv(os.path.join(path, 'dataframe.csv'))
        for idx, label in idx_lbl.items():
            df.loc[idx, 'class'] = label
        df.to_csv(os.path.join(path, 'dataframe.csv'), index=False)

    def export_dataset(self):
        if self.task_type == 'tracker':
            shutil.make_archive(os.path.join(self.dst_path, self.alias), format='zip', root_dir=self.raw_path)
        else:
            df = pd.read_csv(os.path.join(self.tmp_path, 'dataframe.csv'))
            for i in range(len(df)):
                df.loc[i, 'file'] = os.path.join(df.loc[i, 'class'], df.loc[i, 'file'])
            df.to_csv(os.path.join(self.tmp_path, 'dataframe.csv'), index=False)
            for idx, image in enumerate(sorted(os.listdir(self.raw_path))):
                os.rename(os.path.join(self.raw_path, image),
                          os.path.join(self.tmp_path, df.loc[idx, 'class'], image))
            shutil.rmtree(self.raw_path)
            shutil.make_archive(os.path.join(self.dst_path, self.alias), format='zip', root_dir=self.tmp_path)
