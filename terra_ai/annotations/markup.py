import os
import shutil
import pandas as pd
import json
from itertools import product
from datetime import datetime
from pytz import timezone
from terra_ai.data.annotations.markup import MarkupData, AnnotationData


class MarkUp:
    def __init__(self, markup_data: MarkupData):
        inp_dict = markup_data.native()
        self.markup_source_path = '/content/drive/MyDrive/TerraAI/annotations/sources'
        self.dst_path = '/content/drive/MyDrive/TerraAI/datasets/sources'
        ANNOT_EXT = 'annot'
        temp_directory = '/content/drive/MyDrive/TerraAI/annotations'
        self.tmp_path = os.path.join(temp_directory, f"{inp_dict['alias']}.{ANNOT_EXT}")

        self.task_type = inp_dict['task_type']
        self.classes_names = inp_dict['classes_names']
        self.classes_colors = inp_dict['classes_colors']
        self.alias = inp_dict['alias']
        self.source = inp_dict['source']
        self.to_do = inp_dict['to_do']
        self.until = inp_dict['until']

        self.make_folders()
        self.make_none_dataframe()

    def make_folders(self):
        os.makedirs(self.tmp_path, exist_ok=True)
        shutil.unpack_archive(os.path.join(self.markup_source_path, self.source), self.tmp_path)

    def make_none_dataframe(self):
        if self.task_type == 'tracker':
            file_1 = []
            file_2 = []
            for i in range(1):  # range(len(sorted(os.listdir(self.tmp_path)))-1):
                for crop_1, crop_2 in product(
                        sorted(os.listdir(os.path.join(self.tmp_path, sorted(os.listdir(self.tmp_path))[i]))),
                        sorted(os.listdir(os.path.join(self.tmp_path, sorted(os.listdir(self.tmp_path))[i + 1])))):
                    file_1.append(os.path.join(sorted(os.listdir(self.tmp_path))[i], crop_1))
                    file_2.append(os.path.join(sorted(os.listdir(self.tmp_path))[i + 1], crop_2))
            dataframe = pd.DataFrame({'file_1': file_1,
                                      'file_2': file_2,
                                      'class': ['none' for i in file_1]})
        else:
            files = []
            for folder in sorted(os.listdir(self.tmp_path)):
                for fl in sorted(os.listdir(os.path.join(self.tmp_path, folder))):
                    files.append(fl)
            dataframe = pd.DataFrame({'file': sorted(files), 'class': ['none' for i in files]})
        dataframe.to_csv(os.path.join(self.tmp_path, 'dataframe.csv'), index=False)

    def fill_dataframe(self, idx_lbl: dict):
        df = pd.read_csv(os.path.join(self.tmp_path, 'dataframe.csv'))
        for idx, label in idx_lbl.items():
            df.loc[idx, 'class'] = label
        df.to_csv(os.path.join(self.tmp_path, 'dataframe.csv'), index=False)

        self.make_json(df)

    def export_dataset(self):
        pass
        # if self.task_type == 'tracker':
        #     shutil.make_archive(os.path.join(self.dst_path, self.alias), format='zip', root_dir=self.tmp_path)
        # else:
        #     df = pd.read_csv(os.path.join(self.tmp_path, 'dataframe.csv'))
        #     for i in range(len(df)):
        #         df.loc[i, 'file'] = os.path.join(df.loc[i, 'class'], df.loc[i, 'file'])
        #     df.to_csv(os.path.join(self.tmp_path, 'dataframe.csv'), index=False)
        #     for idx, image in enumerate(sorted(os.listdir(self.raw_path))):
        #         os.rename(os.path.join(self.raw_path, image),
        #                   os.path.join(self.tmp_path, df.loc[idx, 'class'], image))
        #     shutil.rmtree(self.raw_path)
        #     shutil.make_archive(os.path.join(self.dst_path, inp_dict['alias']), format='zip', root_dir=self.tmp_path)

    def make_json(self, df):
        DATASET_CONFIG = 'config.json'

        size_bytes = 0
        for path, dirs, files in os.walk(self.tmp_path):
            for fl in files:
                size_bytes += os.path.getsize(os.path.join(path, fl))

        data = {
            'alias': self.alias,
            'created': datetime.now().astimezone(timezone("Europe/Moscow")).isoformat(),
            'until': self.until,
            'classes_names': self.classes_names,
            'classes_colors': self.classes_colors,
            'task_type': self.task_type,
            'progress': [0, len(df[df['class'] != 'none']), len(df)],
            'cover': os.listdir(os.path.join(self.tmp_path, os.listdir(self.tmp_path)[0]))[0],
            'to_do': self.to_do,
            'size': {'value': size_bytes}
        }

        with open(os.path.join(self.tmp_path, DATASET_CONFIG), 'w') as fp:
            json.dump(AnnotationData(**data).native(), fp)
