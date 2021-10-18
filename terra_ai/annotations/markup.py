import os
import shutil
import pandas as pd
import json
from itertools import product
from datetime import datetime
from pytz import timezone
from terra_ai.data.annotations.markup import MarkupData, AnnotationData
from terra_ai.data.annotations.markup import SOURCE, DESTINATION_PATH, ANNOT_DIRECTORY, ANNOT_EXT
from terra_ai.datasets.creating import CreateDataset
from terra_ai.settings import DATASET_CONFIG


class MarkUp:
    def __init__(self, markup_data: MarkupData):
        self.source_path = SOURCE
        self.dst_path = DESTINATION_PATH
        annot_directory = ANNOT_DIRECTORY
        self.annot_path = os.path.join(annot_directory, f"{markup_data.alias}.{ANNOT_EXT}")

        self.task_type = markup_data.task_type
        self.classes_names = markup_data.classes_names
        self.classes_colors = markup_data.classes_colors
        self.alias = markup_data.alias
        self.source = markup_data.source
        self.to_do = markup_data.to_do
        self.until = markup_data.until

        self.make_folders()
        self.make_none_dataframe()

    def make_folders(self):
        os.makedirs(self.annot_path, exist_ok=True)
        shutil.unpack_archive(os.path.join(self.source_path, self.source), self.annot_path)

    def make_none_dataframe(self):
        if self.task_type == 'tracker':
            file_1 = []
            file_2 = []
            for i in range(len(sorted(os.listdir(self.annot_path)))-1):
                for crop_1, crop_2 in product(
                        sorted(os.listdir(os.path.join(self.annot_path, sorted(os.listdir(self.annot_path))[i]))),
                        sorted(os.listdir(os.path.join(self.annot_path, sorted(os.listdir(self.annot_path))[i + 1])))):
                    file_1.append(os.path.join(sorted(os.listdir(self.annot_path))[i], crop_1))
                    file_2.append(os.path.join(sorted(os.listdir(self.annot_path))[i + 1], crop_2))
            dataframe = pd.DataFrame({'file_1': file_1,
                                      'file_2': file_2,
                                      'class': ['none' for i in file_1]})
        else:
            files = []
            for folder in sorted(os.listdir(self.annot_path)):
                for fl in sorted(os.listdir(os.path.join(self.annot_path, folder))):
                    files.append(fl)
            dataframe = pd.DataFrame({'file': sorted(files), 'class': ['none' for i in files]})
        dataframe.to_csv(os.path.join(self.annot_path, 'dataframe.csv'), index=False)

    def fill_dataframe(self, idx_lbl: dict):
        df = pd.read_csv(os.path.join(self.annot_path, 'dataframe.csv'))
        for idx, label in idx_lbl.items():
            df.loc[idx, 'class'] = label
        df.to_csv(os.path.join(self.annot_path, 'dataframe.csv'), index=False)

        self.make_json(df)

    def export_dataset(self):
        if self.task_type == 'tracker':
            src = self.annot_path
        else:
            df = pd.read_csv(os.path.join(self.annot_path, 'dataframe.csv'))
            os.makedirs(os.path.join(self.annot_path, 'Done'))
            for cl in self.classes_names:
                os.makedirs(os.path.join(self.annot_path, 'Done', cl))
            for folder in sorted(os.listdir(self.annot_path)):
                if os.path.isdir(os.path.join(self.annot_path, folder)):
                    for image in sorted(os.listdir(os.path.join(self.annot_path, folder))):
                        for i in range(len(df)):
                            if df.loc[i, 'file'] == image:
                                df.loc[i, 'file'] = os.path.join(df.loc[i, 'class'], df.loc[i, 'file'])
                                os.rename(os.path.join(self.annot_path, folder, image),
                                          os.path.join(self.annot_path, 'Done', df.loc[i, 'class'], image))
            df.to_csv(os.path.join(self.annot_path, 'dataframe.csv'), index=False)
            os.rename(os.path.join(self.annot_path, 'config.json'),
                      os.path.join(self.annot_path, 'Done', 'config.json'))
            os.rename(os.path.join(self.annot_path, 'dataframe.csv'),
                      os.path.join(self.annot_path, 'Done', 'dataframe.csv'))
            src = os.path.join(self.annot_path, 'Done')

        CreateDataset.zip_dataset(src, os.path.join(self.dst_path, self.alias))

    def make_json(self, df):
        size_bytes = 0
        for path, dirs, files in os.walk(self.annot_path):
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
            'cover': os.listdir(os.path.join(self.annot_path, os.listdir(self.annot_path)[0]))[0],
            'to_do': self.to_do,
            'size': {'value': size_bytes}
        }

        with open(os.path.join(self.annot_path, DATASET_CONFIG), 'w') as fp:
            json.dump(AnnotationData(**data).native(), fp)
