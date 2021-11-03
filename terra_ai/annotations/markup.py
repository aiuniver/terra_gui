import os
import shutil
import pandas as pd
import json
from itertools import product
from datetime import datetime, date
from pytz import timezone
from pathlib import Path
from terra_ai.data.annotations.markup import MarkupData, AnnotationData, AnnotationPathsData
from terra_ai.data.annotations.markup import ANNOT_EXT
from terra_ai.datasets.creating import CreateDataset
from terra_ai.settings import DATASET_CONFIG


class MarkUp:
    def __init__(self, markup_data: MarkupData):
        common_path = os.path.join(os.sep, *markup_data.native()['annotations_path'].split(os.sep)[:-1])
        self.source = os.path.join(common_path, 'annotations', 'sources', markup_data.source)
        self.dst_path = os.path.join(common_path, 'datasets', 'sources')
        os.makedirs(os.path.join(markup_data.annotations_path, f'{markup_data.alias}.{ANNOT_EXT}'),
                    exist_ok=True)
        self.path = AnnotationPathsData(
            basepath=Path(markup_data.annotations_path, f'{markup_data.alias}.{ANNOT_EXT}'))
        self.annot_path = self.path.basepath
        self.raw_path = os.path.join(self.annot_path, 'Raw')

        self.task_type = markup_data.task_type
        self.classes_names = sorted(markup_data.classes_names)
        self.classes_colors = markup_data.classes_colors
        self.alias = markup_data.alias
        self.to_do = markup_data.to_do
        self.until = markup_data.until

        shutil.unpack_archive(self.source, self.raw_path)
        self.make_json(self.make_none_dataframe())

    def make_none_dataframe(self):
        if self.task_type == 'tracker':
            file_1 = []
            file_2 = []
            for i in range(len(sorted(os.listdir(self.raw_path))) - 1):
                for crop_1, crop_2 in product(
                        sorted(os.listdir(os.path.join(self.raw_path, sorted(os.listdir(self.raw_path))[i]))),
                        sorted(os.listdir(os.path.join(self.raw_path, sorted(os.listdir(self.raw_path))[i + 1])))):
                    file_1.append(os.path.join(sorted(os.listdir(self.raw_path))[i], crop_1))
                    file_2.append(os.path.join(sorted(os.listdir(self.raw_path))[i + 1], crop_2))
            dataframe = pd.DataFrame({'file_1': file_1,
                                      'file_2': file_2,
                                      'result': ['не размечено' for i in file_1]})
        else:
            files = []
            if os.path.isdir(os.path.join(self.raw_path, sorted(os.listdir(self.raw_path))[0])):
                for folder in sorted(os.listdir(self.raw_path)):
                    for fl in sorted(os.listdir(os.path.join(self.raw_path, folder))):
                        files.append(os.path.join(folder, fl))
            else:
                for fl in sorted(os.listdir(self.raw_path)):
                    files.append(fl)
            dataframe = pd.DataFrame({'file': sorted(files), 'result': ['не размечено' for i in files]})
        dataframe.to_csv(os.path.join(self.annot_path, 'dataframe.csv'), index=False)
        return dataframe

    def fill_dataframe(self, idx_lbl: dict):
        df = pd.read_csv(os.path.join(self.annot_path, 'dataframe.csv'))
        for idx, label in idx_lbl.items():
            df.loc[idx, 'result'] = label

        self.make_json(df)
        df.to_csv(os.path.join(self.annot_path, 'dataframe.csv'), index=False)

    def export_dataset(self):

        if self.task_type == 'tracker':
            dst_src = self.raw_path

        elif self.task_type == 'classification':
            dst_src = os.path.join(self.annot_path, 'Done')

            df = pd.read_csv(os.path.join(self.annot_path, 'dataframe.csv'))
            os.makedirs(os.path.join(self.annot_path, 'Done'), exist_ok=True)
            for cl in self.classes_names:
                os.makedirs(os.path.join(self.annot_path, 'Done', cl), exist_ok=True)

            if os.path.isdir(os.path.join(self.raw_path, sorted(os.listdir(self.raw_path))[0])):
                for folder in sorted(os.listdir(self.raw_path)):
                    for image in sorted(os.listdir(os.path.join(self.raw_path, folder))):
                        for i in range(len(df)):
                            if df.loc[i, 'file'].endswith(image):
                                df.loc[i, 'file'] = os.path.join(df.loc[i, 'result'], image)
                                shutil.copyfile(os.path.join(self.raw_path, folder, image),
                                                os.path.join(dst_src, df.loc[i, 'result'], image))
            else:
                for image in sorted(os.listdir(self.raw_path)):
                    for i in range(len(df)):
                        if df.loc[i, 'file'].endswith(image):
                            df.loc[i, 'file'] = os.path.join(df.loc[i, 'result'], image)
                            shutil.copyfile(os.path.join(self.raw_path, image),
                                            os.path.join(dst_src, df.loc[i, 'result'], image))
            df.to_csv(os.path.join(self.annot_path, 'dataframe.csv'), index=False)

        elif self.task_type == 'object_detection':
            dst_src = os.path.join(self.annot_path, 'Done')
            os.makedirs(os.path.join(self.annot_path, 'Done', 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.annot_path, 'Done', 'annotations'), exist_ok=True)

            if os.path.isdir(os.path.join(self.raw_path, sorted(os.listdir(self.raw_path))[0])):
                for folder in sorted(os.listdir(self.raw_path)):
                    for image in sorted(os.listdir(os.path.join(self.raw_path, folder))):
                        shutil.copyfile(os.path.join(self.raw_path, folder, image),
                                        os.path.join(dst_src, 'images', image))
            else:
                for image in sorted(os.listdir(self.raw_path)):
                    shutil.copyfile(os.path.join(self.raw_path, image),
                                    os.path.join(dst_src, 'images', image))

            df = pd.read_csv(os.path.join(self.annot_path, 'dataframe.csv'))

            for i in range(len(df)):
                df.loc[i, 'file'] = os.path.join('images', df.loc[i, 'file'].split(os.sep)[-1])
                with open(os.path.join(self.annot_path, 'Done', 'annotations',
                                       f"{'.'.join(df.loc[i, 'file'].split(os.sep)[-1].split('.')[:-1])}.txt"),
                          'w') as f:
                    for bbox in df.loc[i, 'result'].split(' '):
                        f.write(f"{bbox}\n")
                df.loc[i, 'result'] = os.path.join('annotations',
                                                f"{'.'.join(df.loc[i, 'file'].split(os.sep)[-1].split('.')[:-1])}.txt")
            df.to_csv(os.path.join(self.annot_path, 'dataframe.csv'), index=False)
            with open(os.path.join(dst_src, 'obj.names'), 'w') as f:
                for item in self.classes_names:
                    f.write("%s\n" % item)

        else:
            dst_src = os.path.join(self.annot_path, 'Done')
            os.makedirs(os.path.join(self.annot_path, 'Done', 'images'), exist_ok=True)

            if os.path.isdir(os.path.join(self.raw_path, sorted(os.listdir(self.raw_path))[0])):
                for folder in sorted(os.listdir(self.raw_path)):
                    for image in sorted(os.listdir(os.path.join(self.raw_path, folder))):
                        shutil.copyfile(os.path.join(self.raw_path, folder, image),
                                        os.path.join(dst_src, 'images', image))
            else:
                for image in sorted(os.listdir(self.raw_path)):
                    shutil.copyfile(os.path.join(self.raw_path, image),
                                    os.path.join(dst_src, 'images', image))

            df = pd.read_csv(os.path.join(self.annot_path, 'dataframe.csv'))
            for i in range(len(df)):
                df.loc[i, 'file'] = os.path.join('images', df.loc[i, 'file'].split(os.sep)[-1])
            df.to_csv(os.path.join(self.annot_path, 'dataframe.csv'), index=False)

        shutil.copyfile(os.path.join(self.annot_path, 'dataframe.csv'),
                        os.path.join(dst_src, 'dataframe.csv'))
        shutil.copyfile(os.path.join(self.annot_path, 'config.json'),
                        os.path.join(dst_src, 'config.json'))

        CreateDataset.zip_dataset(dst_src, os.path.join(self.dst_path, self.alias))
        if self.task_type != 'tracker':
            shutil.rmtree(dst_src)

    def make_json(self, df):
        size_bytes = 0
        for path, dirs, files in os.walk(self.annot_path):
            for fl in files:
                size_bytes += os.path.getsize(os.path.join(path, fl))
        if os.path.isdir(os.path.join(self.raw_path, sorted(os.listdir(self.raw_path))[0])):
            cover = os.listdir(os.path.join(self.raw_path, os.listdir(self.raw_path)[0]))[0]
        else:
            cover = os.listdir(self.raw_path)[0]
        data = {
            'alias': self.alias,
            'created': datetime.now().astimezone(timezone("Europe/Moscow")).isoformat(),
            'until': date(*self.until).isoformat(),
            'classes_names': self.classes_names,
            'classes_colors': self.classes_colors,
            'task_type': self.task_type,
            'progress': [0, len(df[df['result'] != 'не размечено']), len(df)],
            'cover': cover,
            'to_do': self.to_do,
            'size': {'value': size_bytes}
        }

        with open(os.path.join(self.annot_path, DATASET_CONFIG), 'w') as fp:
            json.dump(AnnotationData(**data).native(), fp)
