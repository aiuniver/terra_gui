from terra_ai.utils import decamelize, get_tempdir
from terra_ai.datasets import creating_classes
from terra_ai.datasets.preprocessing import CreatePreprocessing
from terra_ai.data.datasets.creation import CreationData, CreationVersionData, DatasetCreationArchitectureData
from terra_ai.data.datasets.dataset import DatasetCommonPathsData, DatasetVersionPathsData,\
    DatasetVersionExtData, DatasetCommonData
from terra_ai.data.datasets.extra import DatasetGroupChoice
from terra_ai.settings import DATASET_EXT, DATASET_CONFIG, DATASET_VERSION_EXT, DATASET_VERSION_CONFIG, \
    DATASET_PROGRESS_NAME, VERSION_PROGRESS_NAME, DATASET_VERSION_CREATION_DATA
from terra_ai import progress
from terra_ai.settings import TERRA_PATH

import os
import random
import pandas as pd
import json
import shutil
import zipfile
import h5py
from distutils.dir_util import copy_tree
from pathlib import Path
from datetime import datetime
from pytz import timezone
from terra_ai.logging import logger


def zip_dataset(src, dst):
    zf = zipfile.ZipFile("%s.zip" % dst, "w", zipfile.ZIP_DEFLATED)
    abs_src = os.path.abspath(src)
    for dir_name, sub_dirs, files in os.walk(src):
        for filename in files:
            abs_name = os.path.abspath(os.path.join(dir_name, filename))
            arc_name = abs_name[len(abs_src) + 1:]
            zf.write(abs_name, arc_name)
    zf.close()


class CreateDataset(object):

    @progress.threading
    def __init__(self, creation_data: CreationData):

        progress.pool.reset(name=DATASET_PROGRESS_NAME,
                            message='Начало',
                            finished=False)
        logger.info(f'Начало формирования датасета {creation_data.name}.')
        self.temp_directory: Path = get_tempdir()
        os.makedirs(self.temp_directory.joinpath('.'.join([creation_data.alias, DATASET_EXT])), exist_ok=True)
        self.dataset_paths_data: DatasetCommonPathsData = DatasetCommonPathsData(
            basepath=self.temp_directory.joinpath('.'.join([creation_data.alias, DATASET_EXT])))
        progress.pool(name=DATASET_PROGRESS_NAME, message='Копирование файлов', percent=10)
        copy_tree(str(creation_data.source.path), str(self.dataset_paths_data.sources))
        zip_dataset(self.dataset_paths_data.sources, self.temp_directory.joinpath('sources'))
        shutil.move(str(self.temp_directory.joinpath('sources.zip')), self.dataset_paths_data.basepath)
        shutil.rmtree(self.dataset_paths_data.sources)
        dataset_data = self.write_dataset_configure(creation_data)
        if TERRA_PATH.datasets.joinpath('.'.join([creation_data.alias, DATASET_EXT])).is_dir():
            progress.pool(name=DATASET_PROGRESS_NAME,
                          message=f"Удаление существующего датасета "
                                  f"{TERRA_PATH.datasets.joinpath('.'.join([creation_data.alias, DATASET_EXT]))}",
                          percent=70)
            shutil.rmtree(TERRA_PATH.datasets.joinpath('.'.join([creation_data.alias, DATASET_EXT])))
        progress.pool(name=DATASET_PROGRESS_NAME, message=f"Копирование датасета в {TERRA_PATH.datasets}",
                      percent=80)
        shutil.move(str(self.dataset_paths_data.basepath), TERRA_PATH.datasets)
        progress.pool(name=DATASET_PROGRESS_NAME, message=f"Удаление временной папки {self.temp_directory}", percent=95)
        shutil.rmtree(self.temp_directory)
        progress.pool(name=DATASET_PROGRESS_NAME, message='Формирование датасета завершено',
                      data=dataset_data, percent=100)
        logger.info(f'Создан датасет {creation_data.name}.')

        self.version = CreateVersion(creation_data=creation_data)

    def write_dataset_configure(self, creation_data):

        dataset_data = DatasetCommonData(**{'name': creation_data.name,
                                            'alias': creation_data.alias,
                                            'group': DatasetGroupChoice.custom,
                                            'tags': creation_data.tags,
                                            'date': datetime.now().astimezone(timezone("Europe/Moscow")).isoformat(),
                                            'architecture': creation_data.architecture
                                            })

        with open(os.path.join(self.dataset_paths_data.basepath, DATASET_CONFIG), 'w') as fp:
            json.dump(dataset_data.native(), fp)

        return dataset_data


class CreateVersion(object):

    @progress.threading
    def __init__(self, creation_data: CreationData):

        progress.pool.reset(name=VERSION_PROGRESS_NAME, message='Начало создания версии датасета', finished=False)
        version_data = creation_data.version  # не согласен

        self.dataframe: dict = {}
        self.columns: dict = {}
        self.preprocessing = CreatePreprocessing()
        self.y_cls = []
        self.tags = {}

        # Подготовка путей и файлов
        self.temp_directory: Path = get_tempdir()
        self.sources_temp_directory: Path = get_tempdir()
        self.dataset_paths_data = DatasetCommonPathsData(basepath=self.temp_directory)
        self.parent_dataset_paths_data = DatasetCommonPathsData(
            basepath=TERRA_PATH.datasets.joinpath('.'.join([creation_data.alias, DATASET_EXT]))
        )
        progress.pool(name=VERSION_PROGRESS_NAME, message='Копирование исходного архива', percent=0)
        shutil.copyfile(
            self.parent_dataset_paths_data.basepath.joinpath('sources.zip'),
            self.dataset_paths_data.basepath.joinpath('sources.zip')
        )
        progress.pool(name=VERSION_PROGRESS_NAME, message='Распаковка исходного архива', percent=0)
        with zipfile.ZipFile(self.dataset_paths_data.basepath.joinpath('sources.zip'), 'r') as z_file:
            z_file.extractall(self.sources_temp_directory)
        current_version = self.dataset_paths_data.versions.joinpath(f'{version_data.alias}.{DATASET_VERSION_EXT}')
        os.makedirs(current_version)
        self.version_paths_data = DatasetVersionPathsData(basepath=current_version)
        with open(self.parent_dataset_paths_data.basepath.joinpath('config.json'), 'r') as cfg:
            parent_architecture = json.load(cfg)['architecture']

        # Начало создания версии
        architecture_class = getattr(getattr(creating_classes, decamelize(parent_architecture)),
                                     parent_architecture + 'Class')()

        version_data = architecture_class.preprocess_version_data(
            version_data=version_data,
            source_path=creation_data.source.path,
            version_path_data=self.sources_temp_directory
        )
        logger.debug(version_data)

        progress.pool(name=VERSION_PROGRESS_NAME, message='Создание инструкций', percent=0)
        self.instructions, self.tags = architecture_class.create_instructions(
            version_data=version_data,
            sources_temp_directory=self.sources_temp_directory,
            version_paths_data=self.version_paths_data
        )

        progress.pool(name=VERSION_PROGRESS_NAME, message='Создание объектов обработки', percent=0)

        for prep_type in ['numeric', 'text']:
            self.preprocessing = getattr(architecture_class, f"create_{prep_type}_preprocessing")(
                instructions=self.instructions,
                preprocessing=self.preprocessing
            )
        for prep_type in ['numeric', 'text']:
            self.preprocessing = getattr(architecture_class, f"fit_{prep_type}_preprocessing")(
                put_data=self.instructions.inputs,
                preprocessing=self.preprocessing,
                sources_temp_directory=self.sources_temp_directory
            )
            self.preprocessing = getattr(architecture_class, f"fit_{prep_type}_preprocessing")(
                put_data=self.instructions.outputs,
                preprocessing=self.preprocessing,
                sources_temp_directory=self.sources_temp_directory
            )

        self.create_table(version_data)

        self.inputs, inp_col = architecture_class.create_input_parameters(
            input_instr=self.instructions.inputs,
            version_data=version_data,
            preprocessing=self.preprocessing,
            version_paths_data=self.version_paths_data
        )
        self.outputs, out_col = architecture_class.create_output_parameters(
            output_instr=self.instructions.outputs,
            version_data=version_data,
            preprocessing=self.preprocessing,
            version_paths_data=self.version_paths_data
        )
        self.service = architecture_class.create_service_parameters(
            output_instr=self.instructions.outputs,
            version_data=version_data,
            preprocessing=self.preprocessing,
            version_paths_data=self.version_paths_data
        )

        self.columns.update(inp_col)
        self.columns.update(out_col)

        progress.pool(name=VERSION_PROGRESS_NAME, message='Создание массивов данных', percent=0)

        architecture_class.create_arrays(
            instructions=self.instructions,
            version_paths_data=self.version_paths_data,
            dataframe=self.dataframe,
            preprocessing=self.preprocessing
        )

        progress.pool(name=VERSION_PROGRESS_NAME, message='Сохранение', percent=100)
        self.write_instructions_to_files()
        zip_dataset(self.version_paths_data.basepath, os.path.join(self.dataset_paths_data.versions, 'version'))
        version_dir = self.parent_dataset_paths_data.versions.joinpath('.'.join([version_data.alias,
                                                                                 DATASET_VERSION_EXT]))
        if version_dir.is_dir():
            shutil.rmtree(version_dir)
        os.makedirs(version_dir)
        shutil.move(self.dataset_paths_data.versions.joinpath('version.zip'), version_dir.joinpath('version.zip'))
        self.write_version_configure(version_data)
        shutil.rmtree(self.sources_temp_directory)
        shutil.rmtree(self.temp_directory)
        progress.pool(name=VERSION_PROGRESS_NAME, message='Формирование версии датасета завершено', data=version_data,
                      percent=100, finished=True)
        logger.info(f'Создана версия {version_data.name}', extra={'type': "info"})

    #
    # def create_put_instructions(self, put_data, processing):
    #
    #     def instructions(one_path, params):
    #
    #         try:
    #             cut = getattr(getattr(arrays_classes, decamelize(params["type"])), f'{params["type"]}Array')().prepare(
    #                 sources=[one_path],
    #                 dataset_folder=self.version_paths_data.sources,
    #                 **params['parameters'],
    #                 **{'cols_names': col_name, 'put': idx}
    #             )
    #
    #         except Exception:
    #             progress.pool(version_progress_name, error=f'Ошибка создания инструкций для {put_data.get(idx).name}')
    #             logger.debug(f'Создание инструкций провалилось на {one_path}')
    #             raise
    #
    #         return cut  # {'instructions': return_data, 'parameters': cut['parameters']}
    #
    #     put_parameters = {}
    #     for idx in range(put_data[0].id, put_data[0].id + len(put_data)):
    #         self.tags[idx] = {}
    #         put_parameters[idx] = {}
    #         for path, val in put_data.get(idx).parameters.items():
    #             for name, proc in val.items():
    #                 data = []
    #                 data_to_pass = []
    #                 parameters = processing[str(proc[0])].native()  # Аккуратно с [0]
    #                 col_name = f'{idx}_{decamelize(parameters["type"])}'
    #
    #                 if Path(self.sources_temp_directory).joinpath(name.split(':')[0]).is_dir():
    #                     # Собираем все пути к файлам в один список
    #                     for folder_name in name.split(':'):
    #                         current_path = Path(self.sources_temp_directory).joinpath(folder_name)
    #                         for direct, folder, files_name in os.walk(current_path):
    #                             if files_name:
    #                                 for file_name in sorted(files_name):
    #                                     data_to_pass.append(os.path.join(current_path, file_name))
    #                 elif Path(self.sources_temp_directory).joinpath(path.split(':')[0]).is_file():
    #                     col_name = f'{idx}_{name}'
    #                     current_path = Path(self.sources_temp_directory).joinpath(path.split(':')[0])
    #                     # Собираем всю колонку в один список
    #                     _, enc = autodetect_encoding(str(current_path), True)
    #                     data_to_pass = pd.read_csv(current_path, sep=None, usecols=[name],
    #                                                engine='python', encoding=enc)[name].to_list()
    #                     if decamelize(parameters['type']) in PATH_TYPE_LIST:  # in PATH_TYPE_LIST:
    #                         data_to_pass = [str(Path(self.sources_temp_directory).joinpath(Path(x)))
    #                                         for x in data_to_pass]
    #                 print('data_to_pass', data_to_pass[:3])
    #                 with concurrent.futures.ThreadPoolExecutor() as executor:
    #                     results = executor.map(instructions, data_to_pass, repeat(parameters))
    #                     progress.pool(version_progress_name, message=f'Формирование файлов')  # Добавить конкретику
    #                     for i, result in enumerate(results):
    #                         progress.pool(version_progress_name, percent=ceil(i / len(data_to_pass) * 100))
    #                         if decamelize(parameters['type']) in PATH_TYPE_LIST:
    #                             for j in range(len(result['instructions'])):
    #                                 result['instructions'][j] = result['instructions'][j].replace(
    #                                     str(self.version_paths_data.sources), '')[1:]
    #                         data += result['instructions']
    #                         result_params = result['parameters']
    #                         # classes_names += result['parameters']['classes_names']
    #                         if idx == put_data[0].id and parameters['type'] != LayerOutputTypeChoice.Classification:
    #                             self.y_cls += [os.path.basename(os.path.dirname(data_to_pass[i])) for _ in
    #                                            range(len(result['instructions']))]
    #                 if parameters['type'] == LayerOutputTypeChoice.Classification:
    #                     data = self.y_cls
    #                     # ### Дальше идет не очень хороший код
    #                     if parameters['parameters']['type_processing'] == "categorical":
    #                         classes_names = list(dict.fromkeys(data))
    #                     else:
    #                         if len(parameters['parameters']["ranges"].split(" ")) == 1:
    #                             border = max(data) / int(parameters['parameters']["ranges"])
    #                             classes_names = np.linspace(border, max(data), int(parameters['parameters']["ranges"])).tolist()
    #                         else:
    #                             classes_names = parameters['parameters']["ranges"].split(" ")
    #                     result['parameters']['classes_names'] = classes_names
    #                     result['parameters']['num_classes'] = len(classes_names)
    #                     # ###
    #                 instructions_data = InstructionsData(instructions=data, parameters=result_params)
    #                 instructions_data.parameters.update({'put_type': decamelize(parameters['type'])})
    #                 print(instructions_data.instructions[0])
    #                 print(len(instructions_data.instructions[0]))
    #                 put_parameters[idx] = {col_name: instructions_data}
    #                 self.tags[idx].update({col_name: decamelize(parameters['type'])})
    #
    #     return put_parameters

    # def create_preprocessing(self, instructions: DatasetInstructionsData):
    #
    #     for put in list(instructions.inputs.values()) + list(instructions.outputs.values()):
    #         for col_name, data in put.items():
    #             if 'timeseries' in data.parameters.values():
    #                 length = data.parameters['length']
    #                 depth = data.parameters['depth']
    #                 step = data.parameters['step']
    #                 for pt in list(instructions.inputs.values()) + list(instructions.outputs.values()):
    #                     for col_nm, dt in pt.items():
    #                         if 'raw' in dt.parameters.values():
    #                             dt.parameters['length'] = length
    #                             dt.parameters['depth'] = depth
    #                             dt.parameters['step'] = step
    #             if 'scaler' in data.parameters.keys():
    #                 self.preprocessing.create_scaler(**data.parameters)
    #             elif 'prepare_method' in data.parameters.keys():
    #                 if data.parameters['prepare_method'] in [LayerPrepareMethodChoice.embedding,
    #                                                          LayerPrepareMethodChoice.bag_of_words]:
    #                     self.preprocessing.create_tokenizer(text_list=data.instructions, **data.parameters)
    #                 elif data.parameters['prepare_method'] == LayerPrepareMethodChoice.word_to_vec:
    #                     self.preprocessing.create_word2vec(text_list=data.instructions, **data.parameters)
    #             else:
    #                 self.preprocessing.preprocessing.update(
    #                     {data.parameters['put']: {data.parameters['cols_names']: None}}
    #                 )

    # def fit_preprocessing(self, put_data):
    #
    #     for key in put_data.keys():
    #         for col_name, data in put_data[key].items():
    #             if 'scaler' in data.parameters and data.parameters['scaler'] not in [LayerScalerImageChoice.no_scaler,
    #                                                                                  None]:
    #                 progress.pool(version_progress_name, message=f'Обучение {camelize(data.parameters["scaler"])}')
    #                 #                     try:
    #                 if self.tags[key][col_name] in PATH_TYPE_LIST:
    #                     for i in range(len(data.instructions)):
    #                         #                             progress.pool(version_progress_name,
    #                         #                                           percent=ceil(i / len(data.instructions) * 100))
    #
    #                         arr = getattr(CreateArray(), f'create_{self.tags[key][col_name]}')(
    #                             str(self.sources_temp_directory.joinpath(data.instructions[i])),
    #                             **data.parameters
    #                         )
    #
    #                         if data.parameters['put_type'] in [decamelize(LayerInputTypeChoice.Image),
    #                                                            decamelize(LayerOutputTypeChoice.Image)]:
    #                             arr = {'instructions': cv2.resize(arr['instructions'], (data.parameters['width'],
    #                                                                                     data.parameters['height']))}
    #                         if data.parameters['scaler'] == 'terra_image_scaler':
    #                             self.preprocessing.preprocessing[key][col_name].fit(arr['instructions'])
    #                         else:
    #                             self.preprocessing.preprocessing[key][col_name].fit(arr['instructions'].reshape(-1, 1))
    #                 else:
    #                     self.preprocessing.preprocessing[key][col_name].fit(np.array(data.instructions).reshape(-1, 1))
    #
    #                     # except Exception:
    #                     #     progress.pool(version_progress_name, error='Ошибка обучения скейлера')
    #                     #     raise

    def create_table(self, version_data: CreationVersionData):

        classes_dict = {}
        for out in self.instructions.outputs.keys():
            #             if creation_data.columns_processing.get(str(out)) is not None and \
            #                     creation_data.columns_processing.get(str(out)).type == LayerOutputTypeChoice.Classification and \
            #                     creation_data.columns_processing.get(str(out)).parameters.type_processing != \
            #                     LayerTypeProcessingClassificationChoice.ranges or \
            #                     creation_data.outputs.get(out).type == LayerOutputTypeChoice.Classification:
            #                 for col_name, data in self.instructions.outputs[out].items():
            #                     class_names = list(dict.fromkeys(data.instructions))
            #                     classes_dict = {cl_name: [] for cl_name in class_names}
            #                     for idx, cl_name in enumerate(data.instructions):
            #                         classes_dict[cl_name].append(idx)
            #                 break
            #             else:
            for col_name, data in self.instructions.outputs[out].items():
                classes_dict = {'one_class': [idx for idx in range(len(data.instructions))]}

        if version_data.info.shuffle:
            for key in classes_dict.keys():
                random.shuffle(classes_dict[key])

        split_sequence = {"train": [], "val": []}
        for key, value in classes_dict.items():
            train_len = int(version_data.info.part.train * len(classes_dict[key]))
            split_sequence['train'].extend(value[:train_len])
            split_sequence['val'].extend(value[train_len:])

        if version_data.info.shuffle:
            random.shuffle(split_sequence['train'])
            random.shuffle(split_sequence['val'])

        build_dataframe = {}
        for inp in self.instructions.inputs.keys():
            for key, value in self.instructions.inputs[inp].items():
                build_dataframe[key] = value.instructions
                print(len(value.instructions))
        for out in self.instructions.outputs.keys():
            for key, value in self.instructions.outputs[out].items():
                build_dataframe[key] = value.instructions
                print(len(value.instructions))
        try:
            dataframe = pd.DataFrame(build_dataframe)
        except Exception:
            progress.pool(VERSION_PROGRESS_NAME,
                          error='Ошибка создания датасета. Несоответствие количества входных/выходных данных')
            for key, value in build_dataframe.items():
                logger.debug(key, len(value))
            raise
        for key, value in split_sequence.items():
            self.dataframe[key] = dataframe.loc[value, :].reset_index(drop=True)
        print(self.dataframe['train'])

    # def create_put_parameters(self, put_instructions, version_data: CreationVersionData, put: str):  # -> dict:
    #
    #     creating_puts_data = {}
    #     for key in put_instructions.keys():
    #         put_array = []
    #         self.columns[key] = {}
    #         for col_name, data in put_instructions[key].items():
    #             data_to_pass = data.instructions[0]
    #             options_to_pass = data.parameters.copy()
    #             if self.preprocessing.preprocessing.get(key) and \
    #                     self.preprocessing.preprocessing.get(key).get(col_name):
    #                 prep = self.preprocessing.preprocessing.get(key).get(col_name)
    #                 options_to_pass.update([('preprocess', prep)])
    #             if self.tags[key][col_name] in PATH_TYPE_LIST:
    #                 data_to_pass = str(self.version_paths_data.sources.joinpath(data_to_pass))
    #             create = getattr(getattr(arrays_classes, data.parameters["put_type"]),
    #                              f'{camelize(data.parameters["put_type"])}Array')().create(
    #                 source=data_to_pass, **options_to_pass)
    #             array = getattr(getattr(arrays_classes, data.parameters["put_type"]),
    #                             f'{camelize(data.parameters["put_type"])}Array')().preprocess(
    #                 create['instructions'], **create['parameters'])
    #             # array = array[0] if isinstance(array, tuple) else array
    #             # if not array.shape:
    #             #     array = np.expand_dims(array, 0)
    #             put_array.append(array)
    #             if create['parameters'].get('classes_names'):
    #                 classes_names = create['parameters'].get('classes_names')
    #             else:
    #                 classes_names = sorted([os.path.basename(x) for x in version_data.__dict__[put].get(key).parameters.keys()])
    #
    #             # Прописываем параметры для колонки
    #             col_parameters = {'datatype': DataType.get(len(array.shape), 'DIM'),
    #                               'dtype': str(array.dtype),
    #                               'shape': array.shape,
    #                               'name': version_data.__dict__[put].get(key).name,
    #                               'task': camelize(data.parameters.get('put_type')),
    #                               'classes_names': classes_names,
    #                               'classes_colors': data.parameters.get('classes_colors'),
    #                               'num_classes': len(classes_names) if classes_names else 0,
    #                               'encoding': 'none' if not data.parameters.get('encoding') else data.parameters.get('encoding')}
    #             current_column = DatasetInputsData(**col_parameters) if put == 'inputs' else DatasetOutputsData(**col_parameters)
    #             self.columns[key].update([(col_name, current_column.native())])
    #
    #         put_array = np.concatenate(put_array, axis=0)
    #         classes_colors_list, classes_names_list, encoding_list, task_list = [], [], [], []
    #         for value in self.columns[key].values():
    #             if value.get('classes_colors'):
    #                 for c_color in value.get('classes_colors'):
    #                     classes_colors_list.append(c_color)
    #             if value.get('classes_names'):
    #                 for c_name in value.get('classes_names'):
    #                     classes_names_list.append(c_name)
    #             encoding_list.append(value.get('encoding') if value.get('encoding') else 'none')
    #             task_list.append(value.get('task'))
    #         put_parameters = {'datatype': DataType.get(len(put_array.shape), 'DIM'),
    #                           'dtype': str(put_array.dtype),
    #                           'shape': put_array.shape,
    #                           'name': version_data.__dict__[put].get(key).name,
    #                           'task': task_list[0] if len(task_list) == 1 else 'Dataframe',
    #                           'classes_names': classes_names_list if classes_names_list else None,
    #                           'classes_colors': classes_colors_list if classes_colors_list else None,
    #                           'num_classes': len(classes_names_list) if classes_names_list else None,
    #                           'encoding': 'none' if len(encoding_list) > 1 or not encoding_list else encoding_list[0]}
    #
    #         creating_puts_data[key] = DatasetInputsData(**put_parameters).native() if put == 'inputs'\
    #             else DatasetOutputsData(**put_parameters).native()
    #
    #     return creating_puts_data

    # def create_dataset_arrays(self, put_data: dict):
    #
    #     def array_creation(row, instructions):
    #
    #         full_array = []
    #         for h in range(len(row)):
    #             try:
    #                 create = getattr(getattr(arrays_classes, instructions[h]["put_type"]),
    #                                  f'{camelize(instructions[h]["put_type"])}Array')().create(
    #                     source=row[h], **instructions[h])
    #                 prepr = getattr(getattr(arrays_classes, instructions[h]["put_type"]),
    #                                 f'{camelize(instructions[h]["put_type"])}Array')().preprocess(
    #                     create['instructions'], **create['parameters'])
    #                 full_array.append(prepr)
    #             except Exception:
    #                 progress.pool(version_progress_name, error='Ошибка создания массивов данных')
    #                 raise
    #
    #         return full_array
    #
    #     for split in ['train', 'val']:
    #         open_mode = 'w' if not self.version_paths_data.arrays.joinpath('dataset.h5') else 'a'
    #         hdf = h5py.File(self.version_paths_data.arrays.joinpath('dataset.h5'), open_mode)
    #         if split not in list(hdf.keys()):
    #             hdf.create_group(split)
    #         for key in put_data.keys():
    #             col_name = None
    #             length, depth, step = 0, 0, 1
    #
    #             for col_name, data in put_data[key].items():
    #                 depth = data.parameters['depth'] if 'depth' in data.parameters.keys() and \
    #                                                     data.parameters['depth'] else 0
    #                 length = data.parameters['length'] if depth else 0
    #                 step = data.parameters['step'] if depth else 1
    #
    #             data_to_pass = []
    #             dict_to_pass = []
    #             for i in range(0, len(self.dataframe[split]) - length - depth, step):
    #                 tmp_data = []
    #                 tmp_parameter_data = []
    #                 for col_name, data in put_data[key].items():
    #                     parameters_to_pass = data.parameters.copy()
    #                     if self.preprocessing.preprocessing.get(key) and \
    #                             self.preprocessing.preprocessing.get(key).get(col_name):
    #                         prep = self.preprocessing.preprocessing.get(key).get(col_name)
    #                         parameters_to_pass.update([('preprocess', prep)])
    #
    #                     if self.tags[key][col_name] in PATH_TYPE_LIST:
    #                         tmp_data.append(os.path.join(self.version_paths_data.sources, # .self.sources_temp_directory
    #                                                      self.dataframe[split].loc[i, col_name]))
    #                     elif 'depth' in data.parameters.keys() and data.parameters['depth']:
    #                         if 'trend' in data.parameters.keys() and data.parameters['trend']:
    #                             tmp_data.append([self.dataframe[split].loc[i, col_name],
    #                                              self.dataframe[split].loc[i + data.parameters['length'],
    #                                                                        col_name]])
    #                         elif 'trend' in data.parameters.keys():
    #                             tmp_data.append(
    #                                 self.dataframe[split].loc[i + data.parameters['length']:i +
    #                                                                                         data.parameters['length']
    #                                                                                         + data.parameters[
    #                                                                                             'depth'] - 1, col_name])
    #                         else:
    #                             tmp_data.append(self.dataframe[split].loc[i:i + data.parameters['length'] - 1,
    #                                             col_name])
    #
    #                     elif self.tags[key][col_name] == decamelize(LayerOutputTypeChoice.ObjectDetection):
    #                         tmp_data.append(self.dataframe[split].loc[i, col_name])
    #                         height, width = self.dataframe[split].iloc[i, 0].split(';')[1].split(',')
    #                         # tmp_im = Image.open(os.path.join(self.sources_temp_directory,
    #                         #                                  self.dataframe[split].iloc[i, 0]))
    #                         parameters_to_pass.update([('orig_x', int(width)),
    #                                                    ('orig_y', int(height))])
    #                     else:
    #                         tmp_data.append(self.dataframe[split].loc[i, col_name])
    #                     tmp_parameter_data.append(parameters_to_pass)
    #                 data_to_pass.append(tmp_data)
    #                 dict_to_pass.append(tmp_parameter_data)
    #
    #             progress.pool(version_progress_name,
    #                           message=f'Формирование массивов {split.title()} выборки. ID: {key}.',
    #                           percent=0)
    #
    #             if self.tags[key][col_name] == decamelize(LayerOutputTypeChoice.ObjectDetection):
    #                 for n in range(3):
    #                     current_group = f'id_{key + n}'
    #                     current_serv_group = f'id_{key + n}_service'
    #                     if current_group not in list(hdf[split].keys()):
    #                         hdf[split].create_group(current_group)
    #                     if current_serv_group not in list(hdf[split].keys()):
    #                         hdf[split].create_group(current_serv_group)
    #             else:
    #                 hdf[split].create_group(f'id_{key}')
    #
    #             with concurrent.futures.ThreadPoolExecutor() as executor:
    #                 results = executor.map(array_creation, data_to_pass, dict_to_pass)
    #                 for i, result in enumerate(results):
    #                     progress.pool(version_progress_name, percent=ceil(i / len(data_to_pass) * 100))
    #                     if not self.tags[key][col_name] == decamelize(LayerOutputTypeChoice.ObjectDetection):
    #                         if depth:
    #                             if 'trend' in dict_to_pass[i][0].keys() and dict_to_pass[i][0]['trend']:
    #                                 array = np.array(result[0])
    #                             else:
    #                                 array = self.postprocess_timeseries(result)
    #                         else:
    #                             array = np.concatenate(result, axis=0)
    #                         hdf[f'{split}/id_{key}'].create_dataset(str(i), data=array)
    #                     else:
    #                         for n in range(3):
    #                             hdf[f'{split}/id_{key + n}'].create_dataset(str(i), data=result[0][n])
    #                             hdf[f'{split}/id_{key + n}_service'].create_dataset(str(i), data=result[0][n + 3])
    #                     del result
    #         hdf.close()

    def write_instructions_to_files(self):

        parameters_path = self.version_paths_data.instructions.joinpath('parameters')
        os.makedirs(parameters_path, exist_ok=True)

        for cols in self.instructions.inputs.values():
            for col_name, data in cols.items():
                with open(parameters_path.joinpath(f'{col_name}.json'), 'w') as cfg:
                    json.dump(data.parameters, cfg)

        for cols in self.instructions.outputs.values():
            for col_name, data in cols.items():
                with open(parameters_path.joinpath(f'{col_name}.json'), 'w') as cfg:
                    json.dump(data.parameters, cfg)

        tables_path = self.version_paths_data.instructions.joinpath('tables')
        os.makedirs(tables_path, exist_ok=True)

        for key in self.dataframe.keys():
            self.dataframe[key].to_csv(self.version_paths_data.instructions.joinpath('tables', f'{key}.csv'))

    def write_version_configure(self, version_data):
        """
        inputs, outputs, service, size, columns, date
        """
        tags_list = []
        for inp, cols in self.tags.items():
            for col_name, tag in cols.items():
                tags_list.append(tag)
        size_bytes = 0
        for path, dirs, files in os.walk(self.version_paths_data.basepath):
            for file in files:
                size_bytes += os.path.getsize(os.path.join(path, file))

        data = {'alias': version_data.alias,
                'name': version_data.name,
                'tags': tags_list,
                'date': datetime.now().astimezone(timezone("Europe/Moscow")).isoformat(),
                'size': {'value': size_bytes},
                'length': {'train': None,
                           'val': None},
                'inputs': self.inputs,
                'outputs': self.outputs,
                'service': self.service,
                'columns': self.columns
                }

        with h5py.File(self.version_paths_data.arrays.joinpath('dataset.h5'), 'r') as hdf:
            for part in hdf.keys():
                for idx in hdf[part].keys():
                    data['length'][part] = len(hdf[part][idx])

        with open(self.parent_dataset_paths_data.versions.joinpath(f'{version_data.alias}.{DATASET_VERSION_EXT}')
                      .joinpath(DATASET_VERSION_CONFIG), 'w') as fp:
            json.dump(DatasetVersionExtData(**data).native(), fp)

        creation_data = DatasetCreationArchitectureData(
            inputs=version_data.inputs.native(),
            outputs=version_data.outputs.native()
        )

        with open(self.parent_dataset_paths_data.versions.joinpath(f'{version_data.alias}.{DATASET_VERSION_EXT}')
                      .joinpath(DATASET_VERSION_CREATION_DATA), 'w') as fp:
            json.dump(creation_data.native(), fp)
