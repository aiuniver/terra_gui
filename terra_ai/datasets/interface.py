from terra_ai.data.datasets.creation import SourceData, CreationInfoData, CreationData
from terra_ai.data.datasets.creations.column_processing import types as put_types
from terra_ai.data.datasets.extra import SourceModeChoice, LayerInputTypeChoice, LayerOutputTypeChoice
from terra_ai.datasets import loading
from terra_ai.datasets.creating import CreateDataset
from terra_ai.settings import TERRA_PATH, PROJECT_PATH, DATASETS_SOURCE_DIR, DATASETS_LOADED_DIR
from terra_ai import progress
from pathlib import Path
from IPython.display import display
import pandas as pd
import os
import time


class InterfaceCreateDataset(object):
    alias: str
    name: str
    datasets_path: Path
    source_path: Path
    tags: list

    def __init__(self):

        self.datasets_path = TERRA_PATH.datasets
        self.info: CreationInfoData = CreationInfoData()
        self.use_generator: bool = False
        self.processing: dict = {}
        self.processing_table: dict = {}
        self.inputs: list = []
        self.outputs: list = []
        self.required_parameters = ['alias', 'name', 'datasets_path',
                                    'source_path', 'use_generator', 'tags']

    def load_archive(self, archive: str):

        """
        Распаковка архива с исходными данными во временное хранилище.
        Inputs:
            archive: str - название архива в папке ./TerraAI/datasets/sources
        """

        assert archive in os.listdir(TERRA_PATH.sources), f'Данного архива нет в папке {TERRA_PATH.sources}'

        source = {
            "mode": SourceModeChoice.GoogleDrive,
            "value": TERRA_PATH.sources.joinpath(archive)
        }

        self.source_path = DATASETS_SOURCE_DIR.joinpath('googledrive', archive[:archive.rfind('.')])

        start = time.perf_counter()
        print(f'Идёт распаковка исходников датасета в папку {self.source_path}')
        loading.source(SourceData(**source))
        finished = False
        while not finished:
            time.sleep(1)
            p = progress.pool('dataset_source_load')
            print(f"{p.message}. {round(p.percent, 0)}%")
            if p.finished:
                finish = time.perf_counter()
                print(f'Распаковка исходников завершена. '
                      f'Времени потрачено: {round(finish - start, 2)} сек.')
                finished = True

    def manage_info(self, part: list, shuffle: bool):

        """
        Метод создания объекта CreationInfoData
        Inputs:
            part: list - процентное отношение обучающей и валидационной выборок к общему количеству примеров
            shuffle: bool - перемешать выборки
        Пример:
            .manage_info(part=[0.8, 0.2], shuffle=True)
        """

        self.info = CreationInfoData(**{'part': {'train': part[0],
                                                 'validation': part[1]},
                                        'shuffle': shuffle})

    def manage_processing(self, action=None, name=None, put_type=None, options=None):

        if options is None:
            options = {}
        put_type_list = ['Image', 'Video', 'Text', 'Audio', 'Scaler',
                         'Classification', 'Regression', 'Segmentation',
                         'TextSegmentation', 'ObjectDetection']
        if action:
            assert action in ['add', 'delete'], f'Введен недопустимый тип действия.'
            if action == 'add':
                assert put_type in put_type_list, f'Введен недопустимый тип обработчика.'
                assert name not in self.processing_table['Название'], f'Обработчик с таким названием уже существует.'
                self.processing_table['Название'].append(name)
                self.processing_table['Тип'].append(put_type)
                self.processing_table['Параметры'].append(
                    getattr(put_types,
                            f"Parameters{put_type}Data")(**options)
                )
            elif action == 'delete':
                assert name in self.processing_table['Название'], f'Обработчика с таким названием не существует.'
                idx = self.processing_table['Название'].index(name)
                for col in self.processing_table:
                    del self.processing_table[col][idx]
            for proc in self.processing_table:
                for idx in range(len(self.processing_table['Название'])):
                    self.processing[idx] = {'type': self.processing_table['Тип'][idx],
                                            'parameters': self.processing_table['Параметры'][idx].native()}
        else:
            if put_type:
                assert put_type in put_type_list, f'Введен недопустимый тип обработчика.'
                print(getattr(put_types, f"Parameters{put_type}Data").__doc__)
            else:
                display(pd.DataFrame(self.processing_table))

    def manage_inputs(self, action: str = '', name: str = '', inp_type: str = '', folders=None, csv=None,
                      options=None):  # [('Цена', 'Такой обработчик'), ('Площадь', 'Другой обработчик')]

        """
        Управление списком входных слоёв.
        Inputs:
            action: str - тип действия. Варианты: 'add', 'delete'
            name: str - наименование входа
            inp_type: str - тип входных данных. Варианты: 'Image', 'Video', 'Audio', 'Text', 'Dataframe'
            folders: list - список папок с файлами для обработки
            csv: str - ОПЦИОНАЛЬНО. Название csv-файла.
        Пример:
            .manage_inputs(action='add', name='Цветы', inp_type='Image', folders=['Орхидея'],
                        options={'height': 160, 'width': 128, 'image_mode': 'stretch', 'net': 'convolutional',
                                 'scaler': 'min_max_scaler'})
        Для вызова информации о необходимых параметрах для определенного типа данных следует вызвать данный метод,
            указав только тип данных.
            .manage_inputs(inp_type='Image')
        """

        if folders is None:
            folders = []
        if options is None:
            options = {}
        inp_type_list = ['Image', 'Video', 'Audio', 'Text', 'Dataframe']
        if action:
            assert action in ['add', 'delete'], 'Введен недопустимый тип действия.'
            if action == 'add':
                assert inp_type in inp_type_list, 'Веден недопустимый тип входных данных'
                for input in self.inputs:
                    assert input.name != name, 'Вход с таким названием уже существует.'
                id_num = len(self.inputs) + 1
                options.update({'sources_paths': [self.source_path.joinpath(x) for x in folders]})
                input_data = {'alias': f"input_{id_num}",
                              'id': id_num,
                              'name': name,
                              'type': getattr(LayerInputTypeChoice, inp_type),
                              'parameters': options
                              }
                # for pair in link:
                #     w_id = self.processing_table['Название'].index(pair[1])
                #     input_data['parameters'][Path(pair[0])] = {pair[0]: [w_id]}
                self.inputs.append(input_data)  # self.inputs.append(CreationInputData(**input_data))
            elif action == 'delete':
                for i, input in enumerate(self.inputs):
                    if input['name'] == name:
                        del self.inputs[i]

            for i, input in enumerate(self.inputs):
                input['id'] = i + 1
        else:
            if inp_type:
                assert inp_type in inp_type_list, f'Введен недопустимый тип обработчика.'
                print(getattr(put_types, f"Parameters{inp_type}Data").__doc__)
            # else:
            # display(pd.DataFrame(self.inputs))

    def manage_outputs(self, action: str = '', name: str = '', out_type: str = '',
                       folders=None, csv=None, options=None):

        """
        Управление списком выходных слоёв.
        Outputs:
            action: str - тип действия. Варианты: 'add', 'delete'
            name: str - наименование выхода
            out_type: str - тип входных данных. Варианты: 'Image', 'Video', 'Audio', 'Text', 'Dataframe'
            folders: list - список папок с файлами для обработки
            csv: str - ОПЦИОНАЛЬНО. Название csv-файла.
        Пример:
            .manage_outputs(action='add', name='Маски сегментации', out_type='Segmentation', folders=['Маски'],
                            options={'image_mode': 'stretch', 'net': 'convolutional', 'scaler': 'min_max_scaler'})
        Для вызова информации о необходимых параметрах для определенного типа данных следует вызвать данный метод,
            указав только тип данных.
            .manage_outputs(out_type='Segmentation')
        """

        if folders is None:
            folders = []
        if options is None:
            options = {}
        out_type_list = ['Image', 'Text', 'Audio', 'Dataframe', 'Classification',
                         'Segmentation', 'TextSegmentation', 'ObjectDetection',
                         'Tracker', 'Text2Speech', 'Speech2Text']
        if action:
            assert action in ['add', 'delete'], 'Введен недопустимый тип действия.'
            if action == 'add':
                assert out_type in out_type_list
                for output in self.outputs:
                    assert output.name != name, 'Выход с таким названием уже существует.'
                id_num = len(self.inputs) + 1  # СДЕЛАТЬ ЭТО НА ЭТАПЕ ОБЩЕЙ СБОРКИ CreationVersionData
                options.update({'sources_paths': [self.source_path.joinpath(x) for x in folders]})
                output_data = {'alias': f"output_{id_num}",
                               'id': id_num,
                               'name': name,
                               'type': getattr(LayerOutputTypeChoice, out_type),
                               'parameters': options}
                # for pair in link:
                #     w_id = self.processing_table['Название'].index(pair[1])
                #     output_data['parameters'][Path(pair[0])] = {pair[0]: [w_id]}
                self.outputs.append(output_data)  # self.outputs.append(CreationOutputData(**output_data))
            elif action == 'delete':
                for i, output in enumerate(self.outputs):
                    if output['name'] == name:
                        del self.outputs[i]
            id_num = len(self.inputs) + 1
            for output in self.outputs:
                output['id'] = id_num
                id_num += 1
        else:
            if out_type:
                assert out_type in out_type_list, f'Введен недопустимый тип обработчика.'
                print(getattr(put_types, f"Parameters{out_type}Data").__doc__)
            # else:
            # display(pd.DataFrame(self.outputs))

    def status(self):

        """
        Проверка готовности начала процесса создания датасета.
        Output:
            True: bool - в случае, если все параметры заполнены
            False: bool - в случае, если не все параметры заполнены
        """

        for attr in self.required_parameters:
            print(f"- {attr} = {self.__dict__[attr] if self.__dict__.get(attr) else ''}")
        try:
            CreationData(**self.__dict__)
            print('Все параметры заполнены.')
            return True
        except:
            print('Необходимо корректно заполнить параметры.')
            return False

    def start(self, alias: str, name: str, generator: bool):

        """
        Запуск процесса создания датасета
        Inputs:
            alias: str - алиас датасета
            name: str - название датасета
            generator: bool - использовать генератор
        """

        self.tags = []
        self.alias = alias
        self.name = name
        self.use_generator = generator

        if self.status():
            start = time.perf_counter()
            create_dataset = CreateDataset(CreationData(**self.__dict__))
            finished = False
            while not finished:
                time.sleep(1)
                p = progress.pool('create_dataset')
                print(f"{p.message}. {round(p.percent, 0)}%")
                if p.finished:
                    finish = time.perf_counter()
                    print(f'Создание датасета завершено. '
                          f'Времени потрачено: {round(finish - start, 2)} сек.')
                    finished = True

            return create_dataset
