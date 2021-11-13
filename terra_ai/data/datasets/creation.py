"""
## Структура данных для работы с датасетами

### **Загрузка исходных данных датасета**
```
In [1]: from terra_ai.data.datasets.creation import SourceData

In [2]: source = {
   ...:     "mode": "GoogleDrive",
   ...:     "value": "source.zip",
   ...: }

In [3]: data = SourceData(**source)

In [4]: data
Out[4]: SourceData(mode=<SourceModeChoice.GoogleDrive: 'GoogleDrive'>, value=PosixPath('source.zip'))

In [5]: data.dict()
Out[5]:
{'mode': <SourceModeChoice.GoogleDrive: 'GoogleDrive'>,
 'value': PosixPath('source.zip')}

In [6]: data.json()
Out[6]: '{"mode": "GoogleDrive", "value": "source.zip"}'

In [7]: print(data.json(indent=2, ensure_ascii=False))
{
  "mode": "GoogleDrive",
  "value": "source.zip"
}
```

### **Формирование датасета**
```
In [1]: from terra_ai.data.datasets.creation import CreationData

In [2]: source = {
   ...:     "name": "Самолеты",
   ...:     "inputs": [
   ...:         {
   ...:             "alias": "input_1",
   ...:             "name": "Input 1",
   ...:             "type": "text",
   ...:             "parameters": {
   ...:                 "x_len": "100",
   ...:                 "step": "30",
   ...:                 "max_words_count": "20000",
   ...:             },
   ...:         },
   ...:     ],
   ...:     "outputs": [
   ...:         {
   ...:             "alias": "input_1",
   ...:             "name": "Input 1",
   ...:             "type": "image",
   ...:             "parameters": {
   ...:                 "width": "120",
   ...:                 "height": "80",
   ...:             },
   ...:         },
   ...:     ],
   ...: }

In [3]: data = CreationData(**source)

In [4]: data
Out[4]: CreationData(name='Самолеты', info=CreationInfoData(part=CreationInfoPartData(train=0.6, validation=0.4), shuffle=True), tags=[], inputs=[CreationInputData(alias='input_1', name='Input 1', type=<LayerInputTypeChoice.text: 'text'>, parameters=LayerInputTypeTextData(folder_path=None, delete_symbols=None, x_len=100, step=30, max_words_count=20000, pymorphy=False, prepare_method=<LayerPrepareMethodChoice.embedding: 'embedding'>, word_to_vec_size=None))], outputs=[CreationOutputData(alias='input_1', name='Input 1', type=<LayerOutputTypeChoice.image: 'image'>, parameters=LayerOutputTypeImageData(folder_path=None, width=120, height=80, net=<LayerNetChoice.convolutional: 'convolutional'>, scaler=<LayerScalerChoice.no_scaler: 'no_scaler'>))])

In [5]: data.dict()
Out[5]:
{'name': 'Самолеты',
 'info': {'part': {'train': 0.6, 'validation': 0.4},
  'shuffle': True},
 'tags': [],
 'inputs': [{'alias': 'input_1',
   'name': 'Input 1',
   'type': <LayerInputTypeChoice.text: 'text'>,
   'parameters': {'folder_path': None,
    'delete_symbols': None,
    'x_len': 100,
    'step': 30,
    'max_words_count': 20000,
    'pymorphy': False,
    'prepare_method': <LayerPrepareMethodChoice.embedding: 'embedding'>,
    'word_to_vec_size': None}}],
 'outputs': [{'alias': 'input_1',
   'name': 'Input 1',
   'type': <LayerOutputTypeChoice.image: 'image'>,
   'parameters': {'folder_path': None,
    'width': 120,
    'height': 80,
    'net': <LayerNetChoice.convolutional: 'convolutional'>,
    'scaler': <LayerScalerChoice.no_scaler: 'no_scaler'>}}]}

In [6]: data.json()
Out[6]: '{"name": "\\u0421\\u0430\\u043c\\u043e\\u043b\\u0435\\u0442\\u044b", "info": {"part": {"train": 0.6, "validation": 0.4}, "shuffle": true}, "tags": [], "inputs": [{"alias": "input_1", "name": "Input 1", "type": "text", "parameters": {"folder_path": null, "delete_symbols": null, "x_len": 100, "step": 30, "max_words_count": 20000, "pymorphy": false, "prepare_method": "embedding", "word_to_vec_size": null}}], "outputs": [{"alias": "input_1", "name": "Input 1", "type": "image", "parameters": {"folder_path": null, "width": 120, "height": 80, "net": "Convolutional", "scaler": "NoScaler"}}]}'

In [7]: print(data.json(indent=2, ensure_ascii=False))
{
  "name": "Самолеты",
  "info": {
    "part": {
      "train": 0.6,
      "validation": 0.4,
    },
    "shuffle": true
  },
  "tags": [],
  "inputs": [
    {
      "alias": "input_1",
      "name": "Input 1",
      "type": "text",
      "parameters": {
        "folder_path": null,
        "delete_symbols": null,
        "x_len": 100,
        "step": 30,
        "max_words_count": 20000,
        "pymorphy": false,
        "prepare_method": "embedding",
        "word_to_vec_size": null
      }
    }
  ],
  "outputs": [
    {
      "alias": "input_1",
      "name": "Input 1",
      "type": "image",
      "parameters": {
        "folder_path": null,
        "width": 120,
        "height": 80,
        "net": "Convolutional",
        "scaler": "NoScaler"
      }
    }
  ]
}
```
"""

from math import fsum
from pathlib import Path
from typing import Union, Optional, Any, Dict
from pydantic import validator
from pydantic.types import DirectoryPath
from pydantic.networks import HttpUrl
from pydantic.errors import EnumMemberError

from ... import settings as terra_ai_settings
from ..mixins import BaseMixinData, UniqueListMixin, AliasMixinData, IDMixinData
from ..types import (
    confilepath,
    confilename,
    FilePathType,
    ConstrainedFloatValueGe0Le1,
    ConstrainedLayerNameValue,
)
from ..exceptions import (
    ValueTypeException,
    PartTotalException,
    ListEmptyException,
    ObjectDetectionQuantityLayersException,
)
from .extra import (
    SourceModeChoice,
    LayerInputTypeChoice,
    LayerOutputTypeChoice,
    ColumnProcessingTypeChoice,
)
from .creations import column_processing
from .tags import TagsList
from . import creations


class FilePathSourceData(BaseMixinData):
    value: confilepath(ext="zip")
    label: Optional[str]

    @validator("label", allow_reuse=True, always=True)
    def _validate_label(cls, value: str, values) -> str:
        file_path = values.get("value")
        if not file_path:
            return value
        return file_path.name.split(".zip")[0]


class FilePathSourcesList(UniqueListMixin):
    class Meta:
        source = FilePathSourceData
        identifier = "label"


class SourceData(BaseMixinData):
    """
    Информация для загрузки исходников датасета
    """

    mode: SourceModeChoice
    "Режим загрузки исходных данных"
    value: Union[confilepath(ext="zip"), HttpUrl, confilename(ext="zip")]
    "Значение для режим загрузки исходных данных. Тип будет зависеть от выбранного режима `mode`"

    @validator("value", allow_reuse=True)
    def _validate_mode_value(
        cls, value: Union[FilePathType, HttpUrl, str], values
    ) -> Union[FilePathType, HttpUrl, str]:
        mode = values.get("mode")
        if mode == SourceModeChoice.GoogleDrive:
            if not isinstance(value, Path):
                raise ValueTypeException(value, FilePathType)
        if mode == SourceModeChoice.URL:
            if not isinstance(value, HttpUrl):
                raise ValueTypeException(value, HttpUrl)
        return value


class CreationInfoPartData(BaseMixinData):
    """
    Доли использования данных для обучающей, тестовой и валидационной выборок"
    """

    train: ConstrainedFloatValueGe0Le1 = 0.7
    "Обучающая выборка"
    validation: ConstrainedFloatValueGe0Le1 = 0.3
    "Валидационная выборка"

    @property
    def total(self) -> float:
        """
        Сумма всех значений
        """
        return fsum([self.train, self.validation])


class CreationInfoData(BaseMixinData):
    """
    Информация о данных датасета
    """

    part: CreationInfoPartData = CreationInfoPartData()
    "Доли выборок"
    shuffle: Optional[bool] = True
    "Случайным образом перемешивать элементы"

    @validator("part", allow_reuse=True)
    def _validate_part(cls, value: CreationInfoPartData) -> CreationInfoPartData:
        if value.total != float(1):
            raise PartTotalException(value)
        return value


class CreationInputData(IDMixinData):
    """
    Информация о `input`-слое
    """

    name: ConstrainedLayerNameValue
    "Название"
    type: LayerInputTypeChoice
    "Тип данных"
    parameters: Any
    "Параметры. Тип данных будет зависеть от выбранного типа `type`"

    @validator("type", pre=True)
    def _validate_type(cls, value: LayerInputTypeChoice) -> LayerInputTypeChoice:
        if value not in list(LayerInputTypeChoice):
            raise EnumMemberError(enum_values=list(LayerInputTypeChoice))
        name = (
            value
            if isinstance(value, LayerInputTypeChoice)
            else LayerInputTypeChoice(value)
        ).name
        type_ = getattr(
            creations.layers.input, getattr(creations.layers.input.Layer, name)
        )
        cls.__fields__["parameters"].type_ = type_
        return value

    @validator("parameters", always=True)
    def _validate_parameters(cls, value: Any, values, field) -> Any:
        return field.type_(**value or {})


class CreationOutputData(IDMixinData):
    """
    Информация о `output`-слое
    """

    name: ConstrainedLayerNameValue
    "Название"
    type: LayerOutputTypeChoice
    "Тип данных"
    parameters: Any
    "Параметры. Тип данных будет зависеть от выбранного типа `type`"

    @validator("type", pre=True)
    def _validate_type(cls, value: LayerOutputTypeChoice) -> LayerOutputTypeChoice:
        if value not in list(LayerOutputTypeChoice):
            raise EnumMemberError(enum_values=list(LayerOutputTypeChoice))
        name = (
            value
            if isinstance(value, LayerOutputTypeChoice)
            else LayerOutputTypeChoice(value)
        ).name
        type_ = getattr(
            creations.layers.output, getattr(creations.layers.output.Layer, name)
        )
        cls.__fields__["parameters"].type_ = type_
        return value

    @validator("parameters", always=True)
    def _validate_parameters(cls, value: Any, values, field) -> Any:
        return field.type_(**value or {})


class CreationInputsList(UniqueListMixin):
    """
    Список `input`-слоев, основанных на `CreationInputData`
    ```
    class Meta:
        source = CreationInputData
        identifier = "alias"
    ```
    """

    class Meta:
        source = CreationInputData
        identifier = "id"


class CreationOutputsList(UniqueListMixin):
    """
    Список `output`-слоев, основанных на `CreationOutputData`
    ```
    class Meta:
        source = CreationOutputData
        identifier = "alias"
    ```
    """

    class Meta:
        source = CreationOutputData
        identifier = "id"


class ColumnsProcessingData(BaseMixinData):
    type: ColumnProcessingTypeChoice
    parameters: Any

    @validator("type", pre=True)
    def _validate_type(
        cls, value: ColumnProcessingTypeChoice
    ) -> ColumnProcessingTypeChoice:
        if not value:
            return value
        name = (
            value
            if isinstance(value, ColumnProcessingTypeChoice)
            else ColumnProcessingTypeChoice(value)
        ).name
        cls.__fields__["parameters"].type_ = getattr(
            column_processing, column_processing.ColumnProcessing[name].value
        )
        return value

    @validator("parameters", always=True)
    def _validate_parameters(cls, value: Any, values, field) -> Any:
        return field.type_(**(value or {}))


class CreationData(AliasMixinData):
    """
    Полная информация о создании датасета
    """

    name: str
    "Название"
    datasets_path: DirectoryPath
    "Путь к директории датасетов проекта"
    source_path: DirectoryPath
    "Путь к директории с исходниками, полученный после их загрузки"
    info: CreationInfoData = CreationInfoData()
    "Информация о данных"
    tags: TagsList = TagsList()
    "Список тегов"
    use_generator: bool = False
    "Использовать генераторы"
    columns_processing: Dict[str, ColumnsProcessingData] = {}
    "Обработчики колонок"
    inputs: CreationInputsList = CreationInputsList()
    "`input`-слои"
    outputs: CreationOutputsList = CreationOutputsList()
    "`output`-слои"

    @property
    def path(self) -> Path:
        return Path(self.datasets_path, f"{self.alias}.{terra_ai_settings.DATASET_EXT}")

    @validator("inputs", "outputs")
    def _validate_required(cls, value: UniqueListMixin) -> UniqueListMixin:
        if not len(value):
            raise ListEmptyException(type(value))
        return value

    @validator("outputs")
    def _validate_outputs(cls, value: UniqueListMixin) -> UniqueListMixin:
        if not value:
            return value
        is_object_detection = False
        for layer in value:
            if layer.type == LayerOutputTypeChoice.ObjectDetection:
                is_object_detection = True
                break
        if is_object_detection and len(value) > 1:
            raise ObjectDetectionQuantityLayersException(f"{len(value)} output layers")
        return value
