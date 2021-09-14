"""
## Структура данных датасетов

### **Датасет**
```
In [1]: from terra_ai.data.datasets.dataset import DatasetData

In [2]: source = {
   ...:     "alias": "cars",
   ...:     "name": "Cars",
   ...:     "date": "2021-06-26T09:30:24+00:00",
   ...:     "size": {
   ...:         "value": "324133875",
   ...:     },
   ...:     "tags": [
   ...:         {
   ...:             "alias": "tensorflow_keras",
   ...:             "name": "Tensorflow.keras",
   ...:         },
   ...:         {
   ...:             "alias": "segmentation",
   ...:             "name": "Segmentation",
   ...:         },
   ...:     ],
   ...: }

In [3]: data = DatasetData(**source)

In [4]: data
Out[4]: DatasetData(alias='cars', name='Cars', size=FileFileSizeData(value=324133875, short=309.1181516647339, unit='Мб'), date=datetime.datetime(2021, 6, 26, 9, 30, 24, tzinfo=datetime.timezone.utc), tags=[TagData(alias='tensorflow_keras', name='Tensorflow.keras'), TagData(alias='segmentation', name='Segmentation')])

In [5]: data.dict()
Out[5]:
{'alias': 'cars',
 'name': 'Cars',
 'size': {'value': 324133875, 'short': 309.1181516647339, 'unit': 'Мб'},
 'date': datetime.datetime(2021, 6, 26, 9, 30, 24, tzinfo=datetime.timezone.utc),
 'tags': [{'alias': 'tensorflow_keras', 'name': 'Tensorflow.keras'},
  {'alias': 'segmentation', 'name': 'Segmentation'}]}

In [6]: data.json()
Out[6]: '{"alias": "cars", "name": "Cars", "size": {"value": 324133875, "short": 309.1181516647339, "unit": "\\u041c\\u0431"}, "date": "2021-06-26T09:30:24+00:00", "tags": [{"alias": "tensorflow_keras", "name": "Tensorflow.keras"}, {"alias": "segmentation", "name": "Segmentation"}]}'

In [7]: print(data.json(indent=2, ensure_ascii=False))
{
  "alias": "cars",
  "name": "Cars",
  "size": {
    "value": 324133875,
    "short": 309.1181516647339,
    "unit": "Мб"
  },
  "date": "2021-06-26T09:30:24+00:00",
  "tags": [
    {
      "alias": "tensorflow_keras",
      "name": "Tensorflow.keras"
    },
    {
      "alias": "segmentation",
      "name": "Segmentation"
    }
  ]
}
```

### **Список датасетов**
```
In [1]: from terra_ai.data.datasets.dataset import DatasetsList

In [2]: source = {
   ...:     "alias": "cars",
   ...:     "name": "Cars",
   ...:     "date": "2021-06-26T09:30:24+00:00",
   ...:     "size": {
   ...:         "value": "324133875",
   ...:     },
   ...:     "tags": [
   ...:         {
   ...:             "alias": "tensorflow_keras",
   ...:             "name": "Tensorflow.keras",
   ...:         },
   ...:         {
   ...:             "alias": "segmentation",
   ...:             "name": "Segmentation",
   ...:         },
   ...:     ],
   ...:     "alias": "kvartiri",
   ...:     "name": "Квартиры",
   ...:     "date": "2020-12-09T15:34:03+00:00",
   ...:     "size": {
   ...:         "value": "32241733875",
   ...:     },
   ...:     "tags": [
   ...:         {
   ...:             "alias": "tensorflow_keras",
   ...:             "name": "Tensorflow.keras",
   ...:         },
   ...:         {
   ...:             "alias": "segmentation",
   ...:             "name": "Segmentation",
   ...:         },
   ...:     ],
   ...: }

In [3]: data = DatasetsList(source)

In [4]: data
Out[4]:
[DatasetData(alias='cars', name='Cars', size=FileFileSizeData(value=324133875, short=309.1181516647339, unit='Мб'), date=datetime.datetime(2021, 6, 26, 9, 30, 24, tzinfo=datetime.timezone.utc), tags=[TagData(alias='tensorflow_keras', name='Tensorflow.keras'), TagData(alias='segmentation', name='Segmentation')]),
 DatasetData(alias='kvartiri', name='Квартиры', size=FileSizeData(value=32241733875, short=30.02745460253209, unit='Гб'), date=datetime.datetime(2020, 12, 9, 15, 34, 3, tzinfo=datetime.timezone.utc), tags=[TagData(alias='tensorflow_keras', name='Tensorflow.keras'), TagData(alias='segmentation', name='Segmentation')])]

In [5]: data.dict()
Out[5]:
[{'alias': 'cars',
  'name': 'Cars',
  'size': {'value': 324133875, 'short': 309.1181516647339, 'unit': 'Мб'},
  'date': datetime.datetime(2021, 6, 26, 9, 30, 24, tzinfo=datetime.timezone.utc),
  'tags': [{'alias': 'tensorflow_keras', 'name': 'Tensorflow.keras'},
   {'alias': 'segmentation', 'name': 'Segmentation'}]},
 {'alias': 'kvartiri',
  'name': 'Квартиры',
  'size': {'value': 32241733875, 'short': 30.02745460253209, 'unit': 'Гб'},
  'date': datetime.datetime(2020, 12, 9, 15, 34, 3, tzinfo=datetime.timezone.utc),
  'tags': [{'alias': 'tensorflow_keras', 'name': 'Tensorflow.keras'},
   {'alias': 'segmentation', 'name': 'Segmentation'}]}]

In [6]: data.json()
Out[6]: '[{"alias": "cars", "name": "Cars", "size": {"value": 324133875, "short": 309.1181516647339, "unit": "\\u041c\\u0431"}, "date": "2021-06-26T09:30:24+00:00", "tags": [{"alias": "tensorflow_keras", "name": "Tensorflow.keras"}, {"alias": "segmentation", "name": "Segmentation"}]}, {"alias": "kvartiri", "name": "\\u041a\\u0432\\u0430\\u0440\\u0442\\u0438\\u0440\\u044b", "size": {"value": 32241733875, "short": 30.02745460253209, "unit": "\\u0413\\u0431"}, "date": "2020-12-09T15:34:03+00:00", "tags": [{"alias": "tensorflow_keras", "name": "Tensorflow.keras"}, {"alias": "segmentation", "name": "Segmentation"}]}]'

In [7]: print(data.json(indent=2, ensure_ascii=False))
[
  {
    "alias": "cars",
    "name": "Cars",
    "size": {
      "value": 324133875,
      "short": 309.1181516647339,
      "unit": "Мб"
    },
    "date": "2021-06-26T09:30:24+00:00",
    "tags": [
      {
        "alias": "tensorflow_keras",
        "name": "Tensorflow.keras"
      },
      {
        "alias": "segmentation",
        "name": "Segmentation"
      }
    ]
  },
  {
    "alias": "kvartiri",
    "name": "Квартиры",
    "size": {
      "value": 32241733875,
      "short": 30.02745460253209,
      "unit": "Гб"
    },
    "date": "2020-12-09T15:34:03+00:00",
    "tags": [
      {
        "alias": "tensorflow_keras",
        "name": "Tensorflow.keras"
      },
      {
        "alias": "segmentation",
        "name": "Segmentation"
      }
    ]
  }
]
```
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from pydantic import validator, DirectoryPath
from pydantic.types import PositiveInt
from pydantic.color import Color

from .tags import TagsList
from .extra import DatasetGroupChoice, LayerInputTypeChoice, LayerOutputTypeChoice
from ..mixins import AliasMixinData, UniqueListMixin, BaseMixinData
from ..extra import FileSizeData
from ..exceptions import TrdsDirExtException, TrdsConfigFileNotFoundException
from ..training.extra import TaskChoice
from ..modeling.model import ModelDetailsData
from ..modeling.extra import LayerTypeChoice, LayerGroupChoice
from ..modeling.layers.extra import ActivationChoice
from ..presets.models import EmptyModelDetailsData
from ..presets.datasets import OutputLayersDefaults
from ... import settings


class DatasetLoadData(BaseMixinData):
    path: DirectoryPath
    group: DatasetGroupChoice
    alias: str


class CustomDatasetConfigData(BaseMixinData):
    """
    Загрузка конфигурации пользовательского датасета
    """

    path: DirectoryPath
    config: Optional[dict] = {}

    @validator("path")
    def _validate_path(cls, value: DirectoryPath) -> DirectoryPath:
        if not str(value).endswith(f".{settings.DATASET_EXT}"):
            raise TrdsDirExtException(value.name)
        return value

    @validator("config", always=True)
    def _validate_config(cls, value: dict, values) -> dict:
        config_path = Path(values.get("path"), settings.DATASET_CONFIG)
        if not config_path.is_file():
            raise TrdsConfigFileNotFoundException(
                values.get("path").name, config_path.name
            )
        with open(config_path, "r") as config_ref:
            value = json.load(config_ref)
        value.update({"group": DatasetGroupChoice.custom})
        return value


class DatasetLayerData(BaseMixinData):
    datatype: str
    dtype: str
    shape: Tuple[PositiveInt, ...]
    name: str


class DatasetPathsData(BaseMixinData):
    datasets: Optional[DirectoryPath]
    # arrays: str  # DirectoryPath
    # instructions: str  # DirectoryPath
    # dataset_sources: str  # DirectoryPath
    tmp_sources: Optional[DirectoryPath]


class DatasetInputsData(DatasetLayerData):
    task: LayerInputTypeChoice


class DatasetOutputsData(DatasetLayerData):
    task: LayerOutputTypeChoice


class DatasetData(AliasMixinData):
    """
    Информация о датасете
    """

    name: str
    date: Optional[datetime]
    size: Optional[FileSizeData]
    group: Optional[DatasetGroupChoice]
    use_generator: bool = False
    tags: Optional[TagsList] = TagsList()
    num_classes: Dict[PositiveInt, PositiveInt] = {}
    classes_names: Dict[PositiveInt, List[str]] = {}
    classes_colors: Dict[PositiveInt, List[Color]] = {}
    encoding: Dict[PositiveInt, str] = {}
    task_type: Dict[int, TaskChoice] = {}
    inputs: Dict[PositiveInt, DatasetInputsData] = {}
    outputs: Dict[PositiveInt, DatasetOutputsData] = {}
    paths: Optional[DatasetPathsData]

    @property
    def model(self) -> ModelDetailsData:
        data = {**EmptyModelDetailsData}
        data.update({"alias": self.alias, "name": self.name})
        layers = []
        for _id, layer in self.inputs.items():
            layers.append(
                {
                    "id": _id,
                    "name": layer.name,
                    "type": LayerTypeChoice.Input,
                    "group": LayerGroupChoice.input,
                    "shape": {"input": [layer.shape]},
                    "task": layer.task,
                    "num_classes": self.num_classes.get(_id),
                }
            )
        for _id, layer in self.outputs.items():
            output_layer_defaults = OutputLayersDefaults.get(layer.task, {}).get(
                layer.datatype, {}
            )
            activation = output_layer_defaults.get("activation", ActivationChoice.relu)
            units = self.num_classes.get(_id, layer.shape[-1] if layer.shape else 1)
            layers.append(
                {
                    "id": _id,
                    "name": layer.name,
                    "type": output_layer_defaults.get("type", LayerTypeChoice.Dense),
                    "group": LayerGroupChoice.output,
                    "shape": {"output": [layer.shape]},
                    "task": layer.task,
                    "num_classes": self.num_classes.get(_id, None),
                    "parameters": {
                        "main": {
                            "activation": activation,
                            "units": units,
                            "filters": units,
                        },
                        "extra": {
                            "activation": activation,
                            "units": units,
                            "filters": units,
                        },
                    },
                }
            )
        data.update({"layers": layers})
        return ModelDetailsData(**data)


class DatasetsList(UniqueListMixin):
    """
    Список датасетов, основанных на `DatasetData`
    ```
    class Meta:
        source = DatasetData
        identifier = "alias"
    ```
    """

    class Meta:
        source = DatasetData
        identifier = "alias"


class DatasetsGroupData(AliasMixinData):
    """
    Группа датасетов
    """

    name: str
    datasets: DatasetsList = DatasetsList()

    @property
    def tags(self) -> TagsList:
        __tags = TagsList()
        for item in self.datasets:
            __tags += item.tags
        return __tags

    def dict(self, **kwargs):
        data = super().dict()
        data.update({"tags": self.tags.dict()})
        return data


class DatasetsGroupsList(UniqueListMixin):
    """
    Список групп датасетов, основанных на `DatasetsGroupData`
    ```
    class Meta:
        source = DatasetsGroupData
        identifier = "alias"
    ```
    """

    class Meta:
        source = DatasetsGroupData
        identifier = "alias"
