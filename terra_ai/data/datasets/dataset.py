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

from datetime import datetime
from typing import Optional

from ..mixins import AliasMixinData, UniqueListMixin
from ..extra import FileSizeData
from .tags import TagsList


class DatasetData(AliasMixinData):
    """
    Информация о датасете
    """

    name: str
    "Название"
    size: Optional[FileSizeData]
    "Вес"
    date: Optional[datetime]
    "Дата создания"
    tags: Optional[TagsList] = TagsList()
    "Список тегов"


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
