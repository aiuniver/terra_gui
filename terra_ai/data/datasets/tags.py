"""
## Структура данных тегов

### **Тег**
```
In [1]: from terra_ai.data.datasets.tags import TagData

In [2]: source = {
   ...:     "alias": "tensorflow_keras",
   ...:     "name": "Tensorflow.keras",
   ...: }

In [3]: data = TagData(**source)

In [4]: data
Out[4]: TagData(alias='tensorflow_keras', name='Tensorflow.keras')

In [5]: data.dict()
Out[5]: {'alias': 'tensorflow_keras', 'name': 'Tensorflow.keras'}

In [6]: data.json()
Out[6]: '{"alias": "tensorflow_keras", "name": "Tensorflow.keras"}'

In [7]: print(data.json(indent=2, ensure_ascii=False))
{
  "alias": "tensorflow_keras",
  "name": "Tensorflow.keras"
}
```

### **Список тегов**
```
In [1]: from terra_ai.data.datasets.tags import TagsList

In [2]: source = [
   ...:     {
   ...:         "alias": "tensorflow_keras",
   ...:         "name": "Tensorflow.keras",
   ...:     },
   ...:     {
   ...:         "alias": "object_detection",
   ...:         "name": "Object detection",
   ...:     },
   ...:     {
   ...:         "alias": "segmentation",
   ...:         "name": "Segmentation",
   ...:     },
   ...: ]

In [3]: data = TagsList(source)

In [4]: data
Out[4]:
[TagData(alias='tensorflow_keras', name='Tensorflow.keras'),
 TagData(alias='object_detection', name='Object detection'),
 TagData(alias='segmentation', name='Segmentation')]

In [5]: data.dict()
Out[5]:
[{'alias': 'tensorflow_keras', 'name': 'Tensorflow.keras'},
 {'alias': 'object_detection', 'name': 'Object detection'},
 {'alias': 'segmentation', 'name': 'Segmentation'}]

In [6]: data.json()
Out[6]: '[{"alias": "tensorflow_keras", "name": "Tensorflow.keras"}, {"alias": "object_detection", "name": "Object detection"}, {"alias": "segmentation", "name": "Segmentation"}]'

In [7]: print(data.json(indent=2, ensure_ascii=False))
[
  {
    "alias": "tensorflow_keras",
    "name": "Tensorflow.keras"
  },
  {
    "alias": "object_detection",
    "name": "Object detection"
  },
  {
    "alias": "segmentation",
    "name": "Segmentation"
  }
]
```
"""

from ..mixins import AliasMixinData, UniqueListMixin


class TagData(AliasMixinData):
    """
    Информация о теге
    """

    name: str
    "Название"


class TagsList(UniqueListMixin):
    """
    Список тегов, основанных на `TagData`
    ```
    class Meta:
        source = TagData
        identifier = "alias"
    ```
    """

    class Meta:
        source = TagData
        identifier = "alias"
