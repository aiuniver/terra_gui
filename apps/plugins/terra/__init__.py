from terra_ai.data.dataset import DatasetData, DatasetsList, Project
from .exchange import TerraExchange

terra_exchange = TerraExchange()

if terra_exchange.project.dataset:
    dataset_name = terra_exchange.project.dataset
    is_custom = False
    layers = terra_exchange.project.dict().get("layers")
    terra_exchange.call(
        "prepare_dataset",
        dataset=terra_exchange.project.dataset,
        is_custom=is_custom,
        not_load_layers=False,
    )


data = [
    {
        "alias": "mnist",
        "name": "Mnist",
        "size": {"value": "2423.32", "unit": "Мб"},
        "date": "2021-06-20T14:52:16.262838",
        "tags": [
            {"alias": "classification", "name": "Classification"},
            {"alias": "tensorflow_keras", "name": "Tensorflow.Keras"},
            {"alias": "timeseries", "name": "Timeseries"},
            {"alias": "object_detection", "name": "Object detection"},
        ],
    },
    {"alias": "cifar10", "name": "Cifar10"},
    {"alias": "lips", "name": "Губы"},
    {"alias": "disease", "name": "Заболевания"},
]
data_one = {"alias": "contracts", "name": "Договоры"}
data_two = {"alias": "apartments", "name": "Квартиры"}
print("--------------------------")
print("Один датасет")
print(DatasetData(**data[0]))
print("--------------------------")
print("Список датасетов")
dd = DatasetsList(data)
print(dd)
print("--------------------------")
print("Добавили в список")
dd.append(data_one)
dd.insert(2, data_two)
print(dd)
print("--------------------------")
print("Ключи датасетов")
print(dd.ids)
print("--------------------------")
print("Получаем по ключу датасет")
print(dd.get("mnist"))
print("--------------------------")
print("Проект")
p = Project(datasets=data)
print(p)
print("--------------------------")
print("Проект as dict")
print(p.dict())
print("--------------------------")
print("Проект as json")
print(p.json())
print("--------------------------")
