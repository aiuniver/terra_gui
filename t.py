#!/usr/bin/env python


from terra_ai.data.datasets.dataset import DatasetData
from terra_ai.data.datasets import creations, dataset, tags, creation, extra
from terra_ai.data.datasets.creations import layers
from terra_ai.data.datasets.creations.layers import input, output
from terra_ai.data.datasets.creations.layers.output import types
from terra_ai.data.datasets.creations.layers.output.types import Image, Text, Audio, Classification, Segmentation, \
    TextSegmentation, ObjectDetection, Regression, Timeseries
from terra_ai.data.datasets.extra import LayerScalerRegressionChoice
from terra_ai.data.presets.datasets import Tags

data = DatasetData(
    **{
        "alias": "flats",
                "name": "Квартиры",
                "group": "terra",
                "tags": [
                    Tags.text,
                    Tags.regression,
                    Tags.russian,
                    Tags.terra_ai,
                ],
        "num_classes": {2: 100},
        "classes_names": {2: ['rsgr']},
        "encoding": {2: 'ohe'},
        "task_type": {2: "Classification"},
        "inputs": {
            1: {
                "datatype": "3D",
                "dtype": "float32",
                "shape": (32,32,3),
                "name": '',
                "task": "Image",
            },
        },
        "outputs": {
            2: {
                "datatype": "DIM",
                "dtype": "uint8",
                "shape": (100,),
                "name": "Выход 1",
                "task": "Classification",
            },
        },
    }
)
print(data.json(indent=2, ensure_ascii=False))

