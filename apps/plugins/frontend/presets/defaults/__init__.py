from terra_ai.data.modeling.layers import Layer, types as layers_types
from terra_ai.data.modeling.extra import LayerTypeChoice
from terra_ai.data.training.extra import ArchitectureChoice

from apps.plugins.frontend.utils import prepare_pydantic_field
from apps.plugins.frontend.presets.defaults.datasets import (
    DatasetsColumnProcessing,
    DatasetsInput,
    DatasetsOutput,
)
from apps.plugins.frontend.presets.defaults.modeling import (
    ModelingLayerForm,
    ModelingLayersTypes,
)
from apps.plugins.frontend.presets.defaults.cascades import (
    CascadesBlockForm,
    CascadesBlocksTypes,
)
from apps.plugins.frontend.presets.defaults.deploy import (
    DeployTypeGroup,
    DeployServerGroup,
)


Defaults = {
    "datasets": {
        "creation": {
            "column_processing": DatasetsColumnProcessing,
            "input": DatasetsInput,
            "output": DatasetsOutput,
        },
        "architectures": [
            {"value": "ImageClassification", "label": "Классификация изображений"},
            {"value": "TextClassification", "label": "Классификация текстов"},
            {"value": "AudioClassification", "label": "Классификация аудио"},
            {"value": "VideoClassification", "label": "Классификация видео"},
            {
                "value": "DataframeClassification",
                "label": "Классификация табличных данных",
            },
            {"value": "DataframeRegression", "label": "Регрессия табличных данных"},
            {"value": "ImageSegmentation", "label": "Сегментация изображений"},
            {"value": "TextSegmentation", "label": "Сегментация текстов"},
            {"value": "Timeseries", "label": "Временные ряды"},
            {"value": "TimeseriesTrend", "label": "Тренд временного ряда"},
            {"value": "VideoTracker", "label": "Трекер для видео"},
            {"value": "TextTransformer", "label": "Текстовый трансформер"},
            {"value": "YoloV3", "label": "YoloV3"},
            {"value": "YoloV4", "label": "YoloV4"},
            {"value": "Text2Speech", "label": "Синтез речи (Text-to-Speech)"},
            {"value": "Speech2Text", "label": "Озвучка текста (Speech-to-Text)"},
            {
                "value": "ImageGAN",
                "label": "Генеративно-состязательные НС на изображениях",
            },
            {
                "value": "ImageCGAN",
                "label": "Генеративно-состязательные НС с условием на изображениях",
            },
            # {"value":"TextToImageGAN", "label": "TextToImageGAN"},
            # {"value":"ImageToImageGAN", "label": "ImageToImageGAN"},
            # {"value":"ImageSRGAN", "label": "ImageSRGAN"},
        ],
    },
    "modeling": {
        "layer_form": ModelingLayerForm,
        "layers_types": ModelingLayersTypes,
    },
    "training": {"architecture": ArchitectureChoice.Basic},
    "cascades": {
        "block_form": CascadesBlockForm,
        "blocks_types": CascadesBlocksTypes,
    },
    "deploy": {
        "type": DeployTypeGroup,
        "server": DeployServerGroup,
    },
}


def __get_group_type_params(data, group) -> list:
    output = []
    for name in data.__fields__:
        output.append(
            prepare_pydantic_field(
                data.__fields__[name], f"parameters[{group}][{name}]"
            )
        )
    return output


for layer in Layer:
    params = getattr(layers_types, layer.name)
    Defaults["modeling"]["layers_types"].update(
        {
            LayerTypeChoice[layer.name].value: {
                "main": __get_group_type_params(params.ParametersMainData, "main"),
                "extra": __get_group_type_params(params.ParametersExtraData, "extra"),
            }
        }
    )
