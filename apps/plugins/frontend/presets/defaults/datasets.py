from apps.plugins.frontend.presets.defaults import fields

from terra_ai.data.datasets.extra import LayerHandlerChoice


BlockHandlerParameters = {
    LayerHandlerChoice.Audio: [
        fields.SampleRateField,
        fields.AudioModeField,
        fields.AudioFillModeField,
        fields.AudioParameterField,
        fields.AudioResampleField,
        fields.ScalerDefaultField,
    ],
    LayerHandlerChoice.Classification: [
        fields.OneHotEncodingField,
        fields.TypeProcessingField,
    ],
    LayerHandlerChoice.Discriminator: [],
    LayerHandlerChoice.Generator: [],
    LayerHandlerChoice.Image: [
        fields.WidthField,
        fields.HeightField,
        fields.NetField,
        fields.ScalerImageField,
        fields.ImageModeField,
    ],
    LayerHandlerChoice.ImageCGAN: [],
    LayerHandlerChoice.ImageGAN: [],
    LayerHandlerChoice.Noise: [],
    LayerHandlerChoice.YoloV3: [
        fields.ODModelTypeField,
    ],
    LayerHandlerChoice.YoloV4: [
        fields.ODModelTypeField,
    ],
    LayerHandlerChoice.Regression: [
        fields.ScalerDefaultField,
    ],
    LayerHandlerChoice.Scaler: [
        fields.ScalerDefaultField,
    ],
    LayerHandlerChoice.Segmentation: [
        fields.MaskRangeField,
        fields.ClassesField,
    ],
    LayerHandlerChoice.Speech2Text: [],
    LayerHandlerChoice.Text: [
        fields.TextModeField,
        fields.TextPrepareMethodField,
        fields.TextPymorphyField,
        fields.TextFiltersField,
    ],
    LayerHandlerChoice.TextTransformer: [],
    LayerHandlerChoice.Text2Speech: [],
    LayerHandlerChoice.TextSegmentation: [
        fields.TextOpenTagsField,
        fields.TextCloseTagsField,
    ],
    LayerHandlerChoice.TextToImageGAN: [],
    LayerHandlerChoice.Timeseries: [
        fields.TimeseriesLengthField,
        fields.TimeseriesStepField,
        fields.TimeseriesTrendField,
    ],
    LayerHandlerChoice.TimeseriesTrend: [
        fields.TimeseriesLengthField,
        fields.TimeseriesStepField,
        fields.TimeseriesTrendLimitField,
        fields.ScalerDefaultField,
    ],
    LayerHandlerChoice.Tracker: [],
    LayerHandlerChoice.Transformer: [],
    LayerHandlerChoice.Video: [
        fields.VideoWidthField,
        fields.VideoHeightField,
        fields.VideoFillModeField,
        fields.VideoFrameModeField,
        fields.VideoModeField,
        fields.ScalerVideoField,
    ],
}


BlockDataForm = []
BlockHandlerForm = {
    "inputs": [
        {
            "type": "select",
            "label": "Тип обработчика",
            "name": "type",
            "parse": "type",
            "value": LayerHandlerChoice.Image.name,
            "list": list(
                map(
                    lambda item: {"value": item.name, "label": item.value},
                    LayerHandlerChoice.inputs(),
                )
            ),
            "fields": dict(
                map(
                    lambda item: (item, BlockHandlerParameters.get(item)),
                    LayerHandlerChoice.inputs(),
                )
            ),
        }
    ],
    "outputs": [
        {
            "type": "select",
            "label": "Тип обработчика",
            "name": "type",
            "parse": "type",
            "value": LayerHandlerChoice.Classification.name,
            "list": list(
                map(
                    lambda item: {"value": item.name, "label": item.value},
                    LayerHandlerChoice.outputs(),
                )
            ),
            "fields": dict(
                map(
                    lambda item: (item, BlockHandlerParameters.get(item)),
                    LayerHandlerChoice.outputs(),
                )
            ),
        }
    ],
}
BlockInputForm = []
BlockOutputForm = []
