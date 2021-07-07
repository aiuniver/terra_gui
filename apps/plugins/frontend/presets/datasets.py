from ..extra import LayerInputTypeChoice, LayerOutputTypeChoice

from ..datasets import DatasetLayersParameters


dataset_layers_parameters = DatasetLayersParameters(
    input={
        LayerInputTypeChoice.Images: [],
        LayerInputTypeChoice.Text: [],
        LayerInputTypeChoice.Audio: [],
        LayerInputTypeChoice.Dataframe: [],
    },
    output={
        LayerOutputTypeChoice.Images: [],
        LayerOutputTypeChoice.Text: [],
        LayerOutputTypeChoice.Audio: [],
        LayerOutputTypeChoice.Classification: [],
        LayerOutputTypeChoice.Segmentation: [],
        LayerOutputTypeChoice.TextSegmentation: [],
        LayerOutputTypeChoice.Regression: [],
        LayerOutputTypeChoice.Timeseries: [],
    },
)
