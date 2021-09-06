import sys
import os
import numpy as np
import json
import tensorflow
from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.datasets.preparing import PrepareDTS
from terra_ai.datasets.data import Preprocesses
from terra_ai.data.datasets.dataset import DatasetData, DatasetLoadData
from terra_ai.data.datasets.creation import SourceData, CreationData, CreationInputData
from terra_ai.data.datasets.extra import SourceModeChoice, LayerInputTypeChoice, LayerOutputTypeChoice, \
    DatasetGroupChoice, LayerNetChoice, LayerVideoFillModeChoice, LayerVideoFrameModeChoice, LayerYoloChoice, \
    LayerTextModeChoice, LayerAudioModeChoice, LayerVideoModeChoice
from terra_ai.data.datasets.extra import LayerScalerImageChoice, LayerScalerAudioChoice, LayerPrepareMethodChoice, \
    LayerScalerVideoChoice
from terra_ai.datasets.creating import CreateDTS
from terra_ai.datasets import loading
from terra_ai import progress
from terra_ai.utils import decamelize


def _plot_mask_segmentation(predict, num_classes, classes_colors):
    """
    Returns:
        mask_images
    """

    def _index2color(pix, num_cls, cls_colors):
        index = np.argmax(pix)
        color = []
        for i in range(num_cls):
            if index == i:
                color = cls_colors[i]
        return color

    def _get_colored_mask(mask, num_cls, cls_colors):
        """
        Transforms prediction mask to colored mask

        Parameters:
        mask : numpy array                 segmentation mask

        Returns:
        colored_mask : numpy array         mask with colors by classes
        """

        colored_mask = []
        shape_mask = mask.shape
        print(shape_mask)
        mask = mask.reshape(-1, num_cls)
        print(mask.shape)
        for pix in range(len(mask)):
            colored_mask.append(
                _index2color(mask[pix], num_cls, cls_colors)
            )
        colored_mask = np.array(colored_mask).astype(np.uint8)
        print(colored_mask.shape)
        colored_mask = colored_mask.reshape(shape_mask)
        return colored_mask

    image = np.squeeze(_get_colored_mask(predict, num_classes, classes_colors))

    return image


pred = np.zeros(shape=(128, 128, 3))
print(pred.shape)
mask = _plot_mask_segmentation(pred, 2, ((0, 0, 0), (255, 0, 0)))
print(mask)