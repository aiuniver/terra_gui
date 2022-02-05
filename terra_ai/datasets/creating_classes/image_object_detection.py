import concurrent.futures
import os
from itertools import repeat
from math import ceil
from pathlib import Path

from terra_ai import progress
from terra_ai.data.datasets.dataset import VersionPathsData
from terra_ai.data.datasets.extra import LayerODDatasetTypeChoice
from terra_ai.datasets.arrays_create import CreateArray
from terra_ai.datasets.creating import version_progress_name
from terra_ai.datasets.data import InstructionsData
from terra_ai.datasets.utils import PATH_TYPE_LIST, get_od_names
from terra_ai.logging import logger
from terra_ai.utils import decamelize
from terra_ai.datasets.creating_classes.base import BaseClass


class ImageObjectDetectionClass(BaseClass):

    def preprocess_version_data(self, **kwargs):
        version_data = kwargs['version_data']
        version_path_data = kwargs['version_path_data']
        version_data.processing['2'].parameters.frame_mode = version_data.processing['1'].parameters.image_mode
        names_list = get_od_names(version_data, version_path_data)
        version_data.processing['2'].parameters.classes_names = names_list
        version_data.processing['2'].parameters.num_classes = len(names_list)

        return version_data
