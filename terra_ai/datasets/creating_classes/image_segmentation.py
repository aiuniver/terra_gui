from terra_ai.datasets.creating_classes.base import BaseClass, PreprocessingScalerClass


class ImageSegmentationClass(PreprocessingScalerClass, BaseClass):

    @staticmethod
    def preprocess_version_data(**kwargs):

        version_data = kwargs['version_data']
        version_data.processing['1'].parameters.height = version_data.processing['0'].parameters.height
        version_data.processing['1'].parameters.width = version_data.processing['0'].parameters.width

        return version_data
