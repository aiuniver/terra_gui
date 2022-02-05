from terra_ai.datasets.creating_classes.base import BaseClass


class ImageSegmentationClass(BaseClass):

    def preprocess_version_data(self, **kwargs):

        version_data = kwargs['version_data']
        version_data.processing['1'].parameters.height = version_data.processing['0'].parameters.height
        version_data.processing['1'].parameters.width = version_data.processing['0'].parameters.width

        return version_data
