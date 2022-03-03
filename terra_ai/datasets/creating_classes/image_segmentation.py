from terra_ai.datasets.creating_classes.base import BaseClass, PreprocessingNumericClass


class ImageSegmentationClass(PreprocessingNumericClass, BaseClass):

    @staticmethod
    def preprocess_version_data(**kwargs):

        version_data = kwargs['version_data']

        width, height = 0, 0
        for inp_data in version_data.inputs:
            if inp_data.type == 'handler' and inp_data.parameters.type == 'Image':
                width = inp_data.parameters.options.width
                height = inp_data.parameters.options.height
        for out_data in version_data.outputs:
            if out_data.type == 'handler' and out_data.parameters.type == 'Segmentation':
                out_data.parameters.options.width = width
                out_data.parameters.options.height = height

        return version_data
