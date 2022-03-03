from terra_ai.datasets.creating_classes.base import BaseClass, PreprocessingNumericClass, ClassificationClass


class ImageCGANClass(ClassificationClass, PreprocessingNumericClass, BaseClass):

    @staticmethod
    def preprocess_version_data(**kwargs):

        version_data = kwargs['version_data']

        image_shape = ()
        for inp_data in version_data.inputs:
            if inp_data.type == 'handler' and inp_data.parameters.type == 'Image':
                image_shape = (inp_data.parameters.options.height, inp_data.parameters.options.width, 3)
        for out_data in version_data.outputs:
            if out_data.type == 'handler' and out_data.parameters.type == 'Generator':
                out_data.parameters.options.shape = image_shape

        return version_data

    def create_arrays(self, instructions, version_paths_data, dataframe, preprocessing):

        self.create_put_arrays(
            put_data=instructions.inputs,
            version_paths_data=version_paths_data,
            dataframe=dataframe,
            preprocessing=preprocessing
        )
