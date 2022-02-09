from terra_ai.datasets.creating_classes.base import BaseClass, PreprocessingNumericClass


class ImageGANClass(PreprocessingNumericClass, BaseClass):

    def create_arrays(self, instructions, version_paths_data, dataframe, preprocessing):

        self.create_put_arrays(
            put_data=instructions.inputs,
            version_paths_data=version_paths_data,
            dataframe=dataframe,
            preprocessing=preprocessing
        )
