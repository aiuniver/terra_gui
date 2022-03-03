from terra_ai.datasets.creating_classes.base import BaseClass, PreprocessingNumericClass, ClassificationClass


class VideoTrackerClass(ClassificationClass, PreprocessingNumericClass, BaseClass):

    def create_arrays(self, instructions, version_paths_data, dataframe, preprocessing):

        pass
