from terra_ai.datasets.creating_classes.base import BaseClass, PreprocessingTextClass, ClassificationClass


class Text2SpeechClass(ClassificationClass, PreprocessingTextClass, BaseClass):

    def create_arrays(self, instructions, version_paths_data, dataframe, preprocessing):

        pass
