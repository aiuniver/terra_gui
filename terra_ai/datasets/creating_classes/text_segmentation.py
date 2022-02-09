from terra_ai.datasets.creating_classes.base import BaseClass, PreprocessingTextClass


class TextSegmentationClass(PreprocessingTextClass, BaseClass):

    @staticmethod
    def preprocess_version_data(**kwargs):

        version_data = kwargs['version_data']

        version_data.processing['1'].parameters.text_mode = version_data.processing['0'].parameters.text_mode
        version_data.processing['1'].parameters.length = version_data.processing['0'].parameters.length
        version_data.processing['1'].parameters.step = version_data.processing['0'].parameters.step
        version_data.processing['1'].parameters.max_words = version_data.processing['0'].parameters.max_words
        filters = version_data.processing['0'].parameters.filters
        for x in version_data.processing['1'].parameters.open_tags + version_data.processing['1'].parameters.close_tags:
            filters = filters.replace(x, '')
        version_data.processing['1'].parameters.filters = filters
        version_data.processing['0'].parameters.filters = filters
        version_data.processing['0'].parameters.open_tags = version_data.processing['1'].parameters.open_tags
        version_data.processing['0'].parameters.close_tags = version_data.processing['1'].parameters.close_tags

        return version_data
