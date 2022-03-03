from terra_ai.datasets.creating_classes.base import BaseClass, PreprocessingTextClass


class TextSegmentationClass(PreprocessingTextClass, BaseClass):

    @staticmethod
    def preprocess_version_data(**kwargs):

        version_data = kwargs['version_data']

        filters, text_mode, prepare_method, length, step, max_words = '', '', '', 0, 0, 0
        for inp_data in version_data.inputs:
            if inp_data.type == 'handler' and inp_data.parameters.type == 'Text':
                filters = inp_data.parameters.options.filters
                text_mode = inp_data.parameters.options.text_mode
                prepare_method = inp_data.parameters.options.prepare_method
                length = inp_data.parameters.options.length
                step = inp_data.parameters.options.step
                max_words = inp_data.parameters.options.max_words
        for out_data in version_data.outputs:
            if out_data.type == 'handler' and out_data.parameters.type == 'TextSegmentation':
                for x in out_data.parameters.options.open_tags + out_data.parameters.options.close_tags:
                    filters = filters.replace(x, '')
                out_data.parameters.options.filters = filters
                out_data.parameters.options.text_mode = text_mode
                out_data.parameters.options.prepare_method = prepare_method
                out_data.parameters.options.length = length
                out_data.parameters.options.step = step
                out_data.parameters.options.max_words = max_words

        return version_data
