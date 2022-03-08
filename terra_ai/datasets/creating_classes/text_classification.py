from terra_ai.datasets.creating_classes.base import BaseClass, ClassificationClass, PreprocessingTextClass


class TextClassificationClass(ClassificationClass, PreprocessingTextClass, BaseClass):

    @staticmethod
    def preprocess_version_data(**kwargs):

        version_data = kwargs['version_data']

        filters, tok_length = '', 0
        for inp_data in version_data.inputs:
            if inp_data.type == 'handler' and inp_data.parameters.type == 'Text':
                filters = inp_data.parameters.options.filters
                if inp_data.parameters.options.text_mode == 'completely':
                    tok_length = inp_data.parameters.options.max_words
                elif inp_data.parameters.options.text_mode == 'length_and_step':
                    tok_length = inp_data.parameters.options.length
        for inp_data in version_data.inputs:
            if inp_data.type == 'preprocess' and inp_data.parameters.type == 'Tokenizer':
                inp_data.parameters.options.filters = filters
                inp_data.parameters.options.length = tok_length

        return version_data
