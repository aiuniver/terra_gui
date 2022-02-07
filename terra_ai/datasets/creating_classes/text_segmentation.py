from terra_ai import progress
from terra_ai.data.datasets.extra import LayerPrepareMethodChoice
from terra_ai.datasets.creating import version_progress_name
from terra_ai.datasets.creating_classes.base import BaseClass
from terra_ai.utils import camelize


class TextSegmentationClass(BaseClass):

    def preprocess_version_data(self, **kwargs):

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

    @staticmethod
    def create_preprocessing(instructions, preprocessing):

        for put in list(instructions.inputs.values()) + list(instructions.outputs.values()):
            for col_name, data in put.items():
                if data.parameters['prepare_method'] in [LayerPrepareMethodChoice.embedding,
                                                         LayerPrepareMethodChoice.bag_of_words]:
                    preprocessing.create_tokenizer(text_list=data.instructions, **data.parameters)
                elif data.parameters['prepare_method'] == LayerPrepareMethodChoice.word_to_vec:
                    preprocessing.create_word2vec(text_list=data.instructions, **data.parameters)

        return preprocessing

    # @staticmethod
    # def fit_preprocessing(put_data, preprocessing):

        # Из-за невозможности создания Word2Vec без сразу передачи в него корпусов текста, обучение текстовых
        # препроцессингов происходит сразу на этапе создания

        # for key in put_data.keys():
        #     for col_name, data in put_data[key].items():
        #         progress.pool(version_progress_name, message=f'Обучение {camelize(data.parameters["scaler"])}')
        #         preprocessing.preprocessing[key][col_name].fit_on_texts(data.instructions)
        # pass
