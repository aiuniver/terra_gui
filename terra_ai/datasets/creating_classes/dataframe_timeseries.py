from terra_ai.datasets.creating_classes.base import BaseClass, PreprocessingNumericClass, PreprocessingTextClass


class DataframeTimeseriesClass(PreprocessingNumericClass, PreprocessingTextClass, BaseClass):

    @staticmethod
    def preprocess_version_data(**kwargs):

        version_data = kwargs['version_data']

        worker_keys = list(version_data.processing.keys())
        for worker_name in worker_keys:
            if version_data.processing[worker_name].type == 'Timeseries':
                # if version_data.columns_processing[worker_name].parameters.trend:
                #     version_data.columns_processing[worker_name].parameters.depth = 1
                # В скейлер перекидываем инфу о length depth step
                for w_name, w_params in version_data.processing.items():
                    if version_data.processing[w_name].type in ['Classification', 'Scaler']:
                        version_data.processing[w_name].parameters.length = \
                            version_data.processing[worker_name].parameters.length
                        version_data.processing[w_name].parameters.depth = \
                            version_data.processing[worker_name].parameters.depth
                        version_data.processing[w_name].parameters.step = \
                            version_data.processing[worker_name].parameters.step

        return version_data
