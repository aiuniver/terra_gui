from terra_ai.datasets.creating_classes.base import BaseClass, MainTimeseriesClass, PreprocessingNumericClass


class TimeseriesTrendClass(MainTimeseriesClass, PreprocessingNumericClass, BaseClass):

    @staticmethod
    def preprocess_version_data(**kwargs):

        version_data = kwargs['version_data']

        length, step, depth = 0, 0, 1
        for out_data in version_data.outputs:
            if out_data.type == 'handler' and out_data.parameters.type == 'TimeseriesTrend':
                length = out_data.parameters.options.length
                step = out_data.parameters.options.step
        for inp_data in version_data.inputs:
            if inp_data.type == 'handler' and inp_data.parameters.type in ['Classification', 'Scaler', 'Raw']:
                inp_data.parameters.options.length = length
                inp_data.parameters.options.step = step
                inp_data.parameters.options.depth = depth

        return version_data
