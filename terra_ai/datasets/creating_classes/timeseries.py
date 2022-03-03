from terra_ai.datasets.creating_classes.base import BaseClass, MainTimeseriesClass, PreprocessingNumericClass


class TimeseriesClass(MainTimeseriesClass, PreprocessingNumericClass, BaseClass):

    @staticmethod
    def preprocess_version_data(**kwargs):

        version_data = kwargs['version_data']

        length, depth, step = 0, 0, 0
        for out_data in version_data.outputs:
            if out_data.type == 'handler' and out_data.parameters.type == 'Timeseries':
                # if out_data.parameters.options.trend:
                #     out_data.parameters.options.depth = 1
                length = out_data.parameters.options.length
                depth = out_data.parameters.options.depth
                step = out_data.parameters.options.step
        for inp_data in version_data.inputs:
            if inp_data.type == 'handler' and inp_data.parameters.type in ['Classification', 'Scaler', 'Raw']:
                inp_data.parameters.options.length = length
                inp_data.parameters.options.depth = depth
                inp_data.parameters.options.step = step

        return version_data
