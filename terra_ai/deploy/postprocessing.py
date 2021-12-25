import os

from terra_ai.callbacks.classification_callbacks import ImageClassificationCallback, TextClassificationCallback, \
    DataframeClassificationCallback, AudioClassificationCallback, VideoClassificationCallback, TimeseriesTrendCallback
from terra_ai.callbacks.object_detection_callbacks import YoloV3Callback, YoloV4Callback
from terra_ai.callbacks.regression_callbacks import DataframeRegressionCallback
from terra_ai.callbacks.segmentation_callbacks import ImageSegmentationCallback, TextSegmentationCallback
from terra_ai.callbacks.time_series_callbacks import TimeseriesCallback
from terra_ai.data.training.extra import ArchitectureChoice
from terra_ai.exceptions.deploy import PostprocessException
from terra_ai.logging import logger


def postprocess_results(array, options, save_path: str = "", dataset_path: str = "", sensitivity=0.15,
                        threashold=0.1) -> dict:
    method_name = "postprocess_results"
    logger.info("Проводим постобработку результатов предсказания.", extra={"type": "info"})
    try:
        return_data = {}
        if options.data.architecture == ArchitectureChoice.ImageClassification:
            return_data = ImageClassificationCallback.postprocess_deploy(
                array=array, options=options, save_path=save_path, dataset_path=dataset_path
            )
        elif options.data.architecture == ArchitectureChoice.TextClassification:
            return_data = TextClassificationCallback.postprocess_deploy(
                array=array, options=options
            )
        elif options.data.architecture == ArchitectureChoice.DataframeClassification:
            return_data = DataframeClassificationCallback.postprocess_deploy(
                array=array, options=options
            )
        elif options.data.architecture == ArchitectureChoice.AudioClassification:
            return_data = AudioClassificationCallback.postprocess_deploy(
                array=array, options=options, save_path=save_path, dataset_path=dataset_path
            )
        elif options.data.architecture == ArchitectureChoice.VideoClassification:
            return_data = VideoClassificationCallback.postprocess_deploy(
                array=array, options=options, save_path=save_path, dataset_path=dataset_path
            )
        elif options.data.architecture == ArchitectureChoice.TimeseriesTrend:
            return_data = TimeseriesTrendCallback.postprocess_deploy(
                array=array, options=options
            )
        elif options.data.architecture == ArchitectureChoice.ImageSegmentation:
            return_data = ImageSegmentationCallback.postprocess_deploy(
                array=array, options=options, save_path=save_path, dataset_path=dataset_path
            )
        elif options.data.architecture == ArchitectureChoice.TextSegmentation:
            return_data = TextSegmentationCallback.postprocess_deploy(
                array=array, options=options
            )
        elif options.data.architecture == ArchitectureChoice.DataframeRegression:
            return_data = DataframeRegressionCallback.postprocess_deploy(
                array=array, options=options
            )
        elif options.data.architecture == ArchitectureChoice.Timeseries:
            return_data = TimeseriesCallback.postprocess_deploy(
                array=array, options=options
            )
        elif options.data.architecture == ArchitectureChoice.YoloV3:
            return_data = YoloV3Callback.postprocess_deploy(
                array=array, options=options, save_path=save_path, dataset_path=dataset_path,
                sensitivity=sensitivity, threashold=threashold
            )
        elif options.data.architecture == ArchitectureChoice.YoloV4:
            return_data = YoloV4Callback.postprocess_deploy(
                array=array, options=options, save_path=save_path, dataset_path=dataset_path,
                sensitivity=sensitivity, threashold=threashold
            )
        else:
            pass
        logger.info("Постобработка результатов предсказания успешно завершена.", extra={"type": "success"})
        return return_data
    except Exception as error:
        raise PostprocessException(
                        os.path.basename(__file__), method_name
                        ).with_traceback(error.__traceback__)
