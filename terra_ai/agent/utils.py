from enum import Enum

from . import exceptions


class ExceptionClasses(Enum):
    """Вызываемые методы агента и их базовые исключения"""
    unknown = exceptions.ExchangeBaseException
    # Project
    projects_info = exceptions.FailedGetProjectsInfoException
    project_save = exceptions.FailedSaveProjectException
    project_load = exceptions.FailedLoadProjectException
    # Dataset
    dataset_choice = exceptions.FailedChoiceDatasetException
    dataset_choice_progress = exceptions.FailedGetProgressDatasetChoiceException
    dataset_delete = exceptions.FailedDeleteDatasetException
    datasets_info = exceptions.FailedGetDatasetsInfoException
    dataset_source_load = exceptions.FailedLoadDatasetsSourceException
    dataset_source_load_progress = exceptions.FailedLoadProgressDatasetsSource
    dataset_create = exceptions.FailedCreateDatasetException
    datasets_sources = exceptions.FailedGetDatasetsSourcesException
    # Modeling
    models = exceptions.FailedGetModelsListException
    model_get = exceptions.FailedGetModelException
    model_update = exceptions.FailedUpdateModelException
    model_validate = exceptions.FailedValidateModelException
    model_create = exceptions.FailedCreateModelException
    model_delete = exceptions.FailedDeleteModelException
    # Training
    training_start = exceptions.FailedStartTrainException
    training_stop = exceptions.FailedStopTrainException
    training_clear = exceptions.FailedCleanTrainException
    training_interactive = exceptions.FailedSetInteractiveConfigException
    training_progress = exceptions.FailedGetTrainingProgressException
    training_save = exceptions.FailedSaveTrainException
    # Deploy
    deploy_presets = exceptions.FailedGetDeployPresetsException
    deploy_collection = exceptions.FailedGetDeployCollectionException
    deploy_upload = exceptions.FailedUploadDeployException
    deploy_upload_progress = exceptions.FailedGetUploadDeployProgressException
