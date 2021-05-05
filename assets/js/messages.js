"use strict";


let Messages = function() {

    let _values = {
        DATASET_SELECTED: "Dataset «{0}» is selected",
        DATASET_LOADING: "Loading dataset «{0}»...",
        DATASET_LOADED: "Dataset «{0}» is loaded",
        MODEL_SAVED: "Model is saved.",
        INTERNAL_SERVER_ERROR: "Internal Server Error! Please try again later...",
        PARSE_TERRA_PROJECT_CONFIG_ERROR: "Terra project config not loaded. The default config is used.",
        TRY_TO_SAVE_PROJECT_NAME: "Trying to save new project name «{0}»",
        PROJECT_NAME_SAVE_SUCCESS: "Save new project name «{0}» complete successfully",
        LOAD_MODEL: "Загрузить готовую архитектуру",
        TRYING_TO_LOAD_MODEL: "Trying to load model «{0}»...",
        MODEL_LOADED: "Model «{0}» loaded",
        TRYING_TO_LOAD_LAYER: "Loading «{0}» layer...",
        LAYER_LOADED: "«{0}» layer is loaded",
        LAYER_ALREADY_EXISTS: "«{0}» layer already exists",
        SUBMIT_PARAMS_METHOD: "You must define a submit method of Params object like this: _params.submit = (send_data) => {}",
        LAYER_SAVED: "Layer is saved.",
        VALIDATE_MODEL: "Validate model...",
        VALIDATION_MODEL_SUCCESS: "Validation of model complete successfully!",
        VALIDATION_MODEL_ERROR: "Validation of model complete with error! Please correct all error and try again",
    };

    this.get = (name, values) => {
        let value = _values[name];
        if (!value) return "";
        if (values && Object.keys(values).length) {
            for (let index in values) {
                value = value.replace("{" + index + "}", values[index]);
            }
        }

        return value;
    }

}

window.Messages = new Messages();
