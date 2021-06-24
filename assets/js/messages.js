"use strict";


let Messages = function() {

    let _values = {
        DATASET_SELECTED: "Выбран датасет «{0}»",
        DATASET_LOADING: "Загрузка датасета «{0}»...",
        DATASET_LOADED: "Датасет «{0}» загружен",
        DATASET_REMOVED: "Датасет «{0}» удален",
        DATASET_SOURCE_LOADING: "Загрузка исходных данных датасета...",
        DATASET_SOURCE_LOADED: "Исходные данные датасета загружены",
        MODEL_SAVED: "Модель сохранена",
        INTERNAL_SERVER_ERROR: "Внутренняя ошибка сервера",
        PARSE_TERRA_PROJECT_CONFIG_ERROR: "Terra project config not loaded. The default config is used.",
        TRY_TO_SAVE_PROJECT_NAME: "Сохранение названия проекта «{0}»",
        PROJECT_NAME_SAVE_SUCCESS: "Название проекта «{0}» сохранено",
        LOAD_MODEL: "Загрузка модели",
        TRYING_TO_LOAD_MODEL: "Загрузка модели «{0}»...",
        MODEL_LOADED: "Модель «{0}» загружена",
        LAYER_LOADED: "Слой «{0}» загружен",
        LAYER_CLONED: "Слой «{0}» скопирован",
        LAYER_ALREADY_EXISTS: "Слой «{0}» уже существует",
        LAYER_SAVED: "Слой сохранен",
        VALIDATE_MODEL: "Проверка модели...",
        VALIDATION_MODEL_SUCCESS: "Модель проверена успешно",
        VALIDATION_MODEL_ERROR: "Модель содержит ошибки",
        TRAINING_DISCARDED: "Обучение сброшено",
        PROJECT_SAVED: "Проект сохранен",
        CREATING_DATASET: "Создание датасета...",
        DATASET_CREATED: "Датасет создан",
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
