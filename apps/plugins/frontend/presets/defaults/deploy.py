from terra_ai.data.deploy.extra import DeployTypePageChoice

DeployFields = {
    "type":
        {
            "type": "select",
            "label": "Тип",
            "value": DeployTypePageChoice.model.name,
            "list": [field.name for field in DeployTypePageChoice]
        },
    "model": {
        "type": "auto_complete",
        "label": "Модель",
        "value": None,
        "list": []
    },
    "cascade": {
        "type": "auto_complete",
        "label": "Каскад",
        "value": None,
        "list": []
    }
}
