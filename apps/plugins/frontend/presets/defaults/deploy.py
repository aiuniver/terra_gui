from ...choices import DeployTypePageChoice


DeployTypeGroup = {
    "collapsable": False,
    "collapsed": False,
    "fields": [
        {
            "type": "select",
            "label": "Тип",
            "name": "type",
            "parse": "type",
            "value": DeployTypePageChoice.model.name,
            "list": list(
                map(
                    lambda item: {"value": item.name, "label": item.value},
                    list(DeployTypePageChoice),
                )
            ),
            "fields": {
                "model": [
                    {
                        "type": "auto_complete",
                        "label": "Модель",
                        "name": "name",
                        "parse": "name",
                        "value": None,
                        "list": [],
                    }
                ],
                "cascade": [
                    {
                        "type": "auto_complete",
                        "label": "Каскад",
                        "name": "name",
                        "parse": "name",
                        "value": None,
                        "list": [],
                    }
                ],
            },
        },
    ],
}


DeployServerGroup = {
    "name": "Сервер",
    "collapsable": False,
    "collapsed": False,
    "fields": [],
    # "type": {
    #     "type": "select",
    #     "label": "Тип",
    #     "value": DeployTypePageChoice.model.name,
    #     "list": [field.name for field in DeployTypePageChoice],
    # },
    # "model": {"type": "auto_complete", "label": "Модель", "value": None, "list": []},
    # "cascade": {"type": "auto_complete", "label": "Каскад", "value": None, "list": []},
}
