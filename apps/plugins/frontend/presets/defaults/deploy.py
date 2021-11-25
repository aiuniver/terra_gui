from django.conf import settings

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
    "fields": [
        {
            "type": "select",
            "label": "Сервер",
            "name": "server",
            "parse": "server",
            "value": "",
            "list": [{"value": "", "label": "Демо-панель"}]
            + list(
                map(
                    lambda item: {
                        "value": item[0],
                        "label": f'{item[1].get("domain_name")} [{item[1].get("ip_address")}]',
                    },
                    settings.USER_SERVERS.items(),
                )
            ),
        }
    ],
}
