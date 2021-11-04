from terra_ai.data.cascades.extra import BlockGroupChoice


CascadesBlockForm = [
    {
        "type": "text",
        "label": "Название блока",
        "name": "name",
        "parse": "name",
    },
    {
        "type": "select",
        "label": "Тип блока",
        "name": "group",
        "parse": "group",
        "list": list(
            map(
                lambda item: {"value": item.value, "label": item.name},
                list(BlockGroupChoice),
            )
        ),
    },
]


CascadesBlocksTypes = {}
