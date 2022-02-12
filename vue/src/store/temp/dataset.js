const project = {
    "alias": "airplane",
    "name": "Самолеты",
    "datasets_path": "G:\\Мой диск\\TerraAI\\datasets",
    "source_path": "C:\\Users\\Arthur\\AppData\\Local\\Temp\\terraai\\datasets\\sources\\googledrive\\airplane",
    "task_type": "ImageSegmentation",
    "tags": [
        {
            "alias": "proverka",
            "name": "проверка"
        }
    ],
    "version": {
        "alias": "samoleti_drugaja",
        "name": "самолеты другая",
        "datasets_path": "G:\\Мой диск\\TerraAI\\datasets",
        "parent_alias": "airplane",
        "info": {
            "part": {
                "train": 0.7,
                "validation": 0.3
            },
            "shuffle": true
        },
        "inputs": [
            {
                "type": "data",
                "name": "Группа данных",
                "id": 1,
                "position": [
                    412.0,
                    321.0
                ],
                "bind": {
                    "up": [],
                    "down": [
                        3
                    ]
                },
                "parameters": {
                    "type": "table",
                    "filename": "airplane.csv",
                    "items": [
                        "Одна колонка"
                    ]
                }
            },
            {
                "type": "data",
                "name": "Группа данных",
                "id": 2,
                "position": [
                    412.0,
                    321.0
                ],
                "bind": {
                    "up": [],
                    "down": [
                        4
                    ]
                },
                "parameters": {
                    "type": "table",
                    "filename": "airplane.csv",
                    "items": [
                        "Другая колонка"
                    ]
                }
            },
            {
                "type": "handler",
                "name": "Название обработчика",
                "id": 3,
                "position": [
                    412.0,
                    321.0
                ],
                "bind": {
                    "up": [
                        1
                    ],
                    "down": [
                        5
                    ]
                },
                "parameters": {
                    "height": 32,
                    "width": 64,
                    "image_mode": "stretch",
                    "net": "convolutional",
                    "scaler": "min_max_scaler",
                    "augmentation": null,
                    "min_scaler": 0,
                    "max_scaler": 1
                }
            },
            {
                "type": "handler",
                "name": "Название обработчика",
                "id": 4,
                "position": [
                    412.0,
                    321.0
                ],
                "bind": {
                    "up": [
                        2
                    ],
                    "down": [
                        5
                    ]
                },
                "parameters": {
                    "height": 64,
                    "width": 128,
                    "image_mode": "stretch",
                    "net": "convolutional",
                    "scaler": "min_max_scaler",
                    "augmentation": null,
                    "min_scaler": 0,
                    "max_scaler": 1
                }
            },
            {
                "type": "layer",
                "name": "Название слоя",
                "id": 5,
                "position": [
                    412.0,
                    321.0
                ],
                "bind": {
                    "up": [
                        3,
                        4
                    ],
                    "down": []
                },
                "parameters": {}
            }
        ],
        "outputs": [
            {
                "type": "data",
                "name": "Группа данных",
                "id": 1,
                "position": [
                    412.0,
                    321.0
                ],
                "bind": {
                    "up": [],
                    "down": [
                        3
                    ]
                },
                "parameters": {
                    "type": "table",
                    "filename": "airplane.csv",
                    "items": [
                        "Одна колонка"
                    ]
                }
            },
            {
                "type": "data",
                "name": "Группа данных",
                "id": 2,
                "position": [
                    412.0,
                    321.0
                ],
                "bind": {
                    "up": [],
                    "down": [
                        4
                    ]
                },
                "parameters": {
                    "type": "table",
                    "filename": "airplane.csv",
                    "items": [
                        "Другая колонка"
                    ]
                }
            },
            {
                "type": "handler",
                "name": "Название обработчика",
                "id": 3,
                "position": [
                    412.0,
                    321.0
                ],
                "bind": {
                    "up": [
                        1
                    ],
                    "down": [
                        5
                    ]
                },
                "parameters": {
                    "height": 32,
                    "width": 64,
                    "image_mode": "stretch",
                    "net": "convolutional",
                    "scaler": "min_max_scaler",
                    "augmentation": null,
                    "min_scaler": 0,
                    "max_scaler": 1
                }
            },
            {
                "type": "handler",
                "name": "Название обработчика",
                "id": 4,
                "position": [
                    412.0,
                    321.0
                ],
                "bind": {
                    "up": [
                        2
                    ],
                    "down": [
                        5
                    ]
                },
                "parameters": {
                    "height": 64,
                    "width": 128,
                    "image_mode": "stretch",
                    "net": "convolutional",
                    "scaler": "min_max_scaler",
                    "augmentation": null,
                    "min_scaler": 0,
                    "max_scaler": 1
                }
            },
            {
                "type": "layer",
                "name": "Название слоя",
                "id": 5,
                "position": [
                    412.0,
                    321.0
                ],
                "bind": {
                    "up": [
                        3,
                        4
                    ],
                    "down": []
                },
                "parameters": {}
            }
        ]
    }
}

export { project }