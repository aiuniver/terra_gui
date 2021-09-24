const data = {
  "loss_graphs": [
    {
      "id": 1,
      "graph_name": "Выходной слой «2» - График ошибки обучения - Эпоха №2",
      "x_label": "Эпоха",
      "y_label": "Значение",
      "plot_data": [
        {
          "label": "Тренировочная выборка",
          "x": [
            1,
            2
          ],
          "y": [
            0.3554726243019104,
            0.13371901214122772
          ]
        },
        {
          "label": "Проверочная выборка",
          "x": [
            1,
            2
          ],
          "y": [
            0.42863497138023376,
            0.37242254614830017
          ]
        }
      ],
      "progress_state": "normal"
    },
    {
      "id": 2,
      "graph_name": "Выходной слой «2» - График ошибки обучения по классам - Эпоха №2",
      "x_label": "Эпоха",
      "y_label": "Значение",
      "plot_data": [
        {
          "class_label": "Класс 1",
          "x": [
            1,
            2
          ],
          "y": [
            0.36261245608329773,
            0.5447661280632019
          ]
        },
        {
          "class_label": "Класс 2",
          "x": [
            1,
            2
          ],
          "y": [
            0.36261245608329773,
            0.5447661280632019
          ]
        }
      ]
    }
  ],
  "metric_graphs": [
    {
      "id": 1,
      "graph_name": "Выходной слой «2» - График метрики BinaryCrossentropy - Эпоха №2",
      "x_label": "Эпоха",
      "y_label": "Значение",
      "plot_data": [
        {
          "label": "Тренировочная выборка",
          "x": [
            1,
            2
          ],
          "y": [
            0.3554726541042328,
            0.1337190419435501
          ]
        },
        {
          "label": "Проверочная выборка",
          "x": [
            1,
            2
          ],
          "y": [
            0.42863497138023376,
            0.37242254614830017
          ]
        }
      ],
      "progress_state": "normal"
    },
    {
      "id": 2,
      "graph_name": "Выходной слой «2» - График метрики BinaryCrossentropy по классам - Эпоха №2",
      "x_label": "Эпоха",
      "y_label": "Значение",
      "plot_data": [
        {
          "class_label": "Класс 1",
          "x": [
            1,
            2
          ],
          "y": [
            0.428411066532135,
            0.37300699949264526
          ]
        },
        {
          "class_label": "Класс 2",
          "x": [
            1,
            2
          ],
          "y": [
            0.42818865180015564,
            0.3735875189304352
          ]
        }
      ]
    }
  ],
  "intermediate_result": {
    "1": {
      "initial_data": {
        "Входной слой «1»": {
          "type": "image",
          "data": [
            {
              "title": "Изображение",
              "value": "/tmp/tai-project/training/presets/initial_data_image_1_input_1.webp",
              "color_mark": null
            }
          ]
        }
      },
      "true_value": {
        "Выходной слой «2»": {
          "type": "image",
          "data": [
            {
              "title": "Изображение",
              "value": "/tmp/tai-project/training/presets/true_segmentation_data_image_1_output_2.webp",
              "color_mark": null
            }
          ]
        }
      },
      "predict_value": {
        "Выходной слой «2»": {
          "type": "image",
          "data": [
            {
              "title": "Изображение",
              "value": "/tmp/tai-project/training/presets/predict_segmentation_data_image_1_output_2.webp",
              "color_mark": null
            }
          ]
        }
      },
      "tags_color": {},
      "statistic_values": {}
    },
    "2": {
      "initial_data": {
        "Входной слой «1»": {
          "type": "image",
          "data": [
            {
              "title": "Изображение",
              "value": "/tmp/tai-project/training/presets/initial_data_image_2_input_1.webp",
              "color_mark": null
            }
          ]
        }
      },
      "true_value": {
        "Выходной слой «2»": {
          "type": "image",
          "data": [
            {
              "title": "Изображение",
              "value": "/tmp/tai-project/training/presets/true_segmentation_data_image_2_output_2.webp",
              "color_mark": null
            }
          ]
        }
      },
      "predict_value": {
        "Выходной слой «2»": {
          "type": "image",
          "data": [
            {
              "title": "Изображение",
              "value": "/tmp/tai-project/training/presets/predict_segmentation_data_image_2_output_2.webp",
              "color_mark": null
            }
          ]
        }
      },
      "tags_color": {},
      "statistic_values": {}
    }
  },
  "progress_table": {
    "1": {
      "time": 469.8170247077942,
      "data": {
        "Выходной слой «2»": {
          "loss": {
            "loss": 0.3554726243019104,
            "val_loss": 0.42863497138023376
          },
          "metrics": {
            "BinaryCrossentropy": 0.3554726541042328,
            "val_BinaryCrossentropy": 0.42863497138023376,
            "MeanIoU": 0.2810913920402527,
            "val_MeanIoU": 0.25,
            "Hinge": 0.6235536336898804,
            "val_Hinge": 0.7117000818252563,
            "AUC": 0.9465541839599609,
            "val_AUC": 0.9301667213439941
          }
        }
      }
    },
    "2": {
      "time": 549.3132457733154,
      "data": {
        "Выходной слой «2»": {
          "loss": {
            "loss": 0.13371901214122772,
            "val_loss": 0.37242254614830017
          },
          "metrics": {
            "BinaryCrossentropy": 0.1337190419435501,
            "val_BinaryCrossentropy": 0.37242254614830017,
            "MeanIoU": 0.29534977674484253,
            "val_MeanIoU": 0.25,
            "Hinge": 0.5362575054168701,
            "val_Hinge": 0.6374033093452454,
            "AUC": 0.9908504486083984,
            "val_AUC": 0.9190367460250854
          }
        }
      }
    }
  },
  "statistic_data": {
    "2": [
      {
        "id": 1,
        "type": "heatmap",
        "graph_name": "Выходной слой «2» - Confusion matrix",
        "x_label": "Предсказание",
        "y_label": "Истинное значение",
        "labels": [
          "1",
          "2"
        ],
        "data_array": [
          [
            9621680,
            0
          ],
          [
            1417040,
            0
          ]
        ],
        "data_percent_array": [
          [
            100,
            0
          ],
          [
            100,
            0
          ]
        ]
      }
    ]
  },
  "data_balance": {
    "2": [
      {
        "id": 1,
        "type": "Histogram",
        "graph_name": "Тренировочная выборка - баланс присутсвия",
        "x_label": "Название класса",
        "y_label": "Значение",
        "plot_data": [
          {
            "labels": [
              "1",
              "2"
            ],
            "values": [
              686,
              686
            ]
          }
        ]
      },
      {
        "id": 2,
        "type": "Histogram",
        "graph_name": "Проверочная выборка - баланс присутсвия",
        "x_label": "Название класса",
        "y_label": "Значение",
        "plot_data": [
          {
            "labels": [
              "1",
              "2"
            ],
            "values": [
              196,
              196
            ]
          }
        ]
      },
      {
        "id": 3,
        "type": "Histogram",
        "graph_name": "Тренировочная выборка - процент пространства",
        "x_label": "Название класса",
        "y_label": "Значение",
        "plot_data": [
          {
            "labels": [
              "1",
              "2"
            ],
            "values": [
              87,
              13
            ]
          }
        ]
      },
      {
        "id": 4,
        "type": "Histogram",
        "graph_name": "Проверочная выборка - процент пространства",
        "x_label": "Название класса",
        "y_label": "Значение",
        "plot_data": [
          {
            "labels": [
              "1",
              "2"
            ],
            "values": [
              87,
              13
            ]
          }
        ]
      }
    ]
  }
}