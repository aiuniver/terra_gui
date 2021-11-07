const data = {
  "update": "hIgBPeYK36f2ECRF",
  "class_graphics": {
    "2": false
  },
  "loss_graphs": [
    {
      "id": 1,
      "type": "graphic",
      "type_data": null,
      "graph_name": "Выходной слой «2» - График ошибки обучения - Эпоха №1",
      "short_name": "2 - График ошибки обучения",
      "x_label": "Эпоха",
      "y_label": "Значение",
      "plot_data": [
        {
          "label": "Тренировочная выборка",
          "x": [
            1
          ],
          "y": [
            0.0869
          ]
        },
        {
          "label": "Проверочная выборка",
          "x": [
            1
          ],
          "y": [
            0.0204
          ]
        }
      ],
      "best": [
        {
          "label": "Лучший результат на тренировочной выборке",
          "x": [
            1
          ],
          "y": [
            0.0869
          ]
        },
        {
          "label": "Лучший результат на проверочной выборке",
          "x": [
            1
          ],
          "y": [
            0.0204
          ]
        }
      ],
      "progress_state": "normal"
    }
  ],
  "metric_graphs": [
    {
      "id": 1,
      "type": "graphic",
      "type_data": null,
      "graph_name": "Выходной слой «2» - График метрики MeanAbsoluteError - Эпоха №1",
      "short_name": "2 - MeanAbsoluteError",
      "x_label": "Эпоха",
      "y_label": "Значение",
      "plot_data": [
        {
          "label": "Тренировочная выборка",
          "x": [
            1
          ],
          "y": [
            0.256
          ]
        },
        {
          "label": "Проверочная выборка",
          "x": [
            1
          ],
          "y": [
            0.141
          ]
        }
      ],
      "best": [
        {
          "label": "Лучший результат на тренировочной выборке",
          "x": [
            1
          ],
          "y": [
            0.256
          ]
        },
        {
          "label": "Лучший результат на проверочной выборке",
          "x": [
            1
          ],
          "y": [
            0.141
          ]
        }
      ],
      "progress_state": "normal"
    }
  ],
  "intermediate_result": {
    "1": {
      "initial_data": {
        "Входной слой «1»": {
          "type": "graphic",
          "data": [
            {
              "title": "Графики по каналам «Open», «High», «Low», «Close», «Volume»",
              "value": [
                {
                  "id": 1,
                  "graph_name": "График канала «Open»",
                  "x_label": "Время",
                  "y_label": "Значение",
                  "plot_data": [{
                    "x": [
                      0,
                      1,
                      2,
                      3,
                      4
                    ],
                    "y": [
                      8731.64,
                      8732.94,
                      8725.36,
                      8725.37,
                      0.00092724
                    ]
                  }]
                },
                {
                  "id": 2,
                  "graph_name": "График канала «High»",
                  "x_label": "Время",
                  "y_label": "Значение",
                  "plot_data": [{
                    "x": [
                      0,
                      1,
                      2,
                      3,
                      4
                    ],
                    "y": [
                      8731.64,
                      8732.94,
                      8725.36,
                      8725.37,
                      0.00092724
                    ]
                  }]
                },
                {
                  "id": 3,
                  "graph_name": "График канала «Low»",
                  "x_label": "Время",
                  "y_label": "Значение",
                  "plot_data": [{
                    "x": [
                      0,
                      1,
                      2,
                      3,
                      4
                    ],
                    "y": [
                      8731.64,
                      8732.94,
                      8725.36,
                      8725.37,
                      0.00092724
                    ]
                  }]
                },
                {
                  "id": 4,
                  "graph_name": "График канала «Close»",
                  "x_label": "Время",
                  "y_label": "Значение",
                  "plot_data": [{
                    "x": [
                      0,
                      1,
                      2,
                      3,
                      4
                    ],
                    "y": [
                      8731.64,
                      8732.94,
                      8725.36,
                      8725.37,
                      0.00092724
                    ]
                  }]
                },
                {
                  "id": 5,
                  "graph_name": "График канала «Volume»",
                  "x_label": "Время",
                  "y_label": "Значение",
                  "plot_data": [{
                    "x": [
                      0,
                      1,
                      2,
                      3,
                      4
                    ],
                    "y": [
                      8731.64,
                      8732.94,
                      8725.36,
                      8725.37,
                      0.00092724
                    ]
                  }]
                }
              ],
              "color_mark": null
            }
          ]
        }
      },
      "true_value": {},
      "predict_value": {
        "Выходной слой «2»": {
          "type": "graphic",
          "data": [
            {
              "title": "Графики",
              "value": [
                {
                  "id": 1,
                  "type": "graphic",
                  "type_data": null,
                  "graph_name": "График канала «Open»",
                  "short_name": "«Open»",
                  "x_label": "Время",
                  "y_label": "Значение",
                  "plot_data": [
                    {
                      "label": "Исходное значение",
                      "x": [
                        0,
                        1,
                        2,
                        3,
                        4
                      ],
                      "y": [
                        8731.64,
                        8732.94,
                        8725.36,
                        8725.37,
                        0.00092724
                      ]
                    },
                    {
                      "label": "Истинное значение",
                      "x": [
                        5,
                        6
                      ],
                      "y": [
                        9414.08,
                        9404.98
                      ]
                    },
                    {
                      "label": "Предсказанное значение",
                      "x": [
                        5,
                        6
                      ],
                      "y": [
                        16888.345703125,
                        17178.75
                      ]
                    }
                  ],
                  "best": null,
                  "progress_state": null
                },
                {
                  "id": 2,
                  "type": "graphic",
                  "type_data": null,
                  "graph_name": "График канала «Close»",
                  "short_name": "«Close»",
                  "x_label": "Время",
                  "y_label": "Значение",
                  "plot_data": [
                    {
                      "label": "Исходное значение",
                      "x": [
                        0,
                        1,
                        2,
                        3,
                        4
                      ],
                      "y": [
                        9228.58,
                        9228.58,
                        9228.58,
                        9228.58,
                        0.0
                      ]
                    },
                    {
                      "label": "Истинное значение",
                      "x": [
                        5,
                        6
                      ],
                      "y": [
                        6971.2,
                        6971.23
                      ]
                    },
                    {
                      "label": "Предсказанное значение",
                      "x": [
                        5,
                        6
                      ],
                      "y": [
                        16959.544921875,
                        17240.876953125
                      ]
                    }
                  ],
                  "best": null,
                  "progress_state": null
                }
              ],
              "color_mark": null
            }
          ]
        }
      },
      "tags_color": null,
      "statistic_values": {
        "Выходной слой «2»": {
          "type": "table",
          "data": [
            {
              "title": "Open",
              "value": [
                {
                  "Шаг": "1",
                  "Истина": " 9414.08",
                  "Предсказание": " 16888.35",
                  "Отклонение": {
                    "value": " 79.39 %",
                    "color_mark": "wrong"
                  }
                },
                {
                  "Шаг": "2",
                  "Истина": " 9404.98",
                  "Предсказание": " 17178.75",
                  "Отклонение": {
                    "value": " 82.66 %",
                    "color_mark": "wrong"
                  }
                }
              ],
              "color_mark": null
            },
            {
              "title": "Close",
              "value": [
                {
                  "Шаг": "1",
                  "Истина": " 6971.20",
                  "Предсказание": " 16959.54",
                  "Отклонение": {
                    "value": " 143.28 %",
                    "color_mark": "wrong"
                  }
                },
                {
                  "Шаг": "2",
                  "Истина": " 6971.23",
                  "Предсказание": " 17240.88",
                  "Отклонение": {
                    "value": " 147.31 %",
                    "color_mark": "wrong"
                  }
                }
              ],
              "color_mark": null
            }
          ]
        }
      }
    }
  },
  "progress_table": {
    "1": {
      "time": 1274.8710796833038,
      "data": {
        "Выходной слой «2»": {
          "loss": {
            "loss": "0.0869",
            "val_loss": "0.0204"
          },
          "metrics": {
            "MeanAbsoluteError": "0.256",
            "val_MeanAbsoluteError": "0.141"
          }
        }
      }
    }
  },
  "statistic_data": {},
  "addtrain_epochs": []
}

export { data }