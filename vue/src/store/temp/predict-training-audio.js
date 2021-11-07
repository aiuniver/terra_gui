const predict_audio = {

  1: {
    'initial_data': {
      'Входной слой 1': {
        'data': '/tmp/initial_data_audio_0_input_1.webp',
        'type': 'Audio'
      },
      'Входной слой 2': {
        'data': '/tmp/initial_data_audio_0_input_2.webp',
        'type': 'Audio'
      }
    },
    'true_value': {
      'Выходной слой 3': {
        'type': 'str',
        'data': '1_Кондиционер',
        'color_mark': null,
        'tags_color': null
      }
    },
    'predict_value': {
      'Выходной слой 3': {
        'type': 'str',
        'data': '3_Телевизор',
        'color_mark': 'wrong',
        'tags_color': null
      }
    },
    'statistic_values': {
      'Выходной слой 3': {
        '1_Кондиционер': {
          'value': '11.5%',
          'color_mark': null
        },
        '2_Свет': {
          'value': '8.5%',
          'color_mark': null
        },
        '3_Телевизор': {
          'value': '79.5%',
          'color_mark': 'wrong'
        },
        '4_Шум': {
          'value': '0.5%',
          'color_mark': null
        }
      }
    }
  },
  2: {
    'initial_data': {
      'Входной слой 1': {
        'data': '/tmp/initial_data_audio_1_input_1.webp',
        'type': 'Audio'
      },
      'Входной слой 2': {
        'data': '/tmp/initial_data_audio_1_input_2.webp',
        'type': 'Audio'
      }
    },
    'true_value': {
      'Выходной слой 3': {
        'type': 'str',
        'data': '1_Кондиционер',
        'color_mark': null,
        'tags_color': null
      }
    },
    'predict_value': {
      'Выходной слой 3': {
        'type': 'str',
        'data': '3_Телевизор',
        'color_mark': 'wrong',
        'tags_color': null
      }
    },
    'statistic_values': {
      'Выходной слой 3': {
        '1_Кондиционер': {
          'value': '11.5%',
          'color_mark': null
        },
        '2_Свет': {
          'value': '8.5%',
          'color_mark': null
        },
        '3_Телевизор': {
          'value': '79.5%',
          'color_mark': 'wrong'
        },
        '4_Шум': {
          'value': '0.5%',
          'color_mark': null
        }
      }
    }
  }
}

export { predict_audio }
