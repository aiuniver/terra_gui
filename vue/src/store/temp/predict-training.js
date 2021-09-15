const predict = {
  initial_value: {
    'Вход 1': {
      1: {
          type: 'image',
          data: 'predict/img.png'
      },
      2: {
          type: 'image',
          data: 'predict/img_1.png'
      },
      3: {
          type: 'image',
          data: 'predict/img_2.png',
      }
    }
  },
  true_value: {
    'Выход 2': {
       1: {
          type: 'str',
          data: 'frog'
      },
      2: {
          type: 'str',
          data: 'truck'
      },
      3: {
          type: 'str',
          data: 'ship',
      }
    }
  },
  predict_value: {
    'Выход 2': {
      1: {
        type: 'str',
        data: 'frog',
        color_mark: 'success'
      },
      2: {
        type: 'str',
        data: 'bird',
        color_mark: 'wrong'
      },
      3: {
        type: 'str',
        data: 'ship',
        color_mark: 'success'
      }
    }
  },
  statistic_values: {
    'airplane': {
      1: {
        type: 'number',
        data: 0.3
      },
      2: {
        type: 'number',
        data: 0.3
      },
      3: {
        type: 'number',
        data: 0.3
      }
    },
    'bird': {
      1: {
        type: 'number',
        data: 0.3
      },
      2: {
        type: 'number',
        data: 21.8,
        color_mark: 'wrong'
      },
      3: {
        type: 'number',
        data: 0.3
      }
    },
    'cat': {
      1: {
        type: 'number',
        data: 0.3
      },
      2: {
        type: 'number',
        data: 0.3
      },
      3: {
        type: 'number',
        data: 0.3
      }
    },
    'deer': {
      1: {
        type: 'number',
        data: 0.3
      },
      2: {
        type: 'number',
        data: 0.3
      },
      3: {
        type: 'number',
        data: 0.3
      }
    },
    'frog': {
      1: {
        type: 'number',
        data: 46.3,
        color_mark: 'success'
      },
      2: {
        type: 'number',
        data: 0.3
      },
      3: {
        type: 'number',
        data: 0.3
      }
    },
    'ship': {
      1: {
        type: 'number',
        data: 0.3
      },
      2: {
        type: 'number',
        data: 0.3
      },
      3: {
        type: 'number',
        data: 91.3,
        color_mark: 'success'
      }
    },
    'truck': {
      1: {
        type: 'number',
        data: 0.3
      },
      2: {
        type: 'number',
        data: 0.3
      },
      3: {
        type: 'number',
        data: 0.3
      }
    },
  }
}

export { predict }