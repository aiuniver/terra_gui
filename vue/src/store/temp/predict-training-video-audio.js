const predict_video = {
  initial_value: {
    'Input_1 (Видео)': {
      1: {
          type: 'video',
          data: 'https://www.youtube.com/watch?v=BhqJp1yxL0c'
      },
      2: {
          type: 'video',
          data: 'https://www.youtube.com/watch?v=BhqJp1yxL0c'
      },
      3: {
          type: 'video',
          data: 'https://www.youtube.com/watch?v=BhqJp1yxL0c'
      }
    },
    'Input_2 (Аудио)': {
      1: {
          type: 'audio',
          data: 'sound1.mp3'
      },
      2: {
          type: 'audio',
          data: 'sound1.mp3'
      },
      3: {
          type: 'audio',
          data: 'sound1.mp3'
      }
    }
  },
  true_value: {
    'Output_2': {
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
    'Output_3': {
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

export { predict_video }