
const inputs = {
  images: {
    folder_name: {
      type: "str",                       // Тип компонента. Можно оставить, но желательно поменять на соответственно text, select, number, checkbox,
      name: "folder_name",               // Имя компонента, желательно добавить, но не обязательно если ключь будет соответствовать имени
      parse: "[parameters][folder_name][]",// Строка для персера. Как будет формироватся обьект. [inputs][input_$] будет добавлятся к строке 
      label: "Name number",              // Название поля ввода
      default: "",                       // значение по умолчанию
      available: [                       // список. желательно переименовать в lists
        "",
        "Аппендицит",
        "Гастрит",
        "Гепатит",
        "Дуоденит",
        "Колит",
        "Панкреатит",
        "Холецистит",
        "Эзофагит",
        "Энтерит",
        "Язва",
      ],
    },
    height: {
      type: "int",
      default: 176,
      parse: "[parameters][height]",
    },
    width: {
      type: "int",
      default: 220,
      parse: "[parameters][width]",
    },
    net: {
      type: "str",
      default: "Convolutional",
      list: true,
      parse: "[parameters][net]",
      available: ["Convolutional", "Linear"],
    },
    scaler: {
      type: "str",
      default: "No Scaler",
      list: true,
      parse: "[parameters][scaler]",
      available: ["No Scaler", "MinMaxScaler"],
    },
  },
  text: {
    folder_name: {
      type: "str",
      default: "",
      list: true,
      parse: "[parameters][folder_name]",
      available: [
        "",
        "Аппендицит",
        "Гастрит",
        "Гепатит",
        "Дуоденит",
        "Колит",
        "Панкреатит",
        "Холецистит",
        "Эзофагит",
        "Энтерит",
        "Язва",
      ],
    },
    delete_symbols: {
      type: "str",
      parse: "[parameters][delete_symbols]",
      default: "",
    },
    x_len: {
      type: "int",
      parse: "[parameters][x_len]",
      default: 100,
    },
    step: {
      type: "int",
      parse: "[parameters][step]",
      default: 30,
    },
    max_words_count: {
      type: "int",
      parse: "[parameters][max_words_count]",
      default: 20000,
    },
    pymorphy: {
      type: "bool",
      parse: "[parameters][pymorphy]",
      default: false,
    },
    embedding: {
      type: "bool",
      default: true,
      parse: "[parameters][embedding]",
      event: ['word_to_vec', 'bag_of_words']
    },
    bag_of_words: {
      type: "bool",
      default: false,
      parse: "[parameters][bag_of_words]",
      event: ['word_to_vec', 'embedding']
    },
    word_to_vec: {
      type: "bool",
      default: false,
      parse: "[parameters][word_to_vec]",
      event: ['embedding', 'bag_of_words']
    },
    word_to_vec_size: {
      type: "int",
      parse: "[parameters][word_to_vec_size]",
      default: 200,
    },
  },
  audio: {
    folder_name: {
      type: "str",
      default: "",
      list: true,
      parse: "[parameters][folder_name]",
      available: [
        "",
        "Аппендицит",
        "Гастрит",
        "Гепатит",
        "Дуоденит",
        "Колит",
        "Панкреатит",
        "Холецистит",
        "Эзофагит",
        "Энтерит",
        "Язва",
      ],
    },
    length: {
      type: "int",
      parse: "[parameters][length]",
      default: 22050,
    },
    step: {
      type: "int",
      parse: "[parameters][step]",
      default: 2205,
    },
    scaler: {
      type: "str",
      default: "No Scaler",
      list: true,
      parse: "[parameters][scaler]",
      available: ["No Scaler", "StandardScaler", "MinMaxScaler"],
    },
    audio_signal: {
      type: "bool",
      parse: "[parameters][audio_signal]",
      default: true,
    },
    chroma_stft: {
      type: "bool",
      parse: "[parameters][chroma_stft]",
      default: false,
    },
    mfcc: {
      type: "bool",
      parse: "[parameters][mfcc]",
      default: false,
    },
    rms: {
      type: "bool",
      parse: "[parameters][rms]",
      default: false,
    },
    spectral_centroid: {
      type: "bool",
      parse: "[parameters][spectral_centroid]",
      default: false,
    },
    spectral_bandwidth: {
      type: "bool",
      parse: "[parameters][spectral_bandwidth]",
      default: false,
    },
    spectral_rolloff: {
      type: "bool",
      parse: "[parameters][spectral_rolloff]",
      default: false,
    },
    zero_crossing_rate: {
      type: "bool",
      parse: "[parameters][zero_crossing_rate]",
      default: false,
    },
  },
  dataframe: {
    file_name: {
      type: "str",
      default: "",
      list: true,
      parse: "[parameters][dataframe]",
      available: [
        "",
        "Аппендицит",
        "Гастрит",
        "Гепатит",
        "Дуоденит",
        "Колит",
        "Панкреатит",
        "Холецистит",
        "Эзофагит",
        "Энтерит",
        "Язва",
      ],
    },
    separator: {
      type: "str",
      parse: "[parameters][separator]",
      default: "",
    },
    encoding: {
      type: "str",
      parse: "[parameters][encoding]",
      default: "utf-8",
    },
    x_cols: {
      type: "str",
      parse: "[parameters][x_cols]",
      default: "",
    },
    scaler: {
      type: "str",
      default: "No Scaler",
      list: true,
      parse: "[parameters][scaler]",
      available: ["No Scaler", "StandardScaler", "MinMaxScaler"],
    },
  },
  classification: {
    one_hot_encoding: {
      type: "bool",
      parse: "[parameters][classification]",
      default: true,
    },
  },
  segmentation: {
    folder_name: {
      type: "str",
      default: "",
      list: true,
      parse: "[parameters][segmentation]",
      available: [
        "",
        "Аппендицит",
        "Гастрит",
        "Гепатит",
        "Дуоденит",
        "Колит",
        "Панкреатит",
        "Холецистит",
        "Эзофагит",
        "Энтерит",
        "Язва",
      ],
    },
    mask_range: {
      type: "int",
      parse: "[parameters][mask_range]",
      default: 50,
    },
    input_type: {
      type: "str",
      default: "",
      parse: "[parameters][input_type]",
      list: true,
      available: ["", "Ручной ввод", "Автоматический поиск", "Файл аннотации"],
    },
  },
  text_segmentation: {
    open_tags: {
      type: "str",
      parse: "[parameters][open_tags]",
      default: "",
    },
    close_tags: {
      type: "str",
      parse: "[parameters][close_tags]",
      default: "",
    },
  },
  regression: {
    y_col: {
      type: "str",
      parse: "[parameters][regression]",
      default: "",
    },
  },
  timeseries: {
    length: {
      type: "int",
      parse: "[parameters][timeseries]",
      default: 1,
    },
    y_cols: {
      type: "str",
      parse: "[parameters][y_cols]",
      default: "",
    },
    scaler: {
      type: "str",
      default: "No Scaler",
      list: true,
      parse: "[parameters][scaler]",
      available: ["No Scaler", "StandardScaler", "MinMaxScaler"],
    },
    task_type: {
      type: "str",
      default: "timeseries",
      list: true,
      parse: "[parameters][task_type]",
      available: ["timeseries", "regression"],
    },
  },
};

export default inputs;
