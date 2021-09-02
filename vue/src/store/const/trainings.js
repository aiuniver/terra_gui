const toolbar = [
  {
    title: "Графики",
    active: true,
    disabled: false,
    icon: "icon-training-charts",
  },
  {
    title: "Скаттеры",
    active: false,
    disabled: true,
    icon: "icon-training-scatters",
  },
  {
    title: "Изображения",
    active: false,
    disabled: false,
    icon: "icon-training-images",
  },
  {
    title: "Текст",
    active: true,
    disabled: false,
    icon: "icon-training-texts",
  },
]

// const createInputData = function (id, layer, usedColors) {
//   return {
//       id,
//       layer,
//       name: (layer === 'input' ? 'Вход ' : 'Выход ') + id,
//       type: 'Image',
//       color: getColor(usedColors),
//       parameters: {}
//   }
// }

export { toolbar };