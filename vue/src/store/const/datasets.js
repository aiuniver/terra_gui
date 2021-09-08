// const color = [
//     '#FFB054', '#FF9D2A', '#FE8900', '#D47200', '#A95C00', '#7F4500', // orange
//     '#89D764', '#71CE45', '#5DBB31', '#4E9C29', '#3E7D21', '#2F5E18', // green
//     '#54A3FF', '#2A8CFF', '#0075FE', '#0062D4', '#004EAA', '#003B7F', // blue
//     '#8E51F2', '#762BEF', '#6011E2', '#500EBC', '#400B96', '#300871', // purple
// ];
const color = [
  '#FFB054', '#FF9D2A', '#FE8900', '#A95C00', '#552E00', '#2A1700', // orange
  '#89D764', '#5DBB31', '#3A8815', '#2F5E18', '#1F3E10', '#101F08', // green
  '#54A3FF', '#057AFF', '#0B5BCD', '#00419D', '#002755', '#00142A', // blue
  '#FF8F89', '#FF5B54', '#FF322A', '#B81515', '#820803', '#550300', // red
  '#FBF889', '#F6F15E', '#F4ED33', '#C8C100', '#908A00', '#504E00', // yellow
  '#1AFFE0', '#008F7C', '#006B5D', '#00473E', '#00362E', '#00241F', // turquoise  
  '#D67596', '#C84773', ' #9E3156', '#822746', '#521B2E', '#34101C', // purple  
];

const getColor = usedColors => {
  let index;
  do {
    index = Math.floor(Math.random() * 24);
  } while (usedColors.length < color.length && usedColors.includes(color[index]))
  return color[index];
};

const createInputData = function (id, layer, usedColors) {
  return {
    id,
    layer,
    name: (layer === 'input' ? 'Вход ' : 'Выход ') + id,
    type: 'Image',
    color: getColor(usedColors),
    parameters: {}
  }
}

const cloneInputData = function (id, usedColors, { layer, type, parameters }) {
  return {
    id,
    layer,
    name: (layer === 'input' ? 'Вход ' : 'Выход ') + id,
    type: type || 'Image',
    color: getColor(usedColors),
    parameters: parameters || {}
  }
}
export { createInputData, cloneInputData };