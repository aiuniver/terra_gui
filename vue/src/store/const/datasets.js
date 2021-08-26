const color = [
    '#FFB054', '#FF9D2A', '#FE8900', '#D47200', '#A95C00', '#7F4500', // orange
    '#89D764', '#71CE45', '#5DBB31', '#4E9C29', '#3E7D21', '#2F5E18', // green
    '#54A3FF', '#2A8CFF', '#0075FE', '#0062D4', '#004EAA', '#003B7F', // blue
    '#8E51F2', '#762BEF', '#6011E2', '#500EBC', '#400B96', '#300871', // purple
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