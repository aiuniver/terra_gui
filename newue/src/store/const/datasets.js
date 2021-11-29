// const color = [
//     '#FFB054', '#FF9D2A', '#FE8900', '#D47200', '#A95C00', '#7F4500', // orange
//     '#89D764', '#71CE45', '#5DBB31', '#4E9C29', '#3E7D21', '#2F5E18', // green
//     '#54A3FF', '#2A8CFF', '#0075FE', '#0062D4', '#004EAA', '#003B7F', // blue
//     '#8E51F2', '#762BEF', '#6011E2', '#500EBC', '#400B96', '#300871', // purple
// ];
const color = [
  '#54A3FF', '#0B5BCD', '#00419D', '#002755', '#00142A',  // blue
  '#8E51F2', '#4E0CBA', '#3C0792', '#270D52', '#100326',  // purple
  '#89D764', '#3A8815', '#2F5E18', '#1F3E10', '#101F08', // green
  '#FFB054', '#FE8900', '#A95C00', '#552E00', '#2A1700',  // orange
  // '#FF8F89', 
  '#FF322A', '#B81515', '#820803', '#550300',  // red
  // '#FBF889', '#F4ED33', '#C8C100', '#908A00', '#504E00',  // yellow  
  // '#1AFFE0', 
  '#006B5D', '#00473E', '#00362E', '#00241F',  // turquoise  
  '#D67596', '#9E3156', '#822746', '#521B2E', '#34101C',  // pink  
  '#9CBB1E', '#536314', '#3D490E', '#2B330C', '#1A1E0B',  // wtf last column
];

const getColor = usedColors => {
  let index;
  do {
    index = Math.floor(Math.random() * color.length);
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

const changeStructTable = function (data) {
  data.forEach(table => {
    if (table.type === 'table' && table.data) {
      const newarr = [];
      table.data.forEach((el, index) => {
        el.forEach((elm, i) => {
          if (!newarr[i]) {
            newarr[i] = [];
          }
          newarr[i][index] = elm;
        });
      });
      table.data = newarr
    }
  })
  return data;
}
const getNameToId = function (handlers, id) {
  return handlers.find(handler => handler.id === id).name
}
const getIdToName = function (files, { name, table }) {
  const cells = files.find(item => item.title === table)
  // console.log(cells)
  const index = cells.data.findIndex(item => item[0] === name)
  // console.log(index)
  return index

}
export { createInputData, cloneInputData, changeStructTable, getNameToId , getIdToName};