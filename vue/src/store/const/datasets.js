// const index = ['orange', 'green', 'blue', 'purple']
const color = {
    0: ['#FFB054', '#FF9D2A', '#FE8900', '#D47200', '#A95C00', '#7F4500'],
    1: ['#89D764', '#71CE45', '#5DBB31', '#4E9C29', '#3E7D21', '#2F5E18'],
    2: ['#54A3FF', '#2A8CFF', '#0075FE', '#0062D4', '#004EAA', '#003B7F'],
    3: ['#8E51F2', '#762BEF', '#6011E2', '#500EBC', '#400B96', '#300871'],
  };
  
  const getColor = usedColors => {
    let index, tone;
    do {
      index = Math.floor(Math.random() * 4);
      tone = Math.floor(Math.random() * 6);
    } while (usedColors.includes(color[index][tone]))
    return color[index][tone];
  };
  
  const createInputData = function (id, layer, usedColors) {
      return {
          id,
          layer,
          name: 'new_' + id,
          type: 'Image',
          color: getColor(usedColors),
          parameters: {}
      }
  }
  export { createInputData };