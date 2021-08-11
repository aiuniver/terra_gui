// const index = ['orange', 'green', 'blue', 'purple']
const color = {
  0: ['#FFB054', '#FF9D2A', '#FE8900', '#D47200', '#A95C00', '#7F4500'],
  1: ['#89D764', '#71CE45', '#5DBB31', '#4E9C29', '#3E7D21', '#2F5E18'],
  2: ['#54A3FF', '#2A8CFF', '#0075FE', '#0062D4', '#004EAA', '#003B7F'],
  3: ['#8E51F2', '#762BEF', '#6011E2', '#500EBC', '#400B96', '#300871'],
};

const getColor = () => {
  const index = Math.floor(Math.random() * 4);
  const tone = Math.floor(Math.random() * 6);
  return color[index][tone];
};

export { color, getColor };

// const list = [
//   {
//     title: 'Group',
//     options: [
//       { label: 'Text', value: 'text' },
//       { label: 'Text', value: 'text' },
//       { label: 'Text', value: 'text' },
//     ],
//   },
// ];

// const ee = [
//   {
//     type: "select",
//     name: "type",
//     label: "Тип данных",
//     parse: "type",
//     value: "Text",
//     popit: "Какой-то текст",
//     triger: {
//       Text: [
//         {
//           target: "snow",
//           value: true,
//         },
//         {
//           target: "mount",
//           value: 'Эльбрус',
//         },
//       ],
//     },
//     api: {
//       Image: '/reboot'
//     },
//     list: [
//       {
//         value: "Text",
//         label: "Text",
//       },
//       {
//         value: "Image",
//         label: "Image",
//       },
//     ],
//     filds: {
//       Text: [
//         // если значению value = Text, то рендерим эту форму
//         {
//           type: "number",
//           name: "mount",
//           label: "Гора",
//           parse: "mount",
//           value: "",
//         },
//         {
//           type: "text",
//           name: "kids",
//           label: "Дети",
//           parse: "kids",
//           popit: "Какой-то текст",
//           value: "",
//         },
//       ],
//       Image: [
//         // если значению value = Image, то рендерим эту форму
//         {
//           type: "number",
//           name: "mount",
//           label: "Гора",
//           parse: "mount",
//           popit: "Какой-то текст",
//           value: "Эверест",
//         },
//       ],
//     },
//   },
//   {
//     type: "checkbox",
//     name: "snow",
//     label: "Снег",
//     parse: "snow",
//     popit: "Какой-то текст",
//     value: false,
//     list: [],
//     filds: {
//       true: [
//         {
//           type: "text",
//           name: "kids",
//           label: "Дети",
//           parse: "kids",
//           value: "",
//         },
//       ],
//     },
//   },
// ];
