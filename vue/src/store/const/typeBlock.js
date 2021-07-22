const blocks = [
  {
    name: "input",
    title: "Input",
    fields: [
      {
        name: "Output",
        type: "event",
        attr: "output",
      },
    ],
  },
  {
    name: "sloy-one",
    title: 'Sloy one',
    fields: [
      {
        name: "Input",
        type: "event",
        attr: "input",
      },
      {
        name: "output",
        type: "event",
        attr: "output",
      },
    ],
  },
  {
    name: "sloy-two",
    title: 'Sloy two',
    fields: [
      {
        name: "Input",
        type: "event",
        attr: "input",
      },
      {
        name: "Output",
        type: "event",
        attr: "output",
      },
      {
        name: "Output",
        type: "event",
        attr: "output",
      },
    ],
  },
  {
    name: "sloy-three",
    title: 'Sloy three',
    fields: [
      {
        name: "Input",
        type: "event",
        attr: "input",
      },
      {
        name: "Output",
        type: "event",
        attr: "output",
      },
      {
        name: "Output",
        type: "event",
        attr: "output",
      },
      {
        name: "Output",
        type: "event",
        attr: "output",
      },
    ],
  },
  {
    name: "output",
    title: "Output",
    fields: [
      {
        name: "Input",
        type: "event",
        attr: "input",
      },
    ],
  },
];

const scene = {
  blocks: [
    {
      id: 1,
      position: [-900, 50],
      name: "input",
      title: "Input",
      parameters: {
        main: {
          x_cols: {
            type: "string",
            parse: "[main][x_cols]",
            default: "asda",
          },
        },
        extra: {
          x_cols: {
            type: "string",
            parse: "[extra][x_cols]",
            default: "a",
          },
        }
      }
    },
    {
      id: 2,
      position: [-900, 150],
      name: "sloy-one",
      title: "Sloy",
      parameters: {
        main: {
          x_cols: {
            type: "string",
            parse: "[main][x_cols]",
            default: "ddd",
          },
        },
        extra: {
          x_cols: {
            type: "string",
            parse: "[extra][x_cols]",
            default: "h",
          },
        }
      }
    },
    {
      id: 3,
      position: [-900, 250],
      name: "sloy-two",
      title: "Sloy",
      parameters: {
        main: {
          x_cols: {
            type: "string",
            parse: "[main][x_cols]",
            default: "",
          },
        },
        extra: {
          x_cols: {
            type: "string",
            parse: "[extra][x_cols]",
            default: "",
          },
        }
      }
    },
    {
      id: 4,
      position: [-900, 350],
      name: "sloy-three",
      title: "Sloy",
      parameters: {
        main: {
          x_cols: {
            type: "string",
            parse: "[main][x_cols]",
            default: "",
          },
        },
        extra: {
          x_cols: {
            type: "string",
            parse: "[extra][x_cols]",
            default: "",
          },
        }
      }
    },
    {
      id: 5,
      position: [-900, 450],
      name: "output",
      title: "Output",
      parameters: {
        main: {
          x_cols: {
            type: "string",
            parse: "[main][x_cols]",
            default: "",
          },
        },
        extra: {
          x_cols: {
            type: "string",
            parse: "[extra][x_cols]",
            default: "",
          },
        }
      }
    },
  ],
  links: [
    {
      id: 1,
      originID: 1,
      originSlot: 0,
      targetID: 2,
      targetSlot: 0,
    },
    {
      id: 2,
      originID: 2,
      originSlot: 1,
      targetID: 3,
      targetSlot: 0,
    },
    {
      id: 3,
      originID: 1,
      originSlot: 0,
      targetID: 3,
      targetSlot: 0,
    },
  ],
  container: {
    centerX: 1042,
    centerY: 140,
    scale: 1,
  },
};

const typeBlock = blocks.map(({ name, title}) => {
  return { value: name,  title }
});

export { blocks, typeBlock, scene }