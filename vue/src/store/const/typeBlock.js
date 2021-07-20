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

const typeBlock = blocks.map(({ name, title}) => {
  return { value: name,  title }
});

export { blocks, typeBlock }