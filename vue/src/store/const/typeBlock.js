// const blocks = [
//   {
//     group: "input",
//     title: "Input",
//     fields: [
//       {
//         name: "Output",
//         type: "event",
//         attr: "output",
//       },
//     ],
//   },
//   {
//     group: "middle",
//     title: "Sloy",
//     fields: [
//       {
//         name: "Input",
//         type: "event",
//         attr: "input",
//       },
//       {
//         name: "output",
//         type: "event",
//         attr: "output",
//       },
//     ],
//   },
//   {
//     group: "output",
//     title: "Output",
//     fields: [
//       {
//         name: "Input",
//         type: "event",
//         attr: "input",
//       },
//     ],
//   },
// ];

const scene = {
  blocks: [
    {
      id: 1,
      position: [-900, 50],
      name: "input",
      type: "Input",
      parameters: {
        extra: { name: "input_1" },
        main: {},
      },
    },
    {
      id: 2,
      position: [-900, 150],
      name: "middle",
      type: "BatchNormalization",
      parameters: {
        main: {},
        extra: {
          axis: -1,
          momentum: 0.99,
          epsilon: 0.001,
          center: true,
          scale: true,
          beta_initializer: "zeros",
          gamma_initializer: "ones",
          moving_mean_initializer: "zeros",
          moving_variance_initializer: "ones",
          beta_regularizer: "",
          gamma_regularizer: "",
          beta_constraint: "",
          gamma_constraint: "",
        },
      },
    },
    {
      id: 3,
      position: [-900, 250],
      name: "output",
      type: "Dense",
      group: "output",
      bind: [],
      shape: [1, 1, 1],
      location: null,
      parameters: {
        main: {
          units: 100,
          activation: "softmax",
        },
        extra: {
          use_bias: true,
          kernel_initializer: "glorot_uniform",
          bias_initializer: "zeros",
          kernel_regularizer: "",
          bias_regularizer: "",
          activity_regularizer: "",
          kernel_constraint: "",
          bias_constraint: "",
          name: "output_1",
        },
      },
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
      originSlot: 0,
      targetID: 3,
      targetSlot: 0,
    },

  ],

};

// const typeBlock = blocks.map(({ name, title }) => {
//   return { value: name, title };
// });

export {  scene };
