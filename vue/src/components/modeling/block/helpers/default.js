const typeBlock = [
  {
    group: "input",
    inputs: [],
    outputs: [{}],
  },
  {
    group: "middle",
    inputs: [{}],
    outputs: [{}],
  },
  {
    group: "output",
    inputs: [{}],
    outputs: [],
  },
];

const createBlock = function(type, id) {
  // console.log(type, id)
  if (!type || !id) {
    return null;
  }
  const node = typeBlock.find((n) => {
    return n.group === type;
  });
  if (!node) {
    return null;
  }
  return {
    id: id,
    name: "block",
    type: "",
    group: type,
    bind: {
      up: [],
      down: [],
    },
    shape: {
      input: [],
      output: [],
    },
    location: null,
    position: [0, 0],
    parameters: {
      main: {},
      extra: {},
    },
    reference: null,
    selected: false,
    inputs: node.inputs,
    outputs: node.outputs,
  };
};

const prepareBlocks = function(blocks) {
  let last = 0;
  const newBlock = blocks.map((block) => {
      let newBlock = createBlock(block.group, block.id);
      if (!newBlock) {
        console.warn("block not create: " + block)
        return;
      }
      const x = 0; // (this.$el.clientWidth / 2 - this.centerX) / this.scale;
      const y = 0; //(this.$el.clientHeight / 2 - this.centerY) / this.scale;

      newBlock = { ...newBlock, ...block };
      // console.log(newBlock.position);
      if (!newBlock.position) {
        newBlock.position = [x + last, y + last];
        last = last + 20;
      }
      return newBlock;
    })
    .filter((b) => {
      return !!b;
    });
  return JSON.parse(JSON.stringify(newBlock));
};

const prepareLinks = function(blocks) {
  let links = [];

  blocks.forEach(({ bind }) => {
    console.log(bind)        
  });
  return links      
};

export { typeBlock, prepareBlocks, createBlock, prepareLinks };
