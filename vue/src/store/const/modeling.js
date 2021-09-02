const typeBlock = [
  {
    group: 'input',
    name: 'Вход ',
    type: 'Input',
    inputs: [],
    outputs: [{}],
  },
  {
    group: 'middle',
    name: 'Layer ',
    type: 'Conv2D',
    inputs: [{}],
    outputs: [{}],
  },
  {
    group: 'output',
    name: 'Выход  ',
    type: 'Dense',
    inputs: [{}],
    outputs: [],
  },
];

const createBlock = function (type, id, typeLayers, list) {
  // console.log(type, id)
  if (!type || !id) {
    return null;
  }
  const node = typeBlock.find(n => {
    return n.group === type;
  });

  // console.log(list)
  const labelType = list.filter(item => item.value === node.type)
  // console.log(list)
  const mainArr = typeLayers?.[node.type]?.main || []
  const extraArr = typeLayers?.[node.type]?.extra || []
  const main = {}
  const extra = {}
  mainArr.forEach(({ name, value }) => {
    main[name] = value === '__null__' ? null : value
  })
  extraArr.forEach(({ name, value }) => {
    extra[name] = value === '__null__' ? null : value
  })

  if (!node) {
    return null;
  }
  return {
    id: id,
    name: node.name + id,
    type: node.type,
    typeLabel: labelType[0].label,
    group: node.group,
    bind: {
      up: [],
      down: [],
    },
    shape: {
      input: [],
      output: [],
    },
    position: [0, 0],
    parameters: {
      main,
      extra,
    },
    reference: null,
    selected: false,
    inputs: node.inputs,
    outputs: node.outputs,
  };
};

const changeTypeBlock = function (type, block, typeLayers, list) {
  // console.log(type, id)
  if (!type || !block) {
    return null;
  }
  // console.log(type)
  const labelType = list.filter(item => item.value === type)
  const mainArr = typeLayers?.[type]?.main || []
  const extraArr = typeLayers?.[type]?.extra || []
  const main = {}
  const extra = {}
  mainArr.forEach(({ name, value }) => {
    main[name] = value === '__null__' ? null : value
  })
  extraArr.forEach(({ name, value }) => {
    extra[name] = value === '__null__' ? null : value
  })
  block.type = type,
  block.typeLabel = labelType[0].label,
  block.parameters = {
    main,
    extra,
  }
  return block
};

const cloneBlock = function (block, id) {
  return { ...block, ...{ id }, ...{ name: block.name + '(clone)' } };
};

const prepareBlocks = function (blocks,typeLayers, list) {
  // console.log(list)
  let last = 0;
  const newBlock = blocks
    .map(block => {
      let newBlock = createBlock(block.group, block.id, typeLayers, list);
      if (!newBlock) {
        console.warn('block not create: ' + block);
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
    .filter(b => {
      return !!b;
    });
  return JSON.parse(JSON.stringify(newBlock));
};

const prepareLinks = function (blocks) {
  let links = [];
  let linksID = 0;
  blocks.forEach(({ id, bind }) => {
    // console.log(id)
    // console.log(bind)
    if (bind?.down && Array.isArray(bind.down)) {
      const arr = bind.down;
      arr.forEach(item => {
        if (item) {
          links.push({
            id: ++linksID,
            originID: id,
            originSlot: 0,
            targetID: item,
            targetSlot: 0,
          });
        }
      });
    }
  });
  // console.log(links)
  return links;
};

const getOffsetRect = function (element) {
  let box = element.getBoundingClientRect();

  let scrollTop = window.pageYOffset;
  let scrollLeft = window.pageXOffset;

  let top = box.top + scrollTop;
  let left = box.left + scrollLeft;

  return { top: Math.round(top), left: Math.round(left) };
};

const mouseHelper = function (element, event) {
  let mouseX = event.pageX || event.clientX + document.documentElement.scrollLeft;
  let mouseY = event.pageY || event.clientY + document.documentElement.scrollTop;

  let offset = getOffsetRect(element);
  let x = mouseX - offset.left;
  let y = mouseY - offset.top;

  return {
    x: x,
    y: y,
  };
};

export { typeBlock, prepareBlocks, createBlock, prepareLinks, mouseHelper, cloneBlock, changeTypeBlock };
