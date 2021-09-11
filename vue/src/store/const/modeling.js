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

const createBlock = function (group, id, typeLayers, list, [x, y] = []) {
  if (!group || !id) {
    return null;
  }
  const node = typeBlock.find(n => n.group === group);
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
    type: labelType[0].value,
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
    position: [x ?? -90, y ?? 0],
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

const addParamsBlock = function (block, list) {
  const node = typeBlock.find(n => n.group === block.group);
  const labelType = list.filter(item => item.value === block.type)
  return {
    typeLabel: labelType[0].label,
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
  const newBlock = JSON.parse(JSON.stringify(block))
  console.log(id)
  return { ...newBlock, ...{ id }, ...{ name: ~newBlock.name.indexOf('(Clone)') ? newBlock.name : newBlock.name + " (Clone)" } };
};

const prepareBlocks = function (blocks, list) {
  let last = 0;
  const newBlock = blocks.map(block => {
    let newBlock = addParamsBlock(block, list);
    if (!newBlock) {
      console.warn('block not create: ' + block);
      return;
    }
    newBlock = { ...newBlock, ...block };
    if (!newBlock.position) {
      newBlock.position = [-90, last];
      last = last + 60;
    }
    return newBlock;
  }).filter(block => !!block);
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
