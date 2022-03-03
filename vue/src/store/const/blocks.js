
const types = [
  { type: 'data', color: '#ffda60', typeBlock: 'input' },
  { type: 'handler', color: '#45CF81', typeBlock: 'middle' },
  { type: 'input', color: '#FFAC60', typeBlock: 'output' },
  { type: 'output', color: '#7984D7', typeBlock: 'output' },
]



const Block = class {
  constructor({
    id,
    name = 'Block',
    type = 'input',
    position = [0, 0],
    selected = false,
    color = '#6c7883',
    created = 'input',
    removable = true,
    typeBlock = 'middle',
    bind = { up: [], down: [] },
    parameters = {}
  }) {
    this.id = id
    this.name = name
    this.type = type
    this.color = color
    this.typeBlock = typeBlock
    this.position = position
    this.created = created
    this.removable = removable
    this.selected = selected
    this.bind = bind
    this.parameters = parameters
  }
};


const createBlock = (opt) => {
  const type = types.find(i => i.type === opt.type) || {}
  return new Block({ ...opt, ...type })
}

const chengeParametrs = ({ type = '', options = {} }) => {
  return { ...options, type }
}

const preloadBlock = (arr, type) => {
  return arr.map(i => {
    i.created = type
    // if (i.type === 'layer') i.type = type

    if (i.type === 'handler') i.parameters = chengeParametrs(i.parameters)
    console.log(i.parameters)
    return createBlock(i)
  })
}

const setLinks = (blocks, links) => {
  const update = JSON.parse(JSON.stringify(blocks))
  update.forEach(block => {
    block.bind.up = links.map(link => {
      return link.targetID === block.id ? link.originID : null;
    })
      .filter(link => link);
    block.bind.down = links.map(link => {
      return link.originID === block.id ? link.targetID : null;
    })
      .filter(link => link);
  });
  return update
}

const getLinks = function (blocks) {
  let links = [];
  let linksID = 0;
  blocks.forEach(({ id, bind }) => {
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

export { createBlock, types, setLinks, getLinks, preloadBlock };
