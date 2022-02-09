
const types = [
  { type: 'data', color: '#ffda60', typeBlock: 'input'},
  { type: 'handler', color: '#45CF81', typeBlock: 'middle' },
  { type: 'input', color: '#FFAC60', typeBlock: 'output' },
  { type: 'output', color: '#7984D7', typeBlock: 'output' },
]



const Block = class {
  constructor({ id, name = 'Block', type = 'input', position = [0, 0], selected = false, color = '#6c7883', typeBlock = 'middle' }) {
    this.id = id
    this.name = name
    this.type = type
    this.color = color
    this.typeBlock = typeBlock
    this.position = position
    this.selected = selected
    this.bind = { up: [], down: [] }
    this.parameters = {}
  }
};


const getBlock = (opt) => {
  const type = types.find(i => i.type === opt.type) || {}
  return new Block({ ...opt, ...type })
}


export { getBlock, types };
