const Block = class {
  constructor({ id, name = 'Block', type = 'input', position = [0, 0], selected = false }) {
    this.id = id
    this.name = name
    this.type = type
    this.position = position
    this.selected = selected
    this.inputs = [{}]
    this.outputs = [{}]
    this.bind = { up: [], down: [] }
    this.parameters = {}
  }
};

export { Block };
