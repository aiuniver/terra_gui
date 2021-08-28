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
    let linksID = 0
    blocks.forEach(({ id, bind }) => {
      // console.log(id)
      // console.log(bind)
      if (bind?.down && Array.isArray(bind.down)) {
        const arr = bind.down
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
        })
        
      }
  
  
    });
    // console.log(links)
    return links      
  };
  
  const getOffsetRect = function (element) {
    let box = element.getBoundingClientRect()
  
    let scrollTop = window.pageYOffset
    let scrollLeft = window.pageXOffset
  
    let top = box.top + scrollTop
    let left = box.left + scrollLeft
  
    return {top: Math.round(top), left: Math.round(left)}
  }
  
  const mouseHelper = function (element, event) {
    let mouseX = event.pageX || event.clientX + document.documentElement.scrollLeft
    let mouseY = event.pageY || event.clientY + document.documentElement.scrollTop
  
    let offset = getOffsetRect(element)
    let x = mouseX - offset.left
    let y = mouseY - offset.top
  
    return {
      x: x,
      y: y
    }
  }
  
  export { typeBlock, prepareBlocks, createBlock, prepareLinks, mouseHelper };
  