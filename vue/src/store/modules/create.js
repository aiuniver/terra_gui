import { createBlock, setLinks, getLinks, preloadBlock } from '../const/blocks';

export default {
  namespaced: true,
  state: () => ({
    inputs: [
      createBlock({ id: 1, type: 'data', position: [0, 0], bind: { up: [], down: [2, 3] } }),
      createBlock({ id: 2, type: 'handler', position: [0, 100], bind: { up: [1], down: [3] } }),
      createBlock({ id: 3, type: 'handler', position: [200, 100], bind: { up: [2, 1], down: [] } })
    ],
    outputs: [],
    blocks: [],
    links: [],
    key: {},
    // creation: {},
    // default: {}
  }),
  mutations: {
    SET_INPUT_AND_OUTPUT (state, { inputs, outputs }) {
      const input = preloadBlock(inputs, 'input');
      state.blocks = [...input]
      state.inputs = [...input];
      state.outputs = preloadBlock(outputs, 'output');
    },
    SET_BLOCKS (state, value) {
      state.blocks = value;
    },
    SET_LINKS (state, value) {
      state.links = value;
      state.blocks = setLinks(state.blocks, value)
    },
    SET_KEY_EVENT (state, value) {
      state.key = value;
    },
    // SET_DEFAULT (state, value) {
    //   state.default = value;
    // },
    SET_OUTPUT (state, value) {
      state.outputs = value;
    },
    SET_INTPUT (state, value) {
      state.inputs = value;
    },
  },
  actions: {
    // BLOCKS____________________________________________________

    main ({ commit, state: { blocks, inputs, outputs } }, { value, old }) {
      if (value === 2 && old === 1) {
        commit('SET_BLOCKS', [...inputs]);
      }
      if (value === 3 && old === 2) {

        // commit('SET_INTPUT', [...blocks]);
        // const data = blocks.filter(i => i.type === 'data').map(i => {
        //   return { ...i, bind: { up: [], down: [] }, selected: false }
        // })
        // console.log(data)
        // const newOutputs = outputs.filter(i => i.created !== 'input')
        // commit('SET_BLOCKS', [...newOutputs]);
        // data.forEach(b => {
        //   dispatch('clone', b);
        // })


        commit('SET_INTPUT', [...blocks]);
        commit('SET_BLOCKS', [...outputs]);
      }
      if (value === 4 && old === 3) {
        commit('SET_OUTPUT', [...blocks]);
        commit('SET_BLOCKS', []);
      }
      if (value === 3 && old === 4) {
        commit('SET_BLOCKS', [...outputs]);
      }
      if (value === 2 && old === 3) {
        commit('SET_OUTPUT', [...blocks]);
        commit('SET_BLOCKS', [...inputs]);
      }
      if (value === 1 && old === 2) {
        commit('SET_INTPUT', [...blocks]);
        commit('SET_BLOCKS', []);
      }
    },

    setParameters ({ commit, state: { blocks } }, { parameters, id }) {
      console.log(parameters, id)
      const newBlocks = blocks.map(i => {
        if (i.id === id) {
          i.parameters = parameters
        }
        return i
      })
      commit('SET_BLOCKS', [...newBlocks]);
    },

    editBlock ({ commit, state: { blocks } }, block) {
      const newBlocks = blocks.map(i => {
        return (i.id === block.id) ? block : i
      })
      commit('SET_BLOCKS', [...newBlocks]);
    },

    add ({ commit, state: { blocks }, rootState: { createDataset: { pagination } } }, { type, position }) {
      const created = pagination === 2 ? 'input' : 'output'
      console.log(created)
      const id = Math.max(0, ...blocks.map(o => o.id)) + 1;
      const block = createBlock({ id, type, position, selected: true, created })
      commit('SET_BLOCKS', [...blocks, block]);
    },

    update ({ commit }, blocks) {
      commit('SET_BLOCKS', [...blocks]);
    },

    remove ({ dispatch, commit, state: { blocks } }, block) {
      const all = block ? [block.id] : blocks.filter(b => b.selected).map(b => b.id)
      dispatch('removeLinkToBlock', all);
      commit('SET_BLOCKS', [...blocks.filter(b => !all.includes(b.id))]);
    },

    clone ({ commit, state: { blocks } }, oldBlock) {
      let id = Math.max(0, ...blocks.map(o => o.id)) + 1;
      const block = JSON.parse(JSON.stringify(oldBlock))
      block.id = id
      const [x, y] = block.position
      block.position = [x + 5, y + 5]
      commit('SET_BLOCKS', [...blocks, block]);
    },

    cloneAll ({ dispatch, state: { blocks } }) {
      const all = blocks.filter(b => b.selected)
      dispatch('deselect');
      all.forEach(b => {
        dispatch('clone', b);
      })

    },

    select ({ commit, state: { blocks, key: { ctrlKey } } }, { id }) {
      const update = blocks.map((b) => {
        const selected = !ctrlKey ? b.id === id : (b.id === id) ? !b.selected : b.selected
        return { ...b, selected }
      })
      commit('SET_BLOCKS', [...update]);
    },

    deselect ({ commit, state: { blocks } }, value = false) {
      const update = blocks.map(b => {
        const selected = value
        return { ...b, selected }
      })
      commit('SET_BLOCKS', [...update]);
    },

    align ({ commit, state: { blocks } }, code) {
      let tempX = null;
      let tempY = null;
      blocks.filter(i => i.selected).forEach(i => {
        const [x, y] = i.position
        if (code === 'ArrowLeft') {
          if (!tempX || tempX > x) tempX = x
        }
        if (code === 'ArrowRight') {
          if (!tempX || tempX < x) tempX = x
        }
        if (code === 'ArrowUp') {
          if (!tempY || tempY > y) tempY = y
        }
        if (code === 'ArrowDown') {
          if (!tempY || tempY < y) tempY = y
        }
      })
      const update = blocks.map(b => {
        if (b.selected) {
          if (['ArrowLeft', 'ArrowRight'].includes(code)) {
            b.position[0] = tempX
          }
          if (['ArrowUp', 'ArrowDown'].includes(code)) {
            b.position[1] = tempY
          }
        }
        return { ...b }
      })
      commit('SET_BLOCKS', [...update]);
      // console.log(update)
    },


    distance ({ commit, state: { blocks } }) {
      let min = null;
      let max = null;
      let number = 0;
      blocks.filter(i => i.selected).forEach((i, index) => {
        const [x, y] = i.position
        if (!min || min < y) min = y
        if (!max || max > y) max = y
        number = ++index
        console.log(x, y)
      })

      const length = Math.sqrt((max - min) ** 2)
      const delta = length / number
      let start = max
      const update = blocks.map(b => {
        if (b.selected) {
          b.position[1] = start
          start = start + delta
        }
        return { ...b }
      })
      commit('SET_BLOCKS', [...update]);
    },

    // LINKS__________________________________________

    addLink ({ commit, state: { links } }, link) {
      commit('SET_LINKS', [...links, link]);
    },
    updateLink ({ commit }, links) {
      commit('SET_LINKS', [...links]);
    },
    removeLink ({ commit, state: { links } }, id) {
      commit('SET_LINKS', [...links.filter(value => value.id !== id)]);
    },
    removeLinkToBlock ({ commit, state: { links } }, arr) {
      commit('SET_LINKS', [...links.filter(link => (!arr.includes(link.originID) && !arr.includes(link.targetID)))]);
    },

    setKeyEvent ({ commit }, value) {
      commit('SET_KEY_EVENT', value);
    },
  },
  getters: {
    // getDefault: ({ creation }) => (key) => creation[key] || [],
    getBlocks: ({ blocks }) => blocks,
    getKeyEvent: ({ key }) => key,
    getLinks: ({ blocks }) => getLinks(blocks),
    getBlock: ({ blocks }) => id => {
      const index = blocks.findIndex(item => item.id == id);
      return blocks[index] || {};
    },
    getSelected: ({ blocks }) => blocks.filter(i => i.selected),
  },
};
