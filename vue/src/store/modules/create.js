import { Block } from '../const/blocks';

export default {
  namespaced: true,
  state: () => ({
    blocks: [new Block({ id: 1 }), new Block({ id: 2, type: 'middle', position: [20, 20] }), new Block({ id: 3, type: 'handler', position: [40, 40] }), new Block({ id: 4, type: 'output', position: [60, 60] })],
    links: [],
    key: {},
    creation: {},
  }),
  mutations: {
    SET_BLOCKS (state, value) {
      state.blocks = value;
    },
    SET_LINKS (state, value) {
      state.links = value;
    },
    SET_KEY_EVENT (state, value) {
      state.key = value;
    },
    SET_CREATION (state, value) {
      state.creation = value;
    },
  },
  actions: {
    // BLOCKS____________________________________________________
    add ({ commit, state: { blocks } }, { type, position }) {
      const id = Math.max(0, ...blocks.map(o => o.id)) + 1;
      const block = new Block({ id, type, position, selected: true })
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
      all.forEach(b => {
        dispatch('clone', b);
      })
      dispatch('deselect');
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
    getDefault: ({ creation }) => (key) => creation[key] || [],
    getBlocks: ({ blocks }) => blocks,
    getKeyEvent: ({ key }) => key,
    getLinks: ({ links }) => links,
    getBlock: ({ blocks }) => id => {
      const index = blocks.findIndex(item => item.id == id);
      return blocks[index] || {};
    },
    getSelectedLength: ({ blocks }) => blocks.filter(i => i.selected).length,
    getSelected: ({ blocks }) => blocks.find(i => i.selected),
  },
};
