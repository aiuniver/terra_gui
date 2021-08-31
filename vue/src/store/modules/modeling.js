import { prepareBlocks, prepareLinks, createBlock, cloneBlock } from '../const/modeling';

export default {
  namespaced: true,
  state: () => ({
    select: null,
    model: {},
    blocks: [],
    errorsBlocks: {},
    links: [],
    modeling: {
      list: [],
      layers_types: {},
    },
    buttons: {
      save: false,
      clone: false,
    },
  }),
  mutations: {
    SET_MODELING(state, value) {
      state.modeling = { ...value };
    },
    SET_ERRORS_BLOCKS(state, value) {
      state.errorsBlocks = { ...value }
    },
    SET_MODEL(state, value) {
      state.model = value;
      const { layers } = value;
      state.blocks = prepareBlocks(layers);
      state.links = prepareLinks(layers);
    },
    SET_BLOCKS(state, value) {
      state.blocks = [...value];
    },
    SET_LINKS(state, value) {
      state.links = [...value];
    },
    SET_LIST(state, value) {
      state.list = [...value];
    },
    SET_SELECT(state, value) {
      state.select = value;
    },
    SET_BUTTONS(state, value) {
      state.buttons = { ...state.buttons, ...value };
    },
  },
  actions: {
    addBlock({ dispatch, commit, state: { blocks, modeling: { layers_types } } }, type) {
      let maxID = Math.max(0, ...blocks.map(o => o.id));
      let block = createBlock(type, maxID + 1, layers_types);
      if (!block) return;
      blocks.push(block);
      dispatch('updateModel');
      commit('SET_BLOCKS', blocks);
    },
    cloneBlock({ dispatch, commit, state: { blocks } }, oldBlock) {
      let maxID = Math.max(0, ...blocks.map(o => o.id));
      const block = cloneBlock(oldBlock, maxID + 1);
      if (!block) return;
      blocks.push(block);
      commit('SET_BLOCKS', blocks);
      dispatch('updateModel');
    },
    selectBlock({ commit, state: { blocks } }, block) {
      blocks.forEach(item => {
        item.selected = item.id === block.id
      })
      commit('SET_BLOCKS', blocks);
      commit('SET_SELECT', block.id);
    },
    deselectBlocks({ commit, state: { blocks } }) {
      blocks.forEach(item => {
        item.selected = false
      })
      commit('SET_BLOCKS', blocks);
      commit('SET_SELECT', null);
    },
    removeBlock({ dispatch, commit, state: { blocks } }, block) {
      if (block.selected) {
        block.selected = false;
      }
      dispatch('removeLinkToBlock', block);
      commit('SET_BLOCKS', blocks.filter(b => b.id !== block.id));
      dispatch('updateModel');
    },
    removeLink({ commit, state: { links } }, id) {
      console.log(id)
      commit('SET_LINKS', links.filter(value => value.id !== id));
    },
    removeLinkToBlock({ dispatch, commit, state: { links } }, block) {
      console.log(block)
      commit('SET_LINKS', links.filter(link => (link.originID !== block.id && link.targetID !== block.id)));
      dispatch('updateModel');
    },

    async info({ dispatch }, value) {
      return await dispatch('axios', { url: '/modeling/info/', data: value }, { root: true });
    },
    async load({ dispatch }, value) {
      const { data: model } = await dispatch('axios', { url: '/modeling/load/', data: value }, { root: true });
      if (model) {
        await dispatch('projects/get', {}, { root: true });
      }
      return model;
    },
    async createModel({ dispatch }, data) {
      return await dispatch('axios', { url: '/modeling/create/', data }, { root: true });
    },
    async removeModel({ dispatch }, data) {
      return await dispatch('axios', { url: '/modeling/delete/', data }, { root: true });
    },
    async updateModel({ commit, state: { blocks, links }, dispatch }) {
      blocks.forEach(block => {
        block.bind.up = links
          .map(link => {
            return link.targetID === block.id ? link.originID : null;
          })
          .filter(link => link);
        block.bind.down = links
          .map(link => {
            return link.originID === block.id ? link.targetID : null;
          })
          .filter(link => link);
      });
      commit('SET_BUTTONS', { save: false });
      return await dispatch('axios', { url: '/modeling/update/', data: { layers: blocks } }, { root: true });
    },
    async getModel({ dispatch }, value) {
      return await dispatch('axios', { url: '/modeling/get/', data: value }, { root: true });
    },
    async clearModel({ commit, dispatch }) {
      const res = await dispatch('axios', { url: '/modeling/clear/' }, { root: true });
      if (res.success) {
        console.log(res)
        commit('SET_ERRORS_BLOCKS', {})
        await dispatch('projects/get', {}, { root: true });
      }
      return res
    },
    async validateModel({ commit, dispatch }) {
      const { data } = await dispatch('axios', { url: '/modeling/validate/' }, { root: true });
      if (data) {
        commit('SET_ERRORS_BLOCKS', data)
      }
      return data;
    },
    setBlocks({ commit }, value) {
      commit('SET_BLOCKS', value);
    },
    setLinks({ commit }, value) {
      commit('SET_LINKS', value);
    },
    setBlock({ commit, state: { blocks } }, value) {
      const index = blocks.findIndex(item => item.id == value.id);
      blocks[index] = value;
      console.log(blocks);
      commit('SET_BLOCKS', blocks);
    },
    setButtons({ commit }, value) {
      commit('SET_BUTTONS', value);
    },
  },
  getters: {
    getList: ({ modeling: { list } }) => list,
    getLayersType: ({ modeling: { layers_types } }) => layers_types,
    getModel: ({ model }) => model,
    getBlocks: ({ blocks }) => blocks,
    getErrorsBlocks: ({ errorsBlocks }) => errorsBlocks,
    getLinks: ({ links }) => links,
    getSelect: ({ select }) => select,
    getButtons: ({ buttons }) => buttons,
    getBlock: ({ select, blocks }) => {
      const id = blocks.findIndex(item => item.id == select);
      return blocks[id] || {};
    },
  },
};
