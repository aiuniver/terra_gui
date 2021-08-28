import { prepareBlocks, prepareLinks } from '@/components/modeling/block/helpers/default';

export default {
  namespaced: true,
  state: () => ({
    select: null,
    model: {},
    blocks: [],
    links: [],
    modeling: {
      list: [],
      layers_types: {},
    },
    buttons: {
      save: false,
      clone: false
    }
  }),
  mutations: {
    SET_MODELING(state, value) {
      state.modeling = { ...value };
    },
    SET_MODEL(state, value) {
      state.model = value;
      const { layers } = value;
      // console.log(layers);
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
      state.buttons = {...state.buttons, ...value };
    },
  },
  actions: {
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
    async saveModel({ commit, state: { blocks, links }, dispatch }) {
      blocks.forEach(block => {
        block.bind.up = links.map(link => {
          return link.targetID === block.id ? link.originID : null
        }).filter(link => link)
        block.bind.down = links.map(link => {
          return link.originID === block.id ? link.targetID : null
        }).filter(link => link)
      })
      commit('SET_BUTTONS', { save: false});
      return await dispatch('axios', { url: '/modeling/update/', data: { layers: blocks } }, { root: true });
    },
    async getModel({ dispatch }, value) {
      return await dispatch('axios', { url: '/modeling/get/', data: value }, { root: true });
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
    setSelect({ commit }, value) {
      commit('SET_SELECT', value);
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
    getLinks: ({ links }) => links,
    getSelect: ({ select }) => select,
    getButtons: ({ buttons }) => buttons,
    getBlock: ({ select, blocks }) => {
      const id = blocks.findIndex(item => item.id == select);
      return blocks[id];
    },
  },
};
