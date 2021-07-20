import { blocks, typeBlock } from './const/typeBlock'
export default {
  namespaced: true,
  state: () => ({
    dialog: false,
    toolbarEvent:  {}
  }),
  mutations: {
    SET_DIALOG(state, value) {
      state.dialog = value;
    },
    SET_TOOLBAR_EVENT(state, value) {
      state.toolbarEvent = value;
    },
  },
  actions: {
    setDialog({ commit }, value) {
      commit("SET_DIALOG", value);
    },
    setToolbarEvent({ commit }, value) {
      const { event } = value
      if (event === 'load') {
        commit("SET_DIALOG", true);
      }
      commit("SET_TOOLBAR_EVENT", { ...value } );
    },
  },
  getters: {
    getDialog: ({ dialog }) => dialog,
    getToolbarEvent: ({ toolbarEvent }) => toolbarEvent,
    getTypeBlock: () => typeBlock,
    getBlocks: () => blocks,
  },
};
