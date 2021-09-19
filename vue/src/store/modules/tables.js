export default {
  namespaced: true,
  state: () => ({
    handlers: [],
  }),
  mutations: {
    SET_PROJECT(state, value) {
      state.project = { ...state.project, ...value };
    },
    SET_HANDLERS(state, value) {
      state.handlers = [...value]
    },
  },
  actions: {
    setHandlers({ commit }, data) {
      commit("SET_HANDLERS", data);
    },
  },
  getters: {
    getHandlers({ handlers }) {
      return handlers;
    },
  },
};
