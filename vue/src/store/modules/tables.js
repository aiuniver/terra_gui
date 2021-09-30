export default {
  namespaced: true,
  state: () => ({
    handlers: [],
    saveCols: {}
  }),
  mutations: {
    SET_SAVE_COLS(state, value) {
      state.saveCols = { ...value };
    },
    SET_HANDLERS(state, value) {
      state.handlers = [...value]
    },
  },
  actions: {
    resetTable({ commit }) {
      commit("SET_HANDLERS", []);
      commit("SET_SAVE_COLS", {});
    },
    setHandlers({ commit }, data) {
      commit("SET_HANDLERS", data);
    },
    setSaveCols({ commit, state }, { id, value }) {
      state.saveCols[id] = value
      commit("SET_SAVE_COLS", state.saveCols);
    },
  },
  getters: {
    getHandlers({ handlers }) {
      return handlers;
    },
    getSaveCols: (state) => id => {
      return state?.saveCols?.[id] || [];
    },
  },
};
