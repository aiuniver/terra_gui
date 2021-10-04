export default {
  namespaced: true,
  state: () => ({
    errors: [],
  }),
  mutations: {
    SET_ERROR (state, value) {
      state.errors = [...state.errors, value];
    },
  },
  actions: {
    setError ({ commit }, value) {
      commit('SET_ERROR', { date: Date.now(), error: value });
    },
  },
  getters: {
    getErrors: ({ errors }) => errors,
  },
};
