export default {
  namespaced: true,
  state: () => ({
    errors: [],
    error: {},
    warning: []
  }),
  mutations: {
    SET_ERROR (state, value) {
      state.error = value;
    },
    SET_WARNING (state, value) {
      state.warning = value;
    },
    SET_ERRORS (state, value) {
      state.errors = value;
    },
  },
  actions: {
    setError ({ commit, state: { errors } }, value) {
      commit('SET_ERROR', value);
      commit('SET_ERRORS', [...errors, value]);
    },
    setWarning ({ commit }, value) {
      commit('SET_WARNING', value);
    },
  },
  getters: {
    getErrors: ({ errors }) => errors,
    getError: ({ error }) => error,
    getWarning: ({ warning }) => warning,
  },
};
