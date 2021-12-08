export default {
  namespaced: true,
  state: () => ({
    errors: [],
    error: {},
    state: {
      ERORR: 'Ошибка',
    }
  }),
  mutations: {
    SET_ERROR (state, value) {
      state.error = value;
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
  },
  getters: {
    getErrors: ({ errors }) => errors,
    getError: ({ error }) => error,
  },
};
