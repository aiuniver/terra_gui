export default {
  namespaced: true,
  state: () => ({
    errors: [],
    error: {},
    logs: []
  }),
  mutations: {
    SET_ERROR (state, value) {
      state.error = value;
    },
    SET_LOGS (state, value) {
      state.logs = value;
    },
    SET_ERRORS (state, value) {
      state.errors = value;
    },
  },
  actions: {
    async get ({ dispatch, commit }) {
      const { data } = await dispatch("axios", { url: "/common/logs/" }, { root: true });
      if (data) commit('SET_ERRORS', data);
      return data
    },
    setError ({ commit, state: { errors } }, value) {
      commit('SET_ERROR', value);
      commit('SET_ERRORS', [...errors, value]);
    },
    setLogs ({ commit }, value) {
      commit('SET_LOGS', value);
    },
  },
  getters: {
    getErrors: ({ errors }) => errors,
    getError: ({ error }) => error,
    getLogs: ({ logs }) => logs,
  },
};
