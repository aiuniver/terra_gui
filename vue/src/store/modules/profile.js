export default {
  namespaced: true,
  state: () => ({
    color: 'success',
    message: '',
  }),
  mutations: {
    SET_COLOR(state, value) {
      state.color = value;
    },
    SET_MESSAGE(state, value) {
      state.message = value;
    },

  },
  actions: {
    async save({ dispatch }, data) {
        return await dispatch("axios", { url: "/profile/save/", data }, { root: true });
    },
  },
  getters: {
    getProgress: ({ progress }) => progress,
  },
};
