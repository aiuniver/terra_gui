export default {
  namespaced: true,
  state: () => ({
    project: {},
    user: {},
  }),
  mutations: {
    SET_PROJECT(state, value) {
      state.project = { ...state.project, ...value };
    },
    SET_USER(state, value) {
      state.user = { ...state.user, ...value}
    },
  },
  actions: {
    async get({ dispatch, commit }) {
      const { data } = await dispatch("axios", { url: "/config/" }, { root: true });
      console.log(data);
      if (!data) {
        return;
      }
      const { project, user, defaults: { modeling: { layers_types, layer_form }, datasets: { creation } } } = data;
      const { model } = project;
      const list = layer_form[1]['list'] || []
      commit("SET_PROJECT", project);
      commit("SET_USER", user);
      commit("modeling/SET_MODELING", { layers_types, list }, { root: true });
      commit("modeling/SET_MODEL", model, { root: true });
      commit("datasets/SET_CREATION", creation, { root: true });
    },
    async saveProject({ dispatch }, name) {
      console.log(name);
      const res = { url: "/project/name/", data: name };
      const { data } = await dispatch("axios", res, { root: true });
      if (!data) {
        return;
      }
      console.log(data);
    },
    setProject({ commit }, data) {
      commit("SET_PROJECT", data);
    },
  },
  getters: {
    getProject({ project }) {
      return project;
    },
    getUser({ user }) {
      return user;
    },
  },
};
