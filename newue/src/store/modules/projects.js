export default {
  namespaced: true,
  state: () => ({
    project: {},
    user: {},
  }),
  mutations: {
    SET_PROJECT (state, value) {
      state.project = { ...state.project, ...value };
    },
    SET_USER (state, value) {
      state.user = { ...state.user, ...value }
    },
  },
  actions: {
    async get ({ dispatch }) {
      const res = await dispatch("axios", { url: "/config/" }, { root: true });
      if (!res) {
        return;
      }
      const { data } = res;
      console.log(data);
      // if (!data) {
      //   return;
      // }
      // const { project: { model, training, deploy }, user, defaults: { modeling, datasets, training, cascades } } = data;

      // commit("SET_PROJECT", project);
      // commit("SET_USER", user);
      // commit("modeling/SET_MODELING", { layers_types, layer_form }, { root: true });
      // commit("modeling/SET_MODEL", model, { root: true });
      // commit("cascades/SET_CASCADES", cascades, { root: true });
      // commit("datasets/SET_CREATION", creation, { root: true });
      // commit("trainings/SET_PARAMS", base, { root: true });
      // commit("trainings/SET_ARCHITECTURE", architecture, { root: true });
      // commit("trainings/SET_CONFIG", training, { root: true });
      // if (deploy) {
      //   commit("deploy/SET_DEPLOY", deploy.data, { root: true });
      //   commit("deploy/SET_CARDS", deploy.data.data, { root: true });
      //   commit("deploy/SET_DEPLOY_TYPE", deploy.type, { root: true });
      // }
      // if (training?.result) {
      //   commit("trainings/SET_TRAIN", training.result, { root: true });
      // }
      return data
    },
    async saveNameProject ({ dispatch }, name) {
      const res = { url: "/project/name/", data: name };
      const { data } = await dispatch("axios", res, { root: true });
      if (!data) {
        return;
      }
    },
    setProject ({ commit }, data) {
      commit("SET_PROJECT", data);
    },
    async createProject ({ dispatch }, data) {
      localStorage.clear();
      await dispatch('trainings/resetAllTraining', {}, { root: true });
      const res = await dispatch("axios", { url: "/project/create/", data }, { root: true });
      document.location.href = "/"; // "Миша, все хня, давай по новой" 
      return res
    },
    async loadProject ({ dispatch }, data) {
      const res = await dispatch("axios", { url: "/project/load/", data }, { root: true });
      document.location.href = "/"; // "Миша, все хня, давай по новой, снова" 
      return res
    },
    async removeProject ({ dispatch }, data) {
      return await dispatch("axios", { url: "/project/delete/", data }, { root: true });
    },
    async infoProject ({ dispatch }, data) {
      return await dispatch("axios", { url: "/project/info/", data }, { root: true });
    },
    async saveProject ({ dispatch }, data) {
      return await dispatch("axios", { url: "/project/save/", data }, { root: true });
    },
  },
  getters: {
    getProject ({ project }) {
      return project;
    },
    getProjectData: ({ project }) => key => {
      return project[key];
    },
    getUser ({ user }) {
      return user;
    },
  },
};
