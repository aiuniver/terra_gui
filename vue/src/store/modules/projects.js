export default {
  namespaced: true,
  state: () => ({
    project: {},
    user: {},
    projectsList: []
  }),
  mutations: {
    SET_PROJECT (state, value) {
      state.project = { ...state.project, ...value };
    },
    SET_USER (state, value) {
      state.user = { ...state.user, ...value }
    },
    SET_PROJECTS_LIST (state, value) {
      state.projectsList = value;
    },
  },
  actions: {
    async get ({ dispatch, commit }) {
      const res = await dispatch("axios", { url: "/config/" }, { root: true });
      if (!res) {
        return;
      }
      const { data } = res;
      if (!data) {
        return;
      }
      const { project, user, defaults: { modeling: { layers_types, layer_form }, datasets, training: form, cascades: formsCascades, deploy } } = data;
      const { model, training, cascade } = project;

      commit("SET_PROJECT", project);
      commit("SET_USER", user);
      commit("modeling/SET_MODELING", { layers_types, layer_form }, { root: true });
      commit("modeling/SET_MODEL", model, { root: true });
      commit("cascades/SET_CASCADES", formsCascades, { root: true });
      commit("cascades/SET_MODEL", cascade, { root: true });
      // commit("datasets/SET_CREATION", creation, { root: true });
      // commit("create/SET_DEFAULT", datasets, { root: true });
      commit("createDataset/SET_DEFAULT", datasets, { root: true });
      dispatch("trainings/parseStruct", { ...training, form }, { root: true });
      dispatch("deploy/parseStruct", { form: deploy }, { root: true });
      if (training?.deploy) {
        commit("deploy/SET_DEPLOY", training.deploy.data, { root: true });
        commit("deploy/SET_CARDS", training.deploy.data.data, { root: true });
        commit("deploy/SET_DEPLOY_TYPE", training.deploy.type, { root: true });
      }
      return data
    },
    async progress ({ dispatch }, data) {
      const res = await dispatch('axios', { url: '/project/load/progress/', data }, { root: true });
      return res
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
    async load ({ dispatch }, data) {
      const res = await dispatch("axios", { url: "/project/load/", data }, { root: true });
      // document.location.href = "/"; // "Миша, все хня, давай по новой, снова" 
      return res
    },
    async remove ({ dispatch }, data) {
      return await dispatch("axios", { url: "/project/delete/", data }, { root: true });
    },
    async infoProject ({ dispatch, commit }, params) {
      const { data: { projects } } = await dispatch("axios", { url: "/project/info/", params }, { root: true });
      const list = (projects || []).map((p, i) => {
        return {
          id: p.id || i + 1,
          image: p.image || 'https://www.zastavki.com/pictures/1920x1080/2013/Fantasy__038385_23.jpg',
          active: p.active || false,
          created: p.created || '17 апреля 2021',
          edited: p.edited || '3 дня назад',
          value: p.value,
          label: p.label,
        }
      })
      commit('SET_PROJECTS_LIST', list)
      return list
    },
    async saveProject ({ dispatch }, data) {
      const res = await dispatch("axios", { url: "/project/save/", data }, { root: true });
      if (!res?.error) await dispatch("get", {});
      return res
    },
  },
  getters: {
    getProject: ({ project }) => project,
    getProjectsList: ({ projectsList }) => projectsList,
    getProjectData: ({ project }) => key => project[key],
    getUser: ({ user }) => user,
  },
};
