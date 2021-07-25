export default {
  namespaced: true,
  state: () => ({
    project: {},
  }),
  mutations: {
    SET_PROJECT(state, value) {
      state.project = value;
    },
  },
  actions: {
    async get({ dispatch, commit }) {
      const data = await dispatch('axios', {url: "/config/"}, {root: true});
      if (!data) {
        return;
      }
      const { project, defaults:{modeling: {layers_types} } } = data
      const list = Object.keys(layers_types).map((key) => {
        return { label: key, value: key };
      });
      commit("SET_PROJECT", project);
      commit('modeling/SET_LIST', list, { root: true }) 
      commit('modeling/SET_LAYERS', layers_types, { root: true }) 
    },
    setProject({ commit, state }, data) {
      commit("SET_PROJECT", {...state, ...data });
    },
  },
  getters: {
    getProject({ project }) {
      return project
    },
    isMobile() {
      return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
    },
    autoHeight() {
      return {
        height: (document.documentElement.clientHeight - 155) + "px",
      };
    },
    height:() => (value) => {
      return {
        height: (document.documentElement.clientHeight - value) + "px",
      };
    },
  },
};
