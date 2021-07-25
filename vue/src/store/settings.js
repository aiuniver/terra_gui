export default {
  namespaced: true,
  state: () => ({
    app: {
      name: "Terra_ai_vue",
      version: "0.0.1",
    },
    project: {},
    menus: [
      { title: "Home", icon: "mdi-view-dashboard", path: "/" },
      { title: "Datasets", icon: "mdi-account-multiple", path: "/datasets" },
      { title: "Layers", icon: "mdi-help-box", path: "/layers" },
    ],
    drawer: true,
  }),
  mutations: {
    SET_DRAWER(state, value) {
      state.drawer = value;
    },
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
      const { project } = data
      console.log(project)

      commit("SET_PROJECT", project);
      // commit("SET_TAGS", tags);
    },
    setDrawer({ commit }, data) {
      commit("SET_DRAWER", data);
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
    getDrawer(state) {
      return state.drawer;
    },
    getMenus(state) {
      return state.menus;
    },
    getApp(state) {
      return state.app;
    },
  },
};
