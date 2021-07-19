export default {
  namespaced: true,
  state: () => ({
    app: {
      name: "Terra_ai_vue",
      version: "0.0.1",
    },
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
  },
  actions: {
    setDrawer({ commit }, data) {
      commit("SET_DRAWER", data);
    },
  },
  getters: {
    isMobile() {
      return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
    },
    autoHeight() {
      return {
        height: (document.documentElement.clientHeight - 157) + "px",
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
