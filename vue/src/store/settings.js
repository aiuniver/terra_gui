export default {
  namespaced: true,
  state: () => ({
  }),
  mutations: {
  },
  actions: {
  },
  getters: {
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
