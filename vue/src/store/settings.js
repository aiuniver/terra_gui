export default {
  namespaced: true,
  state: () => ({
    filterHeight: 32
  }),
  mutations: {
    SET_FILTER_HEIGHT(state, value) {
      state.filterHeight = value
    }
  },
  actions: {
    setFilterHeight({ commit }, value) {
      commit('SET_FILTER_HEIGHT', value)
    }
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
    getFilterHeight({ filterHeight }) {
      return filterHeight
    },
  },
};
