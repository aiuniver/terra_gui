export default {
  namespaced: true,
  state: () => ({
    filterHeight: 32,
    height: document.documentElement.clientHeight,
    wigth: document.documentElement.clientWidth
  }),
  mutations: {
    SET_FILTER_HEIGHT(state, value) {
      state.filterHeight = value
    },
    SET_ALL_HEIGHT(state, value) {
      state.height = value
    },
    SET_ALL_WIGTH(state, value) {
      state.wigth = value
    }
  },
  actions: {
    setFilterHeight({ commit }, value) {
      commit('SET_FILTER_HEIGHT', value)
    },
    setResize({ commit }, { height, wigth}) {
      commit('SET_ALL_HEIGHT', height)
      commit('SET_ALL_WIGTH', wigth)
    },
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
    height:({ height }) => (value) => {
      return {
        height: (height - value) + "px",
      };
    },
    wigth:({ wigth }) => (value) => {
      return {
        height: (wigth - value) + "px",
      };
    },
    getFilterHeight({ filterHeight }) {
      return filterHeight
    },
  },
};
