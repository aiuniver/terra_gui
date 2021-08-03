export default {
  namespaced: true,
  state: () => ({
    filterHeight: 32,
    height: {
      all: document.documentElement.clientHeight,
      filter: 0, //datasetFilter
      center: 0, //datasetParamsFull
    },
    wigth: {
      all: document.documentElement.clientWidth,
    },
  }),
  mutations: {
    SET_ALL_HEIGHT(state, value) {
      state.height = { ...state.height, ...value };
    },
    SET_ALL_WIGTH(state, value) {
      state.wigth = { ...state.wigth, ...value };
    },
  },
  actions: {
    setResize({ commit }, { height, wigth }) {
      commit("SET_ALL_HEIGHT", { all: height });
      commit("SET_ALL_WIGTH", { all: wigth });
    },
    setHeight({ commit }, obj) {
      commit("SET_ALL_HEIGHT", obj);
    },
  },
  getters: {
    isMobile() {
      return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
        navigator.userAgent
      );
    },
    // autoHeight() {
    //   return {
    //     height: (document.documentElement.clientHeight - 155) + "px",
    //   };
    // },
    height: ({ height }) => (params = {}) => {
      const { key = 'all', deduct, clean = false, style = true, padding = 0 } = params
      const value = height[key] - (deduct ? height[deduct] : 0) - (clean ? 155 : 0) - (padding)
      console.log(height)
      return style ? { height: value + "px" } : value
    },
    wigth: ({ wigth }) => (value) => {
      return {
        height: wigth - value + "px",
      };
    },
    getFilterHeight({ filterHeight }) {
      return filterHeight;
    },
  },
};
