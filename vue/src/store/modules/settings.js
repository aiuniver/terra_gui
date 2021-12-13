export default {
  namespaced: true,
  state: () => ({
    overlay: false,
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
    SET_OVERLAY (state, value) {
      state.overlay = value;
    },
    SET_ALL_HEIGHT (state, value) {
      state.height = { ...state.height, ...value };
    },
    SET_ALL_WIGTH (state, value) {
      state.wigth = { ...state.wigth, ...value };
    },
  },
  actions: {
    setOverlay ({ commit }, value) {
      commit("SET_OVERLAY", value);
    },
    setResize ({ commit }, { height, wigth }) {
      commit("SET_ALL_HEIGHT", { all: height });
      commit("SET_ALL_WIGTH", { all: wigth });
    },
    setHeight ({ commit }, obj) {
      commit("SET_ALL_HEIGHT", obj);
    },
  },
  getters: {
    isMobile () {
      return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
        navigator.userAgent
      );
    },
    height: ({ height }) => (params = {}) => {
      const { key = 'all', deduct, clean = false, style = true, padding = 0 } = params
      const value = height[key] - (deduct ? height[deduct] : 0) - (clean ? 155 : 0) - (padding)
      return style ? { height: value + "px" } : value
    },
    wigth: ({ wigth }) => (value) => {
      return {
        height: wigth - value + "px",
      };
    },
    // getFilterHeight({ filterHeight }) {
    //   return filterHeight;
    // },
    getOverlay: ({ overlay }) => overlay
  },
};
