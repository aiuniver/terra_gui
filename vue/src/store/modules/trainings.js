import temp from "../temp/training";
import { toolbar } from "../const/trainings";

export default {
  namespaced: true,
  state: () => ({
    data: temp.data,
    params: [],
    toolbar,
    stateParams: {} 
  }),
  mutations: {
    SET_PARAMS(state, value) {
      state.params = value;
    },
    SET_STATE_PARAMS(state, value) {
      state.stateParams = {...value};
    },
  },
  actions: {
    setDrawer({ commit }, data) {
      commit("SET_DRAWER", data);
    },
    setStateParams({ commit, state: { stateParams } }, data) {
      commit("SET_STATE_PARAMS", { ...stateParams, ...data });
    },
  },
  getters: {
    getStateParams ({ stateParams }) {
      return stateParams || {}
    },
    getParams ({ params }) {
      return params || []
    },
    getToolbar({ toolbar }) {
      return toolbar;
    },
    getChars({ data: { plots } }) {
      return plots;
    },
    getScatters({ data: {scatters} }) {
      return scatters;
    },
    getImages({ data: { images: { images } } }) {
      return images;
    },
    getTexts({ data: { texts } }) {
      return texts;
    },
  },
};
