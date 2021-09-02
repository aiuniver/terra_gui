import temp from "../temp/training";
import { toolbar } from "../const/trainings";

export default {
  namespaced: true,
  state: () => ({
    data: temp.data,
    params: [],
    toolbar
  }),
  mutations: {
    SET_PARAMS(state, value) {
      state.params = value;
    },
  },
  actions: {
    setDrawer({ commit }, data) {
      commit("SET_DRAWER", data);
    },
  },
  getters: {
    getParams ({ params }) {
      console.log(params)
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
