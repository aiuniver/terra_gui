import { temp, test } from "../temp/training";
import { toolbar } from "../const/trainings";
import { predict } from "../temp/predict-training";


export default {
  namespaced: true,
  state: () => ({
    data: temp.data,
    params: [],
    toolbar,
    stateParams: {},
    test,
    predict: predict
  }),
  mutations: {
    SET_PARAMS(state, value) {
      state.params = value;
    },
    SET_STATE_PARAMS(state, value) {
      state.stateParams = { ...value };
    },
  },
  actions: {
    async start({ dispatch }, parse ) {
      let data = JSON.parse(JSON.stringify(parse))
      console.log(data)
      const arht = data.architecture.parameters.outputs || []
      data.architecture.parameters.outputs = arht.map((item, index) => {
        return item ? { id: index, ...item} : null
      }).filter(item => item)

      return await dispatch('axios', { url: '/training/start/', data}, { root: true });
    },
    async stop({ dispatch }, data) {
      return await dispatch('axios', { url: '/training/stop/', data }, { root: true });
    },
    async clear({ dispatch }, data) {
      return await dispatch('axios', { url: '/training/cler/', data }, { root: true });
    },
    async interactive({ dispatch }, data) {
      return await dispatch('axios', { url: '/training/interactive/', data }, { root: true });
    },
    async progress({ dispatch }, data) {
      return await dispatch('axios', { url: '/training/progress/', data }, { root: true });
    },
    setDrawer({ commit }, data) {
      commit("SET_DRAWER", data);
    },
    setStateParams({ commit, state: { stateParams } }, data) {
      commit("SET_STATE_PARAMS", { ...stateParams, ...data });
    },
  },
  getters: {
    getStateParams({ stateParams }) {
      return stateParams || {}
    },
    getParams({ params }) {
      return params || []
    },
    getToolbar({ toolbar }) {
      return toolbar;
    },
    getChars({ data: { plots } }) {
      return plots;
    },
    getScatters({ data: { scatters } }) {
      return scatters;
    },
    getImages({ data: { images: { images } } }) {
      return images;
    },
    getTexts({ data: { texts } }) {
      return texts;
    },
    getTest: ({ test }) => (key) => {
      return test?.[key] || {};
    },
    getTrainData: ({ test }) => (key) => {
      return test?.train_data?.[key] || {};
    },
    getPredict({ predict }) {
      return predict || {}
    },
  },
};
