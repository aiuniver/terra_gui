import { data } from "../temp/training";
import { toolbar } from "../const/trainings";
import { predict } from "../temp/predict-training";

console.warn(data)
export default {
  namespaced: true,
  state: () => ({
    params: [],
    toolbar,
    stateParams: {},
    predict: predict,

    info: '',
    states: {},
    // trainData: {},
    trainData: data,
    usage: {},
  }),
  mutations: {
    SET_PARAMS(state, value) {
      state.params = value;
    },
    SET_STATE_PARAMS(state, value) {
      state.stateParams = { ...value };
    },
    SET_INFO(state, value) {
      state.info = value;
    },
    SET_STATES(state, value) {
      state.states = { ...value };
    },
    SET_TRAIN(state, value) {
      state.trainData = { ...value };
    },
    SET_USAGE(state, value) {
      state.usage = { ...value };
    },
  },
  actions: {
    async start({ dispatch }, parse) {
      let data = JSON.parse(JSON.stringify(parse))
      console.log(data)
      const arht = data.architecture.parameters.outputs || []
      data.architecture.parameters.outputs = arht.map((item, index) => {
        return item ? { id: index, ...item } : null
      }).filter(item => item)

      return await dispatch('axios', { url: '/training/start/', data }, { root: true });
    },
    async stop({ dispatch }, data) {
      return await dispatch('axios', { url: '/training/stop/', data }, { root: true });
    },
    async clear({ dispatch }, data) {
      return await dispatch('axios', { url: '/training/clear/', data }, { root: true });
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
    setInfo({ commit }, info) {
      commit("SET_INFO", info);
    },
    setStates({ commit }, data) {
      commit("SET_STATES", data);
    },
    setTrainData({ commit }, data) {
      commit("SET_TRAIN", data);
    },
    setUsage({ commit }, data) {
      commit("SET_USAGE", data);
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
    getUsage: ({ usage }) => {
      return usage || {};
    },
    getTrainData: ({ trainData }) => (key) => {
      return trainData?.[key] || {};
    },
    getPredict({ predict }) {
      return predict || {}
    },
  },
};
