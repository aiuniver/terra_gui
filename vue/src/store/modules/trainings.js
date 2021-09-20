import { data, config } from "../temp/training";
import { toolbar } from "../const/trainings";
// import { predict } from "../temp/predict-training";
// import { predict_video } from "../temp/predict-training-video-audio";
import { predict_text } from "../temp/predict-training-text";

console.warn(data)
export default {
  namespaced: true,
  state: () => ({
    params: [],
    toolbar,
    stateParams: {},
    predict: predict_text,
    info: '',
    states: {},
    // trainData: {},
    trainData: data,
    trainUsage: {},
    trainDisplay: config,
    buttons: {
      train: {
        title: "Обучить",
        visible: true
      },
      stop: {
        title: "Остановить",
        visible: false
      },
      clear: {
        title: "Сбросить",
        visible: false
      },
      save: {
        title: "Сохранить",
        visible: false
      }
    }
  }),
  mutations: {
    SET_PARAMS(state, value) {
      state.params = value;
    },
    SET_BUTTONS(state, buttons) {
      state.buttons = { ...buttons };
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
    SET_TRAIN_USAGE(state, value) {
      state.trainUsage = { ...value };
    },
    SET_TRAIN_DISPLAY(state, value) {
      Object.assign(state.trainDisplay, value);
    },
  },
  actions: {
    setButtons({ commit }, res) {
      if (res && res?.data) {
        const { buttons } = res?.data?.data?.states || res?.data
        if (buttons) {
          commit("SET_BUTTONS", buttons);
        }
      }
    },
    async start({ dispatch }, parse) {
      let data = JSON.parse(JSON.stringify(parse))
      console.log(data)
      const arht = data.architecture.parameters.outputs || []
      data.architecture.parameters.outputs = arht.map((item, index) => {
        return item ? { id: index, ...item } : null
      }).filter(item => item)
      const res = await dispatch('axios', { url: '/training/start/', data }, { root: true });
      dispatch('setButtons', res);
      dispatch('setTrainData', {});
      return res
    },
    async stop({ dispatch }, data) {
      const res = await dispatch('axios', { url: '/training/stop/', data }, { root: true });
      dispatch('setButtons', res);
      return res
    },
    async clear({ dispatch }, data) {
      const res = await dispatch('axios', { url: '/training/clear/', data }, { root: true });
      dispatch('setButtons', res);
      return res
    },
    async interactive({ dispatch }, data) {
      return await dispatch('axios', { url: '/training/interactive/', data }, { root: true });
    },
    async progress({ dispatch }, data) {
      const res = await dispatch('axios', { url: '/training/progress/', data }, { root: true });
      console.log();
      dispatch('setButtons', res);
      return res
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
    setTrainUsage({ commit }, data) {
      commit("SET_TRAIN_USAGE", data);
    },
    setTrainDisplay({ commit }, data) {
      commit("SET_TRAIN_DISPLAY", data);
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
    getTrainUsage: ({ trainUsage }) => {
      return trainUsage || {};
    },
    getTrainData: ({ trainData }) => (key) => {
      return trainData?.[key];
    },
    getTrainDisplay: ({ trainDisplay }) => {
      return trainDisplay;
    },
    getPredict({ predict }) {
      return predict || {}
    },
    getButtons({ buttons }) {
      return buttons
    },
  },
};
