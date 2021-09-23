import { data } from "../temp/training";
import { toolbar } from "../const/trainings";
// import { predict } from "../temp/predict-training";
// import { predict_video } from "../temp/predict-training-video-audio";
// import { predict_text } from "../temp/predict-training-text";
// import { predict_audio } from "../temp/predict-training-audio";

console.warn(data)
export default {
  namespaced: true,
  state: () => ({
    params: [],
    toolbar,
    stateParams: {},
    predict: {},
    info: '',
    states: {},
    trainData: {},
    // trainData: process.env.NODE_ENV === 'development' ? data : {},
    trainUsage: {},
    training: {
      base: {},
      interactive: {},
      state: {}
    },
  }),
  mutations: {
    SET_INTERACTIV(state, value) {
      if (state?.training?.interactive) {
        state.training.interactive = { ...value };
        state.training = { ...state.training }
      }
    },
    SET_PARAMS(state, value) {
      state.params = { ...value };
    },
    SET_CONFIG(state, value) {
      state.training = { ...value };
    },
    SET_STATE_PARAMS(state, value) {
      state.stateParams = { ...value };
    },
    SET_INFO(state, value) {
      state.info = value;
    },
    SET_STATE(state, value) {
      state.training.state = value;
      state.training = { ...state.training }
    },
    SET_PREDICT(state, value) {
      state.predict = { ...value };
    },
    SET_TRAIN(state, value) {
      state.trainData = { ...value };
    },
    SET_TRAIN_USAGE(state, value) {
      state.trainUsage = { ...value };
    },
  },
  actions: {
    setState({ commit }, res) {
      console.log(res)
      if (res && res?.data) {
        const state = res?.data?.data?.state || res?.data.state
        if (state) {
          commit("SET_STATE", state);
        }
      }
    },
    async start({ dispatch }, parse) {
      const valid = await dispatch('modeling/validateModel', {}, { root: true })
      const isValid = !Object.values(valid).filter(item => item).length
      if (isValid) {
        let data = JSON.parse(JSON.stringify(parse))
        console.log(data)
        const arht = data.architecture.parameters.outputs || []
        data.architecture.parameters.outputs = arht.map((item, index) => {
          return item ? { id: index, ...item } : null
        }).filter(item => item)
        dispatch('messages/setMessage', { message: `Запуск обучения...` }, { root: true });
        const res = await dispatch('axios', { url: '/training/start/', data }, { root: true });
        await dispatch('projects/get', {}, { root: true })
        dispatch('setState', res);
        dispatch('setTrainData', {});
        return res
      }
      return null
    },
    async stop({ dispatch }, data) {
      const res = await dispatch('axios', { url: '/training/stop/', data }, { root: true });
      dispatch('setState', res);
      return res
    },
    async clear({ dispatch }, data) {
      const res = await dispatch('axios', { url: '/training/clear/', data }, { root: true });
      dispatch('setState', res);
      return res
    },
    async interactive({ commit, state: { training: { interactive } }, dispatch }, part) {
      const data = { ...interactive, ...part }
      commit("SET_INTERACTIV", data);
      return await dispatch('axios', { url: '/training/interactive/', data }, { root: true });
    },
    async progress({ dispatch }, data) {
      const res = await dispatch('axios', { url: '/training/progress/', data }, { root: true });
      if (res) {
        const { data } = res.data;
        if (data) {
          const { info, train_data, train_usage } = data;
          dispatch('setInfo', info);
          dispatch('setState', res);
          dispatch('setTrainData', train_data);
          dispatch('setTrainUsage', train_usage);
        }
      }
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
    // setStates({ commit }, data) {
    //   commit("SET_STATES", data);
    // },
    setTrainData({ commit }, data) {
      commit("SET_TRAIN", data);
    },
    setPredict({ commit }, data) {
      commit("SET_PREDICT", data);
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
    getInteractive({ training: { interactive } }) {
      return interactive || {}
    },
    getStatus({ training: { state: { status } } }) {
      return status || ''
    },
    getButtons({ training: { state: { buttons } } }) {
      return buttons
    },
    getOutputs({ training: { base } }) {
      return base?.architecture?.parameters?.outputs || []
    },
    getParams({ params }) {
      return params || []
    },
    getToolbar({ toolbar }) {
      return toolbar;
    },
    getTrainUsage: ({ trainUsage }) => {
      return trainUsage || {};
    },
    getTrainData: ({ trainData }) => (key) => {
      return trainData?.[key];
    },
    getPredict({ predict }) {
      return predict || {}
    },
  },
};
