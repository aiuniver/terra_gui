// import { data } from "../temp/training";
import { toolbar } from "../const/trainings";
// import { predict } from "../temp/predict-training";
// import { predict_video } from "../temp/predict-training-video-audio";
// import { predict_text } from "../temp/predict-training-text";
// import { predict_audio } from "../temp/predict-training-audio";

// console.warn(data)
export default {
  namespaced: true,
  state: () => ({
    collapse: ['0', '1'],
    // params: {},
    toolbar,
    stateParams: {},
    predict: {},
    info: '',
    states: {},
    trainSettings: {},
    // trainData: {},
    // trainData: process.env.NODE_ENV === 'development' ? data : {},
    // trainUsage: {},
    statusTrain: 'no_train',
    // training: {
    //   base: {},
    //   interactive: {},
    //   state: {}
    // },

    form: {},
    state: {},
    result: {},
    progress: {},
    interactive: {},

    // architecture: '',
    largeImgSrc: null
  }),
  mutations: {
    SET_FORM (state, value) {
      state.form = value;
    },
    SET_STATE (state, value) {
      state.state = value;
    },
    SET_INTERACTIV (state, value) {
      state.interactive = value;
    },
    SET_PROGRESS (state, value) {
      state.progress = value;
    },
    SET_RESULT (state, value) {
      state.result = value;
    },











    // SET_ARCHITECTURE (state, value) {
    //   state.architecture = value;
    // },

    SET_TRAIN_SETTINGS (state, value) {
      state.trainSettings = { ...value };
    },
    // SET_PARAMS (state, value) {
    //   state.params = { ...value };
    // },
    SET_CONFIG (state, value) {
      // state.training = { ...value };
      state.interactive = JSON.parse(JSON.stringify(value.interactive))
    },
    SET_STATE_PARAMS (state, value) {
      state.stateParams = { ...value };
    },
    SET_INFO (state, value) {
      state.info = value;
    },

    SET_PREDICT (state, value) {
      state.predict = { ...value };
    },
    // SET_TRAIN (state, value) {
    //   state.trainData = { ...value };
    // },
    // SET_TRAIN_USAGE (state, value) {
    //   state.trainUsage = { ...value };
    // },
    SET_COLLAPSE (state, value) {
      state.collapse = [...value];
    },
    SET_STATUS_TRAIN (state, value) {
      state.statusTrain = value;
    },
    SET_LARGE_IMAGE (state, value) {
      state.largeImgSrc = value;
    },
  },
  actions: {
    parseStruct ({ commit }, { form, interactive, progress, state, result, test }) {
      console.log(form, interactive, progress, state)
      if (form) commit("SET_FORM", form)
      if (state) commit("SET_STATE", state)
      if (state?.status) commit("SET_STATUS_TRAIN", state.status)
      if (interactive) commit("SET_INTERACTIV", interactive)
      if (progress) commit("SET_PROGRESS", state)
      if (result) commit("SET_RESULT", result)
      if (test) {
        commit('sdsdd', 1)
      }
    },
    // setState ({ dispatch, commit }, res) {
    //   // console.log(res)
    //   if (res && res?.data) {
    //     const state = res?.data?.data?.state || res?.data.state
    //     const base = res?.data?.form?.base
    //     if (state) {
    //       commit("SET_STATE", state);
    //       if (base) {
    //         console.log(base)
    //         commit("SET_PARAMS", base);
    //       }
    //       dispatch('setStatusTrain', state.status);
    //     }
    //   }
    // },
    async start ({ state: { state: { status } }, dispatch }, parse) {
      console.log('1')
      let isValid = true
      if (status === 'no_train') {
        console.log('2')
        const valid = await dispatch('modeling/validateModel', {}, { root: true })
        isValid = !Object.values(valid || {}).filter(item => item).length
        dispatch('setTrainData', {});
      }
      console.log('3')
      if (isValid) {
        dispatch('setStatusTrain', 'start');
        let data = JSON.parse(JSON.stringify(parse))
        console.log(data)
        const arht = data.architecture.parameters.outputs || []
        data.architecture.parameters.outputs = arht.map((item, index) => {
          return item ? { id: index, ...item } : null
        }).filter(item => item)
        dispatch('messages/setMessage', { message: `Запуск обучения...` }, { root: true });
        const res = await dispatch('axios', { url: '/training/start/', data }, { root: true });
        if (res && res?.data) {
          await dispatch('projects/get', {}, { root: true })
          // dispatch('setState', res);
        } else {
          dispatch('setStatusTrain', 'no_train');
        }
        dispatch('parseStruct', res?.data || {});
        return res
      }
      return null
    },
    async stop ({ dispatch }, data) {
      const res = await dispatch('axios', { url: '/training/stop/', data }, { root: true });
      // dispatch('setState', res);
      dispatch('parseStruct', res?.data || {});
      return res
    },
    async save ({ dispatch }, data) {
      const res = await dispatch('axios', { url: '/training/save/', data }, { root: true });
      // dispatch('parseStruct', res?.data || {});
      return res
    },
    async update ({ commit, dispatch }, parse) {
      let data = JSON.parse(JSON.stringify(parse))
      // console.log(data)
      const arht = data.architecture.parameters.outputs || []
      data.architecture.parameters.outputs = arht.map((item, index) => {
        return item ? { id: index, ...item } : null
      }).filter(item => item)
      // console.log(JSON.stringify(data, null, 2))
      const res = await dispatch('axios', { url: '/training/update/', data }, { root: true });
      if (res) {
        const { data, error } = res
        if (data && !error) {
          // commit("SET_PARAMS", data.form.base);
          commit("SET_CONFIG", data.data);
        }
      }
      dispatch('parseStruct', res?.data || {});
      return res
    },
    async clear ({ dispatch }, data) {
      const res = await dispatch('axios', { url: '/training/clear/', data }, { root: true });
      // dispatch('setState', res);
      dispatch('resetTraining', {});
      dispatch('parseStruct', res?.data || {});
      return res
    },
    async interactive ({ state: { interactive }, dispatch }, part) {
      const data = { ...interactive, ...part }
      const res = await dispatch('axios', { url: '/training/interactive/', data }, { root: true });
      if (res) {
        dispatch('setObjectInteractive', data);
      }
      if (res?.data?.train_data) {
        const { data: { train_data } } = res
        dispatch('setTrainData', train_data);
      }
      return res
    },
    async progress ({ dispatch }, data) {
      const res = await dispatch('axios', { url: '/training/progress/', data }, { root: true });
      console.log(res)
      if (res) {
        const { data, error } = res;
        if (data) {
          console.log(data)
          dispatch('parseStruct', data || {});
          // const { info, train_data, train_usage } = data;
          // if (info) dispatch('setInfo', info);
          // dispatch('setState', res);
          // if (train_data) dispatch('setTrainData', train_data);
          // if (train_usage) dispatch('setTrainUsage', train_usage);
        }
        if (error) {
          dispatch('messages/setMessage', { error: error }, { root: true });
          dispatch('logging/setError', JSON.stringify(error, null, 2), { root: true });
        }
      }
      return res
    },
    async resetTraining ({ dispatch }) {
      localStorage.removeItem('settingsTrainings');
      dispatch('messages/resetProgress', {}, { root: true });
      // dispatch('setTrainData', {});
      // dispatch('setTrainUsage', {});
      await dispatch('projects/get', {}, { root: true })
    },
    async resetAllTraining ({ commit, dispatch }) {
      dispatch('resetTraining', {});
      // dispatch('setTrainSettings', {});
      // dispatch('setTrainSettings', {});
      commit("SET_STATE_PARAMS", {});
    },
    setDrawer ({ commit }, data) {
      commit("SET_DRAWER", data);
    },
    setStateParams ({ commit, state: { stateParams } }, data) {
      commit("SET_STATE_PARAMS", { ...stateParams, ...data });
    },
    setInfo ({ commit }, info) {
      commit("SET_INFO", info);
    },
    // setStates({ commit }, data) {
    //   commit("SET_STATES", data);
    // },
    // setTrainData ({ commit }, data) {
    //   commit("SET_TRAIN", data);
    // },
    setPredict ({ commit }, data) {
      commit("SET_PREDICT", data);
    },
    // setTrainUsage ({ commit }, data) {
    //   commit("SET_TRAIN_USAGE", data);
    // },
    setTrainDisplay ({ commit }, data) {
      commit("SET_TRAIN_DISPLAY", data);
    },
    setСollapse ({ commit }, data) {
      commit("SET_COLLAPSE", data);
    },
    setObjectInteractive ({ state, commit }, charts) {
      const data = { ...state.interactive, ...charts }
      commit("SET_INTERACTIV", data);
    },
    setStatusTrain ({ commit }, value) {
      commit("SET_STATUS_TRAIN", value);
    },
    setTrainSettings ({ commit }, value) {
      commit("SET_TRAIN_SETTINGS", value);
    },
    setLargeImg ({ commit }, value = null) {
      commit("SET_LARGE_IMAGE", value);
    },
  },
  getters: {
    getTrainSettings: ({ trainSettings }) => {
      return trainSettings
    },
    getArchitecture: ({ form: { architecture } }) => {
      return architecture
    },
    getStatusTrain: ({ statusTrain }) => {
      return statusTrain
    },
    getObjectInteractive: ({ interactive }) => key => {
      return interactive?.[key] || {}
    },
    getArrayInteractive: ({ interactive }) => key => {
      return interactive?.[key] || []
    },
    getStateParams ({ stateParams }) {
      return stateParams || {}
    },
    getСollapse ({ collapse }) {
      return collapse || []
    },
    getInteractive ({ interactive }) {
      return interactive || {}
    },
    getStatus ({ state: { status } }) {
      return status || ''
    },
    getButtons ({ state: { buttons } }) {
      return buttons
    },
    getOutputs ({ form: { base } }) {
      return base?.architecture?.parameters?.outputs || []
    },
    getParams ({ form }) {
      return form?.base || []
    },
    getToolbar ({ toolbar }) {
      return toolbar;
    },
    getTrainUsage: ({ result: { train_usage } }) => {
      return train_usage || {};
    },
    getTrainData: ({ result: { train_data } }) => (key) => {
      return train_data?.[key];
    },
    getPredict ({ predict }) {
      return predict || {}
    },
    getLargeImg ({ largeImgSrc }) {
      return largeImgSrc
    },
  },
};
