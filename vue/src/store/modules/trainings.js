// import { data } from "../temp/training";
import { toolbar } from "../const/trainings";

export default {
  namespaced: true,
  state: () => ({
    collapse: ['0', '1'],

    toolbar,
    stateParams: {},
    predict: {},
    info: '',
    states: {},
    trainSettings: {},
    statusTrain: 'no_train',

    base: {},
    form: {},
    state: {},
    result: {},
    progress: {},
    interactive: {},

    largeImgSrc: null,

    expandedIdx: []
  }),
  mutations: {
    SET_BASE (state, value) {
      state.base = value;
    },
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
    SET_TRAIN_SETTINGS (state, value) {
      state.trainSettings = { ...value };
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
    SET_COLLAPSE (state, value) {
      state.collapse = [...value];
    },
    SET_STATUS_TRAIN (state, value) {
      state.statusTrain = value;
    },
    SET_LARGE_IMAGE (state, value) {
      state.largeImgSrc = value;
    },
    SET_EXPANDED (state, { value, idx }) {
      if (value) return state.expandedIdx.push(idx)
      state.expandedIdx.splice(state.expandedIdx.indexOf(idx), 1)
    }
  },
  actions: {
    parseStruct ({ commit }, { form, interactive, progress, state, result, base }) {
      // console.log(form, interactive, progress, state)
      if (base) commit("SET_BASE", base)
      if (form) commit("SET_FORM", { ...form })
      if (state) commit("SET_STATE", state)
      if (state?.status) commit("SET_STATUS_TRAIN", state.status)
      if (interactive) commit("SET_INTERACTIV", interactive)
      if (progress) commit("SET_PROGRESS", state)
      if (result) commit("SET_RESULT", result)
    },
    async start ({ state: { state: { status } }, dispatch }, parse) {
      let isValid = true
      if (status === 'no_train') {
        const valid = await dispatch('modeling/validateModel', {}, { root: true })
        isValid = !Object.values(valid || {}).filter(item => item).length
        dispatch('setTrainData', {});
      }
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
    async update ({ dispatch }, parse) {
      let data = JSON.parse(JSON.stringify(parse))
      const arht = data.architecture.parameters.outputs || []
      data.architecture.parameters.outputs = arht.map((item, index) => {
        return item ? { id: index, ...item } : null
      }).filter(item => item)
      const res = await dispatch('axios', { url: '/training/update/', data }, { root: true });
      dispatch('parseStruct', res?.data || {});
      return res
    },
    async clear ({ dispatch }, data) {
      const res = await dispatch('axios', { url: '/training/clear/', data }, { root: true });
      dispatch('resetAllTraining', {});
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
      if (res) {
        const { data, error } = res;
        if (data) {
          dispatch('parseStruct', data || {});
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
      await dispatch('projects/get', {}, { root: true })
    },
    async resetAllTraining ({ commit, dispatch }) {
      dispatch('resetTraining', {});
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
    setPredict ({ commit }, data) {
      commit("SET_PREDICT", data);
    },
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
    setExpandedIdx ({ commit }, data) {
      commit("SET_EXPANDED", data);
    }
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
    getOutputs ({ base }) {
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
    getExpandedIdx ({ expandedIdx }) {
      return expandedIdx
    }
  },
};
