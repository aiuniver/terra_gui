import temp from "../temp/deploy";
import { defLayout, originaltextStyle } from "../const/deploy"
export default {
  namespaced: true,
  state: () => ({
    stateParams: {},
    form: {},
    deploy: {},
    graphicData: temp.data,
    defaultLayout: defLayout,
    origTextStyle: originaltextStyle,
    cards: [],
    deployType: '',
    moduleList: {
      api_text: "",
      url: "",
    }
  }),
  mutations: {
    SET_DEPLOY (state, value) {
      state.deploy = { ...value };
    },
    SET_MODULE_LIST (state, value) {
      state.moduleList = { ...state.moduleList, ...value };
    },
    SET_CARDS (state, value) {
      state.cards = value;
    },
    SET_BASE (state, value) {
      state.form = value;
    },
    SET_DEPLOY_TYPE (state, value) {
      state.deployType = value;
    },
    SET_BLOCK_CARDS (state, { value, id }) {
      state.cards[id].data = value;
      state.cards = { ...state.cards }
    },
    SET_STATE_PARAMS (state, value) {
      state.stateParams = { ...value };
    },
  },
  actions: {
    parseStruct ({ commit }, { form }) {
      if (form) commit("SET_BASE", form)
    },
    async uploadData ({ dispatch }, data) {
      return await dispatch('axios', { url: '/deploy/upload/', data }, { root: true });
    },
    async progressUpload ({ commit, dispatch }) {
      const res = await dispatch('axios', { url: '/deploy/upload/progress/' }, { root: true });
      if (res?.data?.finished) commit("SET_MODULE_LIST", res.data.data);
      return res;
    },
    async getData ({ dispatch }, data) {
      return await dispatch('axios', { url: '/deploy/get/', data }, { root: true });
    },
    async progressData ({ dispatch, commit }, data) {
      const res = await dispatch('axios', { url: '/deploy/get/progress/', data }, { root: true });
      if (res) {
        if (res?.data) {
          if (res.data?.finished) {
            console.log(res.data?.data?.data?.data)
            commit("SET_DEPLOY", res.data?.data?.data || {});
            commit("SET_CARDS", res.data?.data?.data?.data || []);
            commit("SET_DEPLOY_TYPE", res.data?.data?.type || []);
          }
        }
        if (res?.error) {
          // dispatch('messages/setMessage', { error: res.error }, { root: true });
          dispatch('logging/setError', JSON.stringify(res.error, null, 2), { root: true });
        }
      }
      return res
    },
    async reloadCard ({ commit, dispatch }, indexes) {
      const { data } = await dispatch('axios', { url: '/deploy/reload/', data: indexes }, { root: true });
      commit("SET_CARDS", data.data.data);
      commit("SET_DEPLOY", data.data);
    },
    setStateParams ({ commit, state: { stateParams } }, data) {
      commit("SET_STATE_PARAMS", { ...stateParams, ...data });
    },
    clear ({ commit }) {
      commit("SET_DEPLOY", {});
      commit("SET_CARDS", []);
      commit("SET_DEPLOY_TYPE", []);
    },
  },
  getters: {
    getParams: ({ form }) => form || {},
    getStateParams: ({ stateParams }) => stateParams || {},
    getModuleList: ({ moduleList }) => moduleList,
    getDeploy: ({ deploy }) => deploy,
    getGraphicData: ({ graphicData }) => graphicData,
    getDefaultLayout: ({ defaultLayout }) => defaultLayout,
    getOrigTextStyle: ({ origTextStyle }) => origTextStyle,
    getCards: ({ cards }) => cards || [],
    getDeployType: ({ deployType }) => deployType,
  }
}