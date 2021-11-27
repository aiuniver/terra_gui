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
    Cards: [],
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
      state.Cards = value;
    },
    SET_BASE (state, value) {
      state.form = value;
    },
    SET_DEPLOY_TYPE (state, value) {
      state.deployType = value;
    },
    SET_BLOCK_CARDS (state, { value, id }) {
      state.Cards[id].data = value;
      state.Cards = { ...state.Cards }
    },
    SET_STATE_PARAMS (state, value) {
      state.stateParams = { ...value };
    },
  },
  actions: {
    parseStruct ({ commit }, { form }) {
      // console.log(form, interactive, progress, state)
      if (form) commit("SET_BASE", form)
    },
    async SendDeploy ({ dispatch }, data) {
      return await dispatch('axios', { url: '/deploy/upload/', data: data }, { root: true });
    },
    async CheckProgress ({ commit, dispatch }) {
      const { data } = await dispatch('axios', { url: '/deploy/upload/progress/' }, { root: true });
      if (data.finished) {
        commit("SET_MODULE_LIST", data.data);
      }
      return data.finished;
    },
    async DownloadSettings ({ dispatch }, data) {
      const res = await dispatch('axios', { url: '/deploy/get/', data }, { root: true });
      return res
    },
    async progress ({ dispatch, commit }, data) {
      const res = await dispatch('axios', { url: '/deploy/get/progress/', data }, { root: true });
      if (res) {
        const { data, error } = res;
        if (data) {
          if (data?.finished) {
            commit("SET_DEPLOY", data?.data?.data || {});
            commit("deploy/SET_CARDS", data?.data?.data?.data || [], { root: true });
            commit("SET_DEPLOY_TYPE", data?.data?.type || []);
          }
        }
        if (error) {
          dispatch('messages/setMessage', { error: error }, { root: true });
          dispatch('logging/setError', JSON.stringify(error, null, 2), { root: true });
        }
      }
      return res
    },
    async ReloadCard ({ commit, dispatch }, values) {
      const { data } = await dispatch('axios', { url: '/deploy/reload/', data: values }, { root: true });
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
    random () {
      let result = '';
      const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
      for (var i = 0; i < 10; i++) {
        result += characters.charAt(Math.floor(Math.random() * characters.length));
      }
      return result;
    },
  },
  getters: {
    getParams ({ form }) {
      console.log(form)
      return form || {}
    },
    getStateParams ({ stateParams }) {
      return stateParams || {}
    },
    getModuleList: ({ moduleList }) => moduleList,
    getDeploy: ({ deploy }) => deploy,
    getGraphicData: ({ graphicData }) => graphicData,
    getDefaultLayout: ({ defaultLayout }) => defaultLayout,
    getOrigTextStyle: ({ origTextStyle }) => origTextStyle,
    getCards: ({ Cards }) => Cards,
    getDeployType: ({ deployType }) => deployType,
  }
}