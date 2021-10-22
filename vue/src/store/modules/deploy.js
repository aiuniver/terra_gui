import temp from "../temp/deploy";
import {defLayout, originaltextStyle} from "../const/deploy"
export default {
    namespaced: true,
    state: () => ({
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
      SET_DEPLOY(state, value) {
        state.deploy = { ...value};
      },
      SET_MODULE_LIST(state, value) {
        state.moduleList = { ...state.moduleList, ...value};
      },
      SET_CARDS(state, value) {
        state.Cards = value;
      },
      SET_DEPLOY_TYPE(state, value) {
        state.deployType = value;
      },
      SET_BLOCK_CARDS(state, { value, id }) {
        state.Cards[id].data = value;
        state.Cards = { ...state.Cards }
      },
    },
    actions: {
      async SendDeploy({ dispatch }, data) {
        return await dispatch('axios', { url: '/deploy/upload/', data: data }, { root: true });
      },
      async CheckProgress({ commit, dispatch }) {
        const { data } = await dispatch('axios', { url: '/deploy/upload/progress/'}, { root: true });
        if(data.finished){
          commit("SET_MODULE_LIST", data.data);
        }
        return data.finished;
      },
      async ReloadCard({ commit, dispatch }, values) {
        const { data } = await dispatch('axios', { url: '/deploy/reload/', data: values }, { root: true });
        commit("SET_CARDS",  data.data.data);
        commit("SET_DEPLOY",  data.data);
      },
    },
    getters: {
      getModuleList: ({ moduleList }) => moduleList,
      getDeploy: ({ deploy }) => deploy,
      getGraphicData: ({ graphicData }) => graphicData,
      getDefaultLayout: ({ defaultLayout }) => defaultLayout,
      getOrigTextStyle: ({ origTextStyle }) => origTextStyle,
      getCards: ({ Cards }) => Cards,
      getDeployType: ({ deployType }) => deployType,
      getRandId:({ Cards }) => {
        let id = Cards;
        let crypto = require("crypto");
        id = crypto.randomBytes(20).toString('hex');
        console.log(id)
        return id;
      }
    }
}