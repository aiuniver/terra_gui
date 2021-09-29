import temp from "../temp/deploy";
import {defLayout, originaltextStyle} from "../const/deploy"
export default {
    namespaced: true,
    state: () => ({
      graphicData: temp.data,
      defaultLayout: defLayout,
      origTextStyle: originaltextStyle,
      Cards: {},
      deployType: "",
      moduleList: {
        api_text: "",
        url: "",
      }
    }),
    mutations: {
      SET_MODULE_LIST(state, value) {
        state.moduleList = { ...state.moduleList, ...value};
      },
      SET_CARDS(state, value) {
        state.Cards = { ...state.Cards, ...value}
      },
      SET_BLOCK_CARDS(state, { value, id }) {
        console.log(value);
        console.log(id);
        state.Cards[id].data = value;
        state.Cards = { ...state.Cards }
      },
      SET_DEPLOY_TYPE(state, value) {
        state.deployType = value;
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
        commit("SET_BLOCK_CARDS", { value: data, id: values.id });
      },
    },
    getters: {
      getModuleList: ({ moduleList }) => moduleList,
      getGraphicData: ({ graphicData }) => graphicData,
      getDefaultLayout: ({ defaultLayout }) => defaultLayout,
      getOrigTextStyle: ({ origTextStyle }) => origTextStyle,
      getCards: ({ Cards }) => Cards,
      getDeployType: ({ deployType }) => deployType,
      getRandId:({ Cards }) => {
        let id = Cards;
        let crypto = require("crypto");
        id = crypto.randomBytes(20).toString('hex');
        return id;
      }
    }
}