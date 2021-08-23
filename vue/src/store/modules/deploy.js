import temp from "../temp/deploy";
import {defLayout, originaltextStyle} from "../const/deploy"
export default {
    namespaced: true,
    state: () => ({
      graphicData: temp.data,
      defaultLayout: defLayout,
      origTextStyle: originaltextStyle,
      Cards: [
        {
          type: "table"
        }
      ],
      moduleList: {
        api_text: "",
        url: "",
      }
    }),
    mutations: {
      SET_MODULE_LIST(state, value) {
        state.moduleList = { ...state.moduleList, ...value};
      },
    },
    actions: {
      async SendDeploy({ dispatch }, data) {
        await dispatch('axios', { url: '/deploy/upload/', data: data }, { root: true });
        return;
      },
      async CheckProgress({ commit, dispatch }) {
        const { data } = await dispatch('axios', { url: '/deploy/upload/progress/'}, { root: true });
        // console.log(data)
        if(data.finished){
          commit("SET_MODULE_LIST", data.data);
        }
        return data.finished;
      },
    },
    getters: {
      getModuleList: ({ moduleList }) => moduleList,
      getGraphicData: ({ graphicData }) => graphicData,
      getDefaultLayout: ({ defaultLayout }) => defaultLayout,
      getOrigTextStyle: ({ origTextStyle }) => origTextStyle,
      getCards: ({ Cards }) => Cards,
    }
}