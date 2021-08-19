import temp from "../temp/deploy";
export default {
    namespaced: true,
    state: () => ({
      graphicData: temp.data,
      moduleList: {
        api_text: "",
        url: "",
      }
    }),
    mutations: {
      SET_MODULE_LIST(state, value) {
        state.moduleList = value;
      },
    },
    actions: {
      async SendDeploy({ state, dispatch }, data) {
        const { data: model } = await dispatch('axios', { url: '/deploy/upload/', data: data }, { root: true });
        state.moduleList = model;
      },
      async CheckProgress({ dispatch }) {
        const { data } = await dispatch('axios', { url: '/deploy/upload/progress/'}, { root: true });
        return data;
      },
    },
    getters: {
      getModuleList: ({ moduleList }) => moduleList,
      getGraphicData: ({ graphicData }) => graphicData,
    }
}