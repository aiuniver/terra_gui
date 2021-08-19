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
        state.moduleList = { ...state.moduleList, ...value};
      },
    },
    actions: {
<<<<<<< HEAD
      async SendDeploy({ state, dispatch }, data) {
        const { data: model } = await dispatch('axios', { url: '/deploy/upload/', data: data }, { root: true });
        state.moduleList = model;
=======
      async SendDeploy({ dispatch }, data) {
        await dispatch('axios', { url: '/deploy/upload/', data: data }, { root: true });
        return;
>>>>>>> svyat/svyat-dev
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
    }
}