export default {
    namespaced: true,
    state: () => ({
        DataLoaded: false,
        moduleList: {
                api_text: "",
                url: "",
            }
    }),
    mutations: {
      SET_DATALOADED(state, value) {
        state.DataLoaded = value;
      },
    },
    actions: {
      async SendDeploy({ state, dispatch }, data) {
        const model = await dispatch('axios', { url: '/deploy/upload/', data: data }, { root: true });
        state.moduleList = model;
      },
      async CheckProgress({ dispatch }) {
        const data = await dispatch('axios', { url: '/deploy/upload/progress/'}, { root: true });
        return data;
      },
      setDataLoaded({ commit }, value) {
        commit("SET_DATALOADED", value);
      },
    },
    getters: {
        getModuleList: ({ moduleList }) => moduleList,
        getDataLoaded: ({ DataLoaded }) => DataLoaded,
    }
}