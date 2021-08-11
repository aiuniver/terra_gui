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
        console.log(data);
        const model = await dispatch('axios', { url: '/deploy/upload/', data: data }, { root: true });
        console.log(model);
        state.moduleList = model;
        return model;
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