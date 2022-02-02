export default {
  namespaced: true,
  state: () => ({
    filesSource: [],
    selectSource: {},
    file_manager: [],
    source_path: '',
  }),
  mutations: {
    SET_FILES_SOURCE (state, value) {
      state.filesSource = value;
    },
    SET_SELECT_SOURCE (state, value) {
      state.selectSource = value;
    },
    SET_FILE_MANAGER (state, value) {
      state.file_manager = value;
    },
    SET_SOURCE_PATH (state, value) {
      state.source_path = value;
    },
  },
  actions: {
    async sourceLoadProgress ({ dispatch, commit }) {
      const res = await dispatch('axios', { url: '/datasets/source/load/progress/', data: {} }, { root: true });
      if (res?.data?.finished) {
        const { data: { file_manager, source_path } } = res.data;
        commit('SET_FILE_MANAGER', file_manager);
        commit('SET_SOURCE_PATH', source_path);
      }
      return res
    },
    async setSourceLoad ({ dispatch }, { mode, value }) {
      const { success } = await dispatch('axios', { url: '/datasets/source/load/', data: { mode, value } }, { root: true });
      return success
    },
    async getDatasetSources ({ commit, dispatch }) {
      const { data } = await dispatch('axios', { url: '/datasets/sources/' }, { root: true });
      commit('SET_FILES_SOURCE', data || [])
      return data
    },
    async setSelectSource ({ commit }, select) {
      console.log(select)
      commit('SET_SELECT_SOURCE', select)
    },
  },
  getters: {
    getFilesSource: ({ filesSource }) => filesSource,
    getSelectSource: ({ selectSource }) => selectSource,
    getFileManager: ({ file_manager }) => file_manager,

  },
};
