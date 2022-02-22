import { getFiles, createObj, chnageType } from "../const/create";

export default {
  namespaced: true,
  state: () => ({
    filesSource: [],
    file_manager: [],
    configDefault: {},
    source_path: '',
    pagination: 1,
    errorsBlock: {},
    project: {
      active: 0,
      name: 'Самолеты',
      architecture: "ImageClassification",
      source: {
        mode: 'GoogleDrive',
        value: '/home/bondrogeen/github/terra_gui/TerraAI/datasets/sources/Seasons.zip',
        path: ''
      },
      tags: ["проверка"],
      verName: "самолеты другая",
      train: 0.7,
      shuffle: true
    },
    inputs: [],
    outputs: []
  }),
  mutations: {
    SER_ERRORS_BLOCK (state, value) {
      state.errorsBlock = value;
    },
    SET_DEFAULT (state, value) {
      state.configDefault = value;
    },
    SET_PROJECT (state, value) {
      state.project = value;
    },
    SET_PAGINATION (state, value) {
      state.pagination = value;
    },
    SET_FILES_SOURCE (state, value) {
      state.filesSource = value;
    },
    SET_FILE_MANAGER (state, value) {
      state.file_manager = value;
    },
    SET_SOURCE_PATH (state, value) {
      state.project.source.path = value;
    },
  },
  actions: {

    async create ({ dispatch, state: { project }, rootState: { create: { inputs, outputs } } }) {
      const data = createObj({ project, inputs, outputs })
      return await dispatch('axios', { url: '/datasets/create/', data }, { root: true });
    },

    setProject ({ commit }, value) {
      commit('SET_PROJECT', value)
    },
    setPagination ({ commit }, value) {
      commit('SET_PAGINATION', value)
    },
    async sourceLoadProgress ({ dispatch, commit }) {
      const res = await dispatch('axios', { url: '/datasets/source/load/progress/', data: {} }, { root: true });
      dispatch('messages/setProgressMessage', res.data.message, { root: true });
      dispatch('messages/setProgress', res.data.percent, { root: true });
      if (res?.data?.finished) {
        const { data: { file_manager, source_path, blocks } } = res.data;
        commit('create/SET_INPUT_AND_OUTPUT', blocks, { root: true });
        commit('SET_FILE_MANAGER', file_manager);
        commit('SET_SOURCE_PATH', source_path);
        dispatch('messages/resetProgress', '', { root: true });
      }
      return res
    },
    async createLoadProgress ({ dispatch }) {
      const res = await dispatch('axios', { url: '/datasets/create/progress/', data: {} }, { root: true });
      console.log(res)
      if (res?.data?.finished) {
        // const { data: { file_manager, source_path, blocks } } = res.data;
        // commit('create/SET_INPUT_AND_OUTPUT', blocks, { root: true });
        // commit('SET_FILE_MANAGER', file_manager);
        // commit('SET_SOURCE_PATH', source_path);
      }
      return res
    },
    async setSourceLoad ({ dispatch, state: { project } }) {
      const { source: { mode, value }, architecture } = project
      const { success } = await dispatch('axios', { url: '/datasets/source/load/', data: { mode, value, architecture } }, { root: true });
      return success
    },
    async getDatasetSources ({ commit, dispatch }) {
      const { data } = await dispatch('axios', { url: '/datasets/source/' }, { root: true });
      commit('SET_FILES_SOURCE', data || [])
      return data
    },
    async datasetValidate ({ commit, dispatch, state: { project }, rootState: { create } }, type) {
      const struct = {
        type,
        architecture: project.architecture,
        items: chnageType(create[type])
      }
      const { data } = await dispatch('axios', { url: '/datasets/create/validate/', data: struct }, { root: true });
      commit('SER_ERRORS_BLOCK', data || {})
      console.log(data)
      return data
    },
  },
  getters: {
    getFiles: ({ file_manager }) => getFiles(file_manager),
    getFilesSource: ({ filesSource }) => filesSource,
    getFileManager: ({ file_manager }) => file_manager,
    getProject: ({ project }) => project,
    getPagination: ({ pagination }) => pagination,
    getArchitectures: ({ configDefault }) => configDefault?.architectures || [],
    getHandler: ({ configDefault }) => configDefault?.blocks?.handler || {},
    getErrorsBlock: ({ errorsBlock }) => errorsBlock || {},

  },
};
