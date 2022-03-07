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
      name: '',
      architecture: "",
      firstCreation: true,
      source: {
        mode: 'GoogleDrive',
        value: '',
        path: ''
      },
      tags: [],
      verName: "",
      train: 0.7,
      shuffle: true
    },
    inputs: [],
    outputs: []
  }),
  mutations: {
    SET_ERRORS_BLOCK (state, value) {
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
    setProject ({ commit }, value) {
      commit('SET_PROJECT', { ...value })
    },
    setPagination ({ commit }, value) {
      commit('SET_PAGINATION', value)
    },
    parseDataset ({ commit }, data) {
      const { source, name, architecture, tags, version, stage, first_creation } = data || {}
      const project = {
        name: name || '',
        architecture: architecture || '',
        firstCreation: first_creation,
        source: {
          mode: source?.mode || 'GoogleDrive',
          path: source?.path || '',
          value: source?.value || '',
        },
        tags: tags || [],
        verName: version?.name || '',
        train: version?.info?.path?.train || 0.7,
        shuffle: version?.info?.shuffle || true
      }
      const blocks = {
        inputs: version?.inputs || [],
        outputs: version?.outputs || [],
        stage: stage || 1
      }
      commit('SET_FILE_MANAGER', source?.manager || []);
      commit('create/SET_INPUT_AND_OUTPUT', blocks, { root: true });
      commit('SET_PAGINATION', stage || 1)
      commit('SET_PROJECT', project)
    },

    async create ({ dispatch, state: { project, pagination }, rootState: { create } }) {
      const { inputs, outputs } = JSON.parse(JSON.stringify(create))
      const data = createObj({ project, inputs, outputs })
      return await dispatch('axios', { url: '/datasets/create/', data: { ...data, stage: pagination } }, { root: true });
    },

    async sourceLoadProgress ({ dispatch }) {
      const res = await dispatch('axios', { url: '/datasets/source/load/progress/', data: {} }, { root: true });
      if (res?.data?.finished) {
        await dispatch('projects/get', {}, { root: true })
      }
      return res
    },
    async createLoadProgress ({ dispatch }) {
      const res = await dispatch('axios', { url: '/datasets/create/progress/', data: {} }, { root: true });
      if (res?.data?.finished) {
        await dispatch('projects/get', {}, { root: true })
      }
      return res
    },
    async setSourceLoad ({ dispatch, state: { project } }) {
      const { source: { mode, value }, architecture, name, tags } = project
      const { success } = await dispatch('axios', { url: '/datasets/source/load/', data: { mode, value, architecture, name, tags } }, { root: true });
      return success
    },
    async getDatasetSources ({ commit, dispatch }) {
      const { data } = await dispatch('axios', { url: '/datasets/source/' }, { root: true });
      commit('SET_FILES_SOURCE', data || [])
      return data
    },
    async datasetValidate ({ commit, dispatch, state: { project }, rootState: { create } }, type) {
      const blocks = JSON.parse(JSON.stringify(create.blocks))
      const struct = {
        type,
        architecture: project.architecture,
        items: chnageType(blocks)
      }
      const res = await dispatch('axios', { url: '/datasets/create/validate/', data: struct }, { root: true });
      commit('SET_ERRORS_BLOCK', res?.data || {})
      return res
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
