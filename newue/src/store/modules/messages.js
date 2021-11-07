export default {
  namespaced: true,
  state: () => ({
    color: 'success',
    message: '',
    progressMessage: '',
    progress: 0,
  }),
  mutations: {
    SET_COLOR(state, value) {
      state.color = value;
    },
    SET_MESSAGE(state, value) {
      state.message = value;
    },
    SET_PROGRESS_MESSAGE(state, value) {
      state.progressMessage = value;
    },
    SET_PROTSESSOR(state, value) {
      state.protsessor = value;
    },
    SET_PROGRESS(state, value) {
      state.progress = value;
    },
  },
  actions: {
    resetProgress({ commit }) {
      commit('SET_PROGRESS_MESSAGE', '');
      commit('SET_PROGRESS', 0);
    },
    setMessage({ commit }, { error, message, info }) {
      // console.log(message)
      commit('SET_COLOR', error ? 'error' : message ? 'success' : 'info');
      commit('SET_MESSAGE', error || message || info);
    },
    setProgressMessage({ commit }, message) {
      commit('SET_PROGRESS_MESSAGE', message);
    },
    setProgress({ commit }, progress) {
      // console.log(progress)
      commit('SET_PROGRESS', ~~progress);
    },
    async setModel(_, { context, title = 'Предупреждение!', width = 300, okText = 'OK', showClose = true, content = 'Что не так ?!' }) {
      console.log({
        title,
        width,
        content,
        showClose,
        okText,
      })
      try {
        return await context.$Modal.alert({ title, width, content, showClose, okText, });
      } catch (error) {
        return error
      }
    }
  },
  getters: {
    getProgress: ({ progress }) => progress,
    getProgressMessage: ({ progressMessage }) => progressMessage,
    getMessage: ({ message }) => message,
    getColor: ({ color }) => color,
  },
};
