export default {
  namespaced: true,
  state: () => ({
    color: 'success',
    message: '',
    progress: 0
  }),
  mutations: {
    SET_COLOR (state, color) {
      state.color = color
    },
    SET_MESSAGE (state, message) {
      state.message = message
    },
    SET_PROTSESSOR (state, protsessor) {
      state.protsessor = protsessor
    },
    SET_PROGRESS (state, progress) {
      state.progress = progress
    },
  },
  actions: {
    setMessage ({ commit }, { error, message }) {
      commit('SET_COLOR', error ? 'error' : 'success')
      commit('SET_MESSAGE', error || message )
    },
    setProgress ({ commit }, progress) {
      commit('SET_PROGRESS', progress )
    },
  },
  getters: {
    getProgress: state => state.progress,
    getMessage: state => state.message,
    getColor: state => state.color
  }
}
