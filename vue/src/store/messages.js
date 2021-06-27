export default {
  namespaced: true,
  state: () => ({
    color: 'primary',
    snackbar: false,
    message: ''
  }),
  mutations: {
    SET_COLOR (state, color) {
      state.color = color
    },
    SET_MESSAGE (state, message) {
      state.message = message
    },
    SET_SNACKBAR (state, value) {
      state.snackbar = value
    }
  },
  actions: {
    setMessage ({ commit }, { error, message }) {
      commit('SET_COLOR', error ? 'red' : 'primary')
      commit('SET_MESSAGE', error || message )
      commit('SET_SNACKBAR', true )
    },
    setSnackbar ({ commit }, value) {
      commit('SET_SNACKBAR', value)
    }
  },
  getters: {
    getMessage: state => state.message,
    getSnackbar: state => state.snackbar,
    getColor: state => state.color
  }
}
