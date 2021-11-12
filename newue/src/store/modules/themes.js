export default {
  namespaced: true,
  state: () => ({
    theme: 'dark',
    list: [
      'light',
      'dark'
    ]
  }),
  mutations: {
    SET_THEME (state, value) {
      state.theme = value;
    },
  },
  actions: {
    setTheme ({ state: { theme } }) {
      document.documentElement.setAttribute('data-theme', theme)
    },
    changeTheme ({ commit, dispatch, state: { theme } }, value) {
      value = value || theme === 'light' ? 'dark' : 'light'
      commit('SET_THEME', value || theme)
      dispatch('setTheme')
    }
  },
  getters: {
    getTheme: ({ theme }) => theme,
    getList: ({ list }) => list
  },
};
