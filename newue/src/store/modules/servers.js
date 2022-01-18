export default {
  namespaced: true,
  state: () => ({
    servers: []
  }),
  mutations: {
    SET_SERVERS (state, value) {
      state.servers = value
    },
    ADD_SERVER (state, value) {
      state.servers.unshift(value)
    },
    SETUP_SERVER (state, value) {
      const idx = state.servers.findIndex(server => server.id === value.id)
      state.servers.splice(idx, 1, value)
    }
  },
  actions: {
    async getServers ({ commit, dispatch }) {
      const res = await dispatch("axios", { url: "/servers/list/" }, { root: true })
      commit('SET_SERVERS', res.data)
    },
    async addServer ({ commit, dispatch }, data) {
      const res = await dispatch("axios", { url: "/servers/create/", data }, { root: true })
      if (res.success) commit('ADD_SERVER', res.data)
      return res.data
    },
    async getInstruction ({ dispatch }, data) {
      return await dispatch("axios", { url: "/servers/get/", data }, { root: true })
    },
    async setup ({ commit, dispatch }, data) {
      const res = await dispatch("axios", { url: "/servers/setup/", data }, { root: true })
      commit('SETUP_SERVER', res.data)
    },
    async ready ({ dispatch }, data) {
      const res = await dispatch("axios", { url: "/servers/ready/", data }, { root: true })
      return res
    }
  },
  getters: {
    getServers: ({ servers }) => servers
  }
};