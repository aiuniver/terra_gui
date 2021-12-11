export default {
  namespaced: true,
  state: () => ({
    servers: {}
  }),
  mutations: {
    SET_SERVERS(state, value) {
      state.servers = value
    }
  },
  actions: {
    async getServers({ commit, dispatch }) {
      const res = await dispatch("axios", { url: "/servers/list/" }, { root: true })
			commit('SET_SERVERS', res.data)
    },
    async addServer({ commit, dispatch }, data) {
      const res = await dispatch("axios", { url: "/servers/create/", data }, { root: true })
      if (res.success) commit('SET_SERVERS', res.data.servers)
      return res.data
    },
    async getInstruction({ dispatch }, data) {
      return await dispatch("axios", { url: "/servers/get/", data }, { root: true })
    },
    async setup({ commit, dispatch }, data) {
      const res = await dispatch("axios", { url: "/servers/setup/", data }, { root: true })
      commit('SET_SERVERS', res.data)
    }
  },
  getters: {
    getServers: ({ servers }) => servers
  }
};