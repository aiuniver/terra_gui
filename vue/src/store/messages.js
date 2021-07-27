export default {
  namespaced: true,
  state: () => ({
    color: "success",
    message: "",
    progressMessage: "",
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
    setMessage({ commit }, { error, message }) {
      commit("SET_COLOR", error ? "error" : "success");
      commit("SET_MESSAGE", error || message);
    },
    setProgressMessage({ commit }, message ) {
      commit("SET_PROGRESS_MESSAGE", message);
    },
    setProgress({ commit }, progress) {
      commit("SET_PROGRESS", progress);
    },
  },
  getters: {
    getProgress: ({ progress }) => progress,
    getProgressMessage: ({ progressMessage }) => progressMessage,
    getMessage: ({ message }) => message,
    getColor: ({ color }) => color,
  },
};
