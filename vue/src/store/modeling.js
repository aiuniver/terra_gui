import axios from "axios";
import { blocks, typeBlock } from './const/typeBlock'
export default {
  namespaced: true,
  state: () => ({
    dialog: false,
    model_list: [],
    toolbarEvent:  {}
  }),
  mutations: {
    SET_DIALOG(state, value) {
      state.dialog = value;
    },
    SET_TOOLBAR_EVENT(state, value) {
      state.toolbarEvent = value;
    },
  },
  actions: {
    setDialog({ commit }, value) {
      commit("SET_DIALOG", value);
    },
    setToolbarEvent({ commit }, value) {
      const { event } = value
      if (event === 'load') {
        commit("SET_DIALOG", true);
      }
      commit("SET_TOOLBAR_EVENT", { ...value } );
    },
    async axios(_, params) {
      try {
        const { data } = await axios(params);
        return data;
      } catch (error) {
        console.log(error);
      }
    },
    async loadModel() {
      try {
        const data = await axios.get("/api/v1/modeling/models/");
        return data.data.data
      } catch (error) {
        console.log(error);
      }
    },
  },
  getters: {
    getDialog: ({ dialog }) => dialog,
    getToolbarEvent: ({ toolbarEvent }) => toolbarEvent,
    getTypeBlock: () => typeBlock,
    getBlocks: () => blocks,
  },
};
