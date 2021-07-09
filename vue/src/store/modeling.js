import axios from "axios";
import { loadModel } from './temp'
export default {
  namespaced: true,
  state: () => ({
    dialog: false,
  }),
  mutations: {
    SET_DIALOG(state, value) {
      state.dialog = value;
    },
  },
  actions: {
    setDialog({ commit }, value) {
      commit("SET_DIALOG", value);
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
        const data = await axios.post("/api/v1/exchange/get_models/");
        console.log(data)
        return loadModel
      } catch (error) {
        console.log(error);
      }
    },
  },
  getters: {
    getDialog: ({ dialog }) => dialog,
  },
};
