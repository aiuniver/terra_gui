import temp from "../temp/training";
import axios from "axios";
export default {
  namespaced: true,
  state: () => ({
    data: temp.data,
    toolbar: [
      {
        title: "Графики",
        active: true,
        disabled: false,
        icon: "icon-training-charts",
      },
      {
        title: "Скаттеры",
        active: false,
        disabled: true,
        icon: "icon-training-scatters",
      },
      {
        title: "Изображения",
        active: false,
        disabled: false,
        icon: "icon-training-images",
      },
      {
        title: "Текст",
        active: true,
        disabled: false,
        icon: "icon-training-texts",
      },
    ],
  }),
  mutations: {
    SET_DRAWER(state, value) {
      state.drawer = value;
    },
  },
  actions: {
    setDrawer({ commit }, data) {
      commit("SET_DRAWER", data);
    },
    async data() {
      try {
        const { data } = await axios.post("/api/v1/exchange/get_data/", {});
        // commit("SET_SETTINGS", data);
        console.log(data)
        return data;
      } catch (error) {
        console.log(error);
      }
    },
  },
  getters: {
    getData () {
      return 0
    },
    getToolbar({ toolbar }) {
      return toolbar;
    },
    getToolbarChars({ toolbar }) {
      return toolbar[0].active;
    },
    getToolbarScatters({ toolbar }) {
      return toolbar[1].active;
    },
    getToolbarImages({ toolbar }) {
      return toolbar[2].active;
    },
    getToolbarTexts({ toolbar }) {
      return toolbar[3].active;
    },
    getChars({ data: { plots } }) {
      return plots;
    },
    getScatters({ data: {scatters} }) {
      return scatters;
    },
    getImages({ data: { images: { images } } }) {
      return images;
    },
    getTexts({ data: { texts } }) {
      return texts;
    },
  },
};
