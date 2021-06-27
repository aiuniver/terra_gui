import axios from "axios";

export default {
  namespaced: true,
  state: () => ({
    datasets: [],
    tags: [
      "images",
      "tag",
      "smarthome",
      "text",
      "bird",
      "time",
      "cloud",
      "cats",
      "smile",
      "black",
      "nitify",
    ],
    sort: "",
    tagsFilter: [],
    id: null,
    dialog: false,
  }),
  mutations: {
    SET_DATASETS(state, value) {
      state.datasets = [...value];
    },
    SET_ADD_DATASET(state, value) {
      state.datasets.push(value);
      state.datasets = [...state.datasets];
    },
    SET_TAGS_FILTER(state, value) {
      state.tagsFilter = value;
    },
  },
  actions: {
    async get({ commit }) {
      console.log("sdsdsddsdsdsd");
      try {
        const { data } = await axios.get(
          "https://60d20d1f5b017400178f5047.mockapi.io/api/v1/datasets"
        );
        commit("SET_DATASETS", data);
      } catch (error) {
        console.log(error);
      }
    },
    async add({ commit }, user) {
      try {
        const { data } = await axios.post(
          "https://60d20d1f5b017400178f5047.mockapi.io/api/v1/datasets",
          user
        );
        if (data) {
          commit("SET_ADD_DATASET", data);
        }
        return data;
      } catch (error) {
        console.log(error);
      }
    },
    async edit({ dispatch }, dataset) {
      try {
        console.log(dataset);
        const { data } = await axios.put(
          "https://60d20d1f5b017400178f5047.mockapi.io/api/v1/datasets/" + dataset.id,
          dataset
        );
        console.log(data);
        if (data) {
          dispatch("get");
        }
        return data;
      } catch (error) {
        console.log(error);
      }
    },
    async delete({ commit, state }, id) {
      try {
        console.log(id)
        const { data } = await axios.delete(
          "https://60d20d1f5b017400178f5047.mockapi.io/api/v1/datasets/" + id
        );
        if (data) {
          const update = state.datasets.filter((dataset) => {
            return dataset.id !== id;
          });
          commit("SET_DATASETS", update);
        }
        return data;
      } catch (error) {
        console.log(error);
      }
    },
    setTagsFilter({ commit }, value) {
      commit("SET_TAGS_FILTER", value);
    },
  },
  getters: {
    getTags({ tags }) {
      return tags;
    },
    getTagsFilter({ tagsFilter }) {
      return tagsFilter;
    },
    getDatasets({ datasets, tagsFilter, tags }) {
      const arr = tags.filter((_, i) => {
        return tagsFilter.indexOf(i) !== -1;
      })
      if (!tagsFilter.length) {
        return datasets
      } 
      return datasets.filter((dataset) => {
        const index = dataset.tags.filter((tag) => {
          return arr.indexOf(tag) !== -1;
        })
        return index.length
      })
    },
  },
};
