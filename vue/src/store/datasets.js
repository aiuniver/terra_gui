import axios from "axios";
import inputs from "@/json";
export default {
  namespaced: true,
  state: () => ({
    datasets: [],
    tags: [],
    settings: {},
    sort: "",
    tagsFilter: [],
    id: null,
    dialog: false,
    full: false
  }),
  mutations: {
    SET_DATASETS(state, value) {
      state.datasets = [...value];
    },
    SET_SETTINGS(state, value) {
      state.settings = { ...value };
    },
    SET_TAGS(state, tags) {
      state.tags = [...tags];
    },
    SET_TAGS_FILTER(state, value) {
      state.tagsFilter = value;
    },
    SET_FULL(state, value) {
      state.full = value;
    },
  },
  actions: {
    async axios(_, params) {
      try {
        const { data } = await axios(params);
        return data;
      } catch (error) {
        console.log(error);
      }
    },
    async settings({ commit }, { inputs, outputs, name }) {
      try {
        const res = {
          link: "",
          mode: "google_drive",
          name,
          num_links: {
            inputs: +inputs,
            outputs: +outputs,
          },
        };
        const {
          data: { data },
        } = await axios.post("/api/v1/exchange/load_dataset/", res);
        console.log(data);
        commit("SET_SETTINGS", data);
        return data;
      } catch (error) {
        console.log(error);
      }
    },
    async get({ commit }) {
      try {
        const { data: { data: [preset, custom] } } = await axios.get("/api/v1/datasets/info/");
        const { datasets:presetDatasets, tags:presetTags } = preset
        const { datasets:customDatasets, tags:customTags } = custom
        const datasets = [...presetDatasets, ...customDatasets]
        console.log(datasets)
        let tags = [...presetTags, ...customTags]
        console.log(tags)
        tags = tags.map((tag) => {
          return {active: false, ...tag }
        });
        commit("SET_DATASETS", datasets);
        commit("SET_TAGS", tags);
      } catch (error) {
        console.log(error);
      }
    },
    async add({ commit }, name) {
      try {
        console.log(name);
        const { data } = await axios.post(
          "/api/v1/exchange/prepare_dataset/",
          name
        );
        console.log(name);
        commit("SET_DATASETS", name);
        return data;
      } catch (error) {
        console.log(error);
      }
    },
    setTagsFilter({ commit }, value) {
      commit("SET_TAGS_FILTER", value);
    },
    setFull({ commit }, value) {
      commit("SET_FULL", value);
    },
  },
  getters: {
    getSettings() {
      return inputs;
    },
    getFull({ full }) {
      return full;
    },
    getTags({ tags }) {
      return tags;
    },
    getTagsFilter({ tagsFilter }) {
      return tagsFilter;
    },
    getDatasets({ datasets, tagsFilter }) {
      if (!tagsFilter.length) {
        return datasets;
      }
      console.log(tagsFilter)
      return datasets.filter(({tags}) => {
        const index = tags.filter(({alias}) => {
          return tagsFilter.indexOf(alias) !== -1;
        });
        return index.length;
      });
    },
  },
};
