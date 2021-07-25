import inputs from "./temp/json";
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
    async settings({ commit, dispatch }, { inputs, outputs, name }) {
        const res = {
          link: "",
          mode: "google_drive",
          name,
          num_links: {
            inputs: +inputs,
            outputs: +outputs,
          },
        };
        const data = await dispatch('axios', {url: "/exchange/load_dataset/", data: res}, {root: true});
        commit("SET_SETTINGS", data);
        return data;
    },
    async get({ dispatch, commit }) {
      const data = await dispatch('axios', {url: "/datasets/info/"}, {root: true});
      if (!data) {
        return;
      }
      const [ preset, custom ]  = data
      const { datasets:presetDatasets, tags:presetTags } = preset
      const { datasets:customDatasets, tags:customTags } = custom
      let datasets = [...presetDatasets, ...customDatasets]
      datasets = datasets.map((dataset) => {
        return  {...dataset, active: false}
      })
      let tags = [...presetTags, ...customTags]
      tags = tags.map((tag) => {
        return {active: false, ...tag }
      });
      commit("SET_DATASETS", datasets);
      commit("SET_TAGS", tags);
    },
    setSelect({ commit, state:{datasets}  }, dataset){
      const data = datasets.map((item) => {
        item.active = (item.name === dataset.name)
        return item
      })
      commit("SET_DATASETS", data);
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
        console.log(index)
        return index.length === tagsFilter.length
      });
    },
  },
};
