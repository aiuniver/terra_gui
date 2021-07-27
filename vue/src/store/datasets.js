import inputs from "./temp/json";
export default {
  namespaced: true,
  state: () => ({
    datasets: [],
    selected: null,
    tags: [],
    settings: {},
    sort: "",
    tagsFilter: [],
    id: null,
    dialog: false,
    full: false,
  }),
  mutations: {
    SET_DATASETS(state, value) {
      state.datasets = [...value];
    },
    SET_SELECTED(state, value) {
      state.selected = value;
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
    async choice({ commit, dispatch }, dataset) {
      const data = await dispatch("axios", { url: "/datasets/choice/", data: dataset }, { root: true });
      if (data) {
        commit("projects/SET_PROJECT", { dataset: data }, { root: true })
        dispatch('messages/setMessage', { message: `Датасет «${data.name}» загружен`}, { root: true })
      }
      // console.log(rootState)
      return data;
    },
    async sourceLoad({ dispatch }, source ) {
      return await dispatch("axios",{ url: "/datasets/source/load/", data: source }, { root: true });
    },
    async loadProgress({ dispatch }, source ) {
      return await dispatch("axios",{ url: "/datasets/source/load/progress/", data: source }, { root: true });
    },
    async get({ dispatch, commit, rootState }) {
      const data = await dispatch(
        "axios",
        { url: "/datasets/info/" },
        { root: true }
      );
      if (!data) {
        return;
      }
      const [preset, custom] = data;
      const { datasets: presetDatasets, tags: presetTags } = preset;
      const { datasets: customDatasets, tags: customTags } = custom;
      const preDataset = presetDatasets.map((item) => {
        return {...item, group: 'keras'}
      })
      const cusDataset = customDatasets.map((item) => {
        return {...item, group: 'custom'}
      })
      const alias = rootState.projects.project.dataset?.alias
      let datasets = [...preDataset, ...cusDataset];
      datasets = datasets.map((dataset) => {
        return { ...dataset, active: ( dataset.alias === alias) };
      });
      let tags = [...presetTags, ...customTags];
      tags = tags.map((tag) => {
        return { active: false, ...tag };
      });
      commit("SET_DATASETS", datasets);
      commit("SET_TAGS", tags);
    },
    setSelect({ commit, state: { datasets } }, dataset) {
      const data = datasets.map((item) => {
        item.active = item.name === dataset.name;
        return item;
      });
      commit("SET_DATASETS", data);
      commit("SET_SELECTED", dataset);
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
    getSelected({ selected }) {
      return selected;
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
      return datasets.filter(({ tags }) => {
        const index = tags.filter(({ alias }) => {
          return tagsFilter.indexOf(alias) !== -1;
        });
        return index.length === tagsFilter.length;
      });
    },
  },
};
