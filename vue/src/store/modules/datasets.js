import { createInputData } from '../const/datasets';
export default {
  namespaced: true,
  state: () => ({
    inputData: [],

    creation: {},
    datasets: [],
    filesSource: [],
    filesDrop: [],
    selected: null,
    selectedIndex: null,
    loaded: null,
    tags: [],
    tagsFilter: [],
    full: false,
  }),
  mutations: {
    SET_DATASETS(state, value) {
      state.datasets = [...value];
    },
    SET_INPUT_DATA(state, value) {
      state.inputData = value;
    },
    SET_NEW_DATASET(state, value) {
      state.inputData = value;
    },
    SET_SELECTED(state, value) {
      state.selected = value;
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
    SET_FILES_SOURCE(state, value) {
      state.filesSource = value;
    },
    SET_FILES_DROP(state, value) {
      state.filesDrop = value;
    },
    SET_CREATION(state, value) {
      state.creation = value;
    },
    SET_SELECTED_INDEX(state, value) {
      state.selectedIndex = value;
    },
    SET_LOADED(state, value) {
      state.loaded = value;
    },
  },
  actions: {
    async createDataset({ dispatch, state: { inputData } }, data) {
      const newDataset = data
      const inputs = inputData.filter(item => item.layer === 'input')
      const outputs = inputData.filter(item => item.layer === 'output')
      newDataset.inputs = inputs
      newDataset.outputs = outputs
      console.log(newDataset)

      return await dispatch('axios', { url: '/datasets/source/create/', data: newDataset }, { root: true });
    },
    async choice({ dispatch }, dataset) {
      return await dispatch('axios', { url: '/datasets/choice/', data: dataset }, { root: true });
    },
    async choiceProgress({ dispatch }, source) {
      return await dispatch('axios', { url: '/datasets/choice/progress/', data: source }, { root: true });
    },
    async sourceLoad({ dispatch }, source) {
      return await dispatch('axios', { url: '/datasets/source/load/', data: source }, { root: true });
    },
    async loadProgress({ dispatch }, source) {
      return await dispatch('axios', { url: '/datasets/source/load/progress/', data: source }, { root: true });
    },
    async get({ dispatch, commit, rootState }) {
      const data = await dispatch('axios', { url: '/datasets/info/' }, { root: true });
      if (!data) {
        return;
      }
      let datasets = [];
      let tags = [];
      const selectDataset = rootState.projects.project.dataset?.alias;

      data.forEach(function ({ datasets: preDataset, tags: preTags, alias }) {
        const tempDataset = preDataset.map(dataset => {
          return { ...dataset, group: alias, active: dataset.alias === selectDataset };
        });
        datasets = [...datasets, ...tempDataset];
        const tempTags = preTags.filter(tag => {
          const isTrue = tags.filter(({ alias }) => {
            return alias === tag.alias;
          });
          return !isTrue.length;
        });
        tags = [...tags, ...tempTags];
      });

      tags = tags.map(tag => {
        return { active: false, ...tag };
      });
      commit('SET_DATASETS', datasets);
      commit('SET_TAGS', tags);
    },
    setSelect({ commit, state: { datasets } }, dataset) {
      const data = datasets.map(item => {
        item.active = item.name === dataset.name;
        return item;
      });
      commit('SET_DATASETS', data);
      commit('SET_SELECTED', dataset);
    },
    setTagsFilter({ commit }, value) {
      commit('SET_TAGS_FILTER', value);
    },
    setFull({ commit }, value) {
      commit('SET_FULL', value);
    },
    setFilesSource({ commit }, value) {
      commit('SET_FILES_SOURCE', value);
    },
    setFilesDrop({ commit }, value) {
      commit('SET_FILES_DROP', value);
    },
    setSelectedIndex({ commit }, value) {
      commit('SET_SELECTED_INDEX', value);
    },
    setLoaded({ commit }, value) {
      commit('SET_LOADED', value);
    },
    createInputData({ commit, state: { inputData } }, { layer }) {
      let maxID = Math.max(0,...inputData.map(o => o.id));
      commit('SET_INPUT_DATA', [...inputData, createInputData(maxID + 1, layer)]);
    },
    updateInputData({ state: { inputData } }, { id, name, value, root }) {
      const index = inputData.findIndex(item => item.id === id);
      if (index !== -1) {
        if (root && name === 'type') {
          const obj = inputData[index].parameters.sources_paths || []
          inputData[index].parameters = {}
          inputData[index].parameters.sources_paths = obj
        }
        if (root) {
          inputData[index][name] = value
        } else {
          inputData[index].parameters[name] = value
        }
        // console.log(inputData)
      }
    },
    removeInputData({ commit, state: { inputData } }, id) {
      commit(
        'SET_INPUT_DATA',
        inputData.filter(item => item.id !== id)
      );
    },
  },
  getters: {
    getInputData({ inputData }) {
      return inputData;
    },
    getInputDataByID:
      ({ inputData }) =>
      id => {
        console.log(inputData, id);
        return inputData.find(item => item.id === id);
      },
    getTypeInput({ creation: { input } }) {
      return input || [];
    },
    getTypeOutput({ creation: { output } }) {
      return output || [];
    },
    getSelected({ selected }) {
      return selected;
    },
    getFilesSource({ filesSource }) {
      return filesSource;
    },
    getFilesDrop({ filesDrop }) {
      return filesDrop;
    },
    getFull({ full }) {
      return full;
    },
    getTags({ tags }) {
      return tags;
    },
    getSelectedIndex({ selectedIndex }) {
      return selectedIndex;
    },
    getLoaded({ loaded }) {
      return loaded;
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
