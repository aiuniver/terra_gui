import { createInputData, cloneInputData, changeStructTable, getIdToName } from '../const/datasets';
export default {
  namespaced: true,
  state: () => ({
    inputData: [],
    errors: {},
    tableGroup: [],

    creation: {},
    datasets: [],
    filesSource: [],
    sourcePath: '',
    filesDrop: [],
    selected: null,
    selectedIndex: null,
    tags: [],
    tagsFilter: [],
    full: false,
  }),
  mutations: {
    SET_TABLE_GROUP(state, value) {
      state.tableGroup = [...value];
    },
    SET_DATASETS(state, value) {
      state.datasets = [...value];
    },
    SET_ERRORS(state, value) {
      state.errors = { ...value };
    },
    SET_INPUT_DATA(state, value) {
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
    SET_SOURCE_PATH(state, value) {
      state.sourcePath = value;
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
  },
  actions: {
    async createDataset({ commit, dispatch, state: { inputData, sourcePath, filesSource }, rootState: { tables: { saveCols, handlers } } }, data) {
      commit("settings/SET_OVERLAY", true, { root: true });
      const newDataset = data
      const colsNames = {}

      for (let key in saveCols) {
        colsNames[key] = {}
        saveCols[key].forEach(el => {
          console.log(el)
          const index = getIdToName(filesSource, el)
          if (!colsNames[key][index]) {
            colsNames[key][index] = []
          }
          if (el.value && !colsNames[key][index].includes(index)) {
            colsNames[key][index].push(el.value)
          }

          // colsNames[key][el.label].push(el.value)
        })
      }
      console.log(filesSource)
      console.log(colsNames)
      const inputs = inputData.filter(item => item.layer === 'input').map(item => {
        item.parameters.cols_names = colsNames[item.id]
        return item
      })

      const outputs = inputData.filter(item => item.layer === 'output').map(item => {
        item.parameters.cols_names = colsNames[item.id]
        return item
      })
      newDataset.columns_processing = {}
      handlers.forEach(el => {
        console.log(el)
        newDataset.columns_processing[el.id] = el
      })
      newDataset.source_path = sourcePath
      newDataset.inputs = inputs
      newDataset.outputs = outputs
      const res = await dispatch('axios', { url: '/datasets/create/', data: newDataset }, { root: true });
      console.log(res)
      if (res) {
        const { error } = res
        if (error) {
          const { fields } = error
          if (fields) {
            commit('SET_ERRORS', { ...fields?.inputs || {}, ...fields?.outputs || {} })
          }
        } else {
          commit('SET_INPUT_DATA', []);
          commit('SET_FILES_DROP', []);
          commit('SET_ERRORS', {});
          dispatch('get')
        }
      }
      commit("settings/SET_OVERLAY", false, { root: true });
      return res
    },
    async choice({ commit, dispatch }, dataset) {
      commit('trainings/SET_STATE_PARAMS', {}, { root: true });
      return await dispatch('axios', { url: '/datasets/choice/', data: dataset }, { root: true });
    },
    async deleteDataset({ dispatch }, dataset) {
      const { success } = await dispatch('axios', { url: '/datasets/delete/', data: dataset }, { root: true });
      dispatch('get')
      return success
    },
    async choiceProgress({ dispatch }, source) {
      return await dispatch('axios', { url: '/datasets/choice/progress/', data: source }, { root: true });
    },
    async sourceLoad({ dispatch }, source) {
      return await dispatch('axios', { url: '/datasets/source/load/', data: source }, { root: true });
    },

    async classesAnnotation({ dispatch, state: { sourcePath } }) {
      const data = { path: sourcePath }
      return await dispatch('axios', { url: '/datasets/source/segmentation/classes/annotation/', data }, { root: true });
    },
    async classesAutosearch({ dispatch }, data) {
      return await dispatch('axios', { url: '/datasets/source/segmentation/classes/autosearch/', data }, { root: true });
    },

    async loadProgress({ dispatch }, source) {
      return await dispatch('axios', { url: '/datasets/source/load/progress/', data: source }, { root: true });
    },
    async get({ dispatch, commit, rootState }) {
      const { data } = await dispatch('axios', { url: '/datasets/info/' }, { root: true });
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
      console.log(changeStructTable(value))
      commit('SET_FILES_SOURCE', value);
    },
    setSourcePath({ commit }, value) {
      commit('SET_SOURCE_PATH', value);
    },
    setFilesDrop({ commit }, value) {
      commit('SET_FILES_DROP', value);
    },
    setSelectedIndex({ commit }, value) {
      commit('SET_SELECTED_INDEX', value);
    },
    createInputData({ commit, state: { inputData } }, { layer }) {
      let maxID = Math.max(0, ...inputData.map(o => o.id));
      const usedColors = inputData.map(item => item.color)
      commit('SET_INPUT_DATA', [...inputData, createInputData(maxID + 1, layer, usedColors)]);
    },
    cloneInputData({ commit, state: { inputData } }, id) {
      let maxID = Math.max(0, ...inputData.map(o => o.id));
      const usedColors = inputData.map(item => item.color)
      const layer = inputData.find(item => item.id === id)
      commit('SET_INPUT_DATA', [...inputData, cloneInputData(maxID + 1, usedColors, layer)]);
    },
    clearInputData({ commit }) {
      commit('SET_INPUT_DATA', []);
    },
    updateInputData({ commit, state: { inputData } }, { id, name, value, root }) {
      const index = inputData.findIndex(item => item.id === id);
      if (index !== -1) {
        if (root && name === 'type') {
          const obj = inputData[index].parameters.sources_paths || []
          // inputData[index].parameters = {}
          inputData[index].parameters.sources_paths = obj
        }
        if (root) {
          inputData[index][name] = value
        } else {
          inputData[index].parameters[name] = value
        }
        commit('SET_INPUT_DATA', [...inputData]);
        // console.log(inputData)
      }
    },
    removeInputData({ commit, state: { inputData } }, id) {
      commit(
        'SET_INPUT_DATA',
        inputData.filter(item => item.id !== id)
      );
    },
    cleanError({ state: { errors } }, { id, name }) {
      if (errors?.[id]?.[name]) {
        errors[id][name] = ''
      }
      if (errors?.[id]?.['parameters']?.[name]) {
        errors[id]['parameters'][name] = ''
      }
    },
    setErrors({ commit, state: { errors } }, error) {
      commit('SET_ERRORS', { ...errors, ...error })
    },
    setTableGroup({ commit }, data) {
      commit('SET_TABLE_GROUP', data)
    }
  },
  getters: {
    getTableGroup({ tableGroup }) {
      return tableGroup;
    },
    getInputData({ inputData }) {
      // console.log(inputData)
      return inputData;
    },
    getErrors: ({ errors }) => (id) => {
      return errors?.[id] || {};
    },
    getInputDataByID:
      ({ inputData }) =>
        id => {
          // console.log(inputData, id);
          return inputData.find(item => item.id === id);
        },
    getTypeInput({ creation: { input } }) {
      return input || [];
    },
    getFormsHandler({ creation: { column_processing } }) {
      return column_processing || [];
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
