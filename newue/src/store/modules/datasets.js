const state = () => ({
  datasets: []
});

const mutations = {
  SET_DATASETS(state, value) {
    state.datasets = value
  }
}

const actions = {
  async init({ dispatch }) {
    await dispatch('updateDatasets')
  },
  async updateDatasets({ commit, rootState, dispatch }) {
    const { data } = await dispatch('axios', { url: '/datasets/info/' }, { root: true });
    if (!data) return;
    let datasets = [];
    const selectDataset = rootState.projects.project.dataset?.alias;

    data.forEach(({ datasets: preDataset, alias }) => {
      const tempDataset = preDataset.map(dataset => {
        return { ...dataset, group: alias, active: dataset.alias === selectDataset };
      });
      datasets = [...datasets, ...tempDataset];
    });

    commit('SET_DATASETS', datasets);
  }
}

const getters = {
  getDatasets: ({ datasets }) => datasets,
}


export default {
  namespaced: true,
  state,
  mutations,
  actions,
  getters
}