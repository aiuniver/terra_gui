export default {
  namespaced: true,
  state: () => ({
    data: {
      centerX: 1024,
      centerY: 140,
      scale: 1,
      layers: [
        {
          id: 1,
          name: "input_1",
          type: "Input",
          group: "input",
          bind: [2],
          shape: [1, 1, 1],
          location: null,
          position: [320, 100],
          parameters: {}
        },
        {
          id: 2,
          name: "Длинное имя дададада",
          type: "Conv1D",
          group: "middle",
          bind: [3],
          shape: [1, 1, 1],
          location: null,
          position: [400, 200],
          parameters: {}
        },
        {
          id: 3,
          name: "tretiy",
          type: "Conv2D",
          group: "middle",
          bind: [4],
          shape: [1, 1, 1],
          location: null,
          position: [320, 300],
          parameters: {}
        },
        {
          id: 4,
          name: "4ka",
          type: "Conv3D",
          group: "middle",
          bind: [5],
          shape: [1, 1, 1],
          location: null,
          position: [400, 400],
          parameters: {}
        },
        {
          id: 5,
          name: "Output_1",
          type: "Dense",
          group: "output",
          bind: [],
          shape: [1, 1, 1],
          location: null,
          position: [320, 500],
          parameters: {}
        }
      ],

      nodes: [
        {
          id: 1,
          x: -650,
          y: -100,
          type: "input",
          label: "test1",
        },
        {
          id: 2,
          x: -500,
          y: 50,
          type: "middle",
          label: "test2",
        },
        {
          id: 3,
          x: -800,
          y: 50,
          type: "middle",
          label: "test3",
        },
        {
          id: 4,
          x: -650,
          y: 150,
          type: "output",
          label: "test4",
        },
      ],
      links: [
        {
          id: 1,
          from: 1,
          to: 2,
        },
        {
          id: 2,
          from: 2,
          to: 3,
        },
        {
          id: 3,
          from: 3,
          to: 4,
        },
        {
          id: 4,
          from: 4,
          to: 5,
        },
      ],
    },
    
  }),
  mutations: {
    SET_DATA(state, value) {
      state.data = [...value];
    },
    SET_LAYERS(state, layers){
      state.data.layers = layers;
    },
    ADD_LAYER(state, layer){
      state.data.layers.push(layer);
    },
    // UPDATE_LAYER(state, index, data){
    //   state.data.layers[index] = data;
    // },
    // DELETE_LAYER(state, index){
    //   state.data.layers.splice(index, 1);
    // },
  },
  actions: {
    setData ({ commit }, data) {
      commit("SET_DATA", data);
    },
    addLayer({ commit, state }, layer){
      let maxId = Math.max(...state.data.layers.map(item => item.id), 0);
      layer.id = maxId + 1
      commit("ADD_LAYER", layer);
    }
  },
  getters: {
    getData(state) {
      return state.data;
    },
    // getLayer(state, index){
    //   return state.data.layers[index]
    // },
  }
};
