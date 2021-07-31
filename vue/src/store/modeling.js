// import { scene } from "./const/typeBlock";
// import { list, layers } from "./const/typeLayers";
const container = {
  centerX: 1042,
  centerY: 140,
  scale: 1,
};

const links = [
  {
    id: 1,
    originID: 1,
    originSlot: 0,
    targetID: 2,
    targetSlot: 0,
  },

]

const blocks = [
  {
    id: 1,
    name: "Вход 1",
    type: "Input",
    group: "input",
    bind: {
      up: [null],
      down: [],
    },
    shape: {
      input: [],
      output: [],
    },
    location: null,
    position: [-900, 50],
    parameters: {
      main: {
        shape: null,
        batch_size: null,
        name: null,
        dtype: null,
        sparse: null,
        ragged: null,
        type_spec: null,
      },
      extra: {},
    },
    reference: null,
  },
  {
    id: 2,
    name: "Conv2D",
    type: "Conv2D",
    group: "middle",
    bind: {
      up: [],
      down: [],
    },
    shape: {
      input: [],
      output: [],
    },
    location: null,
    position: [-900, 150],
    parameters: {
      main: {
        filters: 32,
        kernel_size: [3, 3],
        strides: [1, 1],
        padding: "same",
        activation: "relu",
      },
      extra: {
        data_format: "channels_last",
        dilation_rate: [1, 1],
        groups: 1,
        use_bias: true,
        kernel_initializer: "glorot_uniform",
        bias_initializer: "zeros",
        kernel_regularizer: null,
        bias_regularizer: null,
        activity_regularizer: null,
        kernel_constraint: null,
        bias_constraint: null,
      },
    },
    reference: null,
  },
  {
    id: 3,
    name: "Выход 1",
    type: "Dense",
    group: "output",
    bind: {
      up: [],
      down: [],
    },
    shape: {
      input: [],
      output: [],
    },
    location: null,
    // position: [-900, 250],
    parameters: {
      main: {
        units: 32,
        activation: "relu",
      },
      extra: {
        use_bias: true,
        kernel_initializer: "glorot_uniform",
        bias_initializer: "zeros",
        kernel_regularizer: null,
        bias_regularizer: null,
        activity_regularizer: null,
        kernel_constraint: null,
        bias_constraint: null,
      },
    },
    reference: null,
  },
];

export default {
  namespaced: true,
  state: () => ({
    dialog: false,
    toolbarEvent: {},
    select: null,
    scene: { links, blocks, container },
    model: {},
    modeling: {
      list: [],
      layers_types: {},
    },
  }),
  mutations: {
    SET_MODELING(state, value) {
      state.modeling = { ...value };
    },
    SET_MODEL(state, value) {
      state.model = { ...value };
      // state.scene.blocks = { ...value };
    },
    // SET_LAYERS(state, value) {
    //   state.layers = {...value};
    // },
    SET_LIST(state, value) {
      state.list = [...value];
    },
    SET_DIALOG(state, value) {
      state.dialog = value;
    },
    SET_SELECT(state, value) {
      state.select = value;
    },
    SET_SCENE(state, value) {
      state.scene = { ...value };
    },
    SET_TOOLBAR_EVENT(state, value) {
      state.toolbarEvent = value;
    },
  },
  actions: {
    async info({ dispatch }, value) {
      return await dispatch(
        "axios",
        { url: "/modeling/info/", data: value },
        { root: true }
      );
    },
    async load({ dispatch }, value) {
      return await dispatch(
        "axios",
        { url: "/modeling/load/", data: value },
        { root: true }
      );
    },
    setDialog({ commit }, value) {
      commit("SET_DIALOG", value);
    },
    setSelect({ commit }, value) {
      commit("SET_SELECT", value);
    },
    setScene({ commit }, value) {
      commit("SET_SCENE", value);
    },
    setToolbarEvent({ commit }, value) {
      const { event } = value;
      if (event === "load") {
        commit("SET_DIALOG", true);
      }
      commit("SET_TOOLBAR_EVENT", { ...value });
    },
  },
  getters: {
    getList: ({ modeling: { list } }) => list,
    getLayersType: ({ modeling: { layers_types } }) => layers_types,
    getDialog: ({ dialog }) => dialog,
    getToolbarEvent: ({ toolbarEvent }) => toolbarEvent,
    // getTypeBlock: () => typeBlock,
    // getBlocks: () => blocks,
    getScene: ({ scene }) => scene,
    getSelect: ({ select }) => select,
    getBlock: ({ select, scene }) => {
      const id = scene.blocks.reduce((value, { id }, i) => {
        if (select === id) {
          value = i;
        }
        return value;
      }, -1);
      return scene.blocks[id] || scene.blocks[0];
    },
  },
};
