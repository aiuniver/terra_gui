import { blocks, typeBlock, scene } from "./const/typeBlock";
import { list, layers } from "./const/typeLayers";
export default {
  namespaced: true,
  state: () => ({
    dialog: false,
    toolbarEvent: {},
    select: null,
    scene: { ...scene },
    list: [...list],
    layers: {...layers}
  }),
  mutations: {
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
    getList: ({ list }) => list,
    getLayers: ({ layers }) => layers,
    getDialog: ({ dialog }) => dialog,
    getToolbarEvent: ({ toolbarEvent }) => toolbarEvent,
    getTypeBlock: () => typeBlock,
    getBlocks: () => blocks,
    getScene: ({ scene }) => scene,
    getSelect: ({ select }) => select,
    getBlock: ({ select, scene }) => {
      const id = scene.blocks.reduce((value, { id }, i) => {
        if (select === id) {
          value = i
        }
        return value;
      }, -1);
      return scene.blocks[id] || scene.blocks[0]
    },
  },
};
