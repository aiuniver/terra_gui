import { prepareBlocks, prepareLinks, createBlock, changeTypeBlock, cloneBlock } from '../const/cascades';

export default {
  namespaced: true,
  state: () => ({
    select: null,
    model: {},
    blocks: [],
    errorsBlocks: {},
    errorsFields: {},
    links: [],
    cascades: {
      list: [
        {
          "value": "InputData",
          "label": "InputData"
        },
        {
          "value": "OutputData",
          "label": "OutputData"
        },
        {
          "value": "Model",
          "label": "Model"
        },
        {
          "value": "Function",
          "label": "Function"
        },
        {
          "value": "Custom",
          "label": "Custom"
        }
      ]
    },
    status: {
      isUpdate: true,
    },
  }),
  mutations: {
    SET_CASCADES (state, value) {
      // const list = value.block_form[1]['list'] || []
      state.cascades = { ...value };
    },
    SET_ERRORS_BLOCKS (state, value) {
      state.errorsBlocks = { ...value }
    },
    SET_ERRORS_FIELDS (state, value) {
      state.errorsFields = { ...value }
    },
    SET_MODEL (state, value) {
      state.model = { ...value };
      const { blocks } = value;
      state.blocks = [...prepareBlocks(blocks, state.cascades.list)];
      state.links = [...prepareLinks(blocks)];
    },
    SET_BLOCKS (state, value) {
      state.blocks = [...value];
    },
    SET_LINKS (state, value) {
      state.links = [...value];
    },
    SET_LIST (state, value) {
      state.list = [...value];
    },
    SET_SELECT (state, value) {
      state.select = value;
    },
    SET_STATUS (state, value) {
      state.status = { ...state.status, ...value };
    },
  },
  actions: {
    addBlock ({ dispatch, commit, state: { blocks, cascades: { blocks_types, list } } }, { type, position }) {
      let maxID = Math.max(0, ...blocks.map(o => o.id));
      console.log(block)
      let block = createBlock(type, maxID + 1, blocks_types, list, position);
      if (!block) return;
      blocks.push(block);
      // dispatch('updateModel');
      commit('SET_BLOCKS', blocks);
      dispatch('selectBlock', block)
    },
    typeBlock ({ dispatch, commit, state: { blocks, cascades: { layers_types, list } } }, { type, block }) {
      let newBlock = changeTypeBlock(type, block, layers_types, list);
      if (!newBlock) return;
      // blocks.push(block);
      commit('SET_BLOCKS', blocks);
      dispatch('updateModel');
    },
    cloneBlock ({ dispatch, commit, state: { blocks } }, oldBlock) {
      let maxID = Math.max(0, ...blocks.map(o => o.id));
      const block = cloneBlock(oldBlock, maxID + 1);
      if (!block) return;
      blocks.push(block);
      commit('SET_BLOCKS', blocks);
      dispatch('updateModel');
    },
    selectBlock ({ commit, state: { blocks } }, block) {
      blocks.forEach(item => {
        item.selected = item.id === block.id
      })
      commit('SET_BLOCKS', blocks);
      commit('SET_SELECT', block.id);
    },
    deselectBlocks ({ commit, state: { blocks } }) {
      blocks.forEach(item => {
        item.selected = false
      })
      commit('SET_BLOCKS', blocks);
      commit('SET_SELECT', null);
    },
    removeBlock ({ dispatch, commit, state: { blocks } }, block) {
      if (block.selected) {
        block.selected = false;
      }
      dispatch('removeLinkToBlock', block);
      commit('SET_BLOCKS', blocks.filter(b => b.id !== block.id));
      dispatch('updateModel');
    },
    removeLink ({ commit, state: { links } }, id) {
      console.log(id)
      commit('SET_LINKS', links.filter(value => value.id !== id));
    },
    removeLinkToBlock ({ dispatch, commit, state: { links } }, block) {
      console.log(block)
      commit('SET_LINKS', links.filter(link => (link.originID !== block.id && link.targetID !== block.id)));
      dispatch('updateModel');
    },

    async info ({ dispatch }, value) {
      return await dispatch('axios', { url: '/cascades/info/', data: value }, { root: true });
    },
    async load ({ commit, dispatch }, { model, reset_dataset }) {
      const { data } = await dispatch('axios', {
        url: '/cascades/load/', data: {
          ...model,
          reset_dataset
        }
      }, { root: true });

      if (data) {
        commit('SET_ERRORS_BLOCKS', {});
        await dispatch('projects/get', {}, { root: true });
        await dispatch('validateModel', {});
      }

      return data
    },
    async createModel ({ dispatch, commit }, data) {
      commit('SET_STATUS', { isUpdate: false });
      return await dispatch('axios', { url: '/cascades/create/', data }, { root: true });
    },
    async getImageModel ({ dispatch }, preview) {
      return await dispatch('axios', {
        url: '/cascades/preview/', data: {
          preview
        }
      }, { root: true });
    },
    async removeModel ({ dispatch }, data) {
      return await dispatch('axios', { url: '/cascades/delete/', data }, { root: true });
    },
    async updateModel ({ commit, state: { blocks, links }, dispatch }, block) {
      const semdBlocks = JSON.parse(JSON.stringify(blocks))
      semdBlocks.forEach(block => {
        // if (block.group !== 'input') block.shape.input = null;
        if (block?.shape?.output && !block.shape.output.length) block.shape.output = null
        block.bind.up = links
          .map(link => {
            return link.targetID === block.id ? link.originID : null;
          })
          .filter(link => link);
        block.bind.down = links
          .map(link => {
            return link.originID === block.id ? link.targetID : null;
          })
          .filter(link => link);
      });
      commit('SET_STATUS', { isUpdate: true });

      const res = await dispatch('axios', { url: '/cascades/update/', data: { blocks: semdBlocks } }, { root: true });
      if (res) {
        const { data, error, success } = res
        console.log(data, error, success, block)
        if (error) {
          // const { general, fields } = error

          // const newError = {}
          // for (const key in error) {
          //   newError[key.replace('fields', block.id)] = error[key]
          // }
          console.log(error)
          // commit('SET_ERRORS_FIELDS', { ...errorsBlocks, ...newError });
        }
      }
      return res
    },
    async getModel ({ dispatch }, value) {
      return await dispatch('axios', { url: '/cascades/get/', data: value }, { root: true });
    },
    async changeId ({ commit, dispatch }, { value, id }) {
      const { data } = await dispatch('axios', {
        url: '/cascades/datatype/', data: {
          "source": id,
          "target": value
        }
      }, { root: true });
      if (data) {
        dispatch('deselectBlocks')
        commit("cascades/SET_MODEL", data, { root: true });
      }
      return data
    },
    resetAll ({ commit },) {
      commit('SET_ERRORS_BLOCKS', {})
      return
    },
    async clearModel ({ commit, dispatch }) {
      const res = await dispatch('axios', { url: '/cascades/clear/' }, { root: true });
      if (res.success) {
        // console.log(res)
        commit('SET_ERRORS_BLOCKS', {})
        await dispatch('projects/get', {}, { root: true });
      }
      return res
    },
    async validateModel ({ commit, dispatch }) {
      const { data } = await dispatch('axios', { url: '/cascades/validate/' }, { root: true });
      if (data) {
        const isValid = !Object.values(data).filter(item => item).length
        commit('SET_ERRORS_BLOCKS', data)
        if (isValid) {
          await dispatch('projects/get', {}, { root: true })
        }
        dispatch('messages/setMessage', isValid ? { message: `Валидация прошла успешно` } : { error: `Валидация не прошла` }, { root: true });
      }
      return data;
    },
    setBlocks ({ commit }, value) {
      commit('SET_BLOCKS', value);
    },
    setLinks ({ commit }, value) {
      commit('SET_LINKS', value);
    },
    setBlock ({ commit, state: { blocks } }, value) {
      const index = blocks.findIndex(item => item.id == value.id);
      blocks[index] = value;
      // console.log(blocks);
      commit('SET_BLOCKS', blocks);
    },
  },
  getters: {
    getList: ({ cascades: { list } }) => list,
    getLayersType: ({ cascades: { blocks_types } }) => blocks_types || {},
    getLayersForm: ({ cascades: { block_form } }) => block_form || [],
    getModel: ({ model }) => model,
    getBlocks: ({ blocks }) => blocks,
    getErrorsBlocks: ({ errorsBlocks }) => errorsBlocks,
    getErrorsFields: ({ errorsFields }) => errorsFields,
    getLinks: ({ links }) => links,
    getSelect: ({ select }) => select,
    getStatus: ({ status }) => status,
    getBlock: ({ select, blocks }) => {
      const id = blocks.findIndex(item => item.id == select);
      return blocks[id] || {};
    },
  },
};
