import Vue from "vue";
import Vuex from "vuex";
import axios from 'axios';

import messages from './modules/messages';
import datasets from './modules/datasets';
import logging from './modules/logging';
import projects from './modules/projects';
import themes from './modules/themes';

Vue.use(Vuex);

export default new Vuex.Store({
  modules: {
    logging,
    messages,
    datasets,
    projects,
    themes
  },
  state: {},
  mutations: {},
  actions: {
    async axios ({ dispatch }, { method = 'post', url, data = {} }) {
      try {
        const response = await axios({ method, url: '/api/v1' + url, data });
        const { error, success } = response.data;
        if (success) {
          dispatch('messages/setMessage', '');
        } else {
          dispatch('messages/setMessage', { error: JSON.stringify(error, null, 2) });
          dispatch('logging/setError', JSON.stringify(error, null, 2));
        }
        return response.data;
      } catch (error) {
        dispatch('messages/setMessage', { error: JSON.stringify(error, null, 2) });
        dispatch('logging/setError', JSON.stringify(error, null, 2));
        dispatch('settings/setOverlay', false);
        return null;
      }
    },
  },
});