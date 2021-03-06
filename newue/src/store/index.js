import Vue from "vue";
import Vuex from "vuex";
import axios from 'axios';

import messages from './modules/messages';
import datasets from './modules/datasets';
import logging from './modules/logging';
import projects from './modules/projects';
import themes from './modules/themes';
import create from './modules/create';
import deploy from './modules/deploy';
import settings from './modules/settings';
import cascades from './modules/cascades';
import trainings from './modules/trainings';
import modeling from './modules/modeling';
import servers from './modules/servers';
import dialogs from './modules/dialogs';

Vue.use(Vuex);

export default new Vuex.Store({
  modules: {
    logging,
    messages,
    datasets,
    projects,
    themes,
    create,
    deploy,
    settings,
    cascades,
    trainings,
    modeling,
    servers,
    dialogs
  },
  state: {},
  mutations: {},
  actions: {
    async init({dispatch}){
      await dispatch('themes/setTheme', {}, {root: true});
      await dispatch('projects/get', {}, {root: true});
      await dispatch('datasets/init', {}, {root: true});
    },
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
