import messages from './modules/messages';
import modeling from './modules/modeling';
import cascades from './modules/cascades';
import datasets from './modules/datasets';
import settings from './modules/settings';
import projects from './modules/projects';
import trainings from './modules/trainings';
import deploy from './modules/deploy';
import tables from './modules/tables';
import profile from './modules/profile';
import logging from './modules/logging';
import dialogs from './modules/dialogs';

import axios from 'axios';
// import Vue from 'vue';
export default {
  modules: {
    dialogs,
    messages,
    modeling,
    cascades,
    datasets,
    settings,
    trainings,
    deploy,
    projects,
    tables,
    profile,
    logging
  },
  actions: {
    async axios ({ dispatch }, { method = 'post', url, data = {} }) {
      try {
        const { data: res } = await axios({ method, url: '/api/v1' + url, data });
        if (res) {
          const { error, success } = res;
          if (success) dispatch('messages/setMessage', '');
          if (error) dispatch('logging/setError', error);
        }
        return res;
      } catch (error) {
        console.error({ error: JSON.stringify(error, null, 2) })
        // dispatch('messages/setMessage', { error: JSON.stringify(error, null, 2) });
        // dispatch('logging/setError', JSON.stringify(error, null, 2));
        dispatch('settings/setOverlay', false);
        return null;
      }
    },
  },
};
