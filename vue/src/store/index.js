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
import servers from './modules/servers';
import create from './modules/create';

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
    logging,
    servers,
    create,
  },
  actions: {
    async axios ({ dispatch }, { method = 'post', url, data = {} }) {
      try {
        const { data: res } = await axios({ method, url: '/api/v1' + url, data });
        if (res) {
          const { error, logs, success } = res;
          // if (success) dispatch('messages/setMessage', '');
          if (error && !success) {
            dispatch('logging/setError', error);
            dispatch('settings/setOverlay', false);
          }
          if (logs && logs.length) dispatch('logging/setLogs', logs);
        }
        return res;
      } catch (error) {
        console.error({ error: JSON.stringify(error, null, 2) })
        dispatch('settings/setOverlay', false);
        return null;
      }
    },
  },
};
