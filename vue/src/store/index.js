import messages from "./modules/messages";
import modeling from "./modules/modeling";
import cascades from "./modules/cascades";
import datasets from "./modules/datasets";
import settings from "./modules/settings";
import projects from "./modules/projects";
import trainings from "./modules/trainings";
import deploy from "./modules/deploy"

import axios from "axios";
import Vue from 'vue';
export default {
  modules: {
    messages,
    modeling,
    cascades,
    datasets,
    settings,
    trainings,
    deploy,
    projects
  },
  actions: {
    async axios( { dispatch } , config ) {
      // dispatch('messages/setMessage', { mesage: '' })
      Vue.prototype.$Loading.start()
      try {
        Vue.prototype.$Loading.start()
        config.method = config.method || 'post'
        config.url = '/api/v1' + config.url,
        config.data = config.data || {}
        // console.log('config: ', config)
        const response = await axios(config);
        // console.log('response', response)
        const { error, success } = response.data
        if (success) {
          Vue.prototype.$Loading.finish()
        } else {
          dispatch('messages/setMessage', {error : JSON.stringify(error, null, 2) })
          Vue.prototype.$Loading.error()
        }
        return response.data
      } catch (error) {
        dispatch('messages/setMessage', { error: JSON.stringify(error, null, 2) })
        dispatch('settings/setOverlay', false);
        Vue.prototype.$Loading.error()
        return null;
      }
    },
  },
};
