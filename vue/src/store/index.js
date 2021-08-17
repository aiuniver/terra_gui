import messages from "./modules/messages";
import modeling from "./modules/modeling";
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
    datasets,
    settings,
    trainings,
    deploy,
    projects
  },
  actions: {
    async axios( { dispatch } , config ) {
      Vue.prototype.$Loading.start()
      try {
        Vue.prototype.$Loading.start()
        config.method = config.method || 'post'
        config.url = '/api/v1' + config.url,
        config.data = config.data || {}
        // console.log('config: ', config)
        const response = await axios(config);
        // console.log('response', response)
        const { data: { data, error, success } } = response
        if (success) {
          Vue.prototype.$Loading.finish()
          return data ?? success;
        } else {
          dispatch('messages/setMessage', {error : JSON.stringify(error, null, 2) })
          Vue.prototype.$Loading.error()
          return null;
        }  
      } catch (error) {
        dispatch('messages/setMessage', { error: JSON.stringify(error, null, 2) })
        Vue.prototype.$Loading.error()
        return null;
      }
    },
  },
};
