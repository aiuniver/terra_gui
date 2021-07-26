import messages from "./messages";
import modeling from "./modeling";
import datasets from "./datasets";
import settings from "./settings";
import projects from "./projects";
import trainings from "./trainings";
import data from "./data";
import axios from "axios";
import Vue from 'vue';
export default {
  modules: {
    messages,
    modeling,
    datasets,
    data,
    settings,
    trainings,
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
        console.log('config: ', config)
        const response = await axios(config);
        console.log('response', response)
        const { data: { data, error, success } } = response
        if (success) {
          Vue.prototype.$Loading.finish()
          return data;
        } else {
          dispatch('messages/setMessage', {error : JSON.stringify(error) })
          Vue.prototype.$Loading.error()
          return null;
        }  
      } catch (error) {
        dispatch('messages/setMessage', { error: JSON.stringify(error) })
        Vue.prototype.$Loading.error()
        return null;
      }
    },
  },
};
