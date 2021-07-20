import messages from "./messages";
import modeling from "./modeling";
import datasets from "./datasets";
import settings from "./settings";
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
  },
  actions: {
    async axios( { dispatch } , { url = '/', method = 'get'}) {
      Vue.prototype.$Loading.start()
      try {
        Vue.prototype.$Loading.start()
        const req = {
          method,
          url: '/api/v1' + url,
        };
        const { data: { data, error, success } } = await axios(req);
        if (success) {
          Vue.prototype.$Loading.finish()
          return data;
        } else {
          dispatch('messages/setMessage', {error : error.general[0]})
          Vue.prototype.$Loading.error()
          return null;
        }  
      } catch (error) {
        dispatch('messages/setMessage', { error })
        Vue.prototype.$Loading.error()
        return null;
      }
    },
  },
};
