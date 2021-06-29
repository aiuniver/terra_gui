import Vue from "vue";
import App from "./App.vue";
import Vuex from "vuex";
import VueRouter from "vue-router";
import axios from "axios";
import VueAxios from "vue-axios";

import routes from "./routers/index";
import store from "./store/index";
import Vuetify from "vuetify";

import "vuetify/dist/vuetify.min.css";
import "./assets/css/main.scss";
import colors from "vuetify/lib/util/colors";

const opts = {
  theme: {
    dark: true,
    themes: {
      light: {
        primary: '#17212b',
        secondary: colors.grey.darken1,
        accent: colors.shades.black,
        error: colors.red.accent3,
        background: '#292930',
      },
      dark: {
        primary: '#65b9f4',
        secondary: '#65b9f4',
        info: '#2b5278',
        accent: '#242f3d',
        error: '#FF5252',
        success: '#A7BED3',
        warning: '#FFC107',
        text: '#65b9f4',
        input: '#f6af54',
        action: '#89d764',
        output: '#9166f2',
        background: '#17212b',
      }
    }
  },
};

Vue.config.productionTip = false;
Vue.use(Vuex);
Vue.use(VueAxios, axios);
Vue.use(VueRouter);
Vue.use(Vuetify);

new Vue({
  router: new VueRouter({
    mode: "history",
    routes,
  }),
  store: new Vuex.Store(store),
  vuetify: new Vuetify(opts),
  render: (h) => h(App),
}).$mount("#app");
