import Vue from "vue";
import App from "./App.vue";
import router from "./router";
import store from "./store";

import { vuescroll, settings } from './assets/js/scrollbar';
Vue.use(vuescroll, settings);

import '@/assets/style/normalize.css'
import '@/assets/style/coolicons.css'
import '@/assets/scss/index.scss'


import pkg from '../package.json'
Vue.prototype.$config = {
  isDev: process.env.NODE_ENV === 'development',
  version: pkg.version
}

// import global components
import './components/global'

// import all directivs
import directives from '@/utils/directives'
directives.forEach(directive=>Vue.directive(directive.name, directive))

Vue.config.productionTip = false;

new Vue({
  router,
  store,
  render: (h) => h(App),
}).$mount("#app");
