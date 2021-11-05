import Vue from "vue";
import App from "./App.vue";
import router from "./router";
import store from "./store";

// import axios from "axios";
// import VueAxios from "vue-axios";

import { vuescroll, settings } from './assets/js/scrollbar';
Vue.use(vuescroll, settings);

import '@/assets/scss/main.scss'
import '@/assets/style/coolicons.css'


// import pkg from '../package.json'
// Vue.prototype.$config = {
//   isDev: process.env.NODE_ENV === 'development',
//   version: pkg.version
// }

// // import all directivs
// import directives from '@/utils/directives'
// directives.forEach(directive=>Vue.directive(directive.name, directive))

// // import global components
// import components from '@/components/global'
// components.forEach(component=>Vue.component(component.name, component))


// Vue.use(Vuex);
// Vue.use(VueAxios, axios);


Vue.config.productionTip = false;

new Vue({
  router,
  store,
  render: (h) => h(App),
}).$mount("#app");
