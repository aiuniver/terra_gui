import Vue from "vue";
import App from "./App.vue";
import router from "./router";
import store from "./store";

import { vuescroll, settings } from './assets/js/scrollbar';
Vue.use(vuescroll, settings);

import '@/assets/style/normalize.css'
import '@/assets/style/coolicons.css'
import '@/assets/scss/main.scss'


import pkg from '../package.json'
Vue.prototype.$config = {
  isDev: process.env.NODE_ENV === 'development',
  version: pkg.version
}

// const files = require.context('./components/global', true, /\.vue$/i, 'lazy').keys();

// files.forEach(file => {
    // Vue.component(file.split('/').pop().split('.')[0], () => import(`${file}`));
// });
// console.log(files)
// // import all directivs
// import directives from '@/utils/directives'
// directives.forEach(directive=>Vue.directive(directive.name, directive))

// import global components
import { components } from './components/global'
console.log(components)
// components.forEach(component=>Vue.component(component.name, component))


Vue.config.productionTip = false;

new Vue({
  router,
  store,
  render: (h) => h(App),
}).$mount("#app");
