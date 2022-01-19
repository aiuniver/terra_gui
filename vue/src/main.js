import Vue from "vue";
import App from "./App.vue";
import Vuex from "vuex";
import axios from "axios";
import VueAxios from "vue-axios";

import router from "./routers/index";
import store from "./store/index";

import AtComponents from '@/at-ui/src'
import '@/at-ui/scss/index.scss'

Vue.use(AtComponents)

import vuescroll from '@/assets/js/vuescroll-native.min.js';

const optionsScroll = {
  ops: {
    vuescroll: {
      mode: 'native',
      sizeStrategy: 'percent',
      detectResize: true,
      locking: true,
    },
    scrollPanel: {
      initialScrollY: 0.1,
      initialScrollX: 0.1
    },
    bar: {
      showDelay: 500,
      onlyShowBarOnScroll: false,
      keepShow: true,
      background: '#242f3d',
      opacity: 1,
      hoverStyle: false,
      specifyBorderRadius: false,
      minSize: 0,
      size: '4px',
      disable: false
    },
    rail: {
      gutterOfEnds: '3px',
      gutterOfSide: '3px',
      background: '#01a99a',
      opacity: 0,
      size: '4px',
    }
  },
  name: 'scrollbar'
};

const scroll = localStorage.getItem('settingsScroll');
if (scroll) {
  try {
    const temp = JSON.parse(scroll);
    temp.size = `${temp.size}px`
    optionsScroll.ops.bar = {...optionsScroll.ops.bar, ...temp}
    console.log(optionsScroll.ops.bar)
  } catch (error) {
    console.log(error)
  }
}
Vue.use(vuescroll, optionsScroll)


import pkg from '../package.json'
Vue.prototype.$config = {
  isDev: process.env.NODE_ENV === 'development',
  version: pkg.version
}

// import all directivs
import directives from '@/utils/directives'
directives.forEach(directive => Vue.directive(directive.name, directive))

// import global components
import components from '@/components/global'
components.forEach(component => Vue.component(component.name, component))

import AudioVisual from 'vue-audio-visual'
Vue.use(AudioVisual)

Vue.config.productionTip = false;
Vue.use(Vuex);
Vue.use(VueAxios, axios);
// Vue.use(VueRouter);
export const bus = new Vue();

import '@/assets/scss/main.scss'
import '@/assets/css/coolicons.css'
import '@/assets/css/reset.css';
import '@/assets/css/fonts.css';
import '@/assets/css/farbtastic.css';
import '@/assets/css/layout.css';
import '@/assets/css/icons.css';
import '@/assets/css/new/icons.css';
import '@/assets/css/project/layout.css';
import '@/assets/css/project/datasets.css';
import '@/assets/css/media.css';
import '@/assets/css/theme-dark.css';
import '@/assets/css/new/main.min.css';

new Vue({
  router,
  store: new Vuex.Store(store),
  render: (h) => h(App),
}).$mount("#app");
