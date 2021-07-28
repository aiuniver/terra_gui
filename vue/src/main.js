import Vue from "vue";
import App from "./App.vue";
import Vuex from "vuex";
// import VueRouter from "vue-router";
import axios from "axios";
import VueAxios from "vue-axios";

import router from "./routers/index";
import store from "./store/index";

// import {
//   Button,
//   ButtonGroup,
//   Tag,
//   Radio,
//   RadioGroup,
//   RadioButton,
//   Checkbox,
//   CheckboxGroup,
//   Input,
//   InputNumber,
//   Textarea,
//   Badge,
//   Switch,
//   Slider,
//   Tooltip,
//   Popover,
//   Alert,
//   Progress,
//   LoadingBar,
//   Modal,
//   Select,
//   Option,
//   OptionGroup,
//   Dropdown,
//   DropdownMenu,
//   DropdownItem,
//   Breadcrumb,
//   BreadcrumbItem,
//   Pagination,
//   Menu,
//   MenuItem,
//   MenuItemGroup,
//   Submenu,
//   Table,
//   Card,
//   Collapse,
//   CollapseItem,
//   Steps,
//   Step,
//   Rate,
//   Tabs,
//   TabPane,
//   Timeline,
//   TimelineItem
// } from '@/at-ui/src'

import AtComponents from '@/at-ui/src'
import '@/at-ui/scss/index.scss' 

Vue.use(AtComponents)

// Vue.prototype.$Notify = Notification
// Vue.prototype.$Loading = LoadingBar
// Vue.prototype.$Modal = Dialog
// Vue.prototype.$Message = Message

import vuescroll from '@/assets/js/vuescroll-native.min.js';

Vue.use(vuescroll, {
  ops: {
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
    }
  },
  name: 'scrollbar'
});


import VuePapaParse from "vue-papa-parse";
Vue.use(VuePapaParse)

Vue.config.productionTip = false;
Vue.use(Vuex);
Vue.use(VueAxios, axios);
// Vue.use(VueRouter);
export const bus = new Vue();

new Vue({
  router,
  store: new Vuex.Store(store),
  render: (h) => h(App),
}).$mount("#app");
