import Vue from "vue";
import App from "./App.vue";
import Vuex from "vuex";
import VueRouter from "vue-router";
import axios from "axios";
import VueAxios from "vue-axios";

import routes from "./routers/index";
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

Vue.config.productionTip = false;
Vue.use(Vuex);
Vue.use(VueAxios, axios);
Vue.use(VueRouter);
// Vue.use(Vuetify);
export const bus = new Vue();

new Vue({
  router: new VueRouter({
    mode: "history",
    routes,
  }),
  store: new Vuex.Store(store),
  // vuetify: new Vuetify(opts),
  render: (h) => h(App),
}).$mount("#app");
