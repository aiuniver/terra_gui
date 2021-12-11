(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-630f4480"],{"26c2":function(e,t,s){"use strict";s.r(t);var n=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("at-modal",{directives:[{name:"show",rawName:"v-show",value:e.show,expression:"show"}],staticClass:"t-modal",attrs:{width:"400",maskClosable:!1,showClose:!1},model:{value:e.dialog,callback:function(t){e.dialog=t},expression:"dialog"}},[s("div",{attrs:{slot:"header"},slot:"header"},[s("span",[e._v("Загрузка проекта")])]),s("div",{staticClass:"t-modal__body"},[s("p",[e._v("Загрузка проекта удалит текущий.")])]),s("div",{staticClass:"t-modal__sub-body"},[s("span",{staticClass:"t-modal__link",on:{click:function(t){return e.$emit("start",!0)}}},[e._v("Сохранить проект")])]),s("div",{ref:"list",staticClass:"t-modal__list"},[s("scrollbar",[s("ul",{staticClass:"loaded-list"},[e._l(e.list,(function(t,n){return s("li",{key:"custom_"+n,class:["loaded-list__item",{"loaded-list__item--active":e.selected.label===t.label}],on:{click:function(s){e.selected=t}}},[s("i",{staticClass:"loaded-list__item--icon"}),s("span",{staticClass:"loaded-list__item--text"},[e._v(e._s(t.label))]),s("div",{staticClass:"loaded-list__item--empty"}),s("div",{staticClass:"loaded-list__item--remove",on:{click:function(s){return s.stopPropagation(),e.remove(t)}}},[s("i")])])})),e.list.length?e._e():s("li",{staticClass:"loaded-list__no-list"},[e._v("Нет проектов для загрузки")])],2)])],1),s("template",{slot:"footer"},[s("t-button",{attrs:{disabled:!e.selected.label},on:{click:function(t){return e.loadProject(e.selected)}}},[e._v("Загрузить")]),s("t-button",{attrs:{cancel:""},on:{click:function(t){e.dialog=!1}}},[e._v("Отменить")])],1)],2)},a=[],o=s("1da1"),r=(s("96cf"),s("eb4c")),i={name:"modal-load-project",props:{value:Boolean},data:function(){return{selected:{},show:!0,list:[],debounce:null}},computed:{dialog:{set:function(e){this.$emit("input",e)},get:function(){return this.value}}},methods:{progress:function(){var e=this;return Object(o["a"])(regeneratorRuntime.mark((function t(){var s,n,a,o,r;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("projects/progress",{});case 2:s=t.sent,s&&null!==s&&void 0!==s&&s.data&&(n=s.data,a=n.finished,o=n.message,r=n.percent,e.$store.dispatch("messages/setProgressMessage",o),e.$store.dispatch("messages/setProgress",r),a?(e.$store.dispatch("projects/get"),e.$store.dispatch("settings/setOverlay",!1),e.$emit("message",{message:"Проект загружен"}),e.dialog=!1):e.debounce(!0)),null!==s&&void 0!==s&&s.error&&e.$store.dispatch("settings/setOverlay",!1);case 5:case"end":return t.stop()}}),t)})))()},remove:function(e){var t=this;this.$Modal.confirm({title:"Удаление проекта",content:"Вы действительно хотите удалить проект «".concat(e.label,"»?"),width:300,maskClosable:!1,showClose:!1}).then((function(){t.removeProject(e)})).catch((function(){}))},loadProject:function(e){var t=this;return Object(o["a"])(regeneratorRuntime.mark((function s(){var n;return regeneratorRuntime.wrap((function(s){while(1)switch(s.prev=s.next){case 0:return console.log(e),s.prev=1,s.next=4,t.$store.dispatch("projects/load",{value:e.value});case 4:n=s.sent,console.log(n),null!==n&&void 0!==n&&n.success&&(t.$store.dispatch("settings/setOverlay",!0),t.debounce(!0)),s.next=12;break;case 9:s.prev=9,s.t0=s["catch"](1),console.log(s.t0);case 12:case"end":return s.stop()}}),s,null,[[1,9]])})))()},removeProject:function(e){var t=this;return Object(o["a"])(regeneratorRuntime.mark((function s(){var n;return regeneratorRuntime.wrap((function(s){while(1)switch(s.prev=s.next){case 0:return console.log(e),s.prev=1,s.next=4,t.$store.dispatch("projects/remove",{path:e.value});case 4:if(n=s.sent,!n||n.error){s.next=9;break}return t.$emit("message",{message:"Проект «".concat(e.label,"» удален")}),s.next=9,t.infoProject();case 9:s.next=14;break;case 11:s.prev=11,s.t0=s["catch"](1),console.log(s.t0);case 14:case"end":return s.stop()}}),s,null,[[1,11]])})))()},infoProject:function(){var e=this;return Object(o["a"])(regeneratorRuntime.mark((function t(){var s,n;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.prev=0,t.next=3,e.$store.dispatch("projects/infoProject",{});case 3:s=t.sent,s&&(n=s.data.projects,e.list=n),e.$nextTick((function(){e.$refs.list.clientHeight>200&&(e.$refs.list.style.height="200px")})),t.next=11;break;case 8:t.prev=8,t.t0=t["catch"](0),console.log(t.t0);case 11:case"end":return t.stop()}}),t,null,[[0,8]])})))()}},created:function(){var e=this;this.debounce=Object(r["a"])((function(t){t&&e.progress()}),1e3),this.debounce(this.isLearning)},beforeDestroy:function(){this.debounce(!1)},watch:{dialog:function(e){e&&this.infoProject()}}},c=i,l=(s("a103"),s("2877")),u=Object(l["a"])(c,n,a,!1,null,"177c4f40",null);t["default"]=u.exports},a103:function(e,t,s){"use strict";s("e968")},e968:function(e,t,s){},eb4c:function(e,t,s){"use strict";s.d(t,"a",(function(){return n}));var n=function(e,t,s){var n;return function(){var a=this,o=arguments,r=function(){n=null,s||e.apply(a,o)},i=s&&!n;clearTimeout(n),n=setTimeout(r,t),i&&e.apply(a,o)}}}}]);