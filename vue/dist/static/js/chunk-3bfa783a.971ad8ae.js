(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-3bfa783a"],{"2a3e":function(e,t,s){},5862:function(e,t,s){},6522:function(e,t,s){"use strict";var a=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{class:["dropdown",{"dropdown--active":e.show}]},[s("label",{attrs:{for:e.name}},[e._v(e._s(e.label))]),s("input",{directives:[{name:"model",rawName:"v-model",value:e.search,expression:"search"}],staticClass:"dropdown__input",attrs:{autocomplete:"off",id:e.name,name:e.name,disabled:e.disabled,placeholder:e.placeholder},domProps:{value:e.search},on:{focus:e.focus,blur:function(t){return e.select(!1)},input:[function(t){t.target.composing||(e.search=t.target.value)},function(t){e.changed=!0}]}}),s("div",{directives:[{name:"show",rawName:"v-show",value:e.show,expression:"show"}],staticClass:"dropdown__content"},[e._l(e.filterList,(function(t,a){return s("div",{key:a,on:{mousedown:function(s){return e.select(t)}}},[e._v(" "+e._s(t.label)+" ")])})),e.filterList.length?e._e():s("div",[e._v("Нет данных")])],2)])},r=[],n=(s("ac1f"),s("841c"),s("4de4"),s("caad"),s("2532"),{name:"Autocomplete",props:{name:String,list:{type:Array,required:!0,default:function(){return[]}},placeholder:String,disabled:Boolean,label:String,value:String},data:function(){return{selected:{},show:!1,search:"",changed:null}},created:function(){this.search=this.value,this.changed=!1},computed:{filterList:function(){var e=this;return this.list?this.list.filter((function(t){var s=e.search;return!s||!e.changed||t.label.toLowerCase().includes(s.toLowerCase())})):[]}},methods:{select:function(e){e?(this.selected=e,this.show=!1,this.search=e.label,this.$emit("input",this.selected.value),this.$emit("change",e)):(this.search=this.selected.label||this.value,this.show=!1,this.changed=!1)},focus:function(e){var t=e.target;t.select(),this.show=!0,this.$emit("focus",!0)}},watch:{value:{handler:function(e){this.show=!1,this.search=e}}}}),i=n,c=(s("9b90"),s("2877")),o=Object(c["a"])(i,a,r,!1,null,"93faebd0",null);t["a"]=o.exports},6645:function(e,t,s){"use strict";s("5862")},"9b90":function(e,t,s){"use strict";s("cfe0")},bb6b:function(e,t,s){"use strict";s.r(t);var a=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"params"},[s("div",{staticClass:"params__btn",on:{click:e.openFull}},[s("i",{staticClass:"params__btn--icon"})]),s("div",{staticClass:"params__items"},[s("div",{staticClass:"params__items--item"},[s("DatasetButton")],1),s("div",{staticClass:"params__items--item pa-0"},[s("DatasetTab",{on:{input:e.saveSet,select:e.select},model:{value:e.tab,callback:function(t){e.tab=t},expression:"tab"}})],1),s("div",{staticClass:"params__items--item"},[s("div",{staticClass:"params__items--btn"},[s("t-button",{attrs:{loading:e.loading,disabled:e.disabled},nativeOn:{click:function(t){return e.download.apply(null,arguments)}}})],1)])])])},r=[],n=s("1da1"),i=s("5530"),c=(s("96cf"),s("b64b"),s("fb6a"),s("2f62")),o=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"tabs"},[s("ul",{staticClass:"tabs__list"},e._l(e.items,(function(t,a){var r=t.title,n=t.active,i=t.mode;return s("li",{key:a,class:["tabs__list--item",{active:n}],on:{click:function(t){return t.preventDefault(),e.click(i)}}},[e._v(" "+e._s(r)+" ")])})),0),s("div",{staticClass:"tabs__title"},[e._v("Создание датасета")]),s("div",{directives:[{name:"show",rawName:"v-show",value:"GoogleDrive"===e.value,expression:"value === 'GoogleDrive'"}],staticClass:"tabs__item"},[s("Autocomplete2",{attrs:{list:e.list,name:"gdrive",label:"Выберите файл из Google-диске"},on:{focus:e.focus,change:e.selected}})],1),s("div",{directives:[{name:"show",rawName:"v-show",value:"URL"===e.value,expression:"value === 'URL'"}],staticClass:"tabs__item"},[s("t-input",{attrs:{label:"Введите URL на архив исходников"},on:{input:e.change}})],1)])},l=[],u=(s("498a"),s("d81d"),s("6522")),d={name:"DatasetTab",components:{Autocomplete2:u["a"]},props:{value:{type:String,default:"GoogleDrive"}},data:function(){return{list:[],items:[{title:"Google drive",active:!0,mode:"GoogleDrive"},{title:"URL",active:!1,mode:"URL"}]}},methods:{focus:function(){var e=this;return Object(n["a"])(regeneratorRuntime.mark((function t(){var s,a;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("axios",{url:"/datasets/sources/"});case 2:if(s=t.sent,a=s.data,a){t.next=6;break}return t.abrupt("return");case 6:e.list=a;case 7:case"end":return t.stop()}}),t)})))()},selected:function(e){var t=e.value,s=e.label;this.$emit("select",{mode:"GoogleDrive",value:t,label:s})},change:function(e){this.$emit("select",{mode:"URL",value:e?e.trim():""})},click:function(e){this.select=e,this.$emit("input",e),this.items=this.items.map((function(t){return Object(i["a"])(Object(i["a"])({},t),{},{active:t.mode===e})}))}}},h=d,p=(s("f88b"),s("2877")),g=Object(p["a"])(h,o,l,!1,null,"3bbad915",null),m=g.exports,v=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",[s("t-button",{attrs:{disabled:!e.selected},nativeOn:{click:function(t){return e.handleClick.apply(null,arguments)}}},[e.selected?s("span",[e._v("Выбрать датасет")]):s("span",[e._v(e._s(e.btnText))])])],1)},f=[],b=(s("b0c0"),{name:"DatasetButton",computed:{isNoTrain:function(){return"no_train"===this.$store.getters["trainings/getStatus"]},selected:function(){return this.$store.getters["datasets/getSelected"]},selectedIndex:function(){return this.$store.getters["datasets/getSelectedIndex"]},btnText:function(){var e,t,s=null===(e=this.$store.getters["projects/getProject"])||void 0===e||null===(t=e.dataset)||void 0===t?void 0:t.name;return s?"Выбран: "+s:"Выберите датасет"}},methods:{createInterval:function(){var e=this;this.interval=setTimeout(Object(n["a"])(regeneratorRuntime.mark((function t(){var s,a,r,n,i,c,o,l;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("datasets/choiceProgress",{});case 2:if(s=t.sent,!s){t.next=29;break}if(a=s.data,!a){t.next=26;break}if(r=a.finished,n=a.message,i=a.percent,c=a.error,e.$store.dispatch("messages/setProgressMessage",n),e.$store.dispatch("messages/setProgress",i),!r){t.next=18;break}return e.$store.dispatch("messages/setProgress",0),e.$store.dispatch("messages/setProgressMessage",""),t.next=14,e.$store.dispatch("projects/get");case 14:e.$store.dispatch("messages/setMessage",{message:"Датасет «".concat((null===a||void 0===a||null===(o=a.data)||void 0===o||null===(l=o.dataset)||void 0===l?void 0:l.name)||"","» выбран")}),e.$store.dispatch("settings/setOverlay",!1),t.next=24;break;case 18:if(!c){t.next=23;break}return e.$store.dispatch("messages/setProgressMessage",""),e.$store.dispatch("messages/setProgress",0),e.$store.dispatch("settings/setOverlay",!1),t.abrupt("return");case 23:e.createInterval();case 24:t.next=27;break;case 26:e.$store.dispatch("settings/setOverlay",!1);case 27:t.next=30;break;case 29:e.$store.dispatch("settings/setOverlay",!1);case 30:case"end":return t.stop()}}),t)}))),1e3)},isTraining:function(){var e=this;return Object(n["a"])(regeneratorRuntime.mark((function t(){return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("dialogs/trining",{ctx:e,page:"датасета"});case 2:return t.abrupt("return",t.sent);case 3:case"end":return t.stop()}}),t)})))()},handleClick:function(){var e=this;return Object(n["a"])(regeneratorRuntime.mark((function t(){var s,a,r,n,c,o;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return s=e.selected,t.next=3,e.isTraining();case 3:if(a=t.sent,!a){t.next=23;break}return e.$store.dispatch("settings/setOverlay",!0),t.next=8,e.$store.dispatch("datasets/validateDatasetOrModel",{dataset:s});case 8:if(r=t.sent,n=r.success,c=r.data,!n||!c){t.next=21;break}return e.$store.dispatch("settings/setOverlay",!1),t.next=15,e.$store.dispatch("dialogs/confirm",{ctx:e,content:c});case 15:if(o=t.sent,"confirm"!=o){t.next=19;break}return t.next=19,e.onChoice(Object(i["a"])(Object(i["a"])({},s),{},{reset_model:!0}));case 19:t.next=23;break;case 21:return t.next=23,e.onChoice(Object(i["a"])(Object(i["a"])({},s),{},{reset_model:!1}));case 23:case"end":return t.stop()}}),t)})))()},onChoice:function(){var e=arguments,t=this;return Object(n["a"])(regeneratorRuntime.mark((function s(){var a,r,n,i,c,o,l;return regeneratorRuntime.wrap((function(s){while(1)switch(s.prev=s.next){case 0:return a=e.length>0&&void 0!==e[0]?e[0]:{},r=a.alias,n=a.group,i=a.name,c=a.reset_model,t.$store.dispatch("settings/setOverlay",!0),s.next=4,t.$store.dispatch("datasets/choice",{alias:r,group:n,reset_model:c});case 4:o=s.sent,l=o.success,l&&(t.$store.dispatch("messages/setMessage",{message:"Загружаю датасет «".concat(i,"»")}),t.createInterval());case 7:case"end":return s.stop()}}),s)})))()}}}),w=b,$=Object(p["a"])(w,v,f,!1,null,null,null),_=$.exports,x={name:"Settings",components:{DatasetTab:m,DatasetButton:_},data:function(){return{tab:"GoogleDrive",loading:!1,dataset:{},prevSet:"",interval:null,inputs:1,outputs:1,rules:{length:function(e){return function(t){return(t||"").length>=e||"Length < ".concat(e)}},required:function(e){return 0!==e.length||"Not be empty"}}}},computed:Object(i["a"])(Object(i["a"])({},Object(c["c"])({settings:"datasets/getSettings"})),{},{disabled:function(){return 0===Object.keys(this.dataset).length&&"GoogleDrive"===this.dataset.mode||(!this.dataset.value&&"URL"===this.dataset.mode||this.tab!==this.dataset.mode)},full:{set:function(e){this.$store.dispatch("datasets/setFull",e)},get:function(){return this.$store.getters["datasets/getFull"]}}}),methods:{createInterval:function(){var e=arguments,t=this;return Object(n["a"])(regeneratorRuntime.mark((function s(){var a;return regeneratorRuntime.wrap((function(s){while(1)switch(s.prev=s.next){case 0:a=e.length>0&&void 0!==e[0]?e[0]:null,t.interval=setTimeout(Object(n["a"])(regeneratorRuntime.mark((function e(){var s,r,n,i,c,o,l,u,d;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.$store.dispatch("datasets/loadProgress",{});case 2:if(s=e.sent,console.log(s),!s){e.next=16;break}if(r=s.data,n=r.finished,i=r.message,c=r.percent,o=r.error,console.log(c),t.$store.dispatch("messages/setProgressMessage",i),t.$store.dispatch("messages/setProgress",c),!o){e.next=13;break}return t.loading=!1,t.$store.dispatch("settings/setOverlay",!1),e.abrupt("return");case 13:n?(l=s.data.data,u=l.file_manager,d=l.source_path,t.$store.dispatch("datasets/setFilesSource",u),t.$store.dispatch("datasets/setSourcePath",d),t.$store.dispatch("datasets/setFilesDrop",[]),t.$store.dispatch("datasets/clearInputData"),t.$store.dispatch("messages/setProgressMessage",""),t.$store.dispatch("messages/setProgress",0),t.loading=!1,t.$store.dispatch("settings/setOverlay",!1),t.$store.dispatch("messages/setMessage",{message:"Исходники dataset ".concat(a,"  загружены ")}),t.full=!0):t.createInterval(a),e.next=18;break;case 16:t.loading=!1,t.$store.dispatch("settings/setOverlay",!1);case 18:case"end":return e.stop()}}),e)}))),1e3);case 2:case"end":return s.stop()}}),s)})))()},saveSet:function(){"GoogleDrive"===this.dataset.mode&&(this.prevSet=this.dataset,this.$el.querySelector(".t-field__input").value=""),"URL"===this.dataset.mode&&(this.dataset=this.prevSet)},select:function(e){this.dataset=e},openFull:function(){this.$store.state.datasets.filesSource.length?this.full=!0:this.$Modal.alert({width:250,title:"Внимание!",maskClosable:!0,content:"Загрузите исходник датасета"})},download:function(){var e=this;return Object(n["a"])(regeneratorRuntime.mark((function t(){var s,a,r,n,i,c,o;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:if(!e.loading){t.next=2;break}return t.abrupt("return");case 2:if(s=e.dataset,a=s.mode,r=s.value,!a||!r){t.next=16;break}return n=~r.lastIndexOf("\\")?"\\":"/",i=r.slice(r.lastIndexOf(n)+1,r.length-4),e.loading=!0,e.$store.dispatch("settings/setOverlay",!0),e.$store.dispatch("messages/setMessage",{message:"Загружаю датасет ".concat(i)}),t.next=11,e.$store.dispatch("datasets/sourceLoad",{mode:a,value:r});case 11:c=t.sent,o=c.success,o?e.createInterval(i):(e.loading=!1,e.$store.dispatch("settings/setOverlay",!1)),t.next=17;break;case 16:e.$store.dispatch("messages/setMessage",{error:"Выберите файл"});case 17:case"end":return t.stop()}}),t)})))()}}},k=x,O=(s("6645"),Object(p["a"])(k,a,r,!1,null,"fdde8cda",null));t["default"]=O.exports},cfe0:function(e,t,s){},f88b:function(e,t,s){"use strict";s("2a3e")}}]);