(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-452bc100"],{"2a3e":function(e,t,s){},"33ad":function(e,t,s){},5997:function(e,t,s){"use strict";s("33ad")},"5a9f":function(e,t,s){"use strict";s("8305")},6522:function(e,t,s){"use strict";var a=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{class:["dropdown",{"dropdown--active":e.show}]},[s("label",{attrs:{for:e.name}},[e._v(e._s(e.label))]),s("input",{directives:[{name:"model",rawName:"v-model",value:e.search,expression:"search"}],staticClass:"dropdown__input",attrs:{id:e.name,name:e.name,disabled:e.disabled,placeholder:e.placeholder,autocomplete:"off"},domProps:{value:e.search},on:{focus:e.focus,blur:function(t){return e.select(!1)},input:function(t){t.target.composing||(e.search=t.target.value)}}}),s("div",{directives:[{name:"show",rawName:"v-show",value:e.show,expression:"show"}],staticClass:"dropdown__content"},[e._l(e.filterList,(function(t,a){return s("div",{key:a,on:{mousedown:function(s){return e.select(t)}}},[e._v(" "+e._s(t.label)+" ")])})),e.filterList.length?e._e():s("div",[e._v("Нет данных")])],2)])},r=[],n=(s("ac1f"),s("841c"),s("4de4"),s("caad"),s("2532"),{name:"Autocomplete",props:{name:String,list:{type:Array,required:!0,default:function(){return[]}},placeholder:String,disabled:Boolean,label:String,value:String},data:function(){return{selected:{},show:!1,search:""}},created:function(){this.search=this.value},computed:{filterList:function(){var e=this;return this.list?this.list.filter((function(t){var s=e.search;return!s||t.label.toLowerCase().includes(s.toLowerCase())})):[]}},methods:{select:function(e){e?(this.selected=e,this.show=!1,this.search=e.label,this.$emit("input",this.selected.value),this.$emit("change",e)):(this.search=this.selected.label||this.value,this.show=!1)},focus:function(e){var t=e.target;t.select(),this.show=!0,this.$emit("focus",!0)}},watch:{value:{handler:function(e){this.show=!1,this.search=e}}}}),i=n,o=(s("5997"),s("2877")),c=Object(o["a"])(i,a,r,!1,null,"3c6819c5",null);t["a"]=c.exports},8305:function(e,t,s){},bb6b:function(e,t,s){"use strict";s.r(t);var a=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"params"},[s("div",{staticClass:"params__btn",on:{click:e.openFull}},[s("i",{staticClass:"params__btn--icon"})]),s("div",{staticClass:"params__items"},[s("div",{staticClass:"params__items--item"},[s("DatasetButton")],1),s("div",{staticClass:"params__items--item pa-0"},[s("DatasetTab",{on:{input:e.saveSet,select:e.select},model:{value:e.tab,callback:function(t){e.tab=t},expression:"tab"}})],1),s("div",{staticClass:"params__items--item"},[s("div",{staticClass:"params__items--btn"},[s("t-button",{attrs:{loading:e.loading,disabled:e.disabled},nativeOn:{click:function(t){return e.download.apply(null,arguments)}}})],1)])])])},r=[],n=s("1da1"),i=s("5530"),o=(s("96cf"),s("b64b"),s("2f62")),c=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"tabs"},[s("ul",{staticClass:"tabs__list"},e._l(e.items,(function(t,a){var r=t.title,n=t.active,i=t.mode;return s("li",{key:a,class:["tabs__list--item",{active:n}],on:{click:function(t){return t.preventDefault(),e.click(i)}}},[e._v(" "+e._s(r)+" ")])})),0),s("div",{staticClass:"tabs__title"},[e._v("Создание датасета")]),s("div",{directives:[{name:"show",rawName:"v-show",value:"GoogleDrive"===e.value,expression:"value === 'GoogleDrive'"}],staticClass:"tabs__item"},[s("Autocomplete2",{attrs:{list:e.list,name:"gdrive",label:"Выберите файл из Google-диске"},on:{focus:e.focus,change:e.selected}})],1),s("div",{directives:[{name:"show",rawName:"v-show",value:"URL"===e.value,expression:"value === 'URL'"}],staticClass:"tabs__item"},[s("t-input",{attrs:{label:"Введите URL на архив исходников"},on:{input:e.change}})],1)])},l=[],u=(s("498a"),s("d81d"),s("6522")),d={name:"DatasetTab",components:{Autocomplete2:u["a"]},props:{value:{type:String,default:"GoogleDrive"}},data:function(){return{list:[],items:[{title:"Google drive",active:!0,mode:"GoogleDrive"},{title:"URL",active:!1,mode:"URL"}]}},methods:{focus:function(){var e=this;return Object(n["a"])(regeneratorRuntime.mark((function t(){var s,a;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("axios",{url:"/datasets/sources/"});case 2:if(s=t.sent,a=s.data,a){t.next=6;break}return t.abrupt("return");case 6:e.list=a;case 7:case"end":return t.stop()}}),t)})))()},selected:function(e){var t=e.value,s=e.label;this.$emit("select",{mode:"GoogleDrive",value:t,label:s})},change:function(e){this.$emit("select",{mode:"URL",value:e?e.trim():""})},click:function(e){this.select=e,this.$emit("input",e),this.items=this.items.map((function(t){return Object(i["a"])(Object(i["a"])({},t),{},{active:t.mode===e})}))}}},h=d,p=(s("f88b"),s("2877")),m=Object(p["a"])(h,c,l,!1,null,"3bbad915",null),g=m.exports,v=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",[s("t-button",{attrs:{disabled:!e.selected},nativeOn:{click:function(t){return e.handleClick.apply(null,arguments)}}},[e.selected?s("span",[e._v("Выбрать датасет")]):s("span",[e._v(e._s(e.btnText))])])],1)},f=[],b=(s("b0c0"),{name:"DatasetButton",computed:{isNoTrain:function(){return"no_train"===this.$store.getters["trainings/getStatus"]},selected:function(){return this.$store.getters["datasets/getSelected"]},selectedIndex:function(){return this.$store.getters["datasets/getSelectedIndex"]},btnText:function(){var e,t,s=null===(e=this.$store.getters["projects/getProject"])||void 0===e||null===(t=e.dataset)||void 0===t?void 0:t.name;return s?"Выбран: "+s:"Выберите датасет"}},methods:{createInterval:function(){var e=this;this.interval=setTimeout(Object(n["a"])(regeneratorRuntime.mark((function t(){var s,a,r,n,i,o,c;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("datasets/choiceProgress",{});case 2:if(s=t.sent,!s){t.next=31;break}if(a=s.data,!a){t.next=28;break}if(r=a.finished,n=a.message,i=a.percent,o=a.error,e.$store.dispatch("messages/setProgressMessage",n),e.$store.dispatch("messages/setProgress",i),!r){t.next=19;break}return e.$store.dispatch("messages/setProgress",0),e.$store.dispatch("messages/setProgressMessage",""),t.next=14,e.$store.dispatch("projects/get");case 14:c=a.data,e.$store.dispatch("messages/setMessage",{message:"Датасет «".concat(c.name,"» выбран")},{root:!0}),e.$store.dispatch("settings/setOverlay",!1),t.next=26;break;case 19:if(!o){t.next=25;break}return e.$store.dispatch("messages/setMessage",{error:o}),e.$store.dispatch("messages/setProgressMessage",""),e.$store.dispatch("messages/setProgress",0),e.$store.dispatch("settings/setOverlay",!1),t.abrupt("return");case 25:e.createInterval();case 26:t.next=29;break;case 28:e.$store.dispatch("settings/setOverlay",!1);case 29:t.next=32;break;case 31:e.$store.dispatch("settings/setOverlay",!1);case 32:case"end":return t.stop()}}),t)}))),1e3)},handleClick:function(){var e=this;return Object(n["a"])(regeneratorRuntime.mark((function t(){var s,a,r,i,o,c,l;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:if(e.isNoTrain){t.next=4;break}return t.next=3,e.$store.dispatch("messages/setModel",{context:e,content:"Для выбора датасета необходимо сбросить/остановить обучение"});case 3:return t.abrupt("return");case 4:return s=e.selected,a=s.alias,r=s.group,i=s.name,t.next=7,e.$store.dispatch("datasets/validateDatasetOrModel",{dataset:{alias:a,group:r}});case 7:if(o=t.sent,c=o.success,l=o.data,!c||!l){t.next=14;break}e.$Modal.confirm({title:"Внимание!",content:l,width:300,callback:function(){var t=Object(n["a"])(regeneratorRuntime.mark((function t(s){return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:if("confirm"!=s){t.next=3;break}return t.next=3,e.onChoice({alias:a,group:r,name:i,reset_model:!0});case 3:case"end":return t.stop()}}),t)})));function s(e){return t.apply(this,arguments)}return s}()}),t.next=16;break;case 14:return t.next=16,e.onChoice({alias:a,group:r,name:i,reset_model:!1});case 16:case"end":return t.stop()}}),t)})))()},onChoice:function(){var e=arguments,t=this;return Object(n["a"])(regeneratorRuntime.mark((function s(){var a,r,n,i,o,c,l;return regeneratorRuntime.wrap((function(s){while(1)switch(s.prev=s.next){case 0:return a=e.length>0&&void 0!==e[0]?e[0]:{},r=a.alias,n=a.group,i=a.name,o=a.reset_model,t.$store.dispatch("settings/setOverlay",!0),s.next=4,t.$store.dispatch("datasets/choice",{alias:r,group:n,reset_model:o});case 4:c=s.sent,l=c.success,l&&(t.$store.dispatch("messages/setMessage",{message:"Загружаю датасет «".concat(i,"»")}),t.createInterval());case 7:case"end":return s.stop()}}),s)})))()}}}),w=b,_=Object(p["a"])(w,v,f,!1,null,null,null),$=_.exports,x={name:"Settings",components:{DatasetTab:g,DatasetButton:$},data:function(){return{tab:"GoogleDrive",loading:!1,dataset:{},prevSet:"",interval:null,inputs:1,outputs:1,rules:{length:function(e){return function(t){return(t||"").length>=e||"Length < ".concat(e)}},required:function(e){return 0!==e.length||"Not be empty"}}}},computed:Object(i["a"])(Object(i["a"])({},Object(o["b"])({settings:"datasets/getSettings"})),{},{disabled:function(){return 0===Object.keys(this.dataset).length&&"GoogleDrive"===this.dataset.mode||(!this.dataset.value&&"URL"===this.dataset.mode||this.tab!==this.dataset.mode)},full:{set:function(e){this.$store.dispatch("datasets/setFull",e)},get:function(){return this.$store.getters["datasets/getFull"]}}}),methods:{createInterval:function(){var e=arguments,t=this;return Object(n["a"])(regeneratorRuntime.mark((function s(){var a;return regeneratorRuntime.wrap((function(s){while(1)switch(s.prev=s.next){case 0:a=e.length>0&&void 0!==e[0]?e[0]:null,t.interval=setTimeout(Object(n["a"])(regeneratorRuntime.mark((function e(){var s,r,n,i,o,c,l,u,d;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.$store.dispatch("datasets/loadProgress",{});case 2:if(s=e.sent,console.log(s),!s){e.next=16;break}if(r=s.data,n=r.finished,i=r.message,o=r.percent,c=r.error,console.log(o),t.$store.dispatch("messages/setProgressMessage",i),t.$store.dispatch("messages/setProgress",o),!c){e.next=13;break}return t.loading=!1,t.$store.dispatch("settings/setOverlay",!1),e.abrupt("return");case 13:n?(l=s.data.data,u=l.file_manager,d=l.source_path,t.$store.dispatch("datasets/setFilesSource",u),t.$store.dispatch("datasets/setSourcePath",d),t.$store.dispatch("datasets/setFilesDrop",[]),t.$store.dispatch("datasets/clearInputData"),t.$store.dispatch("messages/setProgressMessage",""),t.$store.dispatch("messages/setProgress",0),t.loading=!1,t.$store.dispatch("settings/setOverlay",!1),t.$store.dispatch("messages/setMessage",{message:"Исходники dataset ".concat(a,"  загружены ")}),t.full=!0):t.createInterval(a),e.next=18;break;case 16:t.loading=!1,t.$store.dispatch("settings/setOverlay",!1);case 18:case"end":return e.stop()}}),e)}))),1e3);case 2:case"end":return s.stop()}}),s)})))()},saveSet:function(){"GoogleDrive"===this.dataset.mode&&(this.prevSet=this.dataset,this.$el.querySelector(".t-field__input").value=""),"URL"===this.dataset.mode&&(this.dataset=this.prevSet)},select:function(e){this.dataset=e},openFull:function(){this.$store.state.datasets.filesSource.length?this.full=!0:this.$Modal.alert({width:250,title:"Внимание!",maskClosable:!0,content:"Загрузите исходник датасета"})},download:function(){var e=this;return Object(n["a"])(regeneratorRuntime.mark((function t(){var s,a,r,n,i,o;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:if(!e.loading){t.next=2;break}return t.abrupt("return");case 2:if(s=e.dataset,a=s.mode,r=s.value,n=s.label,!a||!r){t.next=14;break}return e.loading=!0,e.$store.dispatch("settings/setOverlay",!0),e.$store.dispatch("messages/setMessage",{message:"Загружаю датасет ".concat(n)}),t.next=9,e.$store.dispatch("datasets/sourceLoad",{mode:a,value:r});case 9:i=t.sent,o=i.success,o?e.createInterval(n):(e.loading=!1,e.$store.dispatch("settings/setOverlay",!1)),t.next=15;break;case 14:e.$store.dispatch("messages/setMessage",{error:"Выберите файл"});case 15:case"end":return t.stop()}}),t)})))()}}},k=x,O=(s("5a9f"),Object(p["a"])(k,a,r,!1,null,"46d67aae",null));t["default"]=O.exports},f88b:function(e,t,s){"use strict";s("2a3e")}}]);