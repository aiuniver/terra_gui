(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-74ec8204"],{"129f":function(e,t){e.exports=Object.is||function(e,t){return e===t?0!==e||1/e===1/t:e!=e&&t!=t}},"157e":function(e,t,s){"use strict";s("a4d9")},"3a02":function(e,t,s){},6522:function(e,t,s){"use strict";var a=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{class:["dropdown",{"dropdown--active":e.show}]},[s("label",{attrs:{for:e.name}},[e._v(e._s(e.label))]),s("input",{directives:[{name:"model",rawName:"v-model",value:e.search,expression:"search"}],staticClass:"dropdown__input",attrs:{id:e.name,name:e.name,disabled:e.disabled,placeholder:e.placeholder,autocomplete:"off"},domProps:{value:e.search},on:{focus:e.focus,blur:function(t){return e.select(!1)},input:function(t){t.target.composing||(e.search=t.target.value)}}}),s("div",{directives:[{name:"show",rawName:"v-show",value:e.show,expression:"show"}],staticClass:"dropdown__content"},[e._l(e.filterList,(function(t,a){return s("div",{key:a,on:{mousedown:function(s){return e.select(t)}}},[e._v(" "+e._s(t.label)+" ")])})),e.filterList.length?e._e():s("div",[e._v("Нет данных")])],2)])},r=[],n=(s("ac1f"),s("841c"),s("4de4"),s("caad"),s("2532"),{name:"Autocomplete",props:{name:String,list:{type:Array,required:!0,default:function(){return[]}},placeholder:String,disabled:Boolean,label:String,value:String},data:function(){return{selected:{},show:!1,search:""}},created:function(){this.search=this.value},computed:{filterList:function(){var e=this;return this.list?this.list.filter((function(t){var s=e.search;return!s||t.label.toLowerCase().includes(s.toLowerCase())})):[]}},methods:{select:function(e){e?(this.selected=e,this.show=!1,this.search=e.label,this.$emit("input",this.selected.value),this.$emit("change",e)):(this.search=this.selected.label||this.value,this.show=!1)},focus:function(){this.show=!0,this.$emit("focus",!0)}},watch:{value:{handler:function(e){this.show=!1,this.search=e}}}}),i=n,c=(s("157e"),s("2877")),o=Object(c["a"])(i,a,r,!1,null,"ea064e8e",null);t["a"]=o.exports},"841c":function(e,t,s){"use strict";var a=s("d784"),r=s("825a"),n=s("1d80"),i=s("129f"),c=s("14c3");a("search",(function(e,t,s){return[function(t){var s=n(this),a=void 0==t?void 0:t[e];return void 0!==a?a.call(t,s):new RegExp(t)[e](String(s))},function(e){var a=s(t,this,e);if(a.done)return a.value;var n=r(this),o=String(e),l=n.lastIndex;i(l,0)||(n.lastIndex=0);var u=c(n,o);return i(n.lastIndex,l)||(n.lastIndex=l),null===u?-1:u.index}]}))},"89d7":function(e,t,s){"use strict";s("dce1")},a4d9:function(e,t,s){},bb6b:function(e,t,s){"use strict";s.r(t);var a=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"params"},[s("div",{staticClass:"params__btn",on:{click:e.openFull}},[s("i",{staticClass:"params__btn--icon"})]),s("div",{staticClass:"params__items"},[s("div",{staticClass:"params__items--item"},[s("DatasetButton")],1),s("div",{staticClass:"params__items--item pa-0"},[s("DatasetTab",{on:{select:e.select},model:{value:e.tab,callback:function(t){e.tab=t},expression:"tab"}})],1),s("div",{staticClass:"params__items--item"},[s("div",{staticClass:"params__items--btn"},[s("t-button",{attrs:{loading:e.loading,disabled:e.disabled},nativeOn:{click:function(t){return e.download.apply(null,arguments)}}})],1)])])])},r=[],n=s("1da1"),i=s("5530"),c=(s("96cf"),s("b64b"),s("2f62")),o=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"tabs"},[s("ul",{staticClass:"tabs__list"},e._l(e.items,(function(t,a){var r=t.title,n=t.active,i=t.mode;return s("li",{key:a,class:["tabs__list--item",{active:n}],on:{click:function(t){return t.preventDefault(),e.click(i)}}},[e._v(" "+e._s(r)+" ")])})),0),s("div",{staticClass:"tabs__title"},[e._v("Создание датасета")]),s("div",{directives:[{name:"show",rawName:"v-show",value:"GoogleDrive"===e.value,expression:"value === 'GoogleDrive'"}],staticClass:"tabs__item"},[s("Autocomplete2",{attrs:{list:e.list,name:"gdrive",label:"Выберите файл из Google-диске"},on:{focus:e.focus,change:e.selected}})],1),s("div",{directives:[{name:"show",rawName:"v-show",value:"URL"===e.value,expression:"value === 'URL'"}],staticClass:"tabs__item"},[s("t-input",{attrs:{label:"Введите URL на архив исходников"},on:{change:e.change}})],1)])},l=[],u=(s("d81d"),s("6522")),d={name:"DatasetTab",components:{Autocomplete2:u["a"]},props:{value:{type:String,default:"GoogleDrive"}},data:function(){return{list:[],items:[{title:"Google drive",active:!0,mode:"GoogleDrive"},{title:"URL",active:!1,mode:"URL"}]}},methods:{focus:function(){var e=this;return Object(n["a"])(regeneratorRuntime.mark((function t(){var s,a;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("axios",{url:"/datasets/sources/"});case 2:if(s=t.sent,a=s.data,a){t.next=6;break}return t.abrupt("return");case 6:e.list=a;case 7:case"end":return t.stop()}}),t)})))()},selected:function(e){var t=e.value,s=e.label;this.$emit("select",{mode:"GoogleDrive",value:t,label:s})},change:function(e){this.$emit("select",{mode:"URL",value:e})},click:function(e){this.select=e,this.$emit("input",e),this.items=this.items.map((function(t){return Object(i["a"])(Object(i["a"])({},t),{},{active:t.mode===e})}))}}},h=d,m=(s("da47"),s("2877")),p=Object(m["a"])(h,o,l,!1,null,"2d388197",null),g=p.exports,f=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",[s("t-button",{attrs:{disabled:!e.selected,loading:e.loading},nativeOn:{click:function(t){return e.click.apply(null,arguments)}}},[e._v("Выбрать датасет")])],1)},v=[],b=(s("b0c0"),{name:"DatasetButton",data:function(){return{loading:!1}},computed:{selected:function(){return this.$store.getters["datasets/getSelected"]},selectedIndex:function(){return this.$store.getters["datasets/getSelectedIndex"]}},methods:{createInterval:function(){var e=this;this.interval=setTimeout(Object(n["a"])(regeneratorRuntime.mark((function t(){var s,a,r,n,i,c,o;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("datasets/choiceProgress",{});case 2:if(s=t.sent,a=s.data,!a){t.next=24;break}if(r=a.finished,n=a.message,i=a.percent,c=a.error,e.$store.dispatch("messages/setProgressMessage",n),e.$store.dispatch("messages/setProgress",i),!r){t.next=18;break}e.loading=!1,o=a.data,e.$store.dispatch("messages/setMessage",{message:"Датасет «".concat(o.alias,"» выбран")},{root:!0}),e.$store.dispatch("projects/setProject",{dataset:o},{root:!0}),e.$store.dispatch("datasets/setLoaded",e.selectedIndex),e.$store.dispatch("messages/setProgress",0),e.$store.dispatch("messages/setProgressMessage",""),t.next=24;break;case 18:if(!c){t.next=22;break}return e.$store.dispatch("messages/setMessage",{error:c}),e.$store.dispatch("messages/setProgress",0),t.abrupt("return");case 22:e.$store.dispatch("messages/setProgress",0),e.createInterval();case 24:console.log(a);case 25:case"end":return t.stop()}}),t)}))),1e3)},click:function(){var e=this;return Object(n["a"])(regeneratorRuntime.mark((function t(){var s,a,r,n,i,c;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:if(!e.loading){t.next=2;break}return t.abrupt("return");case 2:return e.loading=!0,s=e.selected,a=s.alias,r=s.group,n=s.name,e.$store.dispatch("messages/setMessage",{message:"Выбран датасет «".concat(n,"»")}),t.next=7,e.$store.dispatch("datasets/choice",{alias:a,group:r});case 7:i=t.sent,c=i.success,c&&e.createInterval();case 10:case"end":return t.stop()}}),t)})))()}}}),w=b,_=Object(m["a"])(w,f,v,!1,null,null,null),$=_.exports,x={name:"Settings",components:{DatasetTab:g,DatasetButton:$},data:function(){return{tab:"GoogleDrive",loading:!1,dataset:{},interval:null,inputs:1,outputs:1,rules:{length:function(e){return function(t){return(t||"").length>=e||"Length < ".concat(e)}},required:function(e){return 0!==e.length||"Not be empty"}}}},computed:Object(i["a"])(Object(i["a"])({},Object(c["b"])({settings:"datasets/getSettings"})),{},{disabled:function(){return 0===Object.keys(this.dataset).length||this.tab!==this.dataset.mode},full:{set:function(e){this.$store.dispatch("datasets/setFull",e)},get:function(){return this.$store.getters["datasets/getFull"]}}}),methods:{createInterval:function(){var e=this;return Object(n["a"])(regeneratorRuntime.mark((function t(){return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:e.interval=setTimeout(Object(n["a"])(regeneratorRuntime.mark((function t(){var s,a,r,n,i,c,o,l;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("datasets/loadProgress",{});case 2:s=t.sent,a=s.data,console.log(a),a&&(r=a.finished,n=a.message,i=a.percent,e.$store.dispatch("messages/setProgressMessage",n),e.$store.dispatch("messages/setProgress",i),r?(c=a.data,o=c.file_manager,l=c.source_path,e.$store.dispatch("datasets/setFilesSource",o),e.$store.dispatch("datasets/setSourcePath",l),e.$store.dispatch("datasets/setFilesDrop",[]),e.$store.dispatch("datasets/clearInputData"),e.$store.dispatch("messages/setProgressMessage",""),e.$store.dispatch("messages/setProgress",0),e.loading=!1,e.full=!0):e.createInterval());case 6:case"end":return t.stop()}}),t)}))),1e3);case 1:case"end":return t.stop()}}),t)})))()},select:function(e){this.dataset=e},openFull:function(){this.$store.state.datasets.filesSource.length?this.full=!0:this.$Modal.alert({width:250,title:"Внимание!",content:"Загрузите датасет"})},download:function(){var e=this;return Object(n["a"])(regeneratorRuntime.mark((function t(){var s,a,r,n,i,c;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:if(!e.loading){t.next=2;break}return t.abrupt("return");case 2:if(s=e.dataset,a=s.mode,r=s.value,n=s.label,!a||!r){t.next=13;break}return e.loading=!0,e.$store.dispatch("messages/setMessage",{message:"Загружаю датасет ".concat(n)}),t.next=8,e.$store.dispatch("datasets/sourceLoad",{mode:a,value:r});case 8:i=t.sent,c=i.success,c?e.createInterval():e.loading=!1,t.next=14;break;case 13:e.$store.dispatch("messages/setMessage",{error:"Выберите файл"});case 14:case"end":return t.stop()}}),t)})))()}}},k=x,j=(s("89d7"),Object(m["a"])(k,a,r,!1,null,"1b55f5fe",null));t["default"]=j.exports},da47:function(e,t,s){"use strict";s("3a02")},dce1:function(e,t,s){}}]);