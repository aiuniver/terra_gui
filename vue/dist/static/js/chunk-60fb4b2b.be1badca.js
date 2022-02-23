(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-60fb4b2b"],{"08dc":function(e,t,s){},"0a1d":function(e,t,s){"use strict";s("d0b1")},"6c14":function(e,t,s){"use strict";s.r(t);var a=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"page-choice"},[s("div",{staticClass:"page-choice__menu"},[s("ChoiceMenu",{attrs:{selectedType:e.selectedType,selectedTag:e.selectedTag},on:{select:function(t){e.selectedType=t},tagClick:e.handleTag}})],1),s("div",{staticClass:"page-choice__main"},[s("Datasets",{attrs:{datasets:e.filteredList,selectedType:e.selectedType},on:{choice:e.getVersions}})],1),s("at-modal",{attrs:{showConfirmButton:!1,showCancelButton:!1,width:450,title:"Выбор версии"},on:{"on-cancel":e.onCancel},scopedSlots:e._u([{key:"footer",fn:function(){return[s("d-button",{staticStyle:{"flex-basis":"50%"},attrs:{disabled:!e.selectedVersion.alias},on:{click:e.setChoice}},[e._v("Выбрать")]),"custom"===e.selectedSet.group?s("d-button",{staticStyle:{"flex-basis":"50%"},attrs:{color:"secondary",direction:"left"}},[e._v("Создать версию")]):e._e()]},proxy:!0}]),model:{value:e.showModal,callback:function(t){e.showModal=t},expression:"showModal"}},e._l(e.versions,(function(t,a){return s("div",{key:a,staticClass:"page-choice__versions",class:{active:t.alias===e.selectedVersion.alias},on:{click:function(s){e.selectedVersion=t}}},[s("span",{staticClass:"name"},[e._v(e._s(t.name))]),s("span",{staticClass:"info"},[e._v(e._s(e.getSize(t.size)))]),s("span",{staticClass:"info"},[e._v(e._s(e.getDate(t.date)))]),"custom"===e.selectedSet.group?s("i",{staticClass:"t-icon icon-modeling-remove",on:{click:function(s){return s.stopPropagation(),e.deleteVersion(t)}}}):e._e()])})),0)],1)},n=[],r=s("1da1"),i=s("5530"),c=(s("96cf"),s("4de4"),s("d3b7"),s("99af"),s("b0c0"),s("caad"),s("2532"),s("b680"),s("159b"),function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"datasets"},[s("div",{staticClass:"datasets-filter"},[s("t-field",{attrs:{label:""}},[s("d-input-text",{attrs:{small:""},model:{value:e.search,callback:function(t){e.search=t},expression:"search"}})],1),s("div",{staticClass:"datasets-filter-sort"},[s("div",{staticClass:"datasets-filter-header"},[s("div",{staticClass:"flex align-center",on:{click:function(t){e.show=!e.show}}},[s("span",[e._v(e._s(e.selectedSort.title))]),s("d-svg",{attrs:{name:"arrow-chevron-down"}})],1),e.show?s("div",{staticClass:"datasets-filter-dropdown"},e._l(e.options,(function(t,a){return s("div",{key:a,on:{click:function(s){return e.onSelect(t)}}},[e._v(" "+e._s(t.title)+" ")])})),0):e._e()]),s("div",{staticClass:"datasets-filter-display"},[s("d-svg",{class:["ci-tile mr-4",{"ci-tile--selected":e.display}],attrs:{name:"grid-cube-outline"},nativeOn:{click:function(t){e.display=!0}}}),s("d-svg",{class:["ci-tile",{"ci-tile--selected":!e.display}],attrs:{name:"lines-justyfy"},nativeOn:{click:function(t){e.display=!1}}})],1)])],1),s("scrollbar",{staticStyle:{"justify-self":"stretch"},attrs:{ops:{rail:{gutterOfSide:"0px"}}}},[e.display?s("div",{staticClass:"datasets-cards"},e._l(e.sortedList,(function(t,a){return s("DatasetCard",{key:a,staticClass:"datasets-cards__item",attrs:{dataset:t},on:{click:function(s){return e.$emit("choice",t)}}})})),1):s("Table",{attrs:{selectedType:e.selectedType,data:e.sortedList},on:{choice:function(t){return e.$emit("choice",t)}}}),e.sortedList.length?e._e():s("div",{staticClass:"datasets__empty"},[e._v("Не найдено")])],1)],1)}),o=[],l=(s("498a"),s("ac1f"),s("841c"),s("4e82"),function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"dataset-card-new",on:{click:e.onClick}},[s("div",{class:["card-borders",{"card-borders--active":e.isActive}]}),s("div",{staticClass:"dataset-card-new__wrapper"},[s("p",{staticClass:"dataset-card-new__title"},[e._v(e._s(e.dataset.name))]),s("div",{staticClass:"dataset-card-new__info"},[s("p",[e._v(e._s(e.dataset.architecture))]),s("div",{staticClass:"dataset-card-new__info--size"},[s("span",[e._v(e._s(e.group))])])])]),"custom"===e.dataset.group?s("i",{staticClass:"t-icon icon-modeling-remove",on:{click:function(t){return t.stopPropagation(),e.deleteDataset.apply(null,arguments)}}}):e._e()])}),d=[],u={name:"DatasetCard",props:{dataset:{type:Object,default:function(){}}},computed:{isActive:function(){var e=this.$store.getters["projects/getProject"].dataset||{},t=e.alias,s=e.group;return"".concat(s,"_").concat(t)===this.dataset.id},group:function(){return this.$store.getters["datasets/getGroups"]?this.$store.getters["datasets/getGroups"][this.dataset.group]:""}},methods:{onClick:function(){this.$emit("click",this.dataset)},deleteDataset:function(){var e=this;this.$Modal.confirm({title:"Внимание!",content:"Уверены, что хотите удалить этот датасет?",width:300,callback:function(){var t=Object(r["a"])(regeneratorRuntime.mark((function t(s){return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:if("confirm"!==s){t.next=9;break}return e.$store.dispatch("settings/setOverlay",!0),t.next=4,e.$store.dispatch("axios",{url:"/datasets/delete/",data:{group:e.dataset.group,alias:e.dataset.alias}});case 4:return t.next=6,e.$store.dispatch("projects/get");case 6:return t.next=8,e.$store.dispatch("datasets/get");case 8:e.$store.dispatch("settings/setOverlay",!1);case 9:case"end":return t.stop()}}),t)})));function s(e){return t.apply(this,arguments)}return s}()})}}},p=u,h=(s("9973"),s("2877")),f=Object(h["a"])(p,l,d,!1,null,null,null),m=f.exports,g=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("table",[s("thead",[s("tr",e._l(e.headers,(function(t,a){return s("th",{key:"table_dataset_th"+a,on:{click:function(s){return e.handleSort(t.idx)}}},[s("span",[e._v(e._s(t.title))])])})),0)]),s("tbody",e._l(e.data,(function(t,a){return s("tr",{key:"table_dataset_tr"+a,on:{click:function(s){return e.$emit("choice",t)}}},e._l(e.headers,(function(a,n){var r=a.value;return s("td",{key:"table_dataset_td"+n},[s("span","group"!==r?[e._v(e._s(t[r]))]:[e._v(e._s(e.$store.getters["datasets/getGroups"][t[r]]))])])})),0)})),0)])},v=[],_=s("2909"),b=(s("a9e3"),s("a434"),s("fb6a"),{name:"Table",props:{data:{type:Array,default:function(){return[]}},selectedType:{type:[String,Number],default:0}},data:function(){return{list:[{title:"Название",value:"name",idx:0},{title:"Размер",value:"size",idx:1},{title:"Автор",value:"group",idx:2},{title:"Последнее использование",value:"date",idx:3},{title:"Создание",value:"alias",idx:4}],sortId:0,sortReverse:!1}},methods:{handleSort:function(e){this.sortId===e?this.sortReverse?this.sortReverse=!1:this.sortReverse=!0:this.sortReverse=!1,this.sortId=e}},computed:{headers:function(){var e=Object(_["a"])(this.list);return 1===this.selectedType?(e.splice(2,1),e):e.slice(0,4)}}}),y=b,x=(s("9c47"),Object(h["a"])(y,g,v,!1,null,"0bea8bce",null)),k=x.exports,C=s("2f62"),w=[{title:"По алфавиту от А до Я",value:"alphabet",idx:0},{title:"По алфавиту от Я до А",value:"alphabet_reverse",idx:1},{title:"Последние созданные",value:"last_created",idx:2},{title:"Последние использованные",value:"last_used",idx:3},{title:"Популярные",value:"popular",idx:4},{title:"Последние добавленные",value:"last_added",idx:5}],$={name:"Datasets",props:["datasets","selectedType"],components:{DatasetCard:m,Table:k},data:function(){return{list:["Недавние датасеты","Проектные датасеты","Датасеты Terra"],search:"",display:!0,show:!1,selectedSort:w[0],options:w}},computed:Object(i["a"])(Object(i["a"])({},Object(C["c"])({project:"projects/getProject"})),{},{sortedList:function(){var e=this.datasets||[],t=this.search.trim(),s=[];return 0===this.selectedSort.idx&&(s=e.sort((function(e,t){return e.name.localeCompare(t.name)}))),1===this.selectedSort.idx&&(s=e.sort((function(e,t){return t.name.localeCompare(e.name)}))),2===this.selectedSort.idx&&(s=e.sort((function(e,t){return e.name.localeCompare(t.name)}))),3===this.selectedSort.idx&&(s=e.sort((function(e,t){return e.name.localeCompare(t.name)}))),4===this.selectedSort.idx&&(s=e.sort((function(e,t){return e.name.localeCompare(t.name)}))),5===this.selectedSort.idx&&(s=e),s.filter((function(e){return!t||e.name.toLowerCase().includes(t.toLowerCase())}))}}),methods:Object(i["a"])(Object(i["a"])({},Object(C["b"])({choice:"datasets/choice"})),{},{isLoaded:function(e){var t,s;return(null===(t=this.project)||void 0===t||null===(s=t.dataset)||void 0===s?void 0:s.alias)===e.alias},selectDataset:function(e){this.$store.dispatch("datasets/selectDataset",e)},handleChangeFilter:function(e){this.selectedSort=e},randomDate:function(e,t,s,a){var n=new Date(+e+Math.random()*(t-e)),r=s+Math.random()*(a-s)|0;return n.setHours(r),n},onSelect:function(e){this.show=!1,this.selectedSort=e},onChoice:function(e){var t=this;this.$Modal.confirm({title:"Загрузить",content:"Загрузить датасет ".concat(e.name,"?")}).then((function(s){if(s){if(console.log(e),!e.training_available)return;if(!t.isLoaded(e))return void t.choice(e)}}))}})},T=$,j=(s("0a1d"),Object(h["a"])(T,c,o,!1,null,"04b587c9",null)),O=j.exports,S=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"menu"},[s("div",{staticClass:"menu-list"},[s("div",{class:["menu-list__item",{"menu-list__item--selected":1===e.selectedType}],on:{click:function(t){return e.select(1)}}},[s("d-svg",{attrs:{name:"clock"}}),s("span",[e._v("Недавние")])],1),s("div",{class:["menu-list__item",{"menu-list__item--selected":2===e.selectedType}],on:{click:function(t){return e.select(2)}}},[s("d-svg",{attrs:{name:"file-outline"}}),s("span",[e._v("Проектные")])],1),s("div",{class:["menu-list__item",{"menu-list__item--selected":3===e.selectedType}],on:{click:function(t){return e.select(3)}}},[s("d-svg",{attrs:{name:"world"}}),s("span",[e._v("Terra")])],1)]),s("hr"),s("scrollbar",{attrs:{ops:{rail:{gutterOfSide:"0px"}}}},e._l(e.tags,(function(t){return s("ul",{key:t.name,staticClass:"menu-categories"},[s("p",{class:{"menu-categories--selected":e.selectedTag.name===t.name},on:{click:function(s){return e.$emit("tagClick",{type:"group",name:t.name})}}},[e._v(e._s(t.name))]),e._l(t.items,(function(a){return s("li",{key:a,staticClass:"menu-categories__item",class:{"menu-categories--selected":e.selectedTag.name===a&&e.selectedTag.group===t.name},on:{click:function(s){return e.$emit("tagClick",{type:"tag",name:a,group:t.name})}}},[e._v(e._s(a))])}))],2)})),0)],1)},D=[],R={name:"choice-menu",props:["selectedType","selectedTag"],computed:Object(i["a"])({},Object(C["c"])({tags:"datasets/getTags"})),methods:{select:function(e){if(this.selectedType===e)return this.$emit("select",0);this.$emit("select",e)}}},M=R,P=(s("6e3c"),Object(h["a"])(M,S,D,!1,null,null,null)),L=P.exports,V={name:"Choice",components:{Datasets:O,ChoiceMenu:L},data:function(){return{selectedType:0,showModal:!1,versions:[],tID:null,selectedSet:{},selectedVersion:{},selectedTag:{}}},computed:Object(i["a"])(Object(i["a"])({},Object(C["c"])({datasets:"datasets/getDatasets"})),{},{filteredList:function(){var e=this,t=this.datasets;if(0!==this.selectedType){if(2===this.selectedType){var s=this.$store.getters["projects/getProject"].dataset||{};t=t.filter((function(e){return e.id==="".concat(s.group,"_").concat(s.alias)}))}3===this.selectedType&&(t=t.filter((function(e){return"terra"===e.group})))}return this.selectedTag.type?t.filter((function(t){return"group"===e.selectedTag.type?e.selectedTag.name===t.architecture:t.tags.includes(e.selectedTag.name)&&e.selectedTag.group===t.architecture})):t}}),methods:{getDate:function(e){return e?new Date(e).toLocaleString("ru-RU",{dateStyle:"short",timeStyle:"short"}):""},getSize:function(e){return null!==e&&void 0!==e&&e.value?"".concat(e.short.toFixed(2)," ").concat(e.unit):""},onCancel:function(){this.versions=[],this.selectedVersion={},this.selectedSet={}},getVersions:function(e){var t=this;return Object(r["a"])(regeneratorRuntime.mark((function s(){var a,n,r,i;return regeneratorRuntime.wrap((function(s){while(1)switch(s.prev=s.next){case 0:return a=e.alias,n=e.group,t.showModal=!0,s.next=4,t.$store.dispatch("axios",{url:"/datasets/versions/",data:{group:n,alias:a}});case 4:r=s.sent,i=r.data,t.selectedSet={group:n,alias:a},t.versions=i;case 8:case"end":return s.stop()}}),s)})))()},setChoice:function(){var e=this;return Object(r["a"])(regeneratorRuntime.mark((function t(){var s,a,n,r,i;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return e.$store.dispatch("settings/setOverlay",!0),s=e.selectedSet,a=s.group,n=s.alias,t.next=4,e.$store.dispatch("datasets/choice",{group:a,alias:n,version:e.selectedVersion.alias});case 4:r=t.sent,i=r.success,i&&(e.$store.dispatch("messages/setMessage",{message:"Загружаю датасет"}),e.createInterval());case 7:case"end":return t.stop()}}),t)})))()},createInterval:function(){var e=this;this.tID=setTimeout(Object(r["a"])(regeneratorRuntime.mark((function t(){var s,a,n,r,i,c,o,l;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("datasets/choiceProgress",{});case 2:if(s=t.sent,!s){t.next=31;break}if(a=s.data,!a){t.next=28;break}if(n=a.finished,r=a.message,i=a.percent,c=a.error,e.$store.dispatch("messages/setProgressMessage",r),e.$store.dispatch("messages/setProgress",i),!n){t.next=20;break}return e.$store.dispatch("messages/setProgress",0),e.$store.dispatch("messages/setProgressMessage",""),t.next=14,e.$store.dispatch("projects/get");case 14:e.$store.dispatch("messages/setMessage",{message:"Датасет «".concat((null===a||void 0===a||null===(o=a.data)||void 0===o||null===(l=o.dataset)||void 0===l?void 0:l.name)||"","» выбран")}),e.$store.dispatch("settings/setOverlay",!1),e.showModal=!1,e.onCancel(),t.next=26;break;case 20:if(!c){t.next=25;break}return e.$store.dispatch("messages/setProgressMessage",""),e.$store.dispatch("messages/setProgress",0),e.$store.dispatch("settings/setOverlay",!1),t.abrupt("return");case 25:e.createInterval();case 26:t.next=29;break;case 28:e.$store.dispatch("settings/setOverlay",!1);case 29:t.next=32;break;case 31:e.$store.dispatch("settings/setOverlay",!1);case 32:case"end":return t.stop()}}),t)}))),1e3)},handleTag:function(e){if(e.type===this.selectedTag.type&&e.name===this.selectedTag.name)return this.selectedTag={};this.selectedTag=e},deleteVersion:function(e){var t=this;this.$Modal.confirm({title:"Внимание!",content:"Уверены, что хотите удалить эту версию?",width:300,callback:function(){var s=Object(r["a"])(regeneratorRuntime.mark((function s(a){return regeneratorRuntime.wrap((function(s){while(1)switch(s.prev=s.next){case 0:if("confirm"!==a){s.next=6;break}return s.next=3,t.$store.dispatch("axios",{url:"/datasets/delete/version/",data:{group:t.selectedSet.group,alias:t.selectedSet.alias,version:e.alias}});case 3:return t.getVersions(t.selectedSet),s.next=6,t.$store.dispatch("projects/get");case 6:case"end":return s.stop()}}),s)})));function a(e){return s.apply(this,arguments)}return a}()})},getDatasets:function(){var e=this;return Object(r["a"])(regeneratorRuntime.mark((function t(){var s,a;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("axios",{url:"/datasets/info/"});case 2:s=t.sent,a=s.data,a.datasets.forEach((function(t){t.datasets.forEach((function(s){e.datasets.push(Object(i["a"])(Object(i["a"])({},s),{},{group:t.alias}))}))})),e.$store.commit("datasets/SET_GROUPS",a.groups),e.tags=a.tags;case 7:case"end":return t.stop()}}),t)})))()}},created:function(){this.$store.dispatch("datasets/get")}},E=V,I=(s("c44c"),Object(h["a"])(E,a,n,!1,null,null,null));t["default"]=I.exports},"6e3c":function(e,t,s){"use strict";s("08dc")},9973:function(e,t,s){"use strict";s("c6dd")},"9c47":function(e,t,s){"use strict";s("aa5a")},aa5a:function(e,t,s){},ac99:function(e,t,s){},c44c:function(e,t,s){"use strict";s("ac99")},c6dd:function(e,t,s){},d0b1:function(e,t,s){}}]);