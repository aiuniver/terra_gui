(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-763d5dc5"],{"08dc":function(e,t,s){},"6c14":function(e,t,s){"use strict";s.r(t);var a=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"page-choice"},[s("div",{staticClass:"page-choice__menu"},[s("ChoiceMenu",{attrs:{selectedType:e.selectedType},on:{select:function(t){e.selectedType=t}}})],1),s("div",{staticClass:"page-choice__main"},[s("Datasets",{attrs:{datasets:e.datasets,selectedType:e.selectedType},on:{choice:e.getVersions}})],1),s("at-modal",{attrs:{showConfirmButton:!1,showCancelButton:!1,width:400,title:"Выбор версии"},on:{"on-cancel":e.onCancel},scopedSlots:e._u([{key:"footer",fn:function(){return[s("d-button",{staticStyle:{"flex-basis":"50%"},attrs:{disabled:!e.selectedVersion.value},on:{click:e.setChoice}},[e._v("Выбрать")])]},proxy:!0}]),model:{value:e.showModal,callback:function(t){e.showModal=t},expression:"showModal"}},[s("ul",{staticClass:"page-choice__versions"},e._l(e.versions,(function(t,a){return s("li",{key:a,class:{active:t.value===e.selectedVersion.value},on:{click:function(s){e.selectedVersion=t}}},[e._v(e._s(t.label))])})),0)])],1)},i=[],n=s("5530"),r=s("1da1"),c=(s("96cf"),s("d81d"),s("b0c0"),s("159b"),function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"datasets"},[s("p",{staticClass:"datasets-type flex align-center"},[s("span",{staticClass:"mr-4"},[e._v(e._s(e.list[e.selectedType]))]),1===e.selectedType?s("d-svg",{attrs:{name:"file-blank-outline"}}):e._e()],1),s("div",{staticClass:"datasets-filter"},[s("t-field",{attrs:{label:""}},[s("d-input-text",{attrs:{small:""},model:{value:e.search,callback:function(t){e.search=t},expression:"search"}})],1),s("div",{staticClass:"datasets-filter-sort"},[s("div",{staticClass:"datasets-filter-header"},[s("div",{staticClass:"flex align-center",on:{click:function(t){e.show=!e.show}}},[s("span",[e._v(e._s(e.selectedSort.title))]),s("d-svg",{attrs:{name:"arrow-chevron-down"}})],1),e.show?s("div",{staticClass:"datasets-filter-dropdown"},e._l(e.options,(function(t,a){return s("div",{key:a,on:{click:function(s){return e.onSelect(t)}}},[e._v(" "+e._s(t.title)+" ")])})),0):e._e()]),s("div",{staticClass:"datasets-filter-display"},[s("d-svg",{class:["ci-tile mr-4",{"ci-tile--selected":e.display}],attrs:{name:"grid-cube-outline"},nativeOn:{click:function(t){e.display=!0}}}),s("d-svg",{class:["ci-tile",{"ci-tile--selected":!e.display}],attrs:{name:"lines-justyfy"},nativeOn:{click:function(t){e.display=!1}}})],1)])],1),s("scrollbar",{staticStyle:{"justify-self":"stretch"}},[e.display?s("div",{staticClass:"datasets-cards"},e._l(e.sortedList,(function(t,a){return s("DatasetCard",{key:a,attrs:{dataset:t},on:{click:function(s){return e.$emit("choice",t)}}})})),1):s("Table",{attrs:{selectedType:e.selectedType}}),e.sortedList.length?e._e():s("div",{staticClass:"datasets__empty"},[e._v("Не найдено")])],1)],1)}),l=[],o=(s("ac1f"),s("841c"),s("4de4"),s("4e82"),s("caad"),s("2532"),function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"dataset-card-new",on:{click:e.onClick}},[s("div",{class:["card-borders",{"card-borders--active":e.isActive}]}),s("div",{staticClass:"dataset-card-new__wrapper"},[s("p",{staticClass:"dataset-card-new__title"},[e._v(e._s(e.dataset.name))]),s("div",{staticClass:"dataset-card-new__info"},[s("p",[e._v("login_name_user")]),s("div",{staticClass:"dataset-card-new__info--size"},[s("i",{staticClass:"ci-icon ci-image"}),s("span",[e._v(e._s(e.dataset.size?e.dataset.size.short.toFixed(2)+" "+e.dataset.size.unit:"Предустановленный"))])])])])])}),d=[],u=(s("99af"),{name:"DatasetCard",props:{dataset:{type:Object,default:function(){}}},computed:{isActive:function(){var e=this.$store.getters["projects/getProject"].dataset||{},t=e.alias,s=e.group;return"".concat(s,"_").concat(t)===this.dataset.id}},methods:{onClick:function(){this.$emit("click",this.dataset)}}}),v=u,p=(s("9973"),s("2877")),h=Object(p["a"])(v,o,d,!1,null,null,null),f=h.exports,m=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("table",[s("thead",[s("tr",e._l(e.headers,(function(t,a){return s("th",{key:"table_dataset_th"+a,on:{click:function(s){return e.handleSort(t.idx)}}},[s("span",[e._v(e._s(t.title))])])})),0)]),s("tbody",e._l(e.datasets,(function(t,a){return s("tr",{key:"table_dataset_tr"+a},e._l(e.headers,(function(a,i){var n=a.value;return s("td",{key:"table_dataset_td"+i},[s("span",[e._v(e._s(t[n]))])])})),0)})),0)])},_=[],g=s("2909"),b=(s("a9e3"),s("a434"),s("fb6a"),{name:"Table",props:{data:{type:Array,default:function(){return[]}},selectedType:{type:[String,Number],default:0}},data:function(){return{list:[{title:"Название",value:"name",idx:0},{title:"Размер",value:"size",idx:1},{title:"Автор",value:"group",idx:2},{title:"Последнее использование",value:"date",idx:3},{title:"Создание",value:"alias",idx:4}],sortId:0,sortReverse:!1}},methods:{handleSort:function(e){this.sortId===e?this.sortReverse?this.sortReverse=!1:this.sortReverse=!0:this.sortReverse=!1,this.sortId=e}},computed:{datasets:function(){var e=this.data;return 0===this.sortId?this.sortReverse?e.sort((function(e,t){return t.name.localeCompare(e.name)})):e.sort((function(e,t){return e.name.localeCompare(t.name)})):e},headers:function(){var e=Object(g["a"])(this.list);return 1===this.selectedType?(e.splice(2,1),e):e.slice(0,4)}}}),y=b,C=(s("a3ae"),Object(p["a"])(y,m,_,!1,null,"5d499e02",null)),w=C.exports,x=s("2f62"),k={name:"Datasets",props:["datasets","selectedType"],components:{DatasetCard:f,Table:w},data:function(){return{list:["Недавние датасеты","Проектные датасеты","Датасеты Terra"],search:"",display:!0,sortIdx:4,show:!1,selectedSort:{title:"По алфавиту от А до Я",value:"alphabet",idx:0},options:[{title:"По алфавиту от А до Я",value:"alphabet",idx:0},{title:"По алфавиту от Я до А",value:"alphabet_reverse",idx:1},{title:"Последние созданные",value:"last_created",idx:2},{title:"Последние использованные",value:"last_used",idx:3},{title:"Популярные",value:"popular",idx:4},{title:"Последние добавленные",value:"last_added",idx:5}]}},computed:Object(n["a"])(Object(n["a"])({},Object(x["c"])({project:"projects/getProject"})),{},{sortedList:function(){var e=this.datasets||[],t=this.search;return e.sort((function(e,t){return e.name.localeCompare(t.name)})).filter((function(e){return!t||e.name.toLowerCase().includes(t.toLowerCase())}))}}),methods:Object(n["a"])(Object(n["a"])({},Object(x["b"])({choice:"datasets/choice"})),{},{isLoaded:function(e){var t,s;return(null===(t=this.project)||void 0===t||null===(s=t.dataset)||void 0===s?void 0:s.alias)===e.alias},selectDataset:function(e){this.$store.dispatch("datasets/selectDataset",e)},handleChangeFilter:function(e){this.selectedSort=e},randomDate:function(e,t,s,a){var i=new Date(+e+Math.random()*(t-e)),n=s+Math.random()*(a-s)|0;return i.setHours(n),i},onSelect:function(e){this.show=!1,this.selectedSort=e},onChoice:function(e){var t=this;this.$Modal.confirm({title:"Загрузить",content:"Загрузить датасет ".concat(e.name,"?")}).then((function(s){if(s){if(console.log(e),!e.training_available)return;if(!t.isLoaded(e))return void t.choice(e)}}))}})},$=k,j=(s("c194"),Object(p["a"])($,c,l,!1,null,"8db8aa60",null)),O=j.exports,T=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",[s("div",{staticClass:"menu-list"},[s("div",{class:["menu-list__item",{"menu-list__item--selected":0===e.selectedType}],on:{click:function(t){return e.$emit("select",0)}}},[s("d-svg",{attrs:{name:"clock"}}),s("span",[e._v("Недавние")])],1),s("div",{class:["menu-list__item",{"menu-list__item--selected":1===e.selectedType}],on:{click:function(t){return e.$emit("select",1)}}},[s("d-svg",{attrs:{name:"file-outline"}}),s("span",[e._v("Проектные")])],1),s("div",{class:["menu-list__item",{"menu-list__item--selected":2===e.selectedType}],on:{click:function(t){return e.$emit("select",2)}}},[s("d-svg",{attrs:{name:"world"}}),s("span",[e._v("Terra")])],1)]),s("hr"),e._m(0)])},M=[function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"menu-categories"},[s("ul",{staticClass:"menu-categories__item"},[s("li",[e._v("Изображения")]),s("li",[e._v("Машины")]),s("li",[e._v("Круглые")]),s("li",[e._v("Самолеты")]),s("li",[e._v("Квадраты")])]),s("ul",{staticClass:"menu-categories__item"},[s("li",[e._v("Видео")]),s("li",[e._v("Машины")]),s("li",[e._v("Круглые")]),s("li",[e._v("Самолеты")]),s("li",[e._v("Квадраты")])]),s("ul",{staticClass:"menu-categories__item"},[s("li",[e._v("Текст")]),s("li",[e._v("Машины")]),s("li",[e._v("Круглые")]),s("li",[e._v("Самолеты")]),s("li",[e._v("Квадраты")])])])}],S={name:"choice-menu",props:["selectedType"]},R=S,D=(s("6e3c"),Object(p["a"])(R,T,M,!1,null,null,null)),I=D.exports,P={name:"Choice",components:{Datasets:O,ChoiceMenu:I},data:function(){return{selectedType:2,datasets:[],showModal:!1,versions:[],tID:null,selectedSet:{},selectedVersion:{}}},methods:{onCancel:function(){this.versions=[],this.selectedVersion={}},getVersions:function(e){var t=this;return Object(r["a"])(regeneratorRuntime.mark((function s(){var a,i,n,r;return regeneratorRuntime.wrap((function(s){while(1)switch(s.prev=s.next){case 0:return a=e.alias,i=e.group,t.showModal=!0,s.next=4,t.$store.dispatch("axios",{url:"/datasets/versions/",data:{group:i,alias:a}});case 4:n=s.sent,r=n.data,t.selectedSet={group:i,alias:a},t.versions=r.map((function(e){return{label:e.name,value:e.alias}}));case 8:case"end":return s.stop()}}),s)})))()},setChoice:function(){var e=this;return Object(r["a"])(regeneratorRuntime.mark((function t(){var s,a,i,n,r;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return e.$store.dispatch("settings/setOverlay",!0),s=e.selectedSet,a=s.group,i=s.alias,t.next=4,e.$store.dispatch("datasets/choice",{group:a,alias:i,version:e.selectedVersion.value});case 4:n=t.sent,r=n.success,r&&(e.$store.dispatch("messages/setMessage",{message:"Загружаю датасет"}),e.createInterval());case 7:case"end":return t.stop()}}),t)})))()},createInterval:function(){var e=this;this.tID=setTimeout(Object(r["a"])(regeneratorRuntime.mark((function t(){var s,a,i,n,r,c,l,o;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("datasets/choiceProgress",{});case 2:if(s=t.sent,!s){t.next=31;break}if(a=s.data,!a){t.next=28;break}if(i=a.finished,n=a.message,r=a.percent,c=a.error,e.$store.dispatch("messages/setProgressMessage",n),e.$store.dispatch("messages/setProgress",r),!i){t.next=20;break}return e.$store.dispatch("messages/setProgress",0),e.$store.dispatch("messages/setProgressMessage",""),t.next=14,e.$store.dispatch("projects/get");case 14:e.$store.dispatch("messages/setMessage",{message:"Датасет «".concat((null===a||void 0===a||null===(l=a.data)||void 0===l||null===(o=l.dataset)||void 0===o?void 0:o.name)||"","» выбран")}),e.$store.dispatch("settings/setOverlay",!1),e.showModal=!1,e.onCancel(),t.next=26;break;case 20:if(!c){t.next=25;break}return e.$store.dispatch("messages/setProgressMessage",""),e.$store.dispatch("messages/setProgress",0),e.$store.dispatch("settings/setOverlay",!1),t.abrupt("return");case 25:e.createInterval();case 26:t.next=29;break;case 28:e.$store.dispatch("settings/setOverlay",!1);case 29:t.next=32;break;case 31:e.$store.dispatch("settings/setOverlay",!1);case 32:case"end":return t.stop()}}),t)}))),1e3)}},created:function(){var e=this;return Object(r["a"])(regeneratorRuntime.mark((function t(){var s,a;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("axios",{url:"/datasets/info/"});case 2:s=t.sent,a=s.data,a.datasets.forEach((function(t){t.datasets.forEach((function(s){e.datasets.push(Object(n["a"])(Object(n["a"])({},s),{},{group:t.alias}))}))}));case 5:case"end":return t.stop()}}),t)})))()}},E=P,V=(s("c44c"),Object(p["a"])(E,a,i,!1,null,null,null));t["default"]=V.exports},"6e3c":function(e,t,s){"use strict";s("08dc")},9973:function(e,t,s){"use strict";s("c6dd")},a3ae:function(e,t,s){"use strict";s("d034")},ac99:function(e,t,s){},c0cf:function(e,t,s){},c194:function(e,t,s){"use strict";s("c0cf")},c44c:function(e,t,s){"use strict";s("ac99")},c6dd:function(e,t,s){},d034:function(e,t,s){}}]);