(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-25f6b71a"],{"040d":function(t,e,s){},"0587":function(t,e,s){},"0bf9":function(t,e,s){},"129f":function(t,e){t.exports=Object.is||function(t,e){return t===e?0!==t||1/t===1/e:t!=t&&e!=e}},"19a1":function(t,e,s){"use strict";s("22b8")},"1aff":function(t,e,s){"use strict";s("7814")},"22b8":function(t,e,s){},2451:function(t,e,s){"use strict";s("8523")},"2a9c":function(t,e,s){"use strict";s("b7d5")},"3f37":function(t,e,s){},4204:function(t,e,s){},"4a01":function(t,e,s){},"52fb":function(t,e,s){},5551:function(t,e,s){},"5a7c":function(t,e,s){},6522:function(t,e,s){"use strict";var i=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"dropdown"},[s("label",[t._v(t._s(t.label))]),s("input",{directives:[{name:"model",rawName:"v-model",value:t.search,expression:"search"}],staticClass:"dropdown__input",attrs:{name:t.name,disabled:t.disabled,placeholder:t.placeholder},domProps:{value:t.search},on:{focus:t.focus,blur:function(e){return t.select(!1)},input:function(e){e.target.composing||(t.search=e.target.value)}}}),s("div",{directives:[{name:"show",rawName:"v-show",value:t.show,expression:"show"}],staticClass:"dropdown__content"},[t._l(t.filterList,(function(e,i){return s("div",{key:i,on:{mousedown:function(s){return t.select(e)}}},[t._v(" "+t._s(e.label)+" ")])})),t.filterList.length?t._e():s("div",[t._v("Нет данных")])],2)])},a=[],n=(s("ac1f"),s("841c"),s("4de4"),s("caad"),s("2532"),{name:"Autocomplete",props:{name:String,list:{type:Array,required:!0,default:function(){return[]}},placeholder:String,disabled:Boolean,label:String,value:String},data:function(){return{selected:{},show:!1,search:""}},created:function(){this.search=this.value},computed:{filterList:function(){var t=this;return this.list?this.list.filter((function(e){var s=t.search;return!s||e.label.toLowerCase().includes(s.toLowerCase())})):[]}},methods:{select:function(t){t?(this.selected=t,this.show=!1,this.search=t.label,this.$emit("input",this.selected.value),this.$emit("change",t)):(this.search=this.selected.label,this.show=!1)},focus:function(){this.show=!0,this.$emit("focus",!0)}},watch:{value:{handler:function(t){this.show=!1,this.search=t}}}}),r=n,l=(s("9abe"),s("2877")),c=Object(l["a"])(r,i,a,!1,null,"6cd1703e",null);e["a"]=c.exports},6835:function(t,e,s){"use strict";s.r(e);var i=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("main",{staticClass:"page-datasets"},[s("div",{staticClass:"cont"},[s("Dataset",{directives:[{name:"show",rawName:"v-show",value:!t.full,expression:"!full"}]}),s("ParamsFull",{directives:[{name:"show",rawName:"v-show",value:t.full,expression:"full"}]}),s("Params",{directives:[{name:"show",rawName:"v-show",value:!t.full,expression:"!full"}]})],1)])},a=[],n=s("5530"),r=s("2f62"),l=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"board"},[s("div",{staticClass:"wrapper"},[s("Filters"),s("div",{staticClass:"project-datasets-block datasets",style:t.height},[s("div",{staticClass:"title",on:{click:function(e){return t.click("name")}}},[t._v("Выберите датасет")]),s("scrollbar",[s("div",{staticClass:"inner"},[s("div",{staticClass:"dataset-card-container"},[s("div",{staticClass:"dataset-card-wrapper"},[t._l(t.datasets,(function(e,i){return[s("CardDataset",{key:i,attrs:{dataset:e},on:{clickCard:t.click}})]}))],2)])])])],1)],1)])},c=[],o=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{ref:"filters",staticClass:"project-datasets-block filters"},[s("div",{staticClass:"title"},[t._v("Теги")]),s("div",{staticClass:"inner"},[s("ul",t._l(t.tags,(function(e,i){var a=e.name,n=e.alias,r=e.active;return s("li",{key:i,class:{active:r},on:{click:function(e){return t.click(i,n)}}},[s("span",[t._v(" "+t._s(a)+" ")])])})),0)])])},u=[],d={computed:{tags:{set:function(t){this.$store.dispatch("datasets/setTags",t)},get:function(){return this.$store.getters["datasets/getTags"]}},tagsFilter:{set:function(t){this.$store.dispatch("datasets/setTagsFilter",t)},get:function(){return this.$store.getters["datasets/getTagsFilter"]}}},methods:{click:function(t){this.tags[t].active=!this.tags[t].active,this.tagsFilter=this.tags.reduce((function(t,e){var s=e.active,i=e.alias;return s&&t.push(i),t}),[])},myEventHandler:function(){this.$store.dispatch("settings/setHeight",{filter:this.$refs.filters.clientHeight})}},watch:{tags:function(){var t=this;this.$nextTick((function(){t.$store.dispatch("settings/setHeight",{filter:t.$refs.filters.clientHeight})}))}},mounted:function(){var t=this;setTimeout((function(){t.$store.dispatch("settings/setHeight",{filter:t.$refs.filters.clientHeight})}),100)},created:function(){window.addEventListener("resize",this.myEventHandler)},destroyed:function(){window.removeEventListener("resize",this.myEventHandler)}},f=d,h=s("2877"),p=Object(h["a"])(f,o,u,!1,null,null,null),g=p.exports,v=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{class:["dataset-card-item",{active:t.dataset.active}],on:{click:function(e){return t.$emit("clickCard",t.dataset)}}},[s("div",{staticClass:"dataset-card"},[s("div",{staticClass:"card-title"},[t._v(t._s(t.dataset.name))]),s("div",{staticClass:"card-body"},t._l(t.dataset.tags,(function(e,i){var a=e.name;return s("div",{key:"tag_"+i,staticClass:"card-tag"},[t._v(" "+t._s(a)+" ")])})),0),s("div",{class:"card-extra "+(t.dataset.size?"is-custom":"")},[s("div",{staticClass:"wrapper"},[s("span",[t._v(t._s(t.dataset.size?t.dataset.size:"Предустановленный"))])]),s("div",{staticClass:"remove"})])])])},m=[],_={props:{dataset:{type:Object,default:function(){}}}},b=_,k=Object(h["a"])(b,v,m,!1,null,null,null),y=k.exports,C={components:{CardDataset:y,Filters:g},data:function(){return{hight:0}},computed:Object(n["a"])(Object(n["a"])({},Object(r["b"])({datasets:"datasets/getDatasets"})),{},{height:function(){return this.$store.getters["settings/height"]({deduct:"filter",padding:52,clean:!0})}}),methods:{click:function(t){this.$store.dispatch("datasets/setSelect",t)}}},w=C,$=(s("2451"),Object(h["a"])(w,l,c,!1,null,"3968546c",null)),x=$.exports,D=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"params"},[s("div",{staticClass:"params__btn",on:{click:t.openFull}},[s("i",{staticClass:"params__btn--icon"})]),s("div",{staticClass:"params__items"},[s("div",{staticClass:"params__items--item"},[s("DatasetButton")],1),s("div",{staticClass:"params__items--item pa-0"},[s("DatasetTab",{on:{select:t.select}})],1),s("div",{staticClass:"params__items--item"},[s("div",{staticClass:"params__items--btn"},[s("t-button",{attrs:{loading:t.loading},nativeOn:{click:function(e){return t.download.apply(null,arguments)}}})],1)])])])},O=[],j=s("1da1"),S=(s("96cf"),s("b64b"),function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"tabs"},[s("ul",{staticClass:"tabs__list"},t._l(t.items,(function(e,i){var a=e.title,n=e.active,r=e.mode;return s("li",{key:i,class:["tabs__list--item",{active:n}],on:{click:function(e){return e.preventDefault(),t.click(r)}}},[t._v(" "+t._s(a)+" ")])})),0),s("div",{staticClass:"tabs__title"},[t._v("Создание датасета")]),s("div",{directives:[{name:"show",rawName:"v-show",value:"GoogleDrive"===t.select,expression:"select === 'GoogleDrive'"}],staticClass:"tabs__item"},[s("Autocomplete2",{attrs:{list:t.list,label:"Выберите файл из Google-диске"},on:{focus:t.focus,change:t.selected}})],1),s("div",{directives:[{name:"show",rawName:"v-show",value:"URL"===t.select,expression:"select === 'URL'"}],staticClass:"tabs__item"},[s("t-input",{attrs:{label:"Введите URL на архив исходников"},on:{blur:t.blur}})],1)])}),F=[],L=(s("d81d"),s("6522")),E={name:"DatasetTab",components:{Autocomplete2:L["a"]},props:{},data:function(){return{select:"GoogleDrive",list:[],items:[{title:"Google drive",active:!0,mode:"GoogleDrive"},{title:"URL",active:!1,mode:"URL"}]}},methods:{focus:function(){var t=this;return Object(j["a"])(regeneratorRuntime.mark((function e(){var s;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.$store.dispatch("axios",{url:"/datasets/sources/?term="});case 2:if(s=e.sent,s){e.next=5;break}return e.abrupt("return");case 5:console.log(s),t.list=s;case 7:case"end":return e.stop()}}),e)})))()},selected:function(t){var e=t.value;this.$emit("select",{mode:"GoogleDrive",value:e})},blur:function(t){this.$emit("select",{mode:"URL",value:t})},click:function(t){this.select=t,this.items=this.items.map((function(e){return Object(n["a"])(Object(n["a"])({},e),{},{active:e.mode===t})}))}}},T=E,R=(s("19a1"),Object(h["a"])(T,S,F,!1,null,"12a2ed74",null)),B=R.exports,P=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",[s("t-button",{attrs:{disabled:!t.selected,loading:t.loading},nativeOn:{click:function(e){return t.click.apply(null,arguments)}}},[t._v("Выбрать датасет")])],1)},I=[],N=(s("b0c0"),{name:"DatasetButton",data:function(){return{loading:!1}},computed:{selected:function(){return this.$store.getters["datasets/getSelected"]}},methods:{createInterval:function(){var t=this;this.interval=setTimeout(Object(j["a"])(regeneratorRuntime.mark((function e(){var s,i,a,n,r;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.$store.dispatch("datasets/choiceProgress",{});case 2:s=e.sent,i=s.finished,a=s.message,n=s.percent,r=s.data,!s||i?(t.$store.dispatch("messages/setProgressMessage",a),t.$store.dispatch("messages/setProgress",n),t.loading=!1,s&&(t.$store.dispatch("messages/setMessage",{message:"Датасет «".concat(r.alias,"» выбран")},{root:!0}),t.$store.dispatch("projects/setProject",{dataset:r},{root:!0}))):(t.$store.dispatch("messages/setProgress",n),t.$store.dispatch("messages/setProgressMessage",a),t.createInterval()),console.log(s);case 6:case"end":return e.stop()}}),e)}))),1e3)},click:function(){var t=this;return Object(j["a"])(regeneratorRuntime.mark((function e(){var s,i,a,n,r;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return console.log("sdsdsddsd"),s=t.selected,i=s.alias,a=s.group,n=s.name,t.$store.dispatch("messages/setMessage",{message:"Выбран датасет «".concat(n,"»")}),e.next=5,t.$store.dispatch("datasets/choice",{alias:i,group:a});case 5:r=e.sent,r&&(t.loading=!0,t.createInterval());case 7:case"end":return e.stop()}}),e)})))()}}}),M=N,H=Object(h["a"])(M,P,I,!1,null,null,null),A=H.exports,U={name:"Settings",components:{DatasetTab:B,DatasetButton:A},data:function(){return{loading:!1,dataset:{},interval:null,inputs:1,outputs:1,rules:{length:function(t){return function(e){return(e||"").length>=t||"Length < ".concat(t)}},required:function(t){return 0!==t.length||"Not be empty"}}}},computed:Object(n["a"])(Object(n["a"])({},Object(r["b"])({settings:"datasets/getSettings"})),{},{inputLayer:function(){var t=+this.inputs,e=this.settings;return t>0&&t<100&&Object.keys(e).length?t:0},outputLayer:function(){var t=+this.outputs,e=this.settings;return t>0&&t<100&&Object.keys(e).length?t:0},full:{set:function(t){this.$store.dispatch("datasets/setFull",t)},get:function(){return this.$store.getters["datasets/getFull"]}}}),methods:{createInterval:function(){var t=this;return Object(j["a"])(regeneratorRuntime.mark((function e(){return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:t.interval=setTimeout(Object(j["a"])(regeneratorRuntime.mark((function e(){var s,i,a,n,r;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.$store.dispatch("datasets/loadProgress",{});case 2:s=e.sent,i=s.finished,a=s.message,n=s.percent,r=s.data.file_manager,!s||i?(t.$store.dispatch("messages/setProgressMessage",a),t.$store.dispatch("messages/setProgress",n),r&&(t.$store.dispatch("datasets/setFilesSource",r),t.$store.dispatch("datasets/setFilesDrop",[])),t.loading=!1,t.full=!0):(t.$store.dispatch("messages/setProgress",n),t.$store.dispatch("messages/setProgressMessage",a),t.createInterval()),console.log(s);case 6:case"end":return e.stop()}}),e)}))),1e3);case 1:case"end":return e.stop()}}),e)})))()},select:function(t){console.log(t),this.dataset=t},openFull:function(){this.$store.state.datasets.filesSource.length?this.full=!0:this.$Modal.alert({width:250,title:"Внимание!",content:"Загрузите датасет"})},download:function(){var t=this;return Object(j["a"])(regeneratorRuntime.mark((function e(){var s,i,a,n;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:if(console.log("dfdfdfdfdfdf"),s=t.dataset,i=s.mode,a=s.value,!i||!a){e.next=11;break}return t.loading=!0,e.next=6,t.$store.dispatch("datasets/sourceLoad",{mode:i,value:a});case 6:n=e.sent,console.log(n),n&&t.createInterval(),e.next=12;break;case 11:t.$store.dispatch("messages/setMessage",{error:"Выберите файл"});case 12:case"end":return e.stop()}}),e)})))()}}},G=U,z=(s("8f0d"),Object(h["a"])(G,D,O,!1,null,"ef3e874c",null)),q=z.exports,X=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"params-full"},[s("div",{staticClass:"params-full__inner"},[s("div",{staticClass:"params-full__btn",on:{click:function(e){t.full=!t.full}}},[s("i",{staticClass:"params-full__btn--icon"})]),s("div",{class:["params-full__files",{toggle:!t.toggle}]},[s("BlockFiles",{on:{toggle:t.change}})],1),s("div",{staticClass:"params-full__main"},[s("div",{staticClass:"main__header"},[s("BlockHeader")],1),s("div",{staticClass:"main__center",style:t.height},[s("div",{staticClass:"main__center--left"},[s("BlockMainLeft")],1),s("div",{staticClass:"main__center--right"},[s("BlockMainRight")],1)]),s("div",{staticClass:"main__footer"},[s("BlockFooter",{on:{create:t.createObject}})],1)])])])},Y=[],J=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"block-file"},[s("div",{class:["block-file__header",{toggle:!t.toggle}],on:{click:function(e){t.toggle=!t.toggle,t.$emit("toggle",t.toggle)}}},[s("i",{staticClass:"block-file__header--icon"}),t._v(" "+t._s(t.text)+" ")]),s("div",{directives:[{name:"show",rawName:"v-show",value:t.toggle,expression:"toggle"}],staticClass:"block-file__body"},[s("scrollbar",[s("files-menu",{model:{value:t.filesSource,callback:function(e){t.filesSource=e},expression:"filesSource"}})],1)],1)])},V=[],W={name:"BlockFiles",data:function(){return{toggle:!0,nodes:[{title:"Cars",type:"folder",isExpanded:!1,children:[{title:"BMW.jpg",type:"image"},{title:"AUDI.jpg",type:"image"}]},{title:"Music",type:"folder",isExpanded:!0,children:[{title:"1.mp3",type:"audio"},{title:"song.wav",type:"audio"}]},{title:"Text",type:"folder",isExpanded:!1,children:[{title:"Table",type:"text"}]}]}},computed:{text:function(){return this.toggle?"Выбор папки/файла":""},filesSource:{set:function(t){this.$store.dispatch("datasets/setFilesSource",t)},get:function(){return this.$store.getters["datasets/getFilesSource"]}}}},K=W,Q=(s("9f2f"),Object(h["a"])(K,J,V,!1,null,"2eeefed6",null)),Z=Q.exports,tt=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("form",{staticClass:"block-footer",on:{submit:function(t){t.preventDefault()}}},[s("div",{staticClass:"block-footer__item"},[s("t-input",{attrs:{parse:"[name]",small:"",value:"Новый"}},[t._v(" Название датасета ")])],1),s("div",{staticClass:"block-footer__item block-tags"},[s("TTags")],1),s("div",{staticClass:"block-footer__item"},[s("DoubleSlider")],1),s("div",{staticClass:"block-footer__item"},[s("t-checkbox",{attrs:{parse:"subsequence"}},[t._v("Сохранить последовательность")])],1),s("div",{staticClass:"block-footer__item"},[s("t-checkbox",{attrs:{parse:"use_generator"}},[t._v("Использовать генератор")])],1),s("div",{staticClass:"action"},[s("t-button",{nativeOn:{click:function(e){return t.getObj.apply(null,arguments)}}},[t._v("Сформировать")])],1)])},et=[],st=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"t-field"},[s("label",{staticClass:"t-field__label"},[t._v("Train / Val / Test")]),s("div",{staticClass:"slider"},[s("div",{staticClass:"range-slider",on:{mousemove:t.slider}},[s("div",{staticClass:"sliders"},[s("div",{staticClass:"first-slider",style:t.firstSlider,on:{mousedown:function(e){return t.startDrag(e,"first")}}}),s("div",{staticClass:"second-slider",style:t.secondSlider,on:{mousedown:function(e){return t.startDrag(e,"second")}}})]),s("div",{staticClass:"scale"},[s("div",{style:t.firstScale,attrs:{id:"first-scale"}},[t._v(t._s(t.sliders.first))]),s("div",{style:t.secondScale,attrs:{id:"second-scale"}},[t._v(t._s(t.sliders.second-t.sliders.first))]),s("div",{style:t.thirdScale,attrs:{id:"third-scale"}},[t._v(t._s(100-t.sliders.second))])]),s("div",{staticClass:"inputs"},[s("input",{attrs:{name:"[info][part][train]",type:"number"},domProps:{value:t.sliders.first}}),s("input",{attrs:{name:"[info][part][validation]",type:"number"},domProps:{value:t.sliders.second-t.sliders.first}}),s("input",{attrs:{name:"[info][part][test]",type:"number"},domProps:{value:100-t.sliders.second}})])])])])},it=[],at={name:"DoubleSlider",data:function(){return{dragging:!1,draggingObj:null,sliders:{first:50,second:77}}},computed:{firstScale:function(){return{width:this.sliders.first+"%"}},secondScale:function(){return{width:this.sliders.second-this.sliders.first+"%"}},thirdScale:function(){return{width:100-this.sliders.second+"%"}},firstSlider:function(){return{"margin-left":this.sliders.first+"%"}},secondSlider:function(){return{"margin-left":this.sliders.second+"%"}}},methods:{startDrag:function(t,e){this.dragging=!0,this.draggingObj=e,this.CurrentX=t.x},stopDrag:function(){this.dragging=!1,this.draggingObj=null},slider:function(t){if(t.preventDefault(),this.dragging){var e=document.querySelector(".".concat(this.draggingObj,"-slider")),s=t.x-e.parentNode.getBoundingClientRect().x;this.sliders[this.draggingObj]=Math.round(s/231*100),this.sliders.first<5&&(this.sliders.first=5),this.sliders.second>95&&(this.sliders.second=95),this.sliders.first>this.sliders.second-5&&(this.sliders.first=this.sliders.second-5)}}},mounted:function(){window.addEventListener("mouseup",this.stopDrag)},destroyed:function(){window.removeEventListener("mouseup",this.stopDrag)}},nt=at,rt=(s("bc66"),Object(h["a"])(nt,st,it,!1,null,"99ec9e2e",null)),lt=rt.exports,ct=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{class:["t-field",{"t-inline":t.inline}]},[s("label",{staticClass:"t-field__label",attrs:{for:t.parse}},[t._v(t._s(t.label))]),s("div",{staticClass:"tags"},[s("button",{staticClass:"tags__add",attrs:{type:"button"}},[s("i",{staticClass:"tags__add--icon t-icon icon-tag-plus"}),s("input",{staticClass:"tags__add--input",attrs:{type:"text",placeholder:"Добавить"},on:{keypress:function(e){return!e.type.indexOf("key")&&t._k(e.keyCode,"enter",13,e.key,"Enter")?null:(e.preventDefault(),t.create.apply(null,arguments))}}})]),t._l(t.tags,(function(t,e){var i=t.value;return s("input",{key:"tag_"+e,staticClass:"tags__item",attrs:{name:"[tags][][name]",type:"text"},domProps:{value:i}})}))],2)])},ot=[],ut=s("2909"),dt=(s("a9e3"),{name:"t-input",props:{label:{type:String,default:"Теги"},type:{type:String,default:"text"},value:{type:[String,Number]},parse:String,name:String,inline:Boolean,disabled:Boolean},data:function(){return{tags:[]}},methods:{create:function(t){var e=t.target.value;t.target.value="",this.tags.length<3&&(this.tags.push({value:e}),this.tags=Object(ut["a"])(this.tags))},inputLength:function(t){t.target.style.width=8*(t.target.value.length+1)+"px"}}}),ft=dt,ht=(s("6f9a"),Object(h["a"])(ft,ct,ot,!1,null,"615eaf6b",null)),pt=ht.exports,gt=s("da6d"),vt=s.n(gt),mt={name:"BlockFooter",components:{DoubleSlider:lt,TTags:pt},methods:{getObj:function(){this.$emit("create",vt()(this.$el))}}},_t=mt,bt=(s("1aff"),Object(h["a"])(_t,tt,et,!1,null,"0efb58c4",null)),kt=bt.exports,yt=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"block-header",on:{drop:function(e){return t.onDrop(e)},dragover:function(t){t.preventDefault()}}},[t.filesDrop.length?s("div",{staticClass:"block-header__main"},[s("Cards",[t._l(t.filesDrop,(function(e,i){return["folder"===e.type?s("CardFile",t._b({key:"files_"+i},"CardFile",e,!1)):t._e()]}))],2),s("div",{staticClass:"empty"})],1):s("div",{staticClass:"inner"},[t._m(0)])])},Ct=[function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"block-header__overlay"},[s("div",{staticClass:"block-header__overlay--icon"}),s("div",{staticClass:"block-header__overlay--title"},[t._v("Перетащите папку или файл для начала работы с содержимым архива")])])}],wt=(s("c740"),function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"card-file",style:t.bc},[t.id?s("div",{staticClass:"card-file__header",style:t.bg},[t._v(t._s("Входные данные "+t.id))]):t._e(),s("div",{class:["card-file__body",t.type]}),s("div",{staticClass:"card-file__footer"},[t._v(t._s(t.label))])])}),$t=[],xt={name:"card-file",props:{color:{type:String,default:""},label:String,name:String,type:String,id:Number},computed:{bg:function(){return{backgroundColor:this.id?this.color:""}},bc:function(){return{borderColor:this.id?this.color:""}}}},Dt=xt,Ot=(s("abc2"),Object(h["a"])(Dt,wt,$t,!1,null,"6c15e59e",null)),jt=Ot.exports,St=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"t-cards",style:t.style,on:{wheel:t.wheel}},[s("scrollbar",{ref:"scrollCards",attrs:{ops:t.ops}},[s("div",{staticClass:"t-cards__items"},[s("div",{staticClass:"t-cards__items--item"},[t._t("default")],2)])])],1)},Ft=[],Lt={data:function(){return{wight:0,ops:{scrollPanel:{scrollingX:!0,scrollingY:!1}}}},computed:{style:function(){return{wight:this.wight+"px"}}},mounted:function(){this.wight=this.$el.clientWidth,console.log(this.$el.clientWidth)},methods:{wheel:function(t){this.$refs.scrollCards.scrollBy({dx:t.wheelDelta},200)}}},Et=Lt,Tt=(s("e65d"),Object(h["a"])(Et,St,Ft,!1,null,"5a8a3de1",null)),Rt=Tt.exports,Bt={name:"BlockHeader",components:{CardFile:jt,Cards:Rt},data:function(){return{}},computed:{filesDrop:{set:function(t){this.$store.dispatch("datasets/setFilesDrop",t)},get:function(){return this.$store.getters["datasets/getFilesDrop"]}}},methods:{onDrop:function(t){var e=t.dataTransfer,s=JSON.parse(e.getData("CardDataType")),i=this.filesDrop.findIndex((function(t){var e=t.label;return s.label===e}));console.log(i),-1===i?(this.filesDrop.push(s),this.filesDrop=Object(ut["a"])(this.filesDrop)):this.$Notify.warning({title:"Внимание!",message:"Каталог уже выбран"})}},mounted:function(){}},Pt=Bt,It=(s("a6ab"),Object(h["a"])(Pt,yt,Ct,!1,null,"6b6c1607",null)),Nt=It.exports,Mt=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"block-left"},[s("div",{staticClass:"block-left__fab"},[s("Fab",{on:{click:t.add}})],1),s("div",{staticClass:"block-left__header"},[t._v("Входные параметры")]),s("div",{staticClass:"block-left__body"},[s("scrollbar",{ref:"scrollLeft",attrs:{ops:t.ops}},[s("div",{staticClass:"block-left__body--inner",style:t.height},[t._l(t.inputDataInput,(function(e){var i=e.id,a=e.color;return[s("CardLayer",{key:"cardLayersLeft"+i,attrs:{id:i,color:a},on:{"click-btn":function(e){return t.click(e,i)},"click-header":t.clickScroll}},[s("TMultiSelect",{attrs:{lists:t.filesDrop,id:i,label:"Выберите путь",inline:""},on:{change:function(e){return t.check(e,a,i)}}}),t._l(t.input,(function(e,i){return[s("t-auto-field",t._b({key:a+i,attrs:{idKey:a+i},on:{change:t.change}},"t-auto-field",e,!1))]}))],2)]})),s("div",{staticClass:"block-left__body--empty"})],2)])],1)])},Ht=[],At=(s("4de4"),s("7db0"),function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"fab",on:{click:function(e){return t.$emit("click",e)}}},[s("i",{staticClass:"fab__icon"})])}),Ut=[],Gt={name:"fab"},zt=Gt,qt=(s("2a9c"),Object(h["a"])(zt,At,Ut,!1,null,"7e89689d",null)),Xt=qt.exports,Yt=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{directives:[{name:"click-outside",rawName:"v-click-outside",value:t.outside,expression:"outside"}],staticClass:"card-layer",style:t.height},[s("div",{staticClass:"card-layer__header",style:t.bg,on:{click:function(e){return t.$emit("click-header",e)}}},[s("div",{staticClass:"card-layer__header--icon",on:{click:function(e){t.toggle=!t.toggle}}},[s("i",{staticClass:"dot"})]),s("div",{staticClass:"card-layer__header--title"},[t._v("Входные данные "+t._s(t.id))])]),s("div",{directives:[{name:"show",rawName:"v-show",value:t.toggle,expression:"toggle"}],staticClass:"card-layer__dropdown"},t._l(t.items,(function(e,i){var a=e.icon;return s("div",{key:"icon"+i,staticClass:"card-layer__dropdown--item",on:{click:function(e){return t.click(a)}}},[s("i",{class:[a]})])})),0),s("div",{staticClass:"card-layer__body"},[s("scrollbar",{attrs:{ops:t.ops}},[s("div",{ref:"cardBody",staticClass:"card-layer__body--inner"},[t._t("default")],2)])],1)])},Jt=[],Vt={name:"card-layer",props:{id:Number,color:String,name:String},data:function(){return{height:{height:"100%"},toggle:!1,items:[{icon:"remove"},{icon:"copy"}],ops:{bar:{background:"#17212b"},scrollPanel:{scrollingX:!1,scrollingY:!0}}}},computed:{bg:function(){return{backgroundColor:this.color}}},methods:{outside:function(){this.toggle&&(this.toggle=!1)},click:function(t){this.toggle=!1,this.$emit("click-btn",t)}},mounted:function(){var t=this.$el.clientHeight,e=this.$refs.cardBody.clientHeight+36;t>e&&(this.height={height:e+"px"})}},Wt=Vt,Kt=(s("b2f9"),Object(h["a"])(Wt,Yt,Jt,!1,null,"714b1b4f",null)),Qt=Kt.exports,Zt=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{directives:[{name:"click-outside",rawName:"v-click-outside",value:t.outside,expression:"outside"}],class:["t-multi-select",{"t-inline":t.inline}]},[s("label",{staticClass:"t-multi-select__label"},[t._v(t._s(t.label))]),s("div",{staticClass:"t-multi-select__input"},[s("span",{class:["t-multi-select__input--text",{"t-multi-select__input--active":t.input}],attrs:{title:t.input},on:{click:function(e){t.show=!0}}},[t._v(" "+t._s(t.input||"Не выбрано")+" ")])]),s("div",{directives:[{name:"show",rawName:"v-show",value:t.show,expression:"show"}],staticClass:"t-multi-select__content"},[t.filterList.length?s("div",{staticClass:"t-multi__item"},[s("span",{class:["t-multi__item--check",{active:t.checkAll}],on:{click:function(e){return t.select(t.checkAll)}}}),s("span",{staticClass:"t-multi__item--title"},[t._v("Выбрать все")])]):t._e(),t._l(t.filterList,(function(e,i){return[s("div",{key:i,staticClass:"t-multi__item",attrs:{title:e.label}},[s("span",{class:["t-multi__item--check",{active:t.active(e)}],on:{click:function(s){return t.select(e)}}}),s("span",{staticClass:"t-multi__item--title"},[t._v(t._s(e.label))])])]})),t.filterList.length?t._e():s("div",{staticClass:"t-multi__item"},[s("span",{staticClass:"t-multi__item--title"},[t._v("Нет данных")])])],2)])},te=[],ee=(s("a15b"),s("99af"),{name:"TMultiSelect",props:{name:String,lists:{type:Array,required:!0,default:function(){return[]}},placeholder:String,disabled:Boolean,label:{type:String,default:"Label"},inline:Boolean,value:String,id:Number},data:function(){return{selected:[],show:!1,pagination:0}},created:function(){},computed:{input:function(){return this.selected.map((function(t){return t.label})).join()},checkAll:function(){return this.lists.length===this.selected.length},filterList:function(){var t=this;return this.lists.filter((function(e){return!e.id||e.id===t.id}))}},methods:{active:function(t){var e=t.value;return!!this.selected.find((function(t){return t.value===e}))},outside:function(){this.show&&(this.show=!1)},select:function(t){"boolean"===typeof t?this.selected=this.lists.map((function(e){return t?null:e})).filter((function(t){return t})):this.selected.find((function(e){return e.value===t.value}))?this.selected=this.selected.filter((function(e){return e.value!==t.value})):this.selected=[].concat(Object(ut["a"])(this.selected),[t]),this.$emit("change",this.selected)}}}),se=ee,ie=(s("741f"),Object(h["a"])(se,Zt,te,!1,null,"0bf9fe5b",null)),ae=ie.exports,ne={name:"BlockMainLeft",components:{Fab:Xt,CardLayer:Qt,TMultiSelect:ae},data:function(){return{ops:{scrollPanel:{scrollingX:!0,scrollingY:!1},rail:{gutterOfEnds:"6px"}}}},computed:Object(n["a"])(Object(n["a"])({},Object(r["b"])({input:"datasets/getTypeInput",inputData:"datasets/getInputData"})),{},{inputDataInput:function(){return this.inputData.filter((function(t){return"input"===t.layer}))},filesDrop:{set:function(t){this.$store.dispatch("datasets/setFilesDrop",t)},get:function(){return this.$store.getters["datasets/getFilesDrop"]}},height:function(){var t=this.$store.getters["settings/height"]({clean:!0,padding:324});return console.log(t),t}}),methods:{check:function(t,e,s){this.filesDrop=this.filesDrop.map((function(i){return t.find((function(t){return t.value===i.value}))?(i.color=e,i.id=s):i.id===s&&(i.id=0,i.color=""),i})),console.log(this.filesDrop)},add:function(){var t=this;this.$store.dispatch("datasets/createInputData",{layer:"input"}),this.$nextTick((function(){t.$refs.scrollLeft.scrollTo({x:"100%"},100)}))},clickScroll:function(t){this.$refs.scrollLeft.scrollIntoView(t.target,100),console.log(t)},click:function(t,e){console.log(t,e),"remove"===t&&(this.$store.dispatch("datasets/removeInputData",e),this.filesDrop=this.filesDrop.map((function(t){return t.id===e&&(t.color="",t.id=0),t})))},heightForm:function(t){this.$store.dispatch("settings/setHeight",{center:this.$el.clientHeight}),console.log(t,this.$el.clientHeight)},change:function(t){console.log(t)}}},re=ne,le=(s("93b8"),Object(h["a"])(re,Mt,Ht,!1,null,"be650046",null)),ce=le.exports,oe=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"block-right"},[s("div",{staticClass:"block-right__fab"},[s("Fab",{on:{click:t.add}})],1),s("div",{staticClass:"block-right__header"},[t._v("Выходные параметры")]),s("div",{staticClass:"block-right__body"},[s("scrollbar",{ref:"scrollRight",attrs:{ops:t.ops}},[s("div",{staticClass:"block-right__body--inner",style:t.height},[s("div",{staticClass:"block-right__body--empty"}),t._l(t.inputDataOutput,(function(e){var i=e.id,a=e.color;return[s("CardLayer",{key:"cardLayersRight"+i,attrs:{id:i,color:a},on:{"click-btn":function(e){return t.click(e,i)},"click-header":t.clickScroll}},[s("TMultiSelect",{attrs:{inline:"",label:"Выберите путь",lists:t.filesDrop,sloy:i},on:{check:function(e){return t.check(e,a,i)},checkAll:function(e){return t.checkAll(e,a,i)}}}),t._l(t.output,(function(e,i){return[s("t-auto-field",t._b({key:a+i,attrs:{idKey:a+i},on:{change:t.change}},"t-auto-field",e,!1))]}))],2)]}))],2)])],1)])},ue=[],de={name:"BlockMainRight",components:{Fab:Xt,CardLayer:Qt,TMultiSelect:ae},data:function(){return{ops:{scrollPanel:{scrollingX:!0,scrollingY:!1},rail:{gutterOfEnds:"6px"}}}},computed:Object(n["a"])(Object(n["a"])({},Object(r["b"])({output:"datasets/getTypeOutput",inputData:"datasets/getInputData"})),{},{inputDataOutput:function(){return this.inputData.filter((function(t){return"output"===t.layer}))},filesDrop:{set:function(t){this.$store.dispatch("datasets/setFilesDrop",t)},get:function(){return this.$store.getters["datasets/getFilesDrop"]}},height:function(){var t=this.$store.getters["settings/height"]({clean:!0,padding:324});return console.log(t),t}}),methods:{add:function(){var t=this;this.$store.dispatch("datasets/createInputData",{layer:"output"}),this.$nextTick((function(){t.$refs.scrollRight.scrollTo({x:"100%"},100)}))},clickScroll:function(t){this.$refs.scrollRight.scrollIntoView(t.target,100),console.log(t)},click:function(t,e){console.log(t,e),"remove"===t&&(this.cardLayers=this.cardLayers.filter((function(t,s){return s!==e})))},check:function(t,e,s){var i=t.value;console.log(i),this.filesDrop=this.filesDrop.map((function(t){return t.value===i&&(t.active=!t.active,t.color=e,t.sloy=s),t}))},checkAll:function(t,e,s){this.filesDrop=this.filesDrop.map((function(i){return t?i.active||(i.active=!i.active,i.color=e,i.sloy=s):i.sloy===s&&(i.active=!i.active,i.sloy=0),i}))},change:function(t){console.log(t)}}},fe=de,he=(s("fca9"),Object(h["a"])(fe,oe,ue,!1,null,"51f7e895",null)),pe=he.exports,ge={name:"ParamsFull",components:{BlockFiles:Z,BlockFooter:kt,BlockHeader:Nt,BlockMainLeft:ce,BlockMainRight:pe},data:function(){return{toggle:!0}},computed:{full:{set:function(t){this.$store.dispatch("datasets/setFull",t)},get:function(){return this.$store.getters["datasets/getFull"]}},height:function(){var t=this.$store.getters["settings/height"]({style:!1,clean:!0});return t=t-172-96,console.log(t),{flex:"0 0 "+t+"px",height:t+"px"}}},methods:{createObject:function(t){console.log(t)},change:function(t){this.toggle=t}},mounted:function(){}},ve=ge,me=(s("9b1c"),Object(h["a"])(ve,X,Y,!1,null,null,null)),_e=me.exports,be={name:"Datasets",components:{Dataset:x,Params:q,ParamsFull:_e},data:function(){return{}},computed:Object(n["a"])({},Object(r["b"])({full:"datasets/getFull"}))},ke=be,ye=(s("84b3"),Object(h["a"])(ke,i,a,!1,null,"990f09ea",null));e["default"]=ye.exports},"6f9a":function(t,e,s){"use strict";s("5551")},"741f":function(t,e,s){"use strict";s("040d")},7814:function(t,e,s){},"841c":function(t,e,s){"use strict";var i=s("d784"),a=s("825a"),n=s("1d80"),r=s("129f"),l=s("14c3");i("search",(function(t,e,s){return[function(e){var s=n(this),i=void 0==e?void 0:e[t];return void 0!==i?i.call(e,s):new RegExp(e)[t](String(s))},function(t){var i=s(e,this,t);if(i.done)return i.value;var n=a(this),c=String(t),o=n.lastIndex;r(o,0)||(n.lastIndex=0);var u=l(n,c);return r(n.lastIndex,o)||(n.lastIndex=o),null===u?-1:u.index}]}))},"84b3":function(t,e,s){"use strict";s("0bf9")},8523:function(t,e,s){},"8f0d":function(t,e,s){"use strict";s("0587")},"93b8":function(t,e,s){"use strict";s("5a7c")},9825:function(t,e,s){},"9abe":function(t,e,s){"use strict";s("4204")},"9ae5":function(t,e,s){},"9b1c":function(t,e,s){"use strict";s("de39")},"9f2f":function(t,e,s){"use strict";s("b20a")},a6ab:function(t,e,s){"use strict";s("fd42")},abc2:function(t,e,s){"use strict";s("3f37")},b20a:function(t,e,s){},b2f9:function(t,e,s){"use strict";s("9ae5")},b7d5:function(t,e,s){},bc66:function(t,e,s){"use strict";s("9825")},da6d:function(t,e,s){var i=s("7037").default;s("b0c0"),s("fb6a"),s("4d63"),s("ac1f"),s("25f0"),s("466d"),s("5319");var a=/^(?:submit|button|image|reset|file)$/i,n=/^(?:input|select|textarea|keygen)/i,r=/(\[[^\[\]]*\])/g;function l(t,e){e={hash:!0,disabled:!0,empty:!0},"object"!=i(e)?e={hash:!!e}:void 0===e.hash&&(e.hash=!0);for(var s=e.hash?{}:"",r=e.serializer||(e.hash?u:d),l=t&&t.elements?t.elements:[],c=Object.create(null),o=0;o<l.length;++o){var f=l[o];if((e.disabled||!f.disabled)&&f.name&&(n.test(f.nodeName)&&!a.test(f.type))){"checkbox"!==f.type&&"radio"!==f.type||f.checked||(p=void 0);var h=f.name,p=f.value;if("number"===f.type&&(p=+p),"checkbox"===f.type&&(p="true"===p),e.empty){if("checkbox"!==f.type||f.checked||(p=!1),"radio"===f.type&&(c[f.name]||f.checked?f.checked&&(c[f.name]=!0):c[f.name]=!1),void 0==p&&"radio"==f.type)continue}else if(!p)continue;if("select-multiple"!==f.type)s=r(s,h,p);else{p=[];for(var g=f.options,v=!1,m=0;m<g.length;++m){var _=g[m],b=e.empty&&!_.value,k=_.value||b;_.selected&&k&&(v=!0,s=e.hash&&"[]"!==h.slice(h.length-2)?r(s,h+"[]",_.value):r(s,h,_.value))}!v&&e.empty&&(s=r(s,h,""))}}}if(e.empty)for(var h in c)c[h]||(s=r(s,h,""));return s}function c(t){var e=[],s=/^([^\[\]]*)/,i=new RegExp(r),a=s.exec(t);a[1]&&e.push(a[1]);while(null!==(a=i.exec(t)))e.push(a[1]);return e}function o(t,e,s){if(0===e.length)return t=s,t;var i=e.shift(),a=i.match(/^\[(.+?)\]$/);if("[]"===i)return t=t||[],Array.isArray(t)?t.push(o(null,e,s)):(t._values=t._values||[],t._values.push(o(null,e,s))),t;if(a){var n=a[1],r=+n;isNaN(r)?(t=t||{},t[n]=o(t[n],e,s)):(t=t||[],t[r]=o(t[r],e,s))}else t[i]=o(t[i],e,s);return t}function u(t,e,s){var i=e.match(r);if(i){var a=c(e);o(t,a,s)}else{var n=t[e];n?(Array.isArray(n)||(t[e]=[n]),t[e].push(s)):t[e]=s}return t}function d(t,e,s){return s=s.replace(/(\r)?\n/g,"\r\n"),s=encodeURIComponent(s),s=s.replace(/%20/g,"+"),t+(t?"&":"")+encodeURIComponent(e)+"="+s}t.exports=l},de39:function(t,e,s){},e65d:function(t,e,s){"use strict";s("52fb")},fca9:function(t,e,s){"use strict";s("4a01")},fd42:function(t,e,s){}}]);