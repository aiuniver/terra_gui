(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-683d24f8"],{"04e3":function(t,e,i){"use strict";i("df33")},"09b9":function(t,e,i){"use strict";i("938a")},"0fd9":function(t,e,i){"use strict";i("1a25")},"1a25":function(t,e,i){},"1d15":function(t,e,i){},"1ddc":function(t,e,i){"use strict";i("5630")},"2a9c":function(t,e,i){"use strict";i("b7d5")},"305f":function(t,e,i){},"3c1e":function(t,e,i){},"3d4c":function(t,e,i){"use strict";i("a645")},4056:function(t,e,i){"use strict";i("3c1e")},"41ba":function(t,e,i){},5630:function(t,e,i){},"5db7":function(t,e,i){"use strict";var s=i("23e7"),n=i("a2bf"),a=i("7b0b"),r=i("50c4"),l=i("1c0b"),o=i("65f0");s({target:"Array",proto:!0},{flatMap:function(t){var e,i=a(this),s=r(i.length);return l(t),e=o(i,0),e.length=n(e,i,i,s,0,1,t,arguments.length>1?arguments[1]:void 0),e}})},"5dd0":function(t,e,i){"use strict";i.r(e);var s=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"params-full"},[i("div",{staticClass:"params-full__inner"},[i("div",{staticClass:"params-full__btn",on:{click:function(e){t.full=!t.full}}},[i("i",{staticClass:"params-full__btn--icon"})]),i("div",{class:["params-full__files",{toggle:!t.toggle}]},[i("BlockFiles",{on:{toggle:t.change}})],1),i("scrollbar",{staticClass:"params-full__scroll",attrs:{ops:t.ops}},[i("div",{staticClass:"params-full__main"},[i("div",{staticClass:"main__header"},[i("BlockHeader")],1),t.isTable?i("div",{staticClass:"main__handlers"},[i("BlockHandlers")],1):t._e(),i("div",{staticClass:"main__center",style:t.height},[i("div",{staticClass:"main__center--left"},[i("BlockMainLeft")],1),i("div",{staticClass:"main__center--right"},[i("BlockMainRight")],1)]),i("div",{staticClass:"main__footer"},[i("BlockFooter",{on:{create:t.createObject}})],1)])])],1)])},n=[],a=i("1da1"),r=(i("96cf"),i("b0c0"),function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"block-file"},[i("div",{class:["block-file__header",{toggle:!t.toggle}],on:{click:function(e){t.toggle=!t.toggle,t.$emit("toggle",t.toggle)}}},[i("i",{staticClass:"block-file__header--icon"}),t._v(" "+t._s(t.text)+" ")]),i("div",{directives:[{name:"show",rawName:"v-show",value:t.toggle,expression:"toggle"}],staticClass:"block-file__body"},[t.isDir?i("button",{staticClass:"block-file__body--btn",on:{click:t.moveAll}},[t._v("Перенести всё")]):t._e(),i("scrollbar",[i("files-menu",{model:{value:t.filesSource,callback:function(e){t.filesSource=e},expression:"filesSource"}})],1)],1)])}),l=[],o=i("2909"),c=(i("4de4"),i("5db7"),i("73d9"),i("d81d"),i("99af"),{name:"BlockFiles",data:function(){return{toggle:!0,nodes:[{title:"Cars",type:"folder",isExpanded:!1,children:[{title:"BMW.jpg",type:"image"},{title:"AUDI.jpg",type:"image"}]},{title:"Music",type:"folder",isExpanded:!0,children:[{title:"1.mp3",type:"audio"},{title:"song.wav",type:"audio"}]},{title:"Text",type:"folder",isExpanded:!1,children:[{title:"Table",type:"text"}]}]}},computed:{text:function(){return this.toggle?"Выбор папки/файла":""},isDir:function(){return this.filesSource.filter((function(t){return"table"!==t.type})).length},filesSource:{set:function(t){this.$store.dispatch("datasets/setFilesSource",t)},get:function(){return this.$store.getters["datasets/getFilesSource"]}}},methods:{moveAll:function(){var t=this.$store.getters["datasets/getFilesSource"].flatMap(this.getFiles),e=t.filter((function(t){return t.dragndrop&&"folder"===t.type})).map((function(t){return{value:t.path,label:t.title,type:t.type,id:0,cover:t.cover,table:"table"===t.type?t.data:null}}));this.$store.dispatch("datasets/setFilesDrop",e)},getFiles:function(t){return t.children?[].concat(Object(o["a"])(t.children.flatMap(this.getFiles)),[t]):t}}}),u=c,d=(i("4056"),i("2877")),f=Object(d["a"])(u,r,l,!1,null,"6a3e569a",null),h=f.exports,p=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("form",{staticClass:"block-footer",on:{submit:function(t){t.preventDefault()}}},[i("div",{staticClass:"block-footer__item"},[i("t-field",{attrs:{label:"Название датасета"}},[i("t-input-new",{staticClass:"block-footer__input-custom",style:{width:"150px"},attrs:{parse:"[name]",small:"",error:t.nameError},on:{focus:function(e){t.nameError=""}},model:{value:t.nameProject,callback:function(e){t.nameProject=e},expression:"nameProject"}})],1)],1),i("div",{staticClass:"block-footer__item block-tags"},[i("TTags")],1),i("div",{staticClass:"block-footer__item"},[i("Slider",{attrs:{degree:t.degree}})],1),i("div",{staticClass:"block-footer__item block-footer__item--checkbox"},[i("t-checkbox",{attrs:{parse:"[info][shuffle]",reverse:"",inline:""}},[t._v("Сохранить последовательность")]),i("t-checkbox",{attrs:{parse:"use_generator",inline:""}},[t._v("Использовать генератор")])],1),i("div",{staticClass:"action"},[i("t-button",{attrs:{disabled:!!t.disabled},nativeOn:{click:function(e){return t.getObj.apply(null,arguments)}}},[t._v("Сформировать")])],1)])},g=[],v=(i("caad"),i("2532"),function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"t-field"},[i("div",{staticClass:"t-field__label"},[t._v("Train / Val")]),i("div",{ref:"slider",staticClass:"slider",on:{mouseleave:t.stopDrag,mouseup:t.stopDrag}},[i("div",{staticClass:"slider__inputs"},[i("input",{attrs:{name:"[info][part][train]",type:"number","data-degree":t.degree},domProps:{value:t.btnFirstVal}}),i("input",{attrs:{name:"[info][part][validation]",type:"number","data-degree":t.degree},domProps:{value:100-t.btnFirstVal}})]),i("div",{staticClass:"slider__scales"},[i("div",{staticClass:"scales__first",style:t.firstScale},[i("input",{directives:[{name:"autowidth",rawName:"v-autowidth"}],key:t.key1,ref:"key1",attrs:{type:"number",autocomplete:"off"},domProps:{value:t.btnFirstVal},on:{keypress:function(e){return!e.type.indexOf("key")&&t._k(e.keyCode,"enter",13,e.key,"Enter")?null:t.inter(1,e)},blur:function(e){return t.clickInput(1,e)},focus:t.focus}})]),i("div",{staticClass:"scales__second",style:t.secondScale},[i("input",{directives:[{name:"autowidth",rawName:"v-autowidth"}],key:t.key2,ref:"key2",attrs:{type:"number",autocomplete:"off"},domProps:{value:100-t.btnFirstVal},on:{keypress:function(e){return!e.type.indexOf("key")&&t._k(e.keyCode,"enter",13,e.key,"Enter")?null:t.inter(2,e)},blur:function(e){return t.clickInput(2,e)},focus:t.focus}})])]),i("div",{ref:"between",staticClass:"slider__between"},[i("button",{staticClass:"slider__btn-1",style:t.sliderFirstStyle,on:{mousedown:t.startDragFirst,mouseup:t.stopDragFirst}})])])])}),b=[],m=(i("a9e3"),{name:"Slider",data:function(){return{input:0,select:0,btnFirstVal:70,firstBtnDrag:!1,key1:1,key2:1}},props:{degree:Number},methods:{focus:function(t){var e=t.target;e.select()},inter:function(t,e){var i=e.target;i.blur();var s=this.$refs["key".concat(t+1)];s&&(s.focus(),this.$nextTick((function(){s.select()})))},clickInput:function(t,e){var i=e.target,s=+i.value;1===t&&s>=0&&s<=95&&(this.btnFirstVal=s>5?s:5),2===t&&s>=0&&s<=95&&(this.btnFirstVal=s>5?100-s:5),this["key".concat(t)]+=1},stopDrag:function(){this.$refs.slider.removeEventListener("mousemove",this.firstBtn)},startDragFirst:function(){this.firstBtnDrag=!0,this.$refs.slider.addEventListener("mousemove",this.firstBtn)},stopDragFirst:function(){this.$refs.slider.removeEventListener("mousemove",this.firstBtn),this.firstBtnDrag=!1},firstBtn:function(t){if(this.firstBtnDrag){var e=document.querySelector(".slider__btn-1"),i=t.pageX-e.parentNode.getBoundingClientRect().x;this.btnFirstVal=Math.round(i/231*100),this.btnFirstVal<5&&(this.btnFirstVal=5),this.btnFirstVal>95&&(this.btnFirstVal=95)}},diff:function(t){var e=arguments.length>1&&void 0!==arguments[1]?arguments[1]:95,i=arguments.length>2&&void 0!==arguments[2]?arguments[2]:5;return t<i&&(t=i),t>e&&(t=e),t}},computed:{sliderFirstStyle:function(){return{left:this.diff(this.btnFirstVal,95)+"%"}},sliderSecondStyle:function(){return{left:this.diff(this.btnFirstVal,95)+"%"}},firstScale:function(){return{width:this.diff(this.btnFirstVal,95)+"%"}},secondScale:function(){return{width:this.diff(100-this.btnFirstVal,95)+"%"}}}}),_=m,y=(i("d4d8"),Object(d["a"])(_,v,b,!1,null,"179b9903",null)),k=y.exports,C=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{class:["t-field",{"t-inline":t.inline}]},[i("label",{staticClass:"t-field__label",attrs:{for:t.parse}},[t._v(t._s(t.label))]),i("div",{staticClass:"tags"},[i("button",{staticClass:"tags__add",attrs:{type:"button"}},[i("i",{staticClass:"tags__add--icon t-icon icon-tag-plus",on:{click:t.create}}),i("input",{staticClass:"tags__add--input",attrs:{type:"text",disabled:t.tags.length>=3,placeholder:"Добавить"},on:{keypress:function(e){return!e.type.indexOf("key")&&t._k(e.keyCode,"enter",13,e.key,"Enter")?null:(e.preventDefault(),t.create.apply(null,arguments))}}})]),t._l(t.tags,(function(e,s){var n=e.value;return[i("div",{key:"tag_"+s,staticStyle:{display:"flex","border-radius":"4px","align-items":"center",border:"1px solid #6c7883","margin-left":"10px","padding-right":"5px"}},[i("input",{class:["tags__item"],style:{width:8*(n.length+1)<=90?8*(n.length+1)+"px":"90px"},attrs:{"data-index":s,name:"[tags][][name]",type:"text",autocomplete:"off"},domProps:{value:n},on:{input:t.change,blur:t.blur}}),i("i",{staticClass:"tags__remove--icon t-icon icon-tag-plus",on:{click:function(e){return t.removeTag(s)}}})])]}))],2)])},x=[],w={name:"t-input",props:{label:{type:String,default:"Теги"},type:{type:String,default:"text"},value:{type:[String,Number]},parse:String,name:String,inline:Boolean,disabled:Boolean},data:function(){return{tags:[]}},methods:{removeTag:function(t){this.tags=this.tags.filter((function(e,i){return i!==+t}))},create:function(){var t,e=null===(t=this.$el.getElementsByClassName("tags__add--input"))||void 0===t?void 0:t[0];console.log(e.value),e.value&&e.value.length>=3&&this.tags.length<=3&&(this.tags.push({value:e.value}),this.tags=Object(o["a"])(this.tags),e.value="")},change:function(t){var e=t.target.dataset.index;console.log(e),t.target.value.length>=3&&(this.tags[+e].value=t.target.value)},blur:function(t){var e=t.target.dataset.index;t.target.value.length<=2&&(this.tags=this.tags.filter((function(t,i){return i!==+e})))}}},$=w,F=(i("8e25"),Object(d["a"])($,C,x,!1,null,"a489cc22",null)),O=F.exports,S=i("da6d"),j=i.n(S),D={name:"BlockFooter",components:{Slider:k,TTags:O},data:function(){return{degree:100,nameProject:"",nameError:""}},computed:{disabled:function(){var t=this.$store.state.datasets.inputData.map((function(t){return t.layer}));return!(t.includes("input")&&t.includes("output"))}},methods:{getObj:function(){this.nameProject?this.$emit("create",j()(this.$el)):this.nameError="Введите имя"}}},B=D,E=(i("bd41"),Object(d["a"])(B,p,g,!1,null,"9bd26b62",null)),N=E.exports,T=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"block-header",on:{drop:function(e){return t.onDrop(e)},dragover:function(t){t.preventDefault()}}},[t.mixinFiles.length?i("div",{staticClass:"block-header__main"},[i("Cards",[t._l(t.mixinFiles,(function(e,s){return["folder"===e.type?i("CardFile",t._b({key:"files_"+s,on:{event:t.event}},"CardFile",e,!1)):t._e(),"table"===e.type?i("CardTable",t._b({key:"files_"+s,on:{event:t.event}},"CardTable",e,!1)):t._e()]}))],2),i("div",{staticClass:"empty"})],1):i("div",{staticClass:"inner"},[t._m(0)])])},I=[function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"block-header__overlay"},[i("div",{staticClass:"block-header__overlay--icon"}),i("div",{staticClass:"block-header__overlay--title"},[t._v("Перетащите папку или файл для начала работы с содержимым архива")])])}],H=(i("7db0"),function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{directives:[{name:"click-outside",rawName:"v-click-outside",value:t.outside,expression:"outside"}],staticClass:"t-card-file",style:t.bc},[t.id?i("div",{staticClass:"t-card-file__header",style:t.bg},[t._v(t._s(t.title))]):t._e(),i("div",{class:["t-card-file__body","icon-file-"+t.type],style:t.img}),i("div",{staticClass:"t-card-file__footer"},[i("div",{staticClass:"t-card-file__footer--label"},[t._v(t._s(t.label))]),i("div",{staticClass:"t-card-file__footer--btn",on:{click:function(e){t.show=!0}}},[i("i",{staticClass:"t-icon icon-file-dot"})])]),i("div",{directives:[{name:"show",rawName:"v-show",value:t.show,expression:"show"}],staticClass:"t-card-file__dropdown"},t._l(t.items,(function(e,s){var n=e.icon,a=e.event;return i("div",{key:"icon"+s,staticClass:"t-card-file__dropdown--item",on:{click:function(e){t.$emit("event",{label:t.label,event:a}),t.show=!1}}},[i("i",{class:["t-icon",n]})])})),0)])}),P=[],M={name:"t-card-file",props:{label:String,type:String,id:Number,cover:String},data:function(){return{show:!1,items:[{icon:"icon-deploy-remove",event:"remove"}]}},computed:{img:function(){return this.cover?{backgroundImage:"url('".concat(this.cover,"')"),backgroundPosition:"center",backgroundSize:"cover"}:{}},selectInputData:function(){return this.$store.getters["datasets/getInputDataByID"](this.id)||{}},title:function(){var t=this.selectInputData;return"input"===t.layer?"Входные данные "+t.id:"Выходные данные "+t.id},color:function(){return this.selectInputData.color||""},bg:function(){return{backgroundColor:this.id?this.color:""}},bc:function(){return{borderColor:this.id?this.color:""}}},methods:{outside:function(){this.show=!1}}},L=M,V=(i("1ddc"),Object(d["a"])(L,H,P,!1,null,"8cd777c2",null)),R=V.exports,A=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"t-table"},[i("div",{staticClass:"t-table__header"}),i("div",{staticClass:"t-table__data"},[i("div",{staticClass:"t-table__col",style:{padding:"1px 0"}},t._l(6,(function(e,s){return i("div",{key:"idx_r_"+e,staticClass:"t-table__row"},[t._v(t._s(s||""))])})),0),i("div",{staticClass:"t-table__border"},t._l(t.origTable,(function(e,s){return i("div",{key:"row_"+s,class:["t-table__col",{"t-table__col--active":e.active}],style:t.getColor,on:{click:function(i){return t.select(e,i)}}},[t._l(e,(function(e,s){return[s<=5?i("div",{key:"item_"+s,staticClass:"t-table__row"},[t._v(" "+t._s(e)+" ")]):t._e()]})),i("div",{staticClass:"t-table__select"},t._l(t.all(e),(function(t,e){return i("div",{key:"all"+e,staticClass:"t-table__circle",style:t})})),0)],2)})),0)]),i("div",{directives:[{name:"click-outside",rawName:"v-click-outside",value:t.outside,expression:"outside"}],staticClass:"t-table__footer"},[i("span",[t._v(t._s(t.label))]),i("div",{staticClass:"t-table__footer--btn",on:{click:function(e){t.show=!0}}},[i("i",{staticClass:"t-icon icon-file-dot"})]),i("div",{directives:[{name:"show",rawName:"v-show",value:t.show,expression:"show"}],staticClass:"t-table__dropdown"},t._l(t.items,(function(e,s){var n=e.icon,a=e.event;return i("div",{key:"icon"+s,staticClass:"t-table__dropdown--item",on:{click:function(e){t.$emit("event",{label:t.label,event:a}),t.show=!1}}},[i("i",{class:["t-icon",n]})])})),0)])])},X=[],J=i("3835"),Y={name:"CardTable",props:{label:String,type:String,id:Number,cover:String,table:Array,value:String},data:function(){return{show:!1,items:[{icon:"icon-deploy-remove",event:"remove"}]}},computed:{getColor:function(){var t=this.handlers.find((function(t){return t.active}));return{borderColor:(null===t||void 0===t?void 0:t.color)||""}},origTable:function(){var t=this;return this.table.map((function(e){return e.active=t.selected.includes(e[0]),e}))},handlers:{set:function(t){this.$store.dispatch("tables/setHandlers",t)},get:function(){return this.$store.getters["tables/getHandlers"]}},selected:{set:function(t){var e=this;this.handlers=this.handlers.map((function(i){return i.active&&i.table[e.label]&&(i.table[e.label]=t),i}))},get:function(){var t=this.handlers.find((function(t){return t.active}));return t?t.table[this.label]:[]}}},methods:{outside:function(){this.show=!1},all:function(t){var e=this,i=Object(J["a"])(t,1),s=i[0];return this.handlers.filter((function(t){return t.table[e.label].includes(s)})).map((function(t,e){return{backgroundColor:t.color,top:-3*e+"px"}}))},select:function(t){var e=Object(J["a"])(t,1),i=e[0];this.selected.find((function(t){return t===i}))?this.selected=this.selected.filter((function(t){return t!==i})):this.selected.push(i)}}},U=Y,K=(i("ad58"),Object(d["a"])(U,A,X,!1,null,"137e7a5b",null)),W=K.exports,q=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"t-cards",style:t.style,on:{wheel:function(e){return e.preventDefault(),t.wheel.apply(null,arguments)}}},[i("scrollbar",{ref:"scrollCards",attrs:{ops:t.ops}},[i("div",{staticClass:"t-cards__items"},[i("div",{staticClass:"t-cards__items--item"},[t._t("default")],2)])])],1)},z=[],G={data:function(){return{wight:0,ops:{scrollPanel:{scrollingX:!0,scrollingY:!1}}}},computed:{style:function(){return{wight:this.wight+"px"}}},mounted:function(){this.wight=this.$el.clientWidth,console.log(this.$el.clientWidth)},methods:{wheel:function(t){t.stopPropagation(),this.$refs.scrollCards.scrollBy({dx:t.wheelDelta},200)}}},Q=G,Z=(i("d541"),Object(d["a"])(Q,q,z,!1,null,"7681bbae",null)),tt=Z.exports,et=i("083d"),it={name:"BlockHeader",components:{CardFile:R,CardTable:W,Cards:tt},mixins:[et["a"]],methods:{event:function(t){var e=t.label;this.mixinFiles=this.mixinFiles.filter((function(t){return t.label!==e}))},onDrop:function(t){var e=t.dataTransfer,i=JSON.parse(e.getData("CardDataType"));if(this.mixinFiles.length){if(this.mixinFiles.find((function(t){return t.type!==i.type})))return void this.$Notify.warning({title:"Внимание!",message:"Выбрать можно только одинаковый тип данных"});if(this.mixinFiles.find((function(t){return"table"===t.type})))return void this.$Notify.warning({title:"Внимание!",message:"Выбрать можно только одину таблицу"})}this.mixinFiles.find((function(t){return t.value===i.value}))?this.$Notify.warning({title:"Внимание!",message:"Каталог уже выбран"}):this.mixinFiles=[].concat(Object(o["a"])(this.mixinFiles),[i])}}},st=it,nt=(i("3d4c"),Object(d["a"])(st,T,I,!1,null,"4d929803",null)),at=nt.exports,rt=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"block-left"},[i("div",{staticClass:"block-left__fab"},[i("Fab",{on:{click:t.addCard}})],1),i("div",{staticClass:"block-left__header"},[t._v("Входные параметры")]),i("div",{staticClass:"block-left__body"},[i("scrollbar",{ref:"scrollLeft",attrs:{ops:t.ops}},[i("div",{staticClass:"block-left__body--inner",style:t.height},[i("div",{staticClass:"block-left__body--empty"}),t._l(t.inputDataInput,(function(e){return[i("CardLayer",t._b({key:"cardLayersLeft"+e.id,on:{"click-btn":function(i){return t.optionsCard(i,e.id)}},scopedSlots:t._u([{key:"header",fn:function(){return[t._v("Входные данные "+t._s(e.id))]},proxy:!0},{key:"default",fn:function(s){var n=s.data,a=n.parameters,r=n.errors;return[t._l(t.input,(function(s,n){return[i("t-auto-field",t._b({key:e.color+n,attrs:{parameters:a,errors:r,idKey:"key_"+n,id:e.id,update:t.mixinUpdateDate,isAudio:t.isAudio,root:""},on:{multiselect:t.mixinUpdate,change:t.mixinChange}},"t-auto-field",s,!1))]}))]}}],null,!0)},"CardLayer",e,!1))]})),i("div",{staticClass:"block-left__body--empty"})],2)])],1)])},lt=[],ot=i("5530"),ct=i("2f62"),ut=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"fab",on:{click:function(e){return t.$emit("click",e)}}},[i("i",{staticClass:"fab__icon"})])},dt=[],ft={name:"fab"},ht=ft,pt=(i("2a9c"),Object(d["a"])(ht,ut,dt,!1,null,"7e89689d",null)),gt=pt.exports,vt=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"card-layer",style:t.height},[i("div",{staticClass:"card-layer__header",style:t.bg,on:{click:function(e){return t.$emit("click-header",e)}}},[i("div",{directives:[{name:"click-outside",rawName:"v-click-outside",value:t.outside,expression:"outside"}],staticClass:"card-layer__header--icon",on:{click:function(e){t.toggle=!t.toggle}}},[i("i",{staticClass:"t-icon icon-file-dot"})]),i("div",{staticClass:"card-layer__header--title"},[t._t("header",null,{id:t.id})],2)]),i("div",{directives:[{name:"show",rawName:"v-show",value:t.toggle,expression:"toggle"}],staticClass:"card-layer__dropdown"},t._l(t.items,(function(e,s){var n=e.icon;return i("div",{key:"icon"+s,staticClass:"card-layer__dropdown--item",on:{click:function(e){return t.click(n)}}},[i("i",{class:[n]})])})),0),i("div",{staticClass:"card-layer__body"},[i("scrollbar",{attrs:{ops:t.ops}},[i("div",{ref:"cardBody",staticClass:"card-layer__body--inner"},[t._t("default",null,{data:t.data})],2)])],1)])},bt=[],mt={name:"card-layer",props:{id:Number,layer:String,name:String,type:String,color:String,parameters:{type:Object,default:function(){}}},data:function(){return{height:{height:"100%"},toggle:!1,items:[{icon:"remove"},{icon:"copy"}],ops:{bar:{background:"#17212b"},scrollPanel:{scrollingX:!1,scrollingY:!0}}}},computed:{errors:function(){return this.$store.getters["datasets/getErrors"](this.id)},data:function(){return{errors:this.errors,parameters:Object(ot["a"])(Object(ot["a"])({},this.parameters),{},{name:this.name,type:this.type})}},bg:function(){return{backgroundColor:this.color}}},methods:{outside:function(){this.toggle&&(this.toggle=!1)},click:function(t){this.toggle=!1,this.$emit("click-btn",t)}},mounted:function(){var t=this.$el.clientHeight,e=this.$refs.cardBody.clientHeight+36;t>e&&(this.height={height:e+"px"}),this.$emit("mount",!0)}},_t=mt,yt=(i("04e3"),Object(d["a"])(_t,vt,bt,!1,null,"27daa405",null)),kt=yt.exports,Ct={name:"BlockMainLeft",components:{Fab:gt,CardLayer:kt},mixins:[et["a"]],data:function(){return{ops:{scrollPanel:{scrollingX:!0,scrollingY:!1},rail:{gutterOfEnds:"6px"}}}},computed:Object(ot["a"])(Object(ot["a"])({},Object(ct["b"])({input:"datasets/getTypeInput",inputData:"datasets/getInputData"})),{},{isAudio:function(){var t=this.inputDataInput.filter((function(t){return"Audio"===t.type})),e=Object(J["a"])(t,1),i=e[0];return null===i||void 0===i?void 0:i.id},inputDataInput:function(){var t=this.inputData.filter((function(t){return"input"===t.layer}));return t},height:function(){var t=this.$store.getters["settings/height"]({clean:!0,padding:324});return t}}),methods:{error:function(t,e){var i,s,n,a=this.$store.getters["datasets/getErrors"](t);return(null===a||void 0===a||null===(i=a[e])||void 0===i?void 0:i[0])||(null===a||void 0===a||null===(s=a.parameters)||void 0===s||null===(n=s[e])||void 0===n?void 0:n[0])||""},autoScroll:function(){var t=this;this.$nextTick((function(){t.$refs.scrollLeft.scrollTo({x:"100%"},100)}))},addCard:function(){this.$store.dispatch("datasets/createInputData",{layer:"input"}),this.autoScroll()},optionsCard:function(t,e){"remove"===t&&(this.$store.dispatch("datasets/removeInputData",e),this.mixinRemove(e)),"copy"===t&&(this.$store.dispatch("datasets/cloneInputData",e),this.autoScroll())},heightForm:function(t){this.$store.dispatch("settings/setHeight",{center:this.$el.clientHeight}),console.log(t,this.$el.clientHeight)}}},xt=Ct,wt=(i("905c"),Object(d["a"])(xt,rt,lt,!1,null,"17079ac9",null)),$t=wt.exports,Ft=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"block-right"},[i("div",{staticClass:"block-right__fab"},[i("Fab",{on:{click:t.addCard}})],1),i("div",{staticClass:"block-right__header"},[t._v("Выходные параметры")]),i("div",{staticClass:"block-right__body"},[i("scrollbar",{ref:"scrollRight",attrs:{ops:t.ops}},[i("div",{staticClass:"block-right__body--inner",style:t.height},[i("div",{staticClass:"block-right__body--empty"}),t._l(t.inputDataOutput,(function(e){return[i("CardLayer",t._b({key:"cardLayersRight"+e.id,on:{"click-btn":function(i){return t.optionsCard(i,e.id)}},scopedSlots:t._u([{key:"header",fn:function(){return[t._v("Выходные данные "+t._s(e.id))]},proxy:!0},{key:"default",fn:function(s){var n=s.data,a=n.parameters,r=n.errors;return[t._l(t.output,(function(s,n){return[i("t-auto-field",t._b({key:e.color+n,attrs:{parameters:a,errors:r,idKey:"key_"+n,id:e.id,update:t.mixinUpdateDate,root:""},on:{multiselect:t.mixinUpdate,change:t.mixinChange}},"t-auto-field",s,!1))]}))]}}],null,!0)},"CardLayer",e,!1))]})),i("div",{staticClass:"block-right__body--empty"})],2)])],1)])},Ot=[],St={name:"BlockMainRight",components:{Fab:gt,CardLayer:kt},mixins:[et["a"]],data:function(){return{ops:{scrollPanel:{scrollingX:!0,scrollingY:!1},rail:{gutterOfEnds:"6px"}}}},computed:Object(ot["a"])(Object(ot["a"])({},Object(ct["b"])({output:"datasets/getTypeOutput",inputData:"datasets/getInputData"})),{},{inputDataOutput:function(){return this.inputData.filter((function(t){return"output"===t.layer}))},height:function(){var t=this.$store.getters["settings/height"]({clean:!0,padding:324});return t}}),methods:{autoScroll:function(){var t=this;this.$nextTick((function(){t.$refs.scrollRight.scrollTo({x:"0%"},100)}))},addCard:function(){this.$store.dispatch("datasets/createInputData",{layer:"output"}),this.autoScroll()},optionsCard:function(t,e){"remove"===t&&(this.$store.dispatch("datasets/removeInputData",e),this.mixinRemove(e)),"copy"===t&&(this.$store.dispatch("datasets/cloneInputData",e),this.autoScroll())}},mounted:function(){}},jt=St,Dt=(i("df64"),Object(d["a"])(jt,Ft,Ot,!1,null,"5b41cfb6",null)),Bt=Dt.exports,Et=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{class:["block-handlers",{"block-handlers--hide":!t.show}]},[i("div",{staticClass:"block-handlers__header"},[i("div",{staticClass:"block-handlers__item"},[i("Fab",{on:{click:t.handleAdd}}),i("p",[t._v("Обработчики")]),i("div",{staticClass:"block-handlers__item--left",on:{click:function(e){t.show=!t.show}}},[i("i",{class:["t-icon icon-collapsable",{rotate:t.show}]})])],1)]),t.show?i("scrollbar",{attrs:{ops:t.ops}},[i("div",{staticClass:"block-handlers__content"},[t._l(t.handlers,(function(e,s){return[i("CardHandler",t._b({key:"handler"+s,on:{"click-btn":function(i){return t.handleClick(i,e.id)}},nativeOn:{click:function(i){return t.select(e.id)}},scopedSlots:t._u([{key:"header",fn:function(){return[t._v(t._s(""+e.name))]},proxy:!0},{key:"default",fn:function(s){var n=s.data,a=n.parameters,r=n.errors;return[t._l(t.formsHandler,(function(s,n){return[i("t-auto-field-handler",t._b({key:e.color+n,attrs:{parameters:a,errors:r,idKey:"key_"+n,id:e.id,root:""},on:{change:t.change}},"t-auto-field-handler",s,!1))]}))]}}],null,!0)},"CardHandler",e,!1))]}))],2)]):t._e()],1)},Nt=[],Tt=(i("c740"),i("d3b7"),i("25f0"),function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{class:["card-layer",{"card-layer--active":t.active}],style:t.height},[i("div",{staticClass:"card-layer__header",style:t.bg,on:{click:function(e){return t.$emit("click-header",e)}}},[i("div",{directives:[{name:"click-outside",rawName:"v-click-outside",value:t.outside,expression:"outside"}],staticClass:"card-layer__header--icon",on:{click:function(e){t.toggle=!t.toggle}}},[i("i",{staticClass:"t-icon icon-file-dot"})]),i("div",{staticClass:"card-layer__header--title"},[t._t("header",null,{id:t.id})],2)]),i("div",{directives:[{name:"show",rawName:"v-show",value:t.toggle,expression:"toggle"}],staticClass:"card-layer__dropdown"},t._l(t.items,(function(e,s){var n=e.icon;return i("div",{key:"icon"+s,staticClass:"card-layer__dropdown--item",on:{click:function(e){return t.click(n)}}},[i("i",{class:[n]})])})),0),i("div",{staticClass:"card-layer__body"},[i("scrollbar",{attrs:{ops:t.ops}},[i("div",{ref:"cardBody",staticClass:"card-layer__body--inner"},[t._t("default",null,{data:t.data})],2)])],1)])}),It=[],Ht={name:"card-layer",props:{id:Number,layer:String,name:String,type:String,active:Boolean,color:String,parameters:{type:Object,default:function(){}}},data:function(){return{height:{height:"100%"},toggle:!1,items:[{icon:"remove"},{icon:"copy"}],ops:{bar:{background:"#17212b"},scrollPanel:{scrollingX:!1,scrollingY:!0}}}},computed:{errors:function(){return this.$store.getters["datasets/getErrors"](this.id)},data:function(){return{errors:this.errors,parameters:Object(ot["a"])(Object(ot["a"])({},this.parameters),{},{name:this.name,type:this.type})}},bg:function(){return{backgroundColor:this.color}}},methods:{outside:function(){this.toggle&&(this.toggle=!1)},click:function(t){this.toggle=!1,this.$emit("click-btn",t)}},mounted:function(){var t=this.$el.clientHeight,e=this.$refs.cardBody.clientHeight+36;t>e&&(this.height={height:e+"px"}),this.$emit("mount",!0)}},Pt=Ht,Mt=(i("0fd9"),Object(d["a"])(Pt,Tt,It,!1,null,"7ac71446",null)),Lt=Mt.exports,Vt={name:"block-handlers",components:{Fab:gt,CardHandler:Lt},data:function(){return{show:!0,ops:{scrollPanel:{scrollingX:!0,scrollingY:!1}},colors:["#1ea61d","#a51da6","#0d6dea","#fecd05","#d72239","#054f1d","#630e76","#031e70","#b78b01","#660634","#86e372","#e473d0","#6bb5f9","#ffe669","#f38079"],table:{}}},computed:{formsHandler:function(){return this.$store.getters["datasets/getFormsHandler"]},handlers:{set:function(t){this.$store.dispatch("tables/setHandlers",t)},get:function(){return this.$store.getters["tables/getHandlers"]}}},created:function(){var t=this.$store.getters["datasets/getFilesSource"];console.log(t),this.table=t.filter((function(t){return"table"===t.type})).reduce((function(t,e){return t[e.title]=[],t}),{})},methods:{change:function(t){var e=t.id,i=t.value,s=t.name,n=this.handlers.findIndex((function(t){return t.id===e}));"name"===s&&(this.handlers[n].name=i),"type"===s&&(this.handlers[n].type=i),this.handlers[n]&&(this.handlers[n].parameters[s]=i),this.handlers=Object(o["a"])(this.handlers)},select:function(t){this.handlers=this.handlers.map((function(e){return e.active=e.id===t,e}))},deselect:function(){this.handlers=this.handlers.map((function(t){return t.active=!1,t}))},handleAdd:function(){if(this.show){console.log(this.table),this.deselect();var t=Math.max.apply(Math,[0].concat(Object(o["a"])(this.handlers.map((function(t){return t.id})))));this.handlers.push({id:t+1,name:"Name_"+(t+1),active:!0,color:this.colors[this.handlers.length],layer:(this.handlers.length+1).toString(),type:"",table:JSON.parse(JSON.stringify(this.table)),parameters:{}}),console.log(this.handlers)}},handleClick:function(t,e){if("remove"===t&&(this.deselect(),this.handlers=this.handlers.filter((function(t){return t.id!==e}))),console.log(t),"copy"===t){this.deselect();var i=JSON.parse(JSON.stringify(this.handlers.filter((function(t){return t.id==e})))),s=Math.max.apply(Math,[0].concat(Object(o["a"])(this.handlers.map((function(t){return t.id})))));i[0].id=s+1,i[0].name="Name_"+(s+1),this.handlers=[].concat(Object(o["a"])(this.handlers),Object(o["a"])(i))}}}},Rt=Vt,At=(i("09b9"),Object(d["a"])(Rt,Et,Nt,!1,null,"3efa65ea",null)),Xt=At.exports,Jt=i("eb4c"),Yt={name:"ParamsFull",components:{BlockFiles:h,BlockFooter:N,BlockHeader:at,BlockMainLeft:$t,BlockMainRight:Bt,BlockHandlers:Xt},data:function(){return{toggle:!0,debounce:null,ops:{scrollPanel:{scrollingX:!1,scrollingY:!0}}}},computed:{full:{set:function(t){this.$store.dispatch("datasets/setFull",t)},get:function(){return this.$store.getters["datasets/getFull"]}},height:function(){var t=this.$store.getters["settings/height"]({style:!1,clean:!0});return t=t-172-96,{flex:"0 0 "+t+"px",height:t+"px"}},isTable:function(){return this.$store.getters["datasets/getFilesDrop"].some((function(t){return"table"===t.type}))}},methods:{createObject:function(t){var e=this;return Object(a["a"])(regeneratorRuntime.mark((function i(){var s,n;return regeneratorRuntime.wrap((function(i){while(1)switch(i.prev=i.next){case 0:return e.$store.dispatch("messages/setMessage",{info:'Создается датасет "'.concat(t.name,'"')}),i.next=3,e.$store.dispatch("datasets/createDataset",t);case 3:s=i.sent,console.log(s),s&&(n=s.success,n&&e.debounce(!0));case 6:case"end":return i.stop()}}),i)})))()},progress:function(){var t=this;return Object(a["a"])(regeneratorRuntime.mark((function e(){var i,s;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.$store.dispatch("datasets/createProgress",{});case 2:i=e.sent,i&&(s=i.data.finished,s?t.full=!1:t.debounce(!0));case 4:case"end":return e.stop()}}),e)})))()},change:function(t){this.toggle=t}},created:function(){var t=this;this.debounce=Object(Jt["a"])((function(e){console.log(e),e&&t.progress()}),1e3)},beforeDestroy:function(){this.debounce(!1)}},Ut=Yt,Kt=(i("9b1c"),Object(d["a"])(Ut,s,n,!1,null,null,null));e["default"]=Kt.exports},"73d9":function(t,e,i){var s=i("44d2");s("flatMap")},"8e25":function(t,e,i){"use strict";i("a3da")},"905c":function(t,e,i){"use strict";i("d6c8")},"938a":function(t,e,i){},"9b1c":function(t,e,i){"use strict";i("de39")},a2bf:function(t,e,i){"use strict";var s=i("e8b5"),n=i("50c4"),a=i("0366"),r=function(t,e,i,l,o,c,u,d){var f,h=o,p=0,g=!!u&&a(u,d,3);while(p<l){if(p in i){if(f=g?g(i[p],p,e):i[p],c>0&&s(f))h=r(t,e,f,n(f.length),h,c-1)-1;else{if(h>=9007199254740991)throw TypeError("Exceed the acceptable array length");t[h]=f}h++}p++}return h};t.exports=r},a3da:function(t,e,i){},a645:function(t,e,i){},ad58:function(t,e,i){"use strict";i("305f")},b7d5:function(t,e,i){},bd41:function(t,e,i){"use strict";i("41ba")},c893:function(t,e,i){},d4d8:function(t,e,i){"use strict";i("f832")},d541:function(t,e,i){"use strict";i("1d15")},d6c8:function(t,e,i){},de39:function(t,e,i){},df33:function(t,e,i){},df64:function(t,e,i){"use strict";i("c893")},eb4c:function(t,e,i){"use strict";i.d(e,"a",(function(){return s}));var s=function(t,e,i){var s;return function(){var n=this,a=arguments,r=function(){s=null,i||t.apply(n,a)},l=i&&!s;clearTimeout(s),s=setTimeout(r,e),l&&t.apply(n,a)}}},f832:function(t,e,i){}}]);