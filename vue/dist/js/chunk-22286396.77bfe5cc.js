(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-22286396"],{"01c2":function(t,e,i){},"152b":function(t,e,i){},"234a":function(t,e,i){"use strict";i("d97e")},2495:function(t,e,i){"use strict";i("152b")},"2a9c":function(t,e,i){"use strict";i("b7d5")},"3c3f":function(t,e,i){"use strict";i("c150")},"405b":function(t,e,i){"use strict";i("d53a")},"45f5":function(t,e,i){"use strict";i("8217")},"52fb":function(t,e,i){},"5dd0":function(t,e,i){"use strict";i.r(e);var s=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"params-full"},[i("div",{staticClass:"params-full__inner"},[i("div",{staticClass:"params-full__btn",on:{click:function(e){t.full=!t.full}}},[i("i",{staticClass:"params-full__btn--icon"})]),i("div",{class:["params-full__files",{toggle:!t.toggle}]},[i("BlockFiles",{on:{toggle:t.change}})],1),i("div",{staticClass:"params-full__main"},[i("div",{staticClass:"main__header"},[i("BlockHeader")],1),i("div",{staticClass:"main__center",style:t.height},[i("div",{staticClass:"main__center--left"},[i("BlockMainLeft")],1),i("div",{staticClass:"main__center--right"},[i("BlockMainRight")],1)]),i("div",{staticClass:"main__footer"},[i("BlockFooter",{on:{create:t.createObject}})],1)])])])},n=[],a=i("1da1"),r=(i("96cf"),function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"block-file"},[i("div",{class:["block-file__header",{toggle:!t.toggle}],on:{click:function(e){t.toggle=!t.toggle,t.$emit("toggle",t.toggle)}}},[i("i",{staticClass:"block-file__header--icon"}),t._v(" "+t._s(t.text)+" ")]),i("div",{directives:[{name:"show",rawName:"v-show",value:t.toggle,expression:"toggle"}],staticClass:"block-file__body"},[i("scrollbar",[i("files-menu",{model:{value:t.filesSource,callback:function(e){t.filesSource=e},expression:"filesSource"}})],1)],1)])}),l=[],o={name:"BlockFiles",data:function(){return{toggle:!0,nodes:[{title:"Cars",type:"folder",isExpanded:!1,children:[{title:"BMW.jpg",type:"image"},{title:"AUDI.jpg",type:"image"}]},{title:"Music",type:"folder",isExpanded:!0,children:[{title:"1.mp3",type:"audio"},{title:"song.wav",type:"audio"}]},{title:"Text",type:"folder",isExpanded:!1,children:[{title:"Table",type:"text"}]}]}},computed:{text:function(){return this.toggle?"Выбор папки/файла":""},filesSource:{set:function(t){this.$store.dispatch("datasets/setFilesSource",t)},get:function(){return this.$store.getters["datasets/getFilesSource"]}}}},c=o,u=(i("9f2f"),i("2877")),d=Object(u["a"])(c,r,l,!1,null,"2eeefed6",null),f=d.exports,h=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("form",{staticClass:"block-footer",on:{submit:function(t){t.preventDefault()}}},[i("div",{staticClass:"block-footer__item"},[i("t-input",{attrs:{parse:"[name]",small:""},model:{value:t.nameProject,callback:function(e){t.nameProject=e},expression:"nameProject"}},[t._v(" Название датасета ")])],1),i("div",{staticClass:"block-footer__item block-tags"},[i("TTags")],1),i("div",{staticClass:"block-footer__item"},[i("Slider",{attrs:{degree:t.degree}})],1),i("div",{staticClass:"block-footer__item"},[i("t-checkbox",{attrs:{parse:"[info][shuffle]",reverse:""}},[t._v("Сохранить последовательность")])],1),i("div",{staticClass:"block-footer__item"},[i("t-checkbox",{attrs:{parse:"use_generator"}},[t._v("Использовать генератор")])],1),i("div",{staticClass:"action"},[i("t-button",{attrs:{disabled:!!t.disabled},nativeOn:{click:function(e){return t.getObj.apply(null,arguments)}}},[t._v("Сформировать")])],1)])},p=[],_=(i("d81d"),i("caad"),i("2532"),function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"t-field"},[i("div",{staticClass:"t-field__label"},[t._v("Train / Val / Test")]),i("div",{ref:"slider",staticClass:"slider",on:{mouseleave:t.stopDrag,mouseup:t.stopDrag}},[i("div",{staticClass:"slider__inputs"},[i("input",{attrs:{name:"[info][part][train]",type:"number","data-degree":t.degree},domProps:{value:t.btnFirstVal}}),i("input",{attrs:{name:"[info][part][validation]",type:"number","data-degree":t.degree},domProps:{value:t.btnSecondVal-t.btnFirstVal}}),i("input",{attrs:{name:"[info][part][test]",type:"number","data-degree":t.degree},domProps:{value:100-t.btnSecondVal}})]),i("div",{staticClass:"slider__scales"},[i("div",{staticClass:"scales__first",style:t.firstScale},[t._v(t._s(t.btnFirstVal))]),i("div",{staticClass:"scales__second",style:t.secondScale},[t._v(t._s(t.btnSecondVal-t.btnFirstVal))]),i("div",{staticClass:"scales__third",style:t.thirdScale},[t._v(t._s(100-t.btnSecondVal))])]),i("div",{ref:"between",staticClass:"slider__between"},[i("button",{staticClass:"slider__btn-1",style:t.sliderFirstStyle,on:{mousedown:t.startDragFirst,mouseup:t.stopDragFirst}}),i("button",{staticClass:"slider__btn-2",style:t.sliderSecondStyle,on:{mousedown:t.startDragSecond,mouseup:t.stopDragSecond}})])])])}),v=[],m=(i("a9e3"),{name:"Slider",data:function(){return{btnFirstVal:50,btnSecondVal:77,firstBtnDrag:!1,secondBtnDrag:!1}},props:{degree:Number},methods:{stopDrag:function(){this.$refs.slider.removeEventListener("mousemove",this.firstBtn),this.$refs.slider.removeEventListener("mousemove",this.secondBtn)},startDragFirst:function(){this.firstBtnDrag=!0,this.$refs.slider.addEventListener("mousemove",this.firstBtn)},stopDragFirst:function(){this.$refs.slider.removeEventListener("mousemove",this.firstBtn),this.firstBtnDrag=!1},startDragSecond:function(){this.secondBtnDrag=!0,this.$refs.slider.addEventListener("mousemove",this.secondBtn)},stopDragSecond:function(){this.$refs.slider.removeEventListener("mousemove",this.secondBtn),this.secondBtnDrag=!1},firstBtn:function(t){if(this.firstBtnDrag){var e=document.querySelector(".slider__btn-1"),i=t.pageX-e.parentNode.getBoundingClientRect().x;this.btnFirstVal=Math.round(i/231*100),this.btnFirstVal<5&&(this.btnFirstVal=5),this.btnFirstVal>95&&(this.btnFirstVal=95),this.btnFirstVal>this.btnSecondVal-5&&(this.btnFirstVal=this.btnSecondVal-5)}},secondBtn:function(t){if(this.secondBtnDrag){var e=document.querySelector(".slider__btn-2"),i=t.pageX-e.parentNode.getBoundingClientRect().x;this.btnSecondVal=Math.round(i/231*100),this.btnSecondVal<5&&(this.btnSecondVal=5),this.btnSecondVal>95&&(this.btnSecondVal=95),this.btnSecondVal<this.btnFirstVal+5&&(this.btnSecondVal=this.btnFirstVal+5)}}},computed:{sliderFirstStyle:function(){return{left:this.btnFirstVal+"%"}},sliderSecondStyle:function(){return{left:this.btnSecondVal+"%"}},firstScale:function(){return{width:this.btnFirstVal+"%"}},secondScale:function(){return{width:this.btnSecondVal-this.btnFirstVal+"%"}},thirdScale:function(){return{width:100-this.btnSecondVal+"%"}}}}),g=m,b=(i("45f5"),Object(u["a"])(g,_,v,!1,null,"dfee37f2",null)),y=b.exports,C=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{class:["t-field",{"t-inline":t.inline}]},[i("label",{staticClass:"t-field__label",attrs:{for:t.parse}},[t._v(t._s(t.label))]),i("div",{staticClass:"tags"},[i("button",{staticClass:"tags__add",attrs:{type:"button"}},[i("i",{staticClass:"tags__add--icon t-icon icon-tag-plus",on:{click:t.create}}),i("input",{staticClass:"tags__add--input",attrs:{type:"text",disabled:t.tags.length>=3,placeholder:"Добавить"},on:{keypress:function(e){return!e.type.indexOf("key")&&t._k(e.keyCode,"enter",13,e.key,"Enter")?null:(e.preventDefault(),t.create.apply(null,arguments))}}})]),t._l(t.tags,(function(e,s){var n=e.value;return[i("input",{key:"tag_"+s,class:["tags__item"],style:{width:8*(n.length+1)+"px"},attrs:{"data-index":s,name:"[tags][][name]",type:"text"},domProps:{value:n},on:{input:t.change,blur:t.blur}})]}))],2)])},k=[],x=i("2909"),S=(i("4de4"),{name:"t-input",props:{label:{type:String,default:"Теги"},type:{type:String,default:"text"},value:{type:[String,Number]},parse:String,name:String,inline:Boolean,disabled:Boolean},data:function(){return{tags:[]}},methods:{create:function(){var t,e=null===(t=this.$el.getElementsByClassName("tags__add--input"))||void 0===t?void 0:t[0];console.log(e.value),e.value&&e.value.length>=3&&this.tags.length<=3&&(this.tags.push({value:e.value}),this.tags=Object(x["a"])(this.tags),e.value="")},change:function(t){var e=t.target.dataset.index;console.log(e),this.tags[+e].value=t.target.value},blur:function(t){var e=t.target.dataset.index;t.target.value.length<=2&&(this.tags=this.tags.filter((function(t,i){return i!==+e}))),console.log(e)}}}),$=S,w=(i("ce8a"),Object(u["a"])($,C,k,!1,null,"92110c1c",null)),D=w.exports,F=i("da6d"),O=i.n(F),B={name:"BlockFooter",components:{Slider:y,TTags:D},data:function(){return{degree:100,nameProject:""}},computed:{disabled:function(){var t=this.$store.state.datasets.inputData.map((function(t){return t.layer}));return!(this.nameProject&&t.includes("input")&&t.includes("output"))}},methods:{getObj:function(){this.$emit("create",O()(this.$el))}}},j=B,E=(i("3c3f"),Object(u["a"])(j,h,p,!1,null,"05d0ce3c",null)),V=E.exports,L=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"block-header",on:{drop:function(e){return t.onDrop(e)},dragover:function(t){t.preventDefault()}}},[t.mixinFiles.length?i("div",{staticClass:"block-header__main"},[i("Cards",[t._l(t.mixinFiles,(function(e,s){return["folder"===e.type?i("CardFile",t._b({key:"files_"+s},"CardFile",e,!1)):t._e(),"table"===e.type?i("CardTable",t._b({key:"files_"+s},"CardTable",e,!1)):t._e()]}))],2),i("div",{staticClass:"empty"})],1):i("div",{staticClass:"inner"},[t._m(0)])])},T=[function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"block-header__overlay"},[i("div",{staticClass:"block-header__overlay--icon"}),i("div",{staticClass:"block-header__overlay--title"},[t._v("Перетащите папку или файл для начала работы с содержимым архива")])])}],I=(i("7db0"),i("99af"),function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"t-card-file",style:t.bc},[t.id?i("div",{staticClass:"t-card-file__header",style:t.bg},[t._v(t._s(t.title))]):t._e(),i("div",{class:["t-card-file__body","icon-file-"+t.type],style:t.img}),i("div",{staticClass:"t-card-file__footer"},[t._v(t._s(t.label))])])}),N=[],R={name:"t-card-file",props:{label:String,type:String,id:Number,cover:String},computed:{img:function(){return console.log(this.cover),this.cover?{backgroundImage:"url('".concat(this.cover,"')"),backgroundPosition:"center",backgroundSize:"cover"}:{}},selectInputData:function(){return this.$store.getters["datasets/getInputDataByID"](this.id)||{}},title:function(){var t=this.selectInputData;return"input"===t.layer?"Входные данные "+t.id:"Выходные данные "+t.id},color:function(){return this.selectInputData.color||""},bg:function(){return{backgroundColor:this.id?this.color:""}},bc:function(){return{borderColor:this.id?this.color:""}}}},M=R,P=(i("77eb"),Object(u["a"])(M,I,N,!1,null,"f4f5de4c",null)),A=P.exports,q=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"csv-table"},[i("div",{staticClass:"table__data"},[t._m(0),i("div",{staticClass:"selected__cols"}),t._l(t.arr,(function(e,s){return i("div",{key:"row_"+s,staticClass:"table__col",attrs:{"data-index":s},on:{mousedown:function(e){return t.select(s)}}},t._l(e,(function(e,s){return i("div",{key:"item_"+s,staticClass:"table__row"},[t._v(t._s(e))])})),0)}))],2),t._m(1)])},H=[function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"table__col"},[i("div",{staticClass:"table__row"}),i("div",{staticClass:"table__row"},[t._v("0")]),i("div",{staticClass:"table__row"},[t._v("2")]),i("div",{staticClass:"table__row"},[t._v("4")]),i("div",{staticClass:"table__row"},[t._v("6")]),i("div",{staticClass:"table__row"},[t._v("8")])])},function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"table__footer"},[i("span",[t._v("Список файлов")])])}],K=(i("159b"),i("4e82"),i("a434"),{name:"CardTable",props:{label:String,type:String,id:Number,cover:String,table:Array},data:function(){return{table_test:[],selected_cols:[]}},computed:{arr:function(){var t=[];return this.table.forEach((function(e,i){e.forEach((function(e,s){t[s]||(t[s]=[]),t[s][i]=e}))})),console.log(t),t}},created:function(){console.log(this.table)},methods:{compare:function(t,e){return t.dataset.index<e.dataset.index?-1:t.dataset.index>e.dataset.index?1:0},sortOnDataIndex:function(t){var e=[],i=t.children.length;while(i--)e[i]=t.children[i],t.children[i].remove();e.sort(this.compare),i=0;while(e[i])t.appendChild(e[i]),++i},select:function(t){if(event.preventDefault(),1==event.which){var e=this.selected_cols.indexOf(t),i=document.querySelector(".selected__cols"),s=document.querySelector(".table__data"),n=document.querySelector(".table__col[data-index='".concat(t,"']"));-1!==e?(this.selected_cols.splice(e,1),document.querySelector(".selected__cols").removeChild(n),s.append(n),this.sortOnDataIndex(s)):(this.selected_cols.push(t),document.querySelector(".table__data").removeChild(n),i.append(n),this.sortOnDataIndex(i)),0==this.selected_cols.length?i.style.display="none":i.style.display="flex"}}}}),X=K,Y=(i("b114"),Object(u["a"])(X,q,H,!1,null,"b18560b0",null)),J=Y.exports,U=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"t-cards",style:t.style,on:{wheel:t.wheel}},[i("scrollbar",{ref:"scrollCards",attrs:{ops:t.ops}},[i("div",{staticClass:"t-cards__items"},[i("div",{staticClass:"t-cards__items--item"},[t._t("default")],2)])])],1)},W=[],z={data:function(){return{wight:0,ops:{scrollPanel:{scrollingX:!0,scrollingY:!1}}}},computed:{style:function(){return{wight:this.wight+"px"}}},mounted:function(){this.wight=this.$el.clientWidth,console.log(this.$el.clientWidth)},methods:{wheel:function(t){this.$refs.scrollCards.scrollBy({dx:t.wheelDelta},200)}}},G=z,Q=(i("e65d"),Object(u["a"])(G,U,W,!1,null,"5a8a3de1",null)),Z=Q.exports,tt={computed:{mixinFiles:{set:function(t){this.$store.dispatch("datasets/setFilesDrop",t)},get:function(){return this.$store.getters["datasets/getFilesDrop"]}}},methods:{mixinCheck:function(t,e){this.mixinFiles=this.mixinFiles.map((function(i){return t.find((function(t){return t.value===i.value}))?i.id=e:i.id=i.id===e?0:i.id,i}));var i=t.map((function(t){return t.value}));this.mixinChange({id:e,name:"sources_paths",value:i})},mixinRemove:function(t){this.mixinFiles=this.mixinFiles.map((function(e){return e.id=e.id===t?0:e.id,e}))},mixinChange:function(t){this.$store.dispatch("datasets/updateInputData",t)}}},et={name:"BlockHeader",components:{CardFile:A,CardTable:J,Cards:Z},mixins:[tt],methods:{onDrop:function(t){var e=t.dataTransfer,i=JSON.parse(e.getData("CardDataType"));this.mixinFiles.find((function(t){return t.value===i.value}))?this.$Notify.warning({title:"Внимание!",message:"Каталог уже выбран"}):this.mixinFiles=[].concat(Object(x["a"])(this.mixinFiles),[i])}}},it=et,st=(i("2495"),Object(u["a"])(it,L,T,!1,null,"1e830015",null)),nt=st.exports,at=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"block-left"},[i("div",{staticClass:"block-left__fab"},[i("Fab",{on:{click:t.addCard}})],1),i("div",{staticClass:"block-left__header"},[t._v("Входные параметры")]),i("div",{staticClass:"block-left__body"},[i("scrollbar",{ref:"scrollLeft",attrs:{ops:t.ops}},[i("div",{staticClass:"block-left__body--inner",style:t.height},[i("div",{staticClass:"block-left__body--empty"}),t._l(t.inputDataInput,(function(e){return[i("CardLayer",t._b({key:"cardLayersLeft"+e.id,on:{"click-btn":function(i){return t.optionsCard(i,e.id)}},scopedSlots:t._u([{key:"header",fn:function(){return[t._v("Входные данные "+t._s(e.id))]},proxy:!0},{key:"default",fn:function(s){var n=s.data,a=n.parameters,r=n.errors;return[i("TMultiSelect",{attrs:{id:e.id,name:"sources_paths",value:a.sources_paths,lists:t.mixinFiles,label:"Выберите путь",errors:r,inline:""},on:{change:function(i){return t.mixinCheck(i,e.id)}}}),t._l(t.input,(function(s,n){return[i("t-auto-field",t._b({key:e.color+n,attrs:{parameters:a,errors:r,idKey:"key_"+n,id:e.id,root:""},on:{change:t.mixinChange}},"t-auto-field",s,!1))]})),i("t-color",{attrs:{parse:"test",inline:""},on:{change:t.test}}),i("t-button-api")]}}],null,!0)},"CardLayer",e,!1))]})),i("div",{staticClass:"block-left__body--empty"})],2)])],1)])},rt=[],lt=i("5530"),ot=i("2f62"),ct=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"fab",on:{click:function(e){return t.$emit("click",e)}}},[i("i",{staticClass:"fab__icon"})])},ut=[],dt={name:"fab"},ft=dt,ht=(i("2a9c"),Object(u["a"])(ft,ct,ut,!1,null,"7e89689d",null)),pt=ht.exports,_t=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"card-layer",style:t.height},[i("div",{staticClass:"card-layer__header",style:t.bg,on:{click:function(e){return t.$emit("click-header",e)}}},[i("div",{directives:[{name:"click-outside",rawName:"v-click-outside",value:t.outside,expression:"outside"}],staticClass:"card-layer__header--icon",on:{click:function(e){t.toggle=!t.toggle}}},[i("i",{staticClass:"t-icon icon-file-dot"})]),i("div",{staticClass:"card-layer__header--title"},[t._t("header",null,{id:t.id})],2)]),i("div",{directives:[{name:"show",rawName:"v-show",value:t.toggle,expression:"toggle"}],staticClass:"card-layer__dropdown"},t._l(t.items,(function(e,s){var n=e.icon;return i("div",{key:"icon"+s,staticClass:"card-layer__dropdown--item",on:{click:function(e){return t.click(n)}}},[i("i",{class:[n]})])})),0),i("div",{staticClass:"card-layer__body"},[i("scrollbar",{attrs:{ops:t.ops}},[i("div",{ref:"cardBody",staticClass:"card-layer__body--inner"},[t._t("default",null,{data:t.data})],2)])],1)])},vt=[],mt=(i("b0c0"),{name:"card-layer",props:{id:Number,layer:String,name:String,type:String,color:String,parameters:{type:Object,default:function(){}}},data:function(){return{height:{height:"100%"},toggle:!1,items:[{icon:"remove"},{icon:"copy"}],ops:{bar:{background:"#17212b"},scrollPanel:{scrollingX:!1,scrollingY:!0}}}},computed:{errors:function(){return this.$store.getters["datasets/getErrors"](this.id)},data:function(){return{errors:this.errors,parameters:Object(lt["a"])(Object(lt["a"])({},this.parameters),{},{name:this.name,type:this.type})}},bg:function(){return{backgroundColor:this.color}}},methods:{outside:function(){this.toggle&&(this.toggle=!1)},click:function(t){this.toggle=!1,this.$emit("click-btn",t)}},mounted:function(){var t=this.$el.clientHeight,e=this.$refs.cardBody.clientHeight+36;t>e&&(this.height={height:e+"px"}),this.$emit("mount",!0)}}),gt=mt,bt=(i("b2d5"),Object(u["a"])(gt,_t,vt,!1,null,"1c752235",null)),yt=bt.exports,Ct=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{directives:[{name:"click-outside",rawName:"v-click-outside",value:t.outside,expression:"outside"}],class:["t-multi-select",{"t-inline":t.inline}]},[i("label",{staticClass:"t-multi-select__label"},[t._t("default",(function(){return[t._v(t._s(t.label))]}))],2),i("div",{class:["t-multi-select__input",{"t-multi-select__error":t.error}]},[i("span",{class:["t-multi-select__input--text",{"t-multi-select__input--active":t.input}],attrs:{title:t.input},on:{click:t.click}},[t._v(" "+t._s(t.input||t.placeholder)+" ")])]),i("div",{directives:[{name:"show",rawName:"v-show",value:t.show,expression:"show"}],staticClass:"t-multi-select__content"},[t.filterList.length?i("div",{staticClass:"t-multi__item",on:{click:function(e){return t.select(t.checkAll)}}},[i("span",{class:["t-multi__item--check",{"t-multi__item--active":t.checkAll}]}),i("span",{staticClass:"t-multi__item--title"},[t._v("Выбрать все")])]):t._e(),t._l(t.filterList,(function(e,s){return[i("div",{key:s,staticClass:"t-multi__item",attrs:{title:e.label},on:{click:function(i){return t.select(e)}}},[i("span",{class:["t-multi__item--check",{"t-multi__item--active":t.active(e)}]}),i("span",{staticClass:"t-multi__item--title"},[t._v(t._s(e.label))])])]})),t.filterList.length?t._e():i("div",{staticClass:"t-multi__item t-multi__item--empty"},[i("span",{staticClass:"t-multi__item--title"},[t._v("Нет данных")])])],2)])},kt=[],xt=(i("a15b"),{name:"TMultiSelect",props:{name:String,id:Number,label:{type:String,default:"Label"},lists:{type:Array,required:!0,default:function(){return[]}},placeholder:{type:String,default:"Не выбрано"},disabled:Boolean,inline:Boolean,value:Array},data:function(){return{selected:[],show:!1,pagination:0}},computed:{errors:function(){return this.$store.getters["datasets/getErrors"](this.id)},error:function(){var t,e,i,s,n,a=this.name;return(null===(t=this.errors)||void 0===t||null===(e=t[a])||void 0===e?void 0:e[0])||(null===(i=this.errors)||void 0===i||null===(s=i.parameters)||void 0===s||null===(n=s[a])||void 0===n?void 0:n[0])||""},input:function(){return this.selected.map((function(t){return t.label})).join()},checkAll:function(){return this.filterList.length===this.selected.length},filterList:function(){var t=this;return this.lists.filter((function(e){return!e.id||e.id===t.id}))}},methods:{click:function(){this.show=!0,this.error&&this.$store.dispatch("datasets/cleanError",{id:this.id,name:this.name})},active:function(t){var e=t.value;return!!this.selected.find((function(t){return t.value===e}))},outside:function(){this.show&&(this.show=!1)},select:function(t){"boolean"===typeof t?this.selected=this.filterList.map((function(e){return t?null:e})).filter((function(t){return t})):this.selected.find((function(e){return e.value===t.value}))?this.selected=this.selected.filter((function(e){return e.value!==t.value})):this.selected=[].concat(Object(x["a"])(this.selected),[t]),this.$emit("change",this.selected)}},created:function(){var t=this.value;Array.isArray(t)&&(this.selected=this.filterList.filter((function(e){return t.includes(e.value)})))}}),St=xt,$t=(i("234a"),Object(u["a"])(St,Ct,kt,!1,null,"2632cf18",null)),wt=$t.exports,Dt=i("1854"),Ft={name:"BlockMainLeft",components:{Fab:pt,CardLayer:yt,TColor:Dt["a"],TMultiSelect:wt},mixins:[tt],data:function(){return{ops:{scrollPanel:{scrollingX:!0,scrollingY:!1},rail:{gutterOfEnds:"6px"}},testListRadio:[{key:"testKey1",value:!0,label:"Изображения"},{key:"testKey2",value:!1,label:"Текст"},{key:"testKey3",value:!1,label:"Аудио"},{key:"testKey4",value:!1,label:"Классификация"}]}},computed:Object(lt["a"])(Object(lt["a"])({},Object(ot["b"])({input:"datasets/getTypeInput",inputData:"datasets/getInputData"})),{},{inputDataInput:function(){var t=this.inputData.filter((function(t){return"input"===t.layer}));return t},height:function(){var t=this.$store.getters["settings/height"]({clean:!0,padding:324});return t}}),methods:{test:function(t){console.log(t)},error:function(t,e){var i,s,n,a=this.$store.getters["datasets/getErrors"](t);return(null===a||void 0===a||null===(i=a[e])||void 0===i?void 0:i[0])||(null===a||void 0===a||null===(s=a.parameters)||void 0===s||null===(n=s[e])||void 0===n?void 0:n[0])||""},autoScroll:function(){var t=this;this.$nextTick((function(){t.$refs.scrollLeft.scrollTo({x:"100%"},100)}))},addCard:function(){this.$store.dispatch("datasets/createInputData",{layer:"input"}),this.autoScroll()},optionsCard:function(t,e){"remove"===t&&(this.$store.dispatch("datasets/removeInputData",e),this.mixinRemove(e)),"copy"===t&&(this.$store.dispatch("datasets/cloneInputData",e),this.autoScroll())},heightForm:function(t){this.$store.dispatch("settings/setHeight",{center:this.$el.clientHeight}),console.log(t,this.$el.clientHeight)}}},Ot=Ft,Bt=(i("405b"),Object(u["a"])(Ot,at,rt,!1,null,"b8e15564",null)),jt=Bt.exports,Et=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{staticClass:"block-right"},[i("div",{staticClass:"block-right__fab"},[i("Fab",{on:{click:t.addCard}})],1),i("div",{staticClass:"block-right__header"},[t._v("Выходные параметры")]),i("div",{staticClass:"block-right__body"},[i("scrollbar",{ref:"scrollRight",attrs:{ops:t.ops}},[i("div",{staticClass:"block-right__body--inner",style:t.height},[i("div",{staticClass:"block-right__body--empty"}),t._l(t.inputDataOutput,(function(e){return[i("CardLayer",t._b({key:"cardLayersRight"+e.id,on:{"click-btn":function(i){return t.optionsCard(i,e.id)}},scopedSlots:t._u([{key:"header",fn:function(){return[t._v("Выходные данные "+t._s(e.id))]},proxy:!0},{key:"default",fn:function(s){var n=s.data,a=n.parameters,r=n.errors;return[i("TMultiSelect",{attrs:{id:e.id,name:"sources_paths",value:a.sources_paths,lists:t.mixinFiles,errors:r,label:"Выберите путь",inline:""},on:{change:function(i){return t.mixinCheck(i,e.id)}}}),t._l(t.output,(function(s,n){return[i("t-auto-field",t._b({key:e.color+n,attrs:{parameters:a,errors:r,idKey:"key_"+n,id:e.id,root:""},on:{change:t.mixinChange}},"t-auto-field",s,!1))]}))]}}],null,!0)},"CardLayer",e,!1))]})),i("div",{staticClass:"block-right__body--empty"})],2)])],1)])},Vt=[],Lt={name:"BlockMainRight",components:{Fab:pt,CardLayer:yt,TMultiSelect:wt},mixins:[tt],data:function(){return{ops:{scrollPanel:{scrollingX:!0,scrollingY:!1},rail:{gutterOfEnds:"6px"}}}},computed:Object(lt["a"])(Object(lt["a"])({},Object(ot["b"])({output:"datasets/getTypeOutput",inputData:"datasets/getInputData"})),{},{inputDataOutput:function(){return this.inputData.filter((function(t){return"output"===t.layer}))},height:function(){var t=this.$store.getters["settings/height"]({clean:!0,padding:324});return t}}),methods:{autoScroll:function(){var t=this;this.$nextTick((function(){t.$refs.scrollRight.scrollTo({x:"0%"},100)}))},addCard:function(){this.$store.dispatch("datasets/createInputData",{layer:"output"}),this.autoScroll()},optionsCard:function(t,e){"remove"===t&&(this.$store.dispatch("datasets/removeInputData",e),this.mixinRemove(e)),"copy"===t&&(this.$store.dispatch("datasets/cloneInputData",e),this.autoScroll())}},mounted:function(){console.log(this.output)}},Tt=Lt,It=(i("b668"),Object(u["a"])(Tt,Et,Vt,!1,null,"495f0d54",null)),Nt=It.exports,Rt={name:"ParamsFull",components:{BlockFiles:f,BlockFooter:V,BlockHeader:nt,BlockMainLeft:jt,BlockMainRight:Nt},data:function(){return{toggle:!0}},computed:{full:{set:function(t){this.$store.dispatch("datasets/setFull",t)},get:function(){return this.$store.getters["datasets/getFull"]}},height:function(){var t=this.$store.getters["settings/height"]({style:!1,clean:!0});return t=t-172-96,console.log(t),{flex:"0 0 "+t+"px",height:t+"px"}}},methods:{createObject:function(t){var e=this;return Object(a["a"])(regeneratorRuntime.mark((function i(){var s;return regeneratorRuntime.wrap((function(i){while(1)switch(i.prev=i.next){case 0:return i.next=2,e.$store.dispatch("datasets/createDataset",t);case 2:s=i.sent,console.log(s);case 4:case"end":return i.stop()}}),i)})))()},change:function(t){this.toggle=t}},mounted:function(){}},Mt=Rt,Pt=(i("9b1c"),Object(u["a"])(Mt,s,n,!1,null,null,null));e["default"]=Pt.exports},7050:function(t,e,i){},"77eb":function(t,e,i){"use strict";i("7050")},"7b9d":function(t,e,i){},8217:function(t,e,i){},9736:function(t,e,i){},"9b1c":function(t,e,i){"use strict";i("de39")},"9f2f":function(t,e,i){"use strict";i("b20a")},b114:function(t,e,i){"use strict";i("7b9d")},b20a:function(t,e,i){},b2d5:function(t,e,i){"use strict";i("efb6")},b668:function(t,e,i){"use strict";i("9736")},b7d5:function(t,e,i){},c150:function(t,e,i){},ce8a:function(t,e,i){"use strict";i("01c2")},d53a:function(t,e,i){},d97e:function(t,e,i){},da6d:function(t,e,i){var s=i("7037").default;i("b0c0"),i("fb6a"),i("4d63"),i("ac1f"),i("25f0"),i("466d"),i("5319");var n=/^(?:submit|button|image|reset|file)$/i,a=/^(?:input|select|textarea|keygen)/i,r=/(\[[^\[\]]*\])/g;function l(t,e){e={hash:!0,disabled:!0,empty:!0},"object"!=s(e)?e={hash:!!e}:void 0===e.hash&&(e.hash=!0);for(var i=e.hash?{}:"",r=e.serializer||(e.hash?u:d),l=t&&t.elements?t.elements:[],o=Object.create(null),c=0;c<l.length;++c){var f=l[c];if((e.disabled||!f.disabled)&&f.name&&(a.test(f.nodeName)&&!n.test(f.type))){"checkbox"!==f.type&&"radio"!==f.type||f.checked||(p=void 0);var h=f.name,p=f.value;if("number"===f.type){var _=f.dataset.degree;p=+p,_&&(p/=+_)}if("checkbox"===f.type){var v=!!f.dataset.reverse;p="true"===p,p=v?!p:p}if(e.empty){if("checkbox"===f.type&&f.checked,"radio"===f.type&&(o[f.name]||f.checked?f.checked&&(o[f.name]=!0):o[f.name]=!1),void 0==p&&"radio"==f.type)continue}else if(!p)continue;if("select-multiple"!==f.type)i=r(i,h,p);else{p=[];for(var m=f.options,g=!1,b=0;b<m.length;++b){var y=m[b],C=e.empty&&!y.value,k=y.value||C;y.selected&&k&&(g=!0,i=e.hash&&"[]"!==h.slice(h.length-2)?r(i,h+"[]",y.value):r(i,h,y.value))}!g&&e.empty&&(i=r(i,h,""))}}}if(e.empty)for(var h in o)o[h]||(i=r(i,h,""));return i}function o(t){var e=[],i=/^([^\[\]]*)/,s=new RegExp(r),n=i.exec(t);n[1]&&e.push(n[1]);while(null!==(n=s.exec(t)))e.push(n[1]);return e}function c(t,e,i){if(0===e.length)return t=i,t;var s=e.shift(),n=s.match(/^\[(.+?)\]$/);if("[]"===s)return t=t||[],Array.isArray(t)?t.push(c(null,e,i)):(t._values=t._values||[],t._values.push(c(null,e,i))),t;if(n){var a=n[1],r=+a;isNaN(r)?(t=t||{},t[a]=c(t[a],e,i)):(t=t||[],t[r]=c(t[r],e,i))}else t[s]=c(t[s],e,i);return t}function u(t,e,i){var s=e.match(r);if(s){var n=o(e);c(t,n,i)}else{var a=t[e];a?(Array.isArray(a)||(t[e]=[a]),t[e].push(i)):t[e]=i}return t}function d(t,e,i){return i=i.replace(/(\r)?\n/g,"\r\n"),i=encodeURIComponent(i),i=i.replace(/%20/g,"+"),t+(t?"&":"")+encodeURIComponent(e)+"="+i}t.exports=l},de39:function(t,e,i){},e65d:function(t,e,i){"use strict";i("52fb")},efb6:function(t,e,i){}}]);