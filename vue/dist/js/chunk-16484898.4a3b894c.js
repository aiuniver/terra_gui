(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-16484898"],{"07cd":function(t,e,s){"use strict";s("aa8e")},"152b":function(t,e,s){},"1dc3":function(t,e,s){},2495:function(t,e,s){"use strict";s("152b")},"2a9c":function(t,e,s){"use strict";s("b7d5")},"52fb":function(t,e,s){},"5dd0":function(t,e,s){"use strict";s.r(e);var a=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"params-full"},[s("div",{staticClass:"params-full__inner"},[s("div",{staticClass:"params-full__btn",on:{click:function(e){t.full=!t.full}}},[s("i",{staticClass:"params-full__btn--icon"})]),s("div",{class:["params-full__files",{toggle:!t.toggle}]},[s("BlockFiles",{on:{toggle:t.change}})],1),s("div",{staticClass:"params-full__main"},[s("div",{staticClass:"main__header"},[s("BlockHeader")],1),s("div",{staticClass:"main__center",style:t.height},[s("div",{staticClass:"main__center--left"},[s("BlockMainLeft")],1),s("div",{staticClass:"main__center--right"},[s("BlockMainRight")],1)]),s("div",{staticClass:"main__footer"},[s("BlockFooter",{on:{create:t.createObject}})],1)])])])},i=[],n=s("1da1"),r=(s("96cf"),function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"block-file"},[s("div",{class:["block-file__header",{toggle:!t.toggle}],on:{click:function(e){t.toggle=!t.toggle,t.$emit("toggle",t.toggle)}}},[s("i",{staticClass:"block-file__header--icon"}),t._v(" "+t._s(t.text)+" ")]),s("div",{directives:[{name:"show",rawName:"v-show",value:t.toggle,expression:"toggle"}],staticClass:"block-file__body"},[s("scrollbar",[s("files-menu",{model:{value:t.filesSource,callback:function(e){t.filesSource=e},expression:"filesSource"}})],1)],1)])}),l=[],o={name:"BlockFiles",data:function(){return{toggle:!0,nodes:[{title:"Cars",type:"folder",isExpanded:!1,children:[{title:"BMW.jpg",type:"image"},{title:"AUDI.jpg",type:"image"}]},{title:"Music",type:"folder",isExpanded:!0,children:[{title:"1.mp3",type:"audio"},{title:"song.wav",type:"audio"}]},{title:"Text",type:"folder",isExpanded:!1,children:[{title:"Table",type:"text"}]}]}},computed:{text:function(){return this.toggle?"Выбор папки/файла":""},filesSource:{set:function(t){this.$store.dispatch("datasets/setFilesSource",t)},get:function(){return this.$store.getters["datasets/getFilesSource"]}}}},c=o,d=(s("9f2f"),s("2877")),u=Object(d["a"])(c,r,l,!1,null,"2eeefed6",null),f=u.exports,h=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("form",{staticClass:"block-footer",on:{submit:function(t){t.preventDefault()}}},[s("div",{staticClass:"block-footer__item"},[s("t-input",{attrs:{parse:"[name]",small:"",error:t.nameError},on:{focus:function(e){t.nameError=""}},model:{value:t.nameProject,callback:function(e){t.nameProject=e},expression:"nameProject"}},[t._v(" Название датасета ")])],1),s("div",{staticClass:"block-footer__item block-tags"},[s("TTags")],1),s("div",{staticClass:"block-footer__item"},[s("Slider",{attrs:{degree:t.degree}})],1),s("div",{staticClass:"block-footer__item"},[s("t-checkbox",{attrs:{parse:"[info][shuffle]",reverse:""}},[t._v("Сохранить последовательность")])],1),s("div",{staticClass:"block-footer__item"},[s("t-checkbox",{attrs:{parse:"use_generator"}},[t._v("Использовать генератор")])],1),s("div",{staticClass:"action"},[s("t-button",{attrs:{disabled:!!t.disabled},nativeOn:{click:function(e){return t.getObj.apply(null,arguments)}}},[t._v("Сформировать")])],1)])},_=[],p=(s("d81d"),s("caad"),s("2532"),function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"t-field"},[s("div",{staticClass:"t-field__label"},[t._v("Train / Val / Test")]),s("div",{ref:"slider",staticClass:"slider",on:{mouseleave:t.stopDrag,mouseup:t.stopDrag}},[s("div",{staticClass:"slider__inputs"},[s("input",{attrs:{name:"[info][part][train]",type:"number","data-degree":t.degree},domProps:{value:t.btnFirstVal}}),s("input",{attrs:{name:"[info][part][validation]",type:"number","data-degree":t.degree},domProps:{value:t.btnSecondVal-t.btnFirstVal}}),s("input",{attrs:{name:"[info][part][test]",type:"number","data-degree":t.degree},domProps:{value:100-t.btnSecondVal}})]),s("div",{staticClass:"slider__scales"},[s("div",{staticClass:"scales__first",style:t.firstScale},[t._v(t._s(t.btnFirstVal))]),s("div",{staticClass:"scales__second",style:t.secondScale},[t._v(t._s(t.btnSecondVal-t.btnFirstVal))]),s("div",{staticClass:"scales__third",style:t.thirdScale},[t._v(t._s(100-t.btnSecondVal))])]),s("div",{ref:"between",staticClass:"slider__between"},[s("button",{staticClass:"slider__btn-1",style:t.sliderFirstStyle,on:{mousedown:t.startDragFirst,mouseup:t.stopDragFirst}}),s("button",{staticClass:"slider__btn-2",style:t.sliderSecondStyle,on:{mousedown:t.startDragSecond,mouseup:t.stopDragSecond}})])])])}),g=[],b=(s("a9e3"),{name:"Slider",data:function(){return{btnFirstVal:70,btnSecondVal:90,firstBtnDrag:!1,secondBtnDrag:!1}},props:{degree:Number},methods:{stopDrag:function(){this.$refs.slider.removeEventListener("mousemove",this.firstBtn),this.$refs.slider.removeEventListener("mousemove",this.secondBtn)},startDragFirst:function(){this.firstBtnDrag=!0,this.$refs.slider.addEventListener("mousemove",this.firstBtn)},stopDragFirst:function(){this.$refs.slider.removeEventListener("mousemove",this.firstBtn),this.firstBtnDrag=!1},startDragSecond:function(){this.secondBtnDrag=!0,this.$refs.slider.addEventListener("mousemove",this.secondBtn)},stopDragSecond:function(){this.$refs.slider.removeEventListener("mousemove",this.secondBtn),this.secondBtnDrag=!1},firstBtn:function(t){if(this.firstBtnDrag){var e=document.querySelector(".slider__btn-1"),s=t.pageX-e.parentNode.getBoundingClientRect().x;this.btnFirstVal=Math.round(s/231*100),this.btnFirstVal<5&&(this.btnFirstVal=5),this.btnFirstVal>95&&(this.btnFirstVal=95),this.btnFirstVal>this.btnSecondVal-5&&(this.btnFirstVal=this.btnSecondVal-5)}},secondBtn:function(t){if(this.secondBtnDrag){var e=document.querySelector(".slider__btn-2"),s=t.pageX-e.parentNode.getBoundingClientRect().x;this.btnSecondVal=Math.round(s/231*100),this.btnSecondVal<5&&(this.btnSecondVal=5),this.btnSecondVal>95&&(this.btnSecondVal=95),this.btnSecondVal<this.btnFirstVal+5&&(this.btnSecondVal=this.btnFirstVal+5)}}},computed:{sliderFirstStyle:function(){return{left:this.btnFirstVal+"%"}},sliderSecondStyle:function(){return{left:this.btnSecondVal+"%"}},firstScale:function(){return{width:this.btnFirstVal+"%"}},secondScale:function(){return{width:this.btnSecondVal-this.btnFirstVal+"%"}},thirdScale:function(){return{width:100-this.btnSecondVal+"%"}}}}),v=b,m=(s("07cd"),Object(d["a"])(v,p,g,!1,null,"4ee5ff3c",null)),y=m.exports,C=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{class:["t-field",{"t-inline":t.inline}]},[s("label",{staticClass:"t-field__label",attrs:{for:t.parse}},[t._v(t._s(t.label))]),s("div",{staticClass:"tags"},[s("button",{staticClass:"tags__add",attrs:{type:"button"}},[s("i",{staticClass:"tags__add--icon t-icon icon-tag-plus",on:{click:t.create}}),s("input",{staticClass:"tags__add--input",attrs:{type:"text",disabled:t.tags.length>=3,placeholder:"Добавить"},on:{keypress:function(e){return!e.type.indexOf("key")&&t._k(e.keyCode,"enter",13,e.key,"Enter")?null:(e.preventDefault(),t.create.apply(null,arguments))}}})]),t._l(t.tags,(function(e,a){var i=e.value;return[s("div",{key:"tag_"+a,staticStyle:{position:"relative"}},[s("input",{class:["tags__item"],style:{width:8*(i.length+1)+16+"px"},attrs:{"data-index":a,name:"[tags][][name]",type:"text",autocomplete:"off"},domProps:{value:i},on:{input:t.change,blur:t.blur}}),s("i",{staticClass:"tags__remove--icon t-icon icon-tag-plus",on:{click:function(e){return t.removeTag(a)}}})])]}))],2)])},k=[],x=s("2909"),S=(s("4de4"),{name:"t-input",props:{label:{type:String,default:"Теги"},type:{type:String,default:"text"},value:{type:[String,Number]},parse:String,name:String,inline:Boolean,disabled:Boolean},data:function(){return{tags:[]}},methods:{removeTag:function(t){this.tags=this.tags.filter((function(e,s){return s!==+t}))},create:function(){var t,e=null===(t=this.$el.getElementsByClassName("tags__add--input"))||void 0===t?void 0:t[0];console.log(e.value),e.value&&e.value.length>=3&&this.tags.length<=3&&(this.tags.push({value:e.value}),this.tags=Object(x["a"])(this.tags),e.value="")},change:function(t){var e=t.target.dataset.index;t.target.value.length>=3&&(this.tags[+e].value=t.target.value)},blur:function(t){t.target.value.length<3&&this.$forceUpdate()}}}),$=S,D=(s("d7e1"),Object(d["a"])($,C,k,!1,null,"b696ae92",null)),w=D.exports,F=s("da6d"),B=s.n(F),O={name:"BlockFooter",components:{Slider:y,TTags:w},data:function(){return{degree:100,nameProject:"",nameError:""}},computed:{disabled:function(){var t=this.$store.state.datasets.inputData.map((function(t){return t.layer}));return!(t.includes("input")&&t.includes("output"))}},methods:{getObj:function(){this.nameProject?this.$emit("create",B()(this.$el)):this.nameError="Введите имя"}}},E=O,j=(s("d152"),Object(d["a"])(E,h,_,!1,null,"13cdba2c",null)),V=j.exports,T=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"block-header",on:{drop:function(e){return t.onDrop(e)},dragover:function(t){t.preventDefault()}}},[t.mixinFiles.length?s("div",{staticClass:"block-header__main"},[s("Cards",[t._l(t.mixinFiles,(function(e,a){return["folder"===e.type?s("CardFile",t._b({key:"files_"+a},"CardFile",e,!1)):t._e(),"table"===e.type?s("CardTable",t._b({key:"files_"+a},"CardTable",e,!1)):t._e()]}))],2),s("div",{staticClass:"empty"})],1):s("div",{staticClass:"inner"},[t._m(0)])])},I=[function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"block-header__overlay"},[s("div",{staticClass:"block-header__overlay--icon"}),s("div",{staticClass:"block-header__overlay--title"},[t._v("Перетащите папку или файл для начала работы с содержимым архива")])])}],L=(s("7db0"),s("99af"),function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"t-card-file",style:t.bc},[t.id?s("div",{staticClass:"t-card-file__header",style:t.bg},[t._v(t._s(t.title))]):t._e(),s("div",{class:["t-card-file__body","icon-file-"+t.type],style:t.img}),s("div",{staticClass:"t-card-file__footer"},[t._v(t._s(t.label))])])}),P=[],N={name:"t-card-file",props:{label:String,type:String,id:Number,cover:String},computed:{img:function(){return console.log(this.cover),this.cover?{backgroundImage:"url('".concat(this.cover,"')"),backgroundPosition:"center",backgroundSize:"cover"}:{}},selectInputData:function(){return this.$store.getters["datasets/getInputDataByID"](this.id)||{}},title:function(){var t=this.selectInputData;return"input"===t.layer?"Входные данные "+t.id:"Выходные данные "+t.id},color:function(){return this.selectInputData.color||""},bg:function(){return{backgroundColor:this.id?this.color:""}},bc:function(){return{borderColor:this.id?this.color:""}}}},R=N,M=(s("77eb"),Object(d["a"])(R,L,P,!1,null,"f4f5de4c",null)),H=M.exports,q=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"csv-table"},[s("div",{staticClass:"table__data"},[t._m(0),s("div",{staticClass:"selected__cols"}),t._l(t.arr,(function(e,a){return s("div",{key:"row_"+a,staticClass:"table__col",attrs:{"data-index":a},on:{mousedown:function(e){return t.select(a)}}},t._l(e,(function(e,a){return s("div",{key:"item_"+a,staticClass:"table__row"},[t._v(t._s(e))])})),0)}))],2),t._m(1)])},K=[function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"table__col"},[s("div",{staticClass:"table__row"}),s("div",{staticClass:"table__row"},[t._v("0")]),s("div",{staticClass:"table__row"},[t._v("2")]),s("div",{staticClass:"table__row"},[t._v("4")]),s("div",{staticClass:"table__row"},[t._v("6")]),s("div",{staticClass:"table__row"},[t._v("8")])])},function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"table__footer"},[s("span",[t._v("Список файлов")])])}],X=(s("159b"),s("4e82"),s("a434"),{name:"CardTable",props:{label:String,type:String,id:Number,cover:String,table:Array},data:function(){return{table_test:[],selected_cols:[]}},computed:{arr:function(){var t=[];return this.table.forEach((function(e,s){e.forEach((function(e,a){t[a]||(t[a]=[]),t[a][s]=e}))})),console.log(t),t}},created:function(){console.log(this.table)},methods:{compare:function(t,e){return t.dataset.index<e.dataset.index?-1:t.dataset.index>e.dataset.index?1:0},sortOnDataIndex:function(t){var e=[],s=t.children.length;while(s--)e[s]=t.children[s],t.children[s].remove();e.sort(this.compare),s=0;while(e[s])t.appendChild(e[s]),++s},select:function(t){if(event.preventDefault(),1==event.which){var e=this.selected_cols.indexOf(t),s=document.querySelector(".selected__cols"),a=document.querySelector(".table__data"),i=document.querySelector(".table__col[data-index='".concat(t,"']"));-1!==e?(this.selected_cols.splice(e,1),document.querySelector(".selected__cols").removeChild(i),a.append(i),this.sortOnDataIndex(a)):(this.selected_cols.push(t),document.querySelector(".table__data").removeChild(i),s.append(i),this.sortOnDataIndex(s)),0==this.selected_cols.length?s.style.display="none":s.style.display="flex"}}}}),Y=X,J=(s("b114"),Object(d["a"])(Y,q,K,!1,null,"b18560b0",null)),W=J.exports,A=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"t-cards",style:t.style,on:{wheel:t.wheel}},[s("scrollbar",{ref:"scrollCards",attrs:{ops:t.ops}},[s("div",{staticClass:"t-cards__items"},[s("div",{staticClass:"t-cards__items--item"},[t._t("default")],2)])])],1)},U=[],z={data:function(){return{wight:0,ops:{scrollPanel:{scrollingX:!0,scrollingY:!1}}}},computed:{style:function(){return{wight:this.wight+"px"}}},mounted:function(){this.wight=this.$el.clientWidth,console.log(this.$el.clientWidth)},methods:{wheel:function(t){this.$refs.scrollCards.scrollBy({dx:t.wheelDelta},200)}}},G=z,Q=(s("e65d"),Object(d["a"])(G,A,U,!1,null,"5a8a3de1",null)),Z=Q.exports,tt=s("083d"),et={name:"BlockHeader",components:{CardFile:H,CardTable:W,Cards:Z},mixins:[tt["a"]],methods:{onDrop:function(t){var e=t.dataTransfer,s=JSON.parse(e.getData("CardDataType"));this.mixinFiles.find((function(t){return t.value===s.value}))?this.$Notify.warning({title:"Внимание!",message:"Каталог уже выбран"}):this.mixinFiles=[].concat(Object(x["a"])(this.mixinFiles),[s])}}},st=et,at=(s("2495"),Object(d["a"])(st,T,I,!1,null,"1e830015",null)),it=at.exports,nt=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"block-left"},[s("div",{staticClass:"block-left__fab"},[s("Fab",{on:{click:t.addCard}})],1),s("div",{staticClass:"block-left__header"},[t._v("Входные параметры")]),s("div",{staticClass:"block-left__body"},[s("scrollbar",{ref:"scrollLeft",attrs:{ops:t.ops}},[s("div",{staticClass:"block-left__body--inner",style:t.height},[s("div",{staticClass:"block-left__body--empty"}),t._l(t.inputDataInput,(function(e){return[s("CardLayer",t._b({key:"cardLayersLeft"+e.id,on:{"click-btn":function(s){return t.optionsCard(s,e.id)}},scopedSlots:t._u([{key:"header",fn:function(){return[t._v("Входные данные "+t._s(e.id))]},proxy:!0},{key:"multi",fn:function(){return[t._v("Входные данные "+t._s(e.id))]},proxy:!0},{key:"default",fn:function(a){var i=a.data,n=i.parameters,r=i.errors;return[t._l(t.input,(function(a,i){return[s("t-auto-field",t._b({key:e.color+i,attrs:{parameters:n,errors:r,idKey:"key_"+i,id:e.id,root:""},on:{change:t.mixinChange}},"t-auto-field",a,!1))]}))]}}],null,!0)},"CardLayer",e,!1))]})),s("div",{staticClass:"block-left__body--empty"})],2)])],1)])},rt=[],lt=s("5530"),ot=s("2f62"),ct=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"fab",on:{click:function(e){return t.$emit("click",e)}}},[s("i",{staticClass:"fab__icon"})])},dt=[],ut={name:"fab"},ft=ut,ht=(s("2a9c"),Object(d["a"])(ft,ct,dt,!1,null,"7e89689d",null)),_t=ht.exports,pt=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"card-layer",style:t.height},[s("div",{staticClass:"card-layer__header",style:t.bg,on:{click:function(e){return t.$emit("click-header",e)}}},[s("div",{directives:[{name:"click-outside",rawName:"v-click-outside",value:t.outside,expression:"outside"}],staticClass:"card-layer__header--icon",on:{click:function(e){t.toggle=!t.toggle}}},[s("i",{staticClass:"t-icon icon-file-dot"})]),s("div",{staticClass:"card-layer__header--title"},[t._t("header",null,{id:t.id})],2)]),s("div",{directives:[{name:"show",rawName:"v-show",value:t.toggle,expression:"toggle"}],staticClass:"card-layer__dropdown"},t._l(t.items,(function(e,a){var i=e.icon;return s("div",{key:"icon"+a,staticClass:"card-layer__dropdown--item",on:{click:function(e){return t.click(i)}}},[s("i",{class:[i]})])})),0),s("div",{staticClass:"card-layer__body"},[s("scrollbar",{attrs:{ops:t.ops}},[s("div",{ref:"cardBody",staticClass:"card-layer__body--inner"},[t._t("default",null,{data:t.data})],2)])],1)])},gt=[],bt=(s("b0c0"),{name:"card-layer",props:{id:Number,layer:String,name:String,type:String,color:String,parameters:{type:Object,default:function(){}}},data:function(){return{height:{height:"100%"},toggle:!1,items:[{icon:"remove"},{icon:"copy"}],ops:{bar:{background:"#17212b"},scrollPanel:{scrollingX:!1,scrollingY:!0}}}},computed:{errors:function(){return this.$store.getters["datasets/getErrors"](this.id)},data:function(){return{errors:this.errors,parameters:Object(lt["a"])(Object(lt["a"])({},this.parameters),{},{name:this.name,type:this.type})}},bg:function(){return{backgroundColor:this.color}}},methods:{outside:function(){this.toggle&&(this.toggle=!1)},click:function(t){this.toggle=!1,this.$emit("click-btn",t)}},mounted:function(){var t=this.$el.clientHeight,e=this.$refs.cardBody.clientHeight+36;t>e&&(this.height={height:e+"px"}),this.$emit("mount",!0)}}),vt=bt,mt=(s("b2d5"),Object(d["a"])(vt,pt,gt,!1,null,"1c752235",null)),yt=mt.exports,Ct={name:"BlockMainLeft",components:{Fab:_t,CardLayer:yt},mixins:[tt["a"]],data:function(){return{ops:{scrollPanel:{scrollingX:!0,scrollingY:!1},rail:{gutterOfEnds:"6px"}},testListRadio:[{key:"testKey1",value:!0,label:"Изображения"},{key:"testKey2",value:!1,label:"Текст"},{key:"testKey3",value:!1,label:"Аудио"},{key:"testKey4",value:!1,label:"Классификация"}]}},computed:Object(lt["a"])(Object(lt["a"])({},Object(ot["b"])({input:"datasets/getTypeInput",inputData:"datasets/getInputData"})),{},{inputDataInput:function(){var t=this.inputData.filter((function(t){return"input"===t.layer}));return t},height:function(){var t=this.$store.getters["settings/height"]({clean:!0,padding:324});return t}}),methods:{test:function(t){console.log(t)},error:function(t,e){var s,a,i,n=this.$store.getters["datasets/getErrors"](t);return(null===n||void 0===n||null===(s=n[e])||void 0===s?void 0:s[0])||(null===n||void 0===n||null===(a=n.parameters)||void 0===a||null===(i=a[e])||void 0===i?void 0:i[0])||""},autoScroll:function(){var t=this;this.$nextTick((function(){t.$refs.scrollLeft.scrollTo({x:"100%"},100)}))},addCard:function(){this.$store.dispatch("datasets/createInputData",{layer:"input"}),this.autoScroll()},optionsCard:function(t,e){"remove"===t&&(this.$store.dispatch("datasets/removeInputData",e),this.mixinRemove(e)),"copy"===t&&(this.$store.dispatch("datasets/cloneInputData",e),this.autoScroll())},heightForm:function(t){this.$store.dispatch("settings/setHeight",{center:this.$el.clientHeight}),console.log(t,this.$el.clientHeight)}}},kt=Ct,xt=(s("87fa"),Object(d["a"])(kt,nt,rt,!1,null,"82d62d2c",null)),St=xt.exports,$t=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"block-right"},[s("div",{staticClass:"block-right__fab"},[s("Fab",{on:{click:t.addCard}})],1),s("div",{staticClass:"block-right__header"},[t._v("Выходные параметры")]),s("div",{staticClass:"block-right__body"},[s("scrollbar",{ref:"scrollRight",attrs:{ops:t.ops}},[s("div",{staticClass:"block-right__body--inner",style:t.height},[s("div",{staticClass:"block-right__body--empty"}),t._l(t.inputDataOutput,(function(e){return[s("CardLayer",t._b({key:"cardLayersRight"+e.id,on:{"click-btn":function(s){return t.optionsCard(s,e.id)}},scopedSlots:t._u([{key:"header",fn:function(){return[t._v("Выходные данные "+t._s(e.id))]},proxy:!0},{key:"default",fn:function(a){var i=a.data,n=i.parameters,r=i.errors;return[t._l(t.output,(function(a,i){return[s("t-auto-field",t._b({key:e.color+i,attrs:{parameters:n,errors:r,idKey:"key_"+i,id:e.id,root:""},on:{change:t.mixinChange}},"t-auto-field",a,!1))]}))]}}],null,!0)},"CardLayer",e,!1))]})),s("div",{staticClass:"block-right__body--empty"})],2)])],1)])},Dt=[],wt={name:"BlockMainRight",components:{Fab:_t,CardLayer:yt},mixins:[tt["a"]],data:function(){return{ops:{scrollPanel:{scrollingX:!0,scrollingY:!1},rail:{gutterOfEnds:"6px"}}}},computed:Object(lt["a"])(Object(lt["a"])({},Object(ot["b"])({output:"datasets/getTypeOutput",inputData:"datasets/getInputData"})),{},{inputDataOutput:function(){return this.inputData.filter((function(t){return"output"===t.layer}))},height:function(){var t=this.$store.getters["settings/height"]({clean:!0,padding:324});return t}}),methods:{autoScroll:function(){var t=this;this.$nextTick((function(){t.$refs.scrollRight.scrollTo({x:"0%"},100)}))},addCard:function(){this.$store.dispatch("datasets/createInputData",{layer:"output"}),this.autoScroll()},optionsCard:function(t,e){"remove"===t&&(this.$store.dispatch("datasets/removeInputData",e),this.mixinRemove(e)),"copy"===t&&(this.$store.dispatch("datasets/cloneInputData",e),this.autoScroll())}},mounted:function(){console.log(this.output)}},Ft=wt,Bt=(s("becc"),Object(d["a"])(Ft,$t,Dt,!1,null,"1d6fe216",null)),Ot=Bt.exports,Et={name:"ParamsFull",components:{BlockFiles:f,BlockFooter:V,BlockHeader:it,BlockMainLeft:St,BlockMainRight:Ot},data:function(){return{toggle:!0}},computed:{full:{set:function(t){this.$store.dispatch("datasets/setFull",t)},get:function(){return this.$store.getters["datasets/getFull"]}},height:function(){var t=this.$store.getters["settings/height"]({style:!1,clean:!0});return t=t-172-96,console.log(t),{flex:"0 0 "+t+"px",height:t+"px"}}},methods:{createObject:function(t){var e=this;return Object(n["a"])(regeneratorRuntime.mark((function s(){var a,i,n,r;return regeneratorRuntime.wrap((function(s){while(1)switch(s.prev=s.next){case 0:return s.next=2,e.$store.dispatch("datasets/createDataset",t);case 2:a=s.sent,i=a.data,n=a.error,r=a.success,i&&r&&!n&&(e.full=!1),console.log(i);case 8:case"end":return s.stop()}}),s)})))()},change:function(t){this.toggle=t}},mounted:function(){}},jt=Et,Vt=(s("9b1c"),Object(d["a"])(jt,a,i,!1,null,null,null));e["default"]=Vt.exports},7050:function(t,e,s){},"77eb":function(t,e,s){"use strict";s("7050")},"7b9d":function(t,e,s){},"7f6b":function(t,e,s){},"87fa":function(t,e,s){"use strict";s("8e12")},"8e12":function(t,e,s){},"9b1c":function(t,e,s){"use strict";s("de39")},"9f2f":function(t,e,s){"use strict";s("b20a")},aa8e:function(t,e,s){},b114:function(t,e,s){"use strict";s("7b9d")},b20a:function(t,e,s){},b2d5:function(t,e,s){"use strict";s("efb6")},b7d5:function(t,e,s){},becc:function(t,e,s){"use strict";s("cbfa")},cbfa:function(t,e,s){},d152:function(t,e,s){"use strict";s("7f6b")},d7e1:function(t,e,s){"use strict";s("1dc3")},de39:function(t,e,s){},e65d:function(t,e,s){"use strict";s("52fb")},efb6:function(t,e,s){}}]);