(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-59c9caf8"],{"05b2":function(t,e,s){},"07cd":function(t,e,s){"use strict";s("aa8e")},"27f1":function(t,e,s){},"2a9c":function(t,e,s){"use strict";s("b7d5")},"52fb":function(t,e,s){},"567a":function(t,e,s){"use strict";s("a6ac")},5776:function(t,e,s){"use strict";s("d8a8")},"5db7":function(t,e,s){"use strict";var i=s("23e7"),a=s("a2bf"),n=s("7b0b"),r=s("50c4"),l=s("1c0b"),o=s("65f0");i({target:"Array",proto:!0},{flatMap:function(t){var e,s=n(this),i=r(s.length);return l(t),e=o(s,0),e.length=a(e,s,s,i,0,1,t,arguments.length>1?arguments[1]:void 0),e}})},"5dd0":function(t,e,s){"use strict";s.r(e);var i=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"params-full"},[s("div",{staticClass:"params-full__inner"},[s("div",{staticClass:"params-full__btn",on:{click:function(e){t.full=!t.full}}},[s("i",{staticClass:"params-full__btn--icon"})]),s("div",{class:["params-full__files",{toggle:!t.toggle}]},[s("BlockFiles",{on:{toggle:t.change}})],1),s("div",{staticClass:"params-full__main"},[s("div",{staticClass:"main__header"},[s("BlockHeader")],1),s("div",{staticClass:"main__center",style:t.height},[s("div",{staticClass:"main__center--left"},[s("BlockMainLeft")],1),s("div",{staticClass:"main__center--right"},[s("BlockMainRight")],1)]),s("div",{staticClass:"main__footer"},[s("BlockFooter",{on:{create:t.createObject}})],1)])])])},a=[],n=s("1da1"),r=(s("96cf"),s("b0c0"),function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"block-file"},[s("div",{class:["block-file__header",{toggle:!t.toggle}],on:{click:function(e){t.toggle=!t.toggle,t.$emit("toggle",t.toggle)}}},[s("i",{staticClass:"block-file__header--icon"}),t._v(" "+t._s(t.text)+" ")]),s("div",{directives:[{name:"show",rawName:"v-show",value:t.toggle,expression:"toggle"}],staticClass:"block-file__body"},[s("button",{staticClass:"block-file__body--btn",on:{click:t.moveAll}},[t._v("Перенести всё")]),s("scrollbar",[s("files-menu",{model:{value:t.filesSource,callback:function(e){t.filesSource=e},expression:"filesSource"}})],1)],1)])}),l=[],o=s("2909"),c=(s("5db7"),s("73d9"),s("d81d"),s("4de4"),s("99af"),{name:"BlockFiles",data:function(){return{toggle:!0,nodes:[{title:"Cars",type:"folder",isExpanded:!1,children:[{title:"BMW.jpg",type:"image"},{title:"AUDI.jpg",type:"image"}]},{title:"Music",type:"folder",isExpanded:!0,children:[{title:"1.mp3",type:"audio"},{title:"song.wav",type:"audio"}]},{title:"Text",type:"folder",isExpanded:!1,children:[{title:"Table",type:"text"}]}]}},computed:{text:function(){return this.toggle?"Выбор папки/файла":""},filesSource:{set:function(t){this.$store.dispatch("datasets/setFilesSource",t)},get:function(){return this.$store.getters["datasets/getFilesSource"]}}},methods:{moveAll:function(){var t=this.$store.getters["datasets/getFilesSource"].flatMap(this.getFiles),e=t.filter((function(t){return t.dragndrop})).map((function(t){return{value:t.path,label:t.title,type:t.type,id:0,cover:t.cover,table:"table"===t.type?t.data:null}}));this.$store.dispatch("datasets/setFilesDrop",e)},getFiles:function(t){return t.children?[].concat(Object(o["a"])(t.children.flatMap(this.getFiles)),[t]):t}}}),d=c,u=(s("5776"),s("2877")),f=Object(u["a"])(d,r,l,!1,null,"10f112d0",null),h=f.exports,p=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("form",{staticClass:"block-footer",on:{submit:function(t){t.preventDefault()}}},[s("div",{staticClass:"block-footer__item"},[s("t-field",{attrs:{label:"Название датасета"}},[s("t-input-new",{staticClass:"block-footer__input-custom",attrs:{parse:"[name]",small:"",error:t.nameError},on:{focus:function(e){t.nameError=""}},model:{value:t.nameProject,callback:function(e){t.nameProject=e},expression:"nameProject"}})],1)],1),s("div",{staticClass:"block-footer__item block-tags"},[s("TTags")],1),s("div",{staticClass:"block-footer__item"},[s("Slider",{attrs:{degree:t.degree}})],1),s("div",{staticClass:"block-footer__item"},[s("t-checkbox",{attrs:{parse:"[info][shuffle]",reverse:""}},[t._v("Сохранить последовательность")])],1),s("div",{staticClass:"block-footer__item"},[s("t-checkbox",{attrs:{parse:"use_generator"}},[t._v("Использовать генератор")])],1),s("div",{staticClass:"action"},[s("t-button",{attrs:{disabled:!!t.disabled},nativeOn:{click:function(e){return t.getObj.apply(null,arguments)}}},[t._v("Сформировать")])],1)])},_=[],v=(s("caad"),s("2532"),function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"t-field"},[s("div",{staticClass:"t-field__label"},[t._v("Train / Val / Test")]),s("div",{ref:"slider",staticClass:"slider",on:{mouseleave:t.stopDrag,mouseup:t.stopDrag}},[s("div",{staticClass:"slider__inputs"},[s("input",{attrs:{name:"[info][part][train]",type:"number","data-degree":t.degree},domProps:{value:t.btnFirstVal}}),s("input",{attrs:{name:"[info][part][validation]",type:"number","data-degree":t.degree},domProps:{value:t.btnSecondVal-t.btnFirstVal}}),s("input",{attrs:{name:"[info][part][test]",type:"number","data-degree":t.degree},domProps:{value:100-t.btnSecondVal}})]),s("div",{staticClass:"slider__scales"},[s("div",{staticClass:"scales__first",style:t.firstScale},[t._v(t._s(t.btnFirstVal))]),s("div",{staticClass:"scales__second",style:t.secondScale},[t._v(t._s(t.btnSecondVal-t.btnFirstVal))]),s("div",{staticClass:"scales__third",style:t.thirdScale},[t._v(t._s(100-t.btnSecondVal))])]),s("div",{ref:"between",staticClass:"slider__between"},[s("button",{staticClass:"slider__btn-1",style:t.sliderFirstStyle,on:{mousedown:t.startDragFirst,mouseup:t.stopDragFirst}}),s("button",{staticClass:"slider__btn-2",style:t.sliderSecondStyle,on:{mousedown:t.startDragSecond,mouseup:t.stopDragSecond}})])])])}),g=[],b=(s("a9e3"),{name:"Slider",data:function(){return{btnFirstVal:70,btnSecondVal:90,firstBtnDrag:!1,secondBtnDrag:!1}},props:{degree:Number},methods:{stopDrag:function(){this.$refs.slider.removeEventListener("mousemove",this.firstBtn),this.$refs.slider.removeEventListener("mousemove",this.secondBtn)},startDragFirst:function(){this.firstBtnDrag=!0,this.$refs.slider.addEventListener("mousemove",this.firstBtn)},stopDragFirst:function(){this.$refs.slider.removeEventListener("mousemove",this.firstBtn),this.firstBtnDrag=!1},startDragSecond:function(){this.secondBtnDrag=!0,this.$refs.slider.addEventListener("mousemove",this.secondBtn)},stopDragSecond:function(){this.$refs.slider.removeEventListener("mousemove",this.secondBtn),this.secondBtnDrag=!1},firstBtn:function(t){if(this.firstBtnDrag){var e=document.querySelector(".slider__btn-1"),s=t.pageX-e.parentNode.getBoundingClientRect().x;this.btnFirstVal=Math.round(s/231*100),this.btnFirstVal<5&&(this.btnFirstVal=5),this.btnFirstVal>95&&(this.btnFirstVal=95),this.btnFirstVal>this.btnSecondVal-5&&(this.btnFirstVal=this.btnSecondVal-5)}},secondBtn:function(t){if(this.secondBtnDrag){var e=document.querySelector(".slider__btn-2"),s=t.pageX-e.parentNode.getBoundingClientRect().x;this.btnSecondVal=Math.round(s/231*100),this.btnSecondVal<5&&(this.btnSecondVal=5),this.btnSecondVal>95&&(this.btnSecondVal=95),this.btnSecondVal<this.btnFirstVal+5&&(this.btnSecondVal=this.btnFirstVal+5)}}},computed:{sliderFirstStyle:function(){return{left:this.btnFirstVal+"%"}},sliderSecondStyle:function(){return{left:this.btnSecondVal+"%"}},firstScale:function(){return{width:this.btnFirstVal+"%"}},secondScale:function(){return{width:this.btnSecondVal-this.btnFirstVal+"%"}},thirdScale:function(){return{width:100-this.btnSecondVal+"%"}}}}),m=b,y=(s("07cd"),Object(u["a"])(m,v,g,!1,null,"4ee5ff3c",null)),C=y.exports,k=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{class:["t-field",{"t-inline":t.inline}]},[s("label",{staticClass:"t-field__label",attrs:{for:t.parse}},[t._v(t._s(t.label))]),s("div",{staticClass:"tags"},[s("button",{staticClass:"tags__add",attrs:{type:"button"}},[s("i",{staticClass:"tags__add--icon t-icon icon-tag-plus",on:{click:t.create}}),s("input",{staticClass:"tags__add--input",attrs:{type:"text",disabled:t.tags.length>=3,placeholder:"Добавить"},on:{keypress:function(e){return!e.type.indexOf("key")&&t._k(e.keyCode,"enter",13,e.key,"Enter")?null:(e.preventDefault(),t.create.apply(null,arguments))}}})]),t._l(t.tags,(function(e,i){var a=e.value;return[s("div",{key:"tag_"+i,staticStyle:{display:"flex","border-radius":"4px","align-items":"center",border:"1px solid #6c7883","margin-left":"10px","padding-right":"5px"}},[s("input",{class:["tags__item"],style:{width:8*(a.length+1)<=90?8*(a.length+1)+"px":"90px"},attrs:{"data-index":i,name:"[tags][][name]",type:"text",autocomplete:"off"},domProps:{value:a},on:{input:t.change,blur:t.blur}}),s("i",{staticClass:"tags__remove--icon t-icon icon-tag-plus",on:{click:function(e){return t.removeTag(i)}}})])]}))],2)])},x=[],S={name:"t-input",props:{label:{type:String,default:"Теги"},type:{type:String,default:"text"},value:{type:[String,Number]},parse:String,name:String,inline:Boolean,disabled:Boolean},data:function(){return{tags:[]}},methods:{removeTag:function(t){this.tags=this.tags.filter((function(e,s){return s!==+t}))},create:function(){var t,e=null===(t=this.$el.getElementsByClassName("tags__add--input"))||void 0===t?void 0:t[0];console.log(e.value),e.value&&e.value.length>=3&&this.tags.length<=3&&(this.tags.push({value:e.value}),this.tags=Object(o["a"])(this.tags),e.value="")},change:function(t){var e=t.target.dataset.index;console.log(e),t.target.value.length>=3&&(this.tags[+e].value=t.target.value)},blur:function(t){var e=t.target.dataset.index;t.target.value.length<=2&&(this.tags=this.tags.filter((function(t,s){return s!==+e})))}}},w=S,$=(s("8e25"),Object(u["a"])(w,k,x,!1,null,"a489cc22",null)),D=$.exports,F=s("da6d"),O=s.n(F),B={name:"BlockFooter",components:{Slider:C,TTags:D},data:function(){return{degree:100,nameProject:"",nameError:""}},computed:{disabled:function(){var t=this.$store.state.datasets.inputData.map((function(t){return t.layer}));return!(t.includes("input")&&t.includes("output"))}},methods:{getObj:function(){this.nameProject?this.$emit("create",O()(this.$el)):this.nameError="Введите имя"}}},E=B,j=(s("8483"),Object(u["a"])(E,p,_,!1,null,"292512c8",null)),V=j.exports,T=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"block-header",on:{drop:function(e){return t.onDrop(e)},dragover:function(t){t.preventDefault()}}},[t.mixinFiles.length?s("div",{staticClass:"block-header__main"},[s("Cards",[t._l(t.mixinFiles,(function(e,i){return["folder"===e.type?s("CardFile",t._b({key:"files_"+i,on:{event:t.event}},"CardFile",e,!1)):t._e(),"table"===e.type?s("CardTable",t._b({key:"files_"+i,on:{event:t.event}},"CardTable",e,!1)):t._e()]}))],2),s("div",{staticClass:"empty"})],1):s("div",{staticClass:"inner"},[t._m(0)])])},I=[function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"block-header__overlay"},[s("div",{staticClass:"block-header__overlay--icon"}),s("div",{staticClass:"block-header__overlay--title"},[t._v("Перетащите папку или файл для начала работы с содержимым архива")])])}],L=(s("7db0"),function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{directives:[{name:"click-outside",rawName:"v-click-outside",value:t.outside,expression:"outside"}],staticClass:"t-card-file",style:t.bc},[t.id?s("div",{staticClass:"t-card-file__header",style:t.bg},[t._v(t._s(t.title))]):t._e(),s("div",{class:["t-card-file__body","icon-file-"+t.type],style:t.img}),s("div",{staticClass:"t-card-file__footer"},[s("div",{staticClass:"t-card-file__footer--label"},[t._v(t._s(t.label))]),s("div",{staticClass:"t-card-file__footer--btn",on:{click:function(e){t.show=!0}}},[s("i",{staticClass:"t-icon icon-file-dot"})])]),s("div",{directives:[{name:"show",rawName:"v-show",value:t.show,expression:"show"}],staticClass:"t-card-file__dropdown"},t._l(t.items,(function(e,i){var a=e.icon,n=e.event;return s("div",{key:"icon"+i,staticClass:"t-card-file__dropdown--item",on:{click:function(e){t.$emit("event",{label:t.label,event:n}),t.show=!1}}},[s("i",{class:["t-icon",a]})])})),0)])}),M=[],N={name:"t-card-file",props:{label:String,type:String,id:Number,cover:String},data:function(){return{show:!1,items:[{icon:"icon-deploy-remove",event:"remove"}]}},computed:{img:function(){return this.cover?{backgroundImage:"url('".concat(this.cover,"')"),backgroundPosition:"center",backgroundSize:"cover"}:{}},selectInputData:function(){return this.$store.getters["datasets/getInputDataByID"](this.id)||{}},title:function(){var t=this.selectInputData;return"input"===t.layer?"Входные данные "+t.id:"Выходные данные "+t.id},color:function(){return this.selectInputData.color||""},bg:function(){return{backgroundColor:this.id?this.color:""}},bc:function(){return{borderColor:this.id?this.color:""}}},methods:{outside:function(){this.show=!1}}},P=N,R=(s("8fe7"),Object(u["a"])(P,L,M,!1,null,"70a8b496",null)),H=R.exports,q=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"csv-table"},[s("div",{staticClass:"table__data"},[t._m(0),s("div",{staticClass:"selected__cols"}),t._l(t.arr,(function(e,i){return s("div",{key:"row_"+i,staticClass:"table__col",attrs:{"data-index":i},on:{mousedown:function(e){return t.select(i)}}},t._l(e,(function(e,i){return s("div",{key:"item_"+i,staticClass:"table__row"},[t._v(t._s(e))])})),0)}))],2),s("div",{staticClass:"table__footer"},[s("span",[t._v("Список файлов")]),s("div",{staticClass:"table__footer--btn",on:{click:function(e){t.show=!0}}},[s("i",{staticClass:"t-icon icon-file-dot"})]),s("div",{directives:[{name:"show",rawName:"v-show",value:t.show,expression:"show"}],staticClass:"table__dropdown"},t._l(t.items,(function(e,i){var a=e.icon,n=e.event;return s("div",{key:"icon"+i,staticClass:"table__dropdown--item",on:{click:function(e){t.$emit("event",{label:t.label,event:n}),t.show=!1}}},[s("i",{class:["t-icon",a]})])})),0)])])},K=[function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"table__col"},[s("div",{staticClass:"table__row"}),s("div",{staticClass:"table__row"},[t._v("0")]),s("div",{staticClass:"table__row"},[t._v("2")]),s("div",{staticClass:"table__row"},[t._v("4")]),s("div",{staticClass:"table__row"},[t._v("6")]),s("div",{staticClass:"table__row"},[t._v("8")])])}],X=(s("159b"),s("4e82"),s("a434"),{name:"CardTable",props:{label:String,type:String,id:Number,cover:String,table:Array},data:function(){return{table_test:[],selected_cols:[],show:!1,items:[{icon:"icon-deploy-remove",event:"remove"}]}},computed:{arr:function(){var t=[];return this.table.forEach((function(e,s){e.forEach((function(e,i){t[i]||(t[i]=[]),t[i][s]=e}))})),console.log(t),t}},created:function(){console.log(this.table)},methods:{compare:function(t,e){return t.dataset.index<e.dataset.index?-1:t.dataset.index>e.dataset.index?1:0},sortOnDataIndex:function(t){var e=[],s=t.children.length;while(s--)e[s]=t.children[s],t.children[s].remove();e.sort(this.compare),s=0;while(e[s])t.appendChild(e[s]),++s},select:function(t){if(event.preventDefault(),1==event.which){var e=this.selected_cols.indexOf(t),s=document.querySelector(".selected__cols"),i=document.querySelector(".table__data"),a=document.querySelector(".table__col[data-index='".concat(t,"']"));-1!==e?(this.selected_cols.splice(e,1),document.querySelector(".selected__cols").removeChild(a),i.append(a),this.sortOnDataIndex(i)):(this.selected_cols.push(t),document.querySelector(".table__data").removeChild(a),s.append(a),this.sortOnDataIndex(s)),0==this.selected_cols.length?s.style.display="none":s.style.display="flex"}}}}),A=X,Y=(s("6368"),Object(u["a"])(A,q,K,!1,null,"2b0407f6",null)),J=Y.exports,W=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"t-cards",style:t.style,on:{wheel:t.wheel}},[s("scrollbar",{ref:"scrollCards",attrs:{ops:t.ops}},[s("div",{staticClass:"t-cards__items"},[s("div",{staticClass:"t-cards__items--item"},[t._t("default")],2)])])],1)},z=[],U={data:function(){return{wight:0,ops:{scrollPanel:{scrollingX:!0,scrollingY:!1}}}},computed:{style:function(){return{wight:this.wight+"px"}}},mounted:function(){this.wight=this.$el.clientWidth,console.log(this.$el.clientWidth)},methods:{wheel:function(t){this.$refs.scrollCards.scrollBy({dx:t.wheelDelta},200)}}},G=U,Q=(s("e65d"),Object(u["a"])(G,W,z,!1,null,"5a8a3de1",null)),Z=Q.exports,tt=s("083d"),et={name:"BlockHeader",components:{CardFile:H,CardTable:J,Cards:Z},mixins:[tt["a"]],methods:{event:function(t){var e=t.label;this.mixinFiles=this.mixinFiles.filter((function(t){return t.label!==e}))},onDrop:function(t){var e=t.dataTransfer,s=JSON.parse(e.getData("CardDataType"));this.mixinFiles.find((function(t){return t.value===s.value}))?this.$Notify.warning({title:"Внимание!",message:"Каталог уже выбран"}):this.mixinFiles=[].concat(Object(o["a"])(this.mixinFiles),[s])}}},st=et,it=(s("cec3"),Object(u["a"])(st,T,I,!1,null,"6b1bcd41",null)),at=it.exports,nt=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"block-left"},[s("div",{staticClass:"block-left__fab"},[s("Fab",{on:{click:t.addCard}})],1),s("div",{staticClass:"block-left__header"},[t._v("Входные параметры")]),s("div",{staticClass:"block-left__body"},[s("scrollbar",{ref:"scrollLeft",attrs:{ops:t.ops}},[s("div",{staticClass:"block-left__body--inner",style:t.height},[s("div",{staticClass:"block-left__body--empty"}),t._l(t.inputDataInput,(function(e){return[s("CardLayer",t._b({key:"cardLayersLeft"+e.id,on:{"click-btn":function(s){return t.optionsCard(s,e.id)}},scopedSlots:t._u([{key:"header",fn:function(){return[t._v("Входные данные "+t._s(e.id))]},proxy:!0},{key:"multi",fn:function(){return[t._v("Входные данные "+t._s(e.id))]},proxy:!0},{key:"default",fn:function(i){var a=i.data,n=a.parameters,r=a.errors;return[t._l(t.input,(function(i,a){return[s("t-auto-field",t._b({key:e.color+a,attrs:{parameters:n,errors:r,idKey:"key_"+a,id:e.id,root:""},on:{change:t.mixinChange}},"t-auto-field",i,!1))]}))]}}],null,!0)},"CardLayer",e,!1))]})),s("div",{staticClass:"block-left__body--empty"})],2)])],1)])},rt=[],lt=s("5530"),ot=s("2f62"),ct=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"fab",on:{click:function(e){return t.$emit("click",e)}}},[s("i",{staticClass:"fab__icon"})])},dt=[],ut={name:"fab"},ft=ut,ht=(s("2a9c"),Object(u["a"])(ft,ct,dt,!1,null,"7e89689d",null)),pt=ht.exports,_t=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"card-layer",style:t.height},[s("div",{staticClass:"card-layer__header",style:t.bg,on:{click:function(e){return t.$emit("click-header",e)}}},[s("div",{directives:[{name:"click-outside",rawName:"v-click-outside",value:t.outside,expression:"outside"}],staticClass:"card-layer__header--icon",on:{click:function(e){t.toggle=!t.toggle}}},[s("i",{staticClass:"t-icon icon-file-dot"})]),s("div",{staticClass:"card-layer__header--title"},[t._t("header",null,{id:t.id})],2)]),s("div",{directives:[{name:"show",rawName:"v-show",value:t.toggle,expression:"toggle"}],staticClass:"card-layer__dropdown"},t._l(t.items,(function(e,i){var a=e.icon;return s("div",{key:"icon"+i,staticClass:"card-layer__dropdown--item",on:{click:function(e){return t.click(a)}}},[s("i",{class:[a]})])})),0),s("div",{staticClass:"card-layer__body"},[s("scrollbar",{attrs:{ops:t.ops}},[s("div",{ref:"cardBody",staticClass:"card-layer__body--inner"},[t._t("default",null,{data:t.data})],2)])],1)])},vt=[],gt={name:"card-layer",props:{id:Number,layer:String,name:String,type:String,color:String,parameters:{type:Object,default:function(){}}},data:function(){return{height:{height:"100%"},toggle:!1,items:[{icon:"remove"},{icon:"copy"}],ops:{bar:{background:"#17212b"},scrollPanel:{scrollingX:!1,scrollingY:!0}}}},computed:{errors:function(){return this.$store.getters["datasets/getErrors"](this.id)},data:function(){return{errors:this.errors,parameters:Object(lt["a"])(Object(lt["a"])({},this.parameters),{},{name:this.name,type:this.type})}},bg:function(){return{backgroundColor:this.color}}},methods:{outside:function(){this.toggle&&(this.toggle=!1)},click:function(t){this.toggle=!1,this.$emit("click-btn",t)}},mounted:function(){var t=this.$el.clientHeight,e=this.$refs.cardBody.clientHeight+36;t>e&&(this.height={height:e+"px"}),this.$emit("mount",!0)}},bt=gt,mt=(s("aebd"),Object(u["a"])(bt,_t,vt,!1,null,"3e2add98",null)),yt=mt.exports,Ct={name:"BlockMainLeft",components:{Fab:pt,CardLayer:yt},mixins:[tt["a"]],data:function(){return{ops:{scrollPanel:{scrollingX:!0,scrollingY:!1},rail:{gutterOfEnds:"6px"}},testListRadio:[{key:"testKey1",value:!0,label:"Изображения"},{key:"testKey2",value:!1,label:"Текст"},{key:"testKey3",value:!1,label:"Аудио"},{key:"testKey4",value:!1,label:"Классификация"}]}},computed:Object(lt["a"])(Object(lt["a"])({},Object(ot["b"])({input:"datasets/getTypeInput",inputData:"datasets/getInputData"})),{},{inputDataInput:function(){var t=this.inputData.filter((function(t){return"input"===t.layer}));return t},height:function(){var t=this.$store.getters["settings/height"]({clean:!0,padding:324});return t}}),methods:{test:function(t){console.log(t)},error:function(t,e){var s,i,a,n=this.$store.getters["datasets/getErrors"](t);return(null===n||void 0===n||null===(s=n[e])||void 0===s?void 0:s[0])||(null===n||void 0===n||null===(i=n.parameters)||void 0===i||null===(a=i[e])||void 0===a?void 0:a[0])||""},autoScroll:function(){var t=this;this.$nextTick((function(){t.$refs.scrollLeft.scrollTo({x:"100%"},100)}))},addCard:function(){this.$store.dispatch("datasets/createInputData",{layer:"input"}),this.autoScroll()},optionsCard:function(t,e){"remove"===t&&(this.$store.dispatch("datasets/removeInputData",e),this.mixinRemove(e)),"copy"===t&&(this.$store.dispatch("datasets/cloneInputData",e),this.autoScroll())},heightForm:function(t){this.$store.dispatch("settings/setHeight",{center:this.$el.clientHeight}),console.log(t,this.$el.clientHeight)}}},kt=Ct,xt=(s("c8b2"),Object(u["a"])(kt,nt,rt,!1,null,"4d0bbaec",null)),St=xt.exports,wt=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"block-right"},[s("div",{staticClass:"block-right__fab"},[s("Fab",{on:{click:t.addCard}})],1),s("div",{staticClass:"block-right__header"},[t._v("Выходные параметры")]),s("div",{staticClass:"block-right__body"},[s("scrollbar",{ref:"scrollRight",attrs:{ops:t.ops}},[s("div",{staticClass:"block-right__body--inner",style:t.height},[s("div",{staticClass:"block-right__body--empty"}),t._l(t.inputDataOutput,(function(e){return[s("CardLayer",t._b({key:"cardLayersRight"+e.id,on:{"click-btn":function(s){return t.optionsCard(s,e.id)}},scopedSlots:t._u([{key:"header",fn:function(){return[t._v("Выходные данные "+t._s(e.id))]},proxy:!0},{key:"default",fn:function(i){var a=i.data,n=a.parameters,r=a.errors;return[t._l(t.output,(function(i,a){return[s("t-auto-field",t._b({key:e.color+a,attrs:{parameters:n,errors:r,idKey:"key_"+a,id:e.id,root:""},on:{change:t.mixinChange}},"t-auto-field",i,!1))]}))]}}],null,!0)},"CardLayer",e,!1))]})),s("div",{staticClass:"block-right__body--empty"})],2)])],1)])},$t=[],Dt={name:"BlockMainRight",components:{Fab:pt,CardLayer:yt},mixins:[tt["a"]],data:function(){return{ops:{scrollPanel:{scrollingX:!0,scrollingY:!1},rail:{gutterOfEnds:"6px"}}}},computed:Object(lt["a"])(Object(lt["a"])({},Object(ot["b"])({output:"datasets/getTypeOutput",inputData:"datasets/getInputData"})),{},{inputDataOutput:function(){return this.inputData.filter((function(t){return"output"===t.layer}))},height:function(){var t=this.$store.getters["settings/height"]({clean:!0,padding:324});return t}}),methods:{autoScroll:function(){var t=this;this.$nextTick((function(){t.$refs.scrollRight.scrollTo({x:"0%"},100)}))},addCard:function(){this.$store.dispatch("datasets/createInputData",{layer:"output"}),this.autoScroll()},optionsCard:function(t,e){"remove"===t&&(this.$store.dispatch("datasets/removeInputData",e),this.mixinRemove(e)),"copy"===t&&(this.$store.dispatch("datasets/cloneInputData",e),this.autoScroll())}},mounted:function(){console.log(this.output)}},Ft=Dt,Ot=(s("567a"),Object(u["a"])(Ft,wt,$t,!1,null,"34b4e8bd",null)),Bt=Ot.exports,Et={name:"ParamsFull",components:{BlockFiles:h,BlockFooter:V,BlockHeader:at,BlockMainLeft:St,BlockMainRight:Bt},data:function(){return{toggle:!0}},computed:{full:{set:function(t){this.$store.dispatch("datasets/setFull",t)},get:function(){return this.$store.getters["datasets/getFull"]}},height:function(){var t=this.$store.getters["settings/height"]({style:!1,clean:!0});return t=t-172-96,console.log(t),{flex:"0 0 "+t+"px",height:t+"px"}}},methods:{createObject:function(t){var e=this;return Object(n["a"])(regeneratorRuntime.mark((function s(){var i,a,n,r;return regeneratorRuntime.wrap((function(s){while(1)switch(s.prev=s.next){case 0:return e.$store.dispatch("messages/setMessage",{info:'Создается датасет "'.concat(t.name,'"')}),s.next=3,e.$store.dispatch("datasets/createDataset",t);case 3:i=s.sent,i&&(a=i.data,n=i.error,r=i.success,a&&r&&!n?(e.full=!1,e.$store.dispatch("messages/setMessage",{message:'Датасет "'.concat(t.name,'" создан')})):e.$store.dispatch("messages/setMessage",{error:"Ошибка создания датасета"}),console.log(a));case 5:case"end":return s.stop()}}),s)})))()},change:function(t){this.toggle=t}},mounted:function(){}},jt=Et,Vt=(s("9b1c"),Object(u["a"])(jt,i,a,!1,null,null,null));e["default"]=Vt.exports},6368:function(t,e,s){"use strict";s("27f1")},6509:function(t,e,s){},"73d9":function(t,e,s){var i=s("44d2");i("flatMap")},8483:function(t,e,s){"use strict";s("6509")},8837:function(t,e,s){},"8e25":function(t,e,s){"use strict";s("a3da")},"8fe7":function(t,e,s){"use strict";s("ca54")},"90f1":function(t,e,s){},"9b1c":function(t,e,s){"use strict";s("de39")},a2bf:function(t,e,s){"use strict";var i=s("e8b5"),a=s("50c4"),n=s("0366"),r=function(t,e,s,l,o,c,d,u){var f,h=o,p=0,_=!!d&&n(d,u,3);while(p<l){if(p in s){if(f=_?_(s[p],p,e):s[p],c>0&&i(f))h=r(t,e,f,a(f.length),h,c-1)-1;else{if(h>=9007199254740991)throw TypeError("Exceed the acceptable array length");t[h]=f}h++}p++}return h};t.exports=r},a3da:function(t,e,s){},a6ac:function(t,e,s){},aa8e:function(t,e,s){},aebd:function(t,e,s){"use strict";s("05b2")},b7d5:function(t,e,s){},c8b2:function(t,e,s){"use strict";s("90f1")},ca54:function(t,e,s){},cec3:function(t,e,s){"use strict";s("8837")},d8a8:function(t,e,s){},de39:function(t,e,s){},e65d:function(t,e,s){"use strict";s("52fb")}}]);