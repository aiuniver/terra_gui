(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-43ab3c5a"],{"0052":function(t,e,n){"use strict";n("37d0")},"0213":function(t,e,n){},"0660":function(t,e,n){"use strict";n.r(e);var i=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"page-create"},[n("div",{staticClass:"page-create__toolbar"},[n("Toolbar",{on:{action:t.onToolbar}})],1),n("div",{staticClass:"page-create__main"},[n("Blocks")],1),n("div",{staticClass:"page-create__params"},[n("Params",{model:{value:t.state,callback:function(e){t.state=e},expression:"state"}})],1)])},s=[],a=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"d-create-toolbar"},[n("div",{staticClass:"d-create-toolbar__workspace mt-10"},[t._l(t.filter,(function(e){var i=e.color,s=e.type,a=e.typeBlock;return[n("div",{key:i,staticClass:"d-create-toolbar__item d-create-toolbar__item--no-hover",on:{click:function(e){return t.onAdd({type:s})}}},[n("d-icon-layer",t._b({},"d-icon-layer",{color:i,type:s,typeBlock:a},!1))],1)]}))],2)])},l=[],r=n("5530"),o=(n("caad"),n("7db0"),n("4de4"),n("2532"),n("2f62")),c=n("9e1e"),u={name:"d-toolbar",data:function(){return{types:c["b"],toolbar:[{id:3,filter:["data","handler","input"]},{id:4,filter:["data","handler","output"]}]}},computed:Object(r["a"])(Object(r["a"])({},Object(o["c"])({getPagination:"createDataset/getPagination"})),{},{isActive:function(){return Boolean([3,4].includes(this.getPagination))},filter:function(){var t,e=this,n=(null===(t=this.toolbar.find((function(t){return t.id===e.getPagination})))||void 0===t?void 0:t.filter)||[];return this.types.filter((function(t){return n.includes(t.type)}))}}),methods:Object(r["a"])(Object(r["a"])({},Object(o["b"])({add:"create/add"})),{},{onAdd:function(t){this.isActive&&this.add(t)}})},d=u,h=(n("6d05"),n("2877")),f=Object(h["a"])(d,a,l,!1,null,"0ade1807",null),v=f.exports,p=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{key:t.key,staticClass:"blocks",on:{contextmenu:t.contextmenu}},[t.isActive?t._e():n("div",{staticClass:"blocks__overlay"}),n("net",{staticClass:"blocks__center",attrs:{x:t.centerX,y:t.centerY,scale:t.scale}}),n("Link",{staticClass:"blocks__lines",attrs:{lines:t.lines}}),t._l(t.blocks,(function(e){return n("Block",t._b({key:e.id,ref:"block_"+e.id,refInFor:!0,attrs:{options:t.optionsForChild,linkingCheck:t.tempLink},on:{linkingStart:function(n){return t.linkingStart(e,n)},linkingStop:function(n){return t.linkingStop(e,n)},linkingBreak:function(n){return t.linkingBreak(e,n)},select:function(n){return t.blockSelect(e)},position:function(e){return t.position(e)}}},"Block",e,!1))}))],2)},g=[],m=n("3835"),b=n("2909"),_=n("b85c"),y=(n("159b"),n("99af"),n("d81d"),n("c8fb")),k=n("00b9"),S=n("2523"),C={name:"Blocks",components:{Block:k["default"],Link:S["default"]},props:{},data:function(){return{key:1,save:[],menu:{},dragging:!1,centerX:0,centerY:0,scale:1,tempLink:null,hasDragged:!1,mouseX:0,mouseY:0,lastMouseX:0,lastMouseY:0,minScale:.2,maxScale:5,linking:!1,linkStart:null,inputSlotClassName:"inputSlot"}},computed:Object(r["a"])(Object(r["a"])({},Object(o["c"])({getKeyEvent:"create/getKeyEvent",blocks:"create/getBlocks",links:"create/getLinks",selectLength:"create/getSelectedLength",getPagination:"createDataset/getPagination"})),{},{isActive:function(){return Boolean([3,4].includes(this.getPagination))},keyEvent:{set:function(t){this.setKeyEvent(t)},get:function(){return this.getKeyEvent}},optionsForChild:function(){return{scale:this.scale,x:this.centerX,y:this.centerY}},container:function(){return{centerX:this.centerX,centerY:this.centerY,scale:this.scale}},lines:function(){var t,e=this,n=[],i=Object(_["a"])(this.links);try{var s=function(){var i=t.value,s=e.blocks.find((function(t){var e=t.id;return e===i.originID})),a=e.blocks.find((function(t){var e=t.id;return e===i.targetID}));if(!s||!a||s.id===a.id)return console.warn("Remove invalid link",i),e.removeLink(i.id),"continue";var l=e.getConnectionPos(s,i.originSlot,!1),r=e.getConnectionPos(a,i.targetSlot,!0);if(!l||!r)return console.log("Remove invalid link (slot not exist)",i),e.removeLink(i.id),"continue";var o=l.x,c=l.y,u=r.x,d=r.y;n.push({x1:o,y1:c,x2:u,y2:d,slot:i.originSlot,scale:e.scale,style:{stroke:"rgb(101, 124, 244)",strokeWidth:2*e.scale,fill:"none",zIndex:999},outlineStyle:{stroke:"#666",strokeWidth:2*e.scale,strokeOpacity:.6,fill:"none",zIndex:999}})};for(i.s();!(t=i.n()).done;)s()}catch(a){i.e(a)}finally{i.f()}return this.tempLink&&n.push(this.tempLink),n}}),methods:Object(r["a"])(Object(r["a"])({},Object(o["b"])("create",["add","cloneAll","align","distance","remove","clone","select","setKeyEvent","deselect","position","update","addLink","updateLink","removeLink"])),{},{event:function(t){this.menu={},console.log(t),"add"===t&&this.add({}),"delete"===t&&this.remove(),"clone"===t&&this.cloneAll(),"left"===t&&this.align("ArrowLeft"),"right"===t&&this.align("ArrowRight"),"up"===t&&this.align("ArrowUp"),"down"===t&&this.align("ArrowDown"),"center"===t&&this.align("center"),"vertical"===t&&this.distance("vertical"),"horizon"===t&&this.distance("horizon"),"select"===t&&this.deselect(!0),this.key+=1},contextmenu:function(t){console.log(t.clientX,t.clientY)},handleMauseOver:function(t){this.mouseIsOver="mouseenter"===t.type},keyup:function(t){var e=this;this.keyEvent=t;var n=t.code,i=t.ctrlKey,s=t.shiftKey,a=this.mouseIsOver;console.log(t),"keyup"===t.type&&(a&&"Delete"===n&&this.remove(),a&&"KeyA"===n&&i&&this.deselect(!0),a&&["ArrowLeft","ArrowRight","ArrowDown","ArrowUp"].includes(n)&&i&&!s&&(this.align(n),this.key+=1),a&&["ArrowUp"].includes(n)&&i&&s&&(this.distance("vertical"),this.key+=1),a&&"KeyC"===n&&i&&(this.save=this.blocks.filter((function(t){return t.selected}))),a&&"KeyX"===n&&i&&(this.save=JSON.parse(JSON.stringify(this.blocks.filter((function(t){return t.selected})))),console.log(this.save),this.remove()),a&&"KeyV"===n&&i&&(this.deselect(),this.save.forEach((function(t){e.clone(t)}))))},handleMove:function(t){var e=Object(y["b"])(this.$el,t);if(this.mouseX=e.x,this.mouseY=e.y,this.dragging){console.log("handleMove");var n=this.mouseX-this.lastMouseX,i=this.mouseY-this.lastMouseY;this.lastMouseX=this.mouseX,this.lastMouseY=this.mouseY,this.centerX+=n,this.centerY+=i,this.hasDragged=!0}if(this.linking&&this.linkStart){var s=this.getConnectionPos(this.linkStart.block,this.linkStart.slot,!1);this.tempLink={x1:s.x,y1:s.y,x2:this.mouseX,y2:this.mouseY,slot:this.linkStart.slot,style:{stroke:"#8f8f8f",strokeWidth:2*this.scale,fill:"none"}}}},handleDown:function(t){console.log("handleDown"),console.log(t);var e=t.target||t.srcElement;if((e===this.$el||e.matches("svg, svg *"))&&1===t.which){var n;this.dragging=!0;var i=Object(y["b"])(this.$el,t);this.mouseX=i.x,this.mouseY=i.y,this.lastMouseX=this.mouseX,this.lastMouseY=this.mouseY,null!==(n=this.keyEvent)&&void 0!==n&&n.ctrlKey||this.deselect(),t.preventDefault&&t.preventDefault()}},handleUp:function(t){console.log("handleUp");var e=t.target||t.srcElement;this.dragging&&(this.dragging=!1,this.hasDragged&&(this.hasDragged=!1)),!this.$el.contains(e)||"string"===typeof e.className&&e.className.includes(this.inputSlotClassName)||(this.linking=!1,this.tempLink=null,this.linkStart=null)},handleWheel:function(t){var e=t.target||t.srcElement;if(this.$el.contains(e)){var n=Math.pow(1.1,-.01*t.deltaY);if(this.scale*=n,this.scale<this.minScale)return void(this.scale=this.minScale);if(this.scale>this.maxScale)return void(this.scale=this.maxScale);var i=(this.mouseX-this.centerX)*(n-1),s=(this.mouseY-this.centerY)*(n-1);this.centerX-=i,this.centerY-=s}},getConnectionPos:function(t,e,n){var i,s,a;if(t&&-1!==e){var l=0,r=0;l+=t.position[0],r+=t.position[1];var o=(null===(i=this.$refs)||void 0===i||null===(s=i["block_"+t.id])||void 0===s||null===(a=s[0])||void 0===a?void 0:a.getH())||{},c=o.width,u=void 0===c?0:c,d=o.heigth,h=void 0===d?0:d;return n?(l+=u/2,r+=-3):(l+=u/2,r+=h),l*=this.scale,r*=this.scale,l+=this.centerX,r+=this.centerY,{x:l,y:r}}},linkingStart:function(t,e){console.log("linkingStart"),this.linkStart={block:t,slot:e};var n=this.getConnectionPos(t,e,!1);this.tempLink={x1:n.x,y1:n.y,x2:this.mouseX,y2:this.mouseY,style:{stroke:"#8f8f8f",strokeWidth:2*this.scale,fill:"none"}},this.linking=!0},linkingStop:function(t,e){if(console.log("linkingStop"),this.linkStart&&t&&e>-1){var n=this.linkStart,i=n.slot,s=n.block.id,a=t.id,l=e,r=this.links.filter((function(t){return!(t.targetID===a&&t.targetSlot===l&&t.originID===s&&t.originSlot===i)&&!(t.targetID===s&&t.originID===a||t.originID===s&&t.targetID===a)}));this.updateLink(r);var o=Math.max.apply(Math,[0].concat(Object(b["a"])(this.links.map((function(t){return t.id})))));if(this.linkStart.block.id!==t.id){var c=this.linkStart.block.id,u=this.linkStart.slot,d=t.id,h=e;this.addLink({id:o+1,originID:c,originSlot:u,targetID:d,targetSlot:h}),this.updateModel()}}this.linking=!1,this.tempLink=null,this.linkStart=null},linkingBreak:function(t,e){if(console.log("linkingBreak"),t&&e>-1){var n=this.links.find((function(n){var i=n.targetID,s=n.targetSlot;return i===t.id&&s===e}));if(n){var i=this.blocks.find((function(t){var e=t.id;return e===n.originID})),s=this.links.filter((function(n){var i=n.targetID,s=n.targetSlot;return!(i===t.id&&s===e)}));this.updateLink(s),this.linkingStart(i,n.originSlot),this.updateModel()}}},position:function(t){var e=t.left,n=t.top;if(!this.keyEvent.ctrlKey){var i=this.blocks.map((function(t){var i=Object(m["a"])(t.position,2),s=i[0],a=i[1],l=t.selected?[s+e,a+n]:[s,a];return Object(r["a"])(Object(r["a"])({},t),{},{position:l})}));this.update(i)}},blockSelect:function(t){var e=t.id,n=t.selected;console.log(e,n),n&&!this.keyEvent.ctrlKey||this.select({id:e})},updateModel:function(){}}),mounted:function(){var t=document.documentElement;this.$el.addEventListener("mouseenter",this.handleMauseOver),this.$el.addEventListener("mouseleave",this.handleMauseOver),t.addEventListener("keydown",this.keyup),t.addEventListener("keyup",this.keyup),t.addEventListener("mousemove",this.handleMove,!0),t.addEventListener("mousedown",this.handleDown,!0),t.addEventListener("mouseup",this.handleUp,!0),t.addEventListener("wheel",this.handleWheel,!0),this.centerX=this.$el.clientWidth/2,this.centerY=this.$el.clientHeight/2},beforeDestroy:function(){var t=document.documentElement;this.$el.removeEventListener("mouseenter",this.handleMauseOver),this.$el.removeEventListener("mouseleave",this.handleMauseOver),t.removeEventListener("keydown",this.keyup),t.removeEventListener("keyup",this.keyup),t.removeEventListener("mousemove",this.handleMove,!0),t.removeEventListener("mousedown",this.handleDown,!0),t.removeEventListener("mouseup",this.handleUp,!0),t.removeEventListener("wheel",this.handleWheel,!0)}},j=C,x=(n("8639"),Object(h["a"])(j,p,g,!1,null,null,null)),O=x.exports,w=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"params"},[n("div",{staticClass:"params__body"},[n("div",{staticClass:"params__header"},[t._v("Данные")]),n("scrollbar",[n("div",{staticClass:"params__inner"},[n(t.getComp.component,{tag:"component"})],1)])],1),n("div",{staticClass:"params__footer"},[n("Pagination",{attrs:{value:t.value,list:t.list},on:{next:t.onNext,prev:t.onPrev}})],1)])},D=[],E=n("1da1"),$=(n("96cf"),n("eb4c")),F=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"preview"},[n("div",{staticClass:"preview__cards"},[t._l(t.mixinFiles,(function(e,i){return["folder"===e.type?n("CardFile",t._b({key:"files_"+i},"CardFile",e,!1)):t._e(),"table"===e.type?n("CardTable",t._b({key:"files_"+i},"CardTable",e,!1)):t._e()]}))],2),n("div",{staticClass:"preview__title mb-2"},[t._v("Файлы")]),n("div",{staticClass:"preview__files"},[n("files-menu",{model:{value:t.filesSource,callback:function(e){t.filesSource=e},expression:"filesSource"}})],1)])},L=[],P=n("b79b"),M=n("4328"),I={components:{CardFile:P["a"],CardTable:M["a"]},props:{list:{type:Array,default:function(){return[]}}},data:function(){return{ops:{scrollPanel:{scrollingX:!0,scrollingY:!1}}}},computed:Object(r["a"])(Object(r["a"])({},Object(o["c"])({getFileManager:"createDataset/getFileManager"})),{},{mixinFiles:function(){return this.getFileManager.map((function(t){return{id:t.id,cover:t.cover,label:t.title,type:t.type,table:t.table,value:t.path}}))},filesSource:{set:function(t){this.$store.dispatch("datasets/setFilesSource",t)},get:function(){return this.getFileManager}}})},A=I,B=(n("a1a3"),Object(h["a"])(A,F,L,!1,null,"57134ebf",null)),X=B.exports,N=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"panel-settings"},[t._v(" "+t._s(t.type)+" "),"input"===t.type?n("SettingInput",{attrs:{selected:t.getSelected}}):t._e(),"handler"===t.type?n("SettingHandler",{attrs:{selected:t.getSelected}}):t._e(),"middle"===t.type?n("SettingMiddle",{attrs:{selected:t.getSelected}}):t._e(),"middle"===t.type?n("SettingMiddle",{attrs:{selected:t.getSelected}}):t._e(),"output"===t.type?n("SettingOutput",{attrs:{selected:t.getSelected}}):t._e(),t.type?t._e():n("SettingEmpty")],1)},Y=[],T=(n("b0c0"),n("c740"),n("d3b7"),n("25f0"),function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"panel-settings-empty"},[n("div",{staticClass:"panel-settings-empty__title mb-4"},[t._v("Для создания датасета необходимо:")]),n("div",{staticClass:"panel-settings-empty__info pl-4"},[n("ul",[n("li",[n("span",[t._v("1. Выбрать входные чистые данные")]),n("d-svg",{attrs:{name:"sloy-start-add"}})],1),n("li",[n("span",[t._v("2. Задать обработчик")]),n("d-svg",{attrs:{name:"sloy-middle-add"}})],1),n("li",[n("span",[t._v("3. Соединить все в инпут вход один")]),n("d-svg",{attrs:{name:"sloy-handler-add"}})],1)])])])}),V=[],K={},R=K,U=(n("f244"),Object(h["a"])(R,T,V,!1,null,null,null)),H=U.exports,W=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"panel-input"},[n("div",{staticClass:"panel-input__view"},[t._l(t.mixinFiles,(function(e,i){return["folder"===e.type?n("CardFile",t._b({key:"files_"+i},"CardFile",e,!1)):t._e(),"table"===e.type?n("CardTable",t._b({key:"files_"+i},"CardTable",e,!1)):t._e()]}))],2),n("div",{staticClass:"panel-input__forms"},[n("t-field",{attrs:{label:"Данные"}},[n("d-select",{attrs:{value:1,list:t.listFiles}})],1)],1)])},J=[],z={components:{CardFile:P["a"],CardTable:M["a"]},props:{selected:{type:Object}},computed:Object(r["a"])(Object(r["a"])({},Object(o["c"])({getFileManager:"createDataset/getFileManager"})),{},{mixinFiles:function(){return this.getFileManager.map((function(t){return console.log(t),{id:t.id,cover:t.cover,label:t.title,type:t.type,table:t.table,value:t.path}}))},listFiles:function(){return this.getFileManager.map((function(t){return{label:t.title,value:t.value}}))}})},G=z,q=(n("3916"),Object(h["a"])(G,W,J,!1,null,null,null)),Q=q.exports,Z=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",[t._l(t.formsHandler,(function(e,i){return[n("t-auto-field-handler",t._b({key:i,attrs:{parameters:{},idKey:"key_"+i,root:""},on:{change:t.change}},"t-auto-field-handler",e,!1))]}))],2)},tt=[],et={components:{},computed:Object(r["a"])({},Object(o["c"])({formsHandler:"datasets/getFormsHandler"}))},nt=et,it=Object(h["a"])(nt,Z,tt,!1,null,null,null),st=it.exports,at=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",[t._l(t.input,(function(e,i){return[n("t-auto-field",t._b({key:"inputData.color"+i,attrs:{parameters:t.parameters,idKey:"key_"+i,id:i+3,update:t.mixinUpdateDate,isAudio:t.isAudio,root:""},on:{multiselect:t.mixinUpdate,change:t.mixinChange}},"t-auto-field",e,!1))]}))],2)},lt=[],rt={components:{},computed:Object(r["a"])(Object(r["a"])({},Object(o["c"])({getDefault:"create/getDefault"})),{},{input:function(){return this.getDefault("input")}})},ot=rt,ct=Object(h["a"])(ot,at,lt,!1,null,null,null),ut=ct.exports,dt=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",[t._l(t.formsHandler,(function(e,i){return[n("t-auto-field-handler",t._b({key:i,attrs:{parameters:{},idKey:"key_"+i,root:""},on:{change:t.change}},"t-auto-field-handler",e,!1))]}))],2)},ht=[],ft={components:{},computed:Object(r["a"])({},Object(o["c"])({getSelected:"create/getSelected",getFileManager:"createDataset/getFileManager",getDefault:"create/getDefault"}))},vt=ft,pt=Object(h["a"])(vt,dt,ht,!1,null,null,null),gt=pt.exports,mt={name:"DatasetSettings",components:{SettingEmpty:H,SettingInput:Q,SettingMiddle:ut,SettingHandler:st,SettingOutput:gt},data:function(){return{show:!0,ops:{scrollPanel:{scrollingX:!0,scrollingY:!1}}}},computed:Object(r["a"])(Object(r["a"])({},Object(o["c"])({getSelected:"create/getSelected"})),{},{type:function(){var t;return(null===this||void 0===this||null===(t=this.getSelected)||void 0===t?void 0:t.type)||""}}),created:function(){var t=this.$store.getters["datasets/getFilesSource"];console.log(t),this.table=t.filter((function(t){return"table"===t.type})).reduce((function(t,e){return t[e.title]=[],t}),{})},methods:{change:function(t){var e=t.id,n=t.value,i=t.name,s=this.handlers.findIndex((function(t){return t.id===e}));"name"===i&&(this.handlers[s].name=n),"type"===i&&(this.handlers[s].type=n),this.handlers[s]&&(this.handlers[s].parameters[i]=n),this.handlers=Object(b["a"])(this.handlers)},select:function(t){this.handlers=this.handlers.map((function(e){return e.active=e.id===t,e}))},deselect:function(){this.handlers=this.handlers.map((function(t){return t.active=!1,t}))},handleAdd:function(){if(this.show){console.log(this.table),this.deselect();var t=Math.max.apply(Math,[0].concat(Object(b["a"])(this.handlers.map((function(t){return t.id})))));this.handlers.push({id:t+1,name:"Name_"+(t+1),active:!0,color:this.colors[this.handlers.length],layer:(this.handlers.length+1).toString(),type:"",table:JSON.parse(JSON.stringify(this.table)),parameters:{}}),console.log(this.handlers)}},handleClick:function(t,e){if("remove"===t&&(this.deselect(),this.handlers=this.handlers.filter((function(t){return t.id!==e}))),console.log(t),"copy"===t){this.deselect();var n=JSON.parse(JSON.stringify(this.handlers.filter((function(t){return t.id==e})))),i=Math.max.apply(Math,[0].concat(Object(b["a"])(this.handlers.map((function(t){return t.id})))));n[0].id=i+1,n[0].name="Name_"+(i+1),this.handlers=[].concat(Object(b["a"])(this.handlers),Object(b["a"])(n))}}}},bt=mt,_t=(n("4b70"),Object(h["a"])(bt,N,Y,!1,null,null,null)),yt=_t.exports,kt=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"tabs-download"},[n("div",{staticClass:"tabs-download-list flex align-center"},t._l(t.items,(function(e){var i=e.text,s=e.tab;return n("div",{key:"tab_"+s,class:["tabs-download-list__item",{"tabs-download-list__item--active":t.isActive(s)}],on:{click:function(e){return t.onTabs(s)}}},[t._v(" "+t._s(i)+" ")])})),0),n("div",{staticClass:"tabs-download-content mt-10"},[0===t.project.active?n("t-field",{attrs:{icon:"google",label:"Выберите файл на Google диске"}},[n("d-auto-complete",{attrs:{icon:"google-drive",placeholder:"Введите имя файла",list:t.getFilesSource},on:{click:t.getDatasetSources,change:function(e){return t.onSelect({mode:"GoogleDrive",value:e.value})}},model:{value:t.project.google,callback:function(e){t.$set(t.project,"google",e)},expression:"project.google"}})],1):t._e(),1===t.project.active?n("t-field",{attrs:{icon:"link",label:"Загрузите по ссылке"}},[n("d-input-text",{attrs:{placeholder:"URL"},on:{blur:function(e){return t.onSelect({mode:"URL",value:e.target.value})}},model:{value:t.project.url,callback:function(e){t.$set(t.project,"url",e)},expression:"project.url"}})],1):t._e()],1),n("div",[n("t-field",{attrs:{label:"Название датасета"}},[n("d-input-text",{model:{value:t.project.name,callback:function(e){t.$set(t.project,"name",e)},expression:"project.name"}})],1),n("t-field",{attrs:{label:"Версия"}},[n("d-input-text",{model:{value:t.project.version,callback:function(e){t.$set(t.project,"version",e)},expression:"project.version"}})],1),n("div",{staticClass:"mb-2"},[n("DTags",{model:{value:t.project.tags,callback:function(e){t.$set(t.project,"tags",e)},expression:"project.tags"}})],1),n("div",[n("DSlider",{model:{value:t.project.train,callback:function(e){t.$set(t.project,"train",e)},expression:"project.train"}})],1),n("t-field",{attrs:{label:"Сохранить последовательность"}},[n("d-checkbox",{model:{value:t.project.shuffle,callback:function(e){t.$set(t.project,"shuffle",e)},expression:"project.shuffle"}})],1),n("t-field",{attrs:{label:"Использовать генератор"}},[n("d-checkbox",{model:{value:t.project.use_generator,callback:function(e){t.$set(t.project,"use_generator",e)},expression:"project.use_generator"}})],1)],1)])},St=[],Ct=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"t-field"},[n("div",{staticClass:"t-field__label"},[t._v("Train / Val")]),n("div",{ref:"slider",class:["d-slider",{"d-slider--disable":t.disable}],on:{mouseleave:t.stopDrag,mouseup:t.stopDrag}},[n("div",{staticClass:"d-slider__inputs"},[n("input",{attrs:{name:"[info][part][train]",type:"number","data-degree":t.degree},domProps:{value:t.btnFirstVal}}),n("input",{attrs:{name:"[info][part][validation]",type:"number","data-degree":t.degree},domProps:{value:100-t.btnFirstVal}})]),n("div",{staticClass:"d-slider__scales"},[n("div",{staticClass:"scales__first",style:t.firstScale},[n("input",{directives:[{name:"autowidth",rawName:"v-autowidth"}],key:t.key1,ref:"key1",attrs:{type:"number",autocomplete:"off"},domProps:{value:t.btnFirstVal},on:{keypress:function(e){return!e.type.indexOf("key")&&t._k(e.keyCode,"enter",13,e.key,"Enter")?null:t.inter(1,e)},blur:function(e){return t.clickInput(1,e)},focus:t.focus}})]),n("div",{staticClass:"scales__second",style:t.secondScale},[n("input",{directives:[{name:"autowidth",rawName:"v-autowidth"}],key:t.key2,ref:"key2",attrs:{type:"number",autocomplete:"off"},domProps:{value:100-t.btnFirstVal},on:{keypress:function(e){return!e.type.indexOf("key")&&t._k(e.keyCode,"enter",13,e.key,"Enter")?null:t.inter(2,e)},blur:function(e){return t.clickInput(2,e)},focus:t.focus}})])]),n("div",{ref:"between",staticClass:"d-slider__between"},[n("button",{staticClass:"d-slider__btn-1",style:t.sliderFirstStyle,on:{mousedown:t.startDragFirst,mouseup:t.stopDragFirst}})])])])},jt=[],xt=(n("a9e3"),{name:"d-slider",props:{degree:Number,disable:Boolean,value:{type:Number,default:70}},data:function(){return{input:0,select:0,firstBtnDrag:!1,key1:1,key2:1}},methods:{focus:function(t){var e=t.target;e.select()},inter:function(t,e){var n=e.target;n.blur();var i=this.$refs["key".concat(t+1)];i&&(i.focus(),this.$nextTick((function(){i.select()})))},clickInput:function(t,e){var n=e.target,i=+n.value;1===t&&i>=0&&i<=90&&(this.btnFirstVal=i>10?i:10),2===t&&i>=0&&i<=90&&(this.btnFirstVal=i>10?100-i:10),this["key".concat(t)]+=1},stopDrag:function(){this.$refs.slider.removeEventListener("mousemove",this.firstBtn)},startDragFirst:function(){this.firstBtnDrag=!0,this.$refs.slider.addEventListener("mousemove",this.firstBtn)},stopDragFirst:function(){this.$refs.slider.removeEventListener("mousemove",this.firstBtn),this.firstBtnDrag=!1},firstBtn:function(t){if(this.firstBtnDrag){var e=document.querySelector(".d-slider__btn-1"),n=t.pageX-e.parentNode.getBoundingClientRect().x,i=this.$refs.slider.clientWidth;this.btnFirstVal=Math.round(n/i*100),console.log(this.btnFirstVal),this.btnFirstVal<10&&(this.btnFirstVal=10),this.btnFirstVal>90&&(this.btnFirstVal=90)}},diff:function(t){var e=arguments.length>1&&void 0!==arguments[1]?arguments[1]:90,n=arguments.length>2&&void 0!==arguments[2]?arguments[2]:10;return t<n&&(t=n),t>e&&(t=e),t}},computed:{sliderFirstStyle:function(){return{left:this.diff(this.btnFirstVal,90)+"%"}},sliderSecondStyle:function(){return{left:this.diff(this.btnFirstVal,90)+"%"}},firstScale:function(){return{width:this.diff(this.btnFirstVal,90)+"%"}},secondScale:function(){return{width:this.diff(100-this.btnFirstVal,90)+"%"}},btnFirstVal:{set:function(t){this.$emit("input",t)},get:function(){return this.value}}},watch:{disable:function(t){this.btnFirstVal=t?0:70}}}),Ot=xt,wt=(n("b0da"),Object(h["a"])(Ot,Ct,jt,!1,null,"133910f9",null)),Dt=wt.exports,Et=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{class:["t-field",{"t-inline":t.inline}]},[n("label",{staticClass:"t-field__label",attrs:{for:t.parse}},[t._v(t._s(t.label))]),n("div",{staticClass:"d-tags"},[n("button",{staticClass:"d-tags__add mr-2",attrs:{type:"button"}},[n("i",{staticClass:"d-tags__add--icon t-icon icon-tag-plus",on:{click:t.create}}),n("input",{staticClass:"d-tags__add--input",attrs:{type:"text",disabled:t.tags.length>=3,placeholder:"Добавить"},on:{keypress:function(e){return!e.type.indexOf("key")&&t._k(e.keyCode,"enter",13,e.key,"Enter")?null:(e.preventDefault(),t.create.apply(null,arguments))}}})]),t._l(t.tags,(function(e,i){var s=e.value;return[n("div",{key:"tag_"+i,staticClass:"d-tags__item mr-2"},[n("input",{staticClass:"d-tags__input",style:{width:8*(s.length+1)<=90?8*(s.length+1)+"px":"90px"},attrs:{"data-index":i,name:"[tags][][name]",type:"text",autocomplete:"off"},domProps:{value:s},on:{input:t.change,blur:t.blur}}),n("i",{staticClass:"d-tags__remove--icon t-icon icon-tag-plus",on:{click:function(e){return t.removeTag(i)}}})])]}))],2)])},$t=[],Ft={name:"t-input",props:{label:{type:String,default:"Теги"},type:{type:String,default:"text"},value:{type:Array},parse:String,name:String,inline:Boolean,disabled:Boolean},data:function(){return{}},computed:{tags:{set:function(t){this.$emit("input",t)},get:function(){return this.value}}},methods:{removeTag:function(t){this.tags=this.tags.filter((function(e,n){return n!==+t}))},create:function(){var t,e=null===(t=this.$el.getElementsByClassName("d-tags__add--input"))||void 0===t?void 0:t[0];console.log(e.value),e.value&&e.value.length>=3&&this.tags.length<=3&&(this.tags.push({value:e.value}),this.tags=Object(b["a"])(this.tags),e.value="")},change:function(t){var e=t.target.dataset.index;console.log(e),t.target.value.length>=3&&(this.tags[+e].value=t.target.value)},blur:function(t){var e=t.target.dataset.index;t.target.value.length<=2&&(this.tags=this.tags.filter((function(t,n){return n!==+e})))}}},Lt=Ft,Pt=(n("7a68"),Object(h["a"])(Lt,Et,$t,!1,null,"d57d1126",null)),Mt=Pt.exports,It={name:"DatasetDownloadTabs",components:{DSlider:Dt,DTags:Mt},data:function(){return{items:[{text:"Google диск",tab:0},{text:"URL",tab:1}],active:0}},computed:Object(r["a"])(Object(r["a"])({},Object(o["c"])("createDataset",["getFilesSource","getProject"])),{},{project:{set:function(t){this.setProject(t)},get:function(){return this.getProject}}}),methods:Object(r["a"])(Object(r["a"])({},Object(o["b"])("createDataset",["getDatasetSources","setSelectSource"])),{},{isActive:function(t){return this.project.active===t},onSelect:function(t){this.setSelectSource(t)},onTabs:function(t){this.project.active=t,this.project.google="",this.project.url="",this.setSelectSource({})}})},At=It,Bt=(n("4a73"),Object(h["a"])(At,kt,St,!1,null,"d62906f0",null)),Xt=Bt.exports,Nt=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"dataset-helpers"},[t._m(0),n("div",{staticClass:"dataset-helpers-list mt-5 ml-5"},[n("ul",{staticClass:"dataset-helpers-list__menu"},[n("li",{staticClass:"flex align-center"},[n("span",{staticClass:"panel-text-gray"},[t._v("1. Выбрать входные чистые данные")]),n("d-svg",{attrs:{name:"sloy-start-add"}})],1),n("li",{staticClass:"flex align-center"},[n("span",{staticClass:"panel-text-gray"},[t._v("2. Задать обработчик")]),n("d-svg",{attrs:{name:"sloy-middle-add"}})],1),n("li",{staticClass:"flex align-center"},[n("span",{staticClass:"panel-text-gray"},[t._v("3. Соединить все в инпут вход один")]),n("d-svg",{attrs:{name:"sloy-end-add"}})],1)])])])},Yt=[function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"dataset-helpers-header"},[n("p",[t._v("Для создания датасета необходимо:")])])}],Tt={name:"DatasetHelpers"},Vt=Tt,Kt=(n("2855"),Object(h["a"])(Vt,Nt,Yt,!1,null,"3b890230",null)),Rt=Kt.exports,Ut=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"d-pagination"},[n("button",{staticClass:"d-pagination__btn",attrs:{disabled:t.isDisabled},on:{click:function(e){return t.$emit("prev",e)}}},[n("d-svg",{attrs:{name:"arrow-carret-left-longer-big"}})],1),n("div",{staticClass:"d-pagination__inner"},[n("div",{staticClass:"d-pagination__list"},t._l(t.list.length,(function(e){return n("div",{key:e,class:["d-pagination__item",{"d-pagination__item--active":t.isActive(e)}]})})),0),n("div",{staticClass:"d-pagination__title"},[n("span",[t._v(t._s(t.getTitle))])])]),n("d-button",{staticStyle:{width:"40%"},attrs:{color:"secondary",direction:"left",text:"Далее",disabled:t.isStatus},on:{click:function(e){return t.$emit("next",e)}}})],1)},Ht=[],Wt={name:"DPagination",props:{list:{type:Array,default:function(){return[]}},value:{type:Number,default:0}},data:function(){return{}},computed:Object(r["a"])(Object(r["a"])({},Object(o["c"])({project:"createDataset/getProject"})),{},{isDisabled:function(){return 1===this.value},isStatus:function(){var t,e,n,i;return!(1!==this.value||(null!==this&&void 0!==this&&null!==(t=this.project)&&void 0!==t&&t.url||null!==this&&void 0!==this&&null!==(e=this.project)&&void 0!==e&&e.google)&&null!==this&&void 0!==this&&null!==(n=this.project)&&void 0!==n&&n.name&&null!==this&&void 0!==this&&null!==(i=this.project)&&void 0!==i&&i.version)},getTitle:function(){var t=this;return this.list.find((function(e){return e.id===t.value})).title}}),methods:{isActive:function(t){return this.value===t}}},Jt=Wt,zt=(n("f96a"),Object(h["a"])(Jt,Ut,Ht,!1,null,null,null)),Gt=zt.exports,qt={components:{StateOne:Xt,StateTwo:X,StateThree:yt,Pagination:Gt,StateFour:Rt},data:function(){return{debounce:null,list:[{id:1,title:"Данные",component:"state-one"},{id:2,title:"Предпросмотр",component:"state-two"},{id:3,title:"Input",component:"state-three"},{id:4,title:"Output",component:"state-three"},{id:5,title:"Завершение",component:"state-four"}]}},computed:Object(r["a"])(Object(r["a"])({},Object(o["c"])({select:"createDataset/getSelectSource",getPagination:"createDataset/getPagination"})),{},{getComp:function(){var t=this;return this.list.find((function(e){return e.id===t.value}))},value:{set:function(t){this.setPagination(t)},get:function(){return this.getPagination}}}),methods:Object(r["a"])(Object(r["a"])({},Object(o["b"])({setSourceLoad:"createDataset/setSourceLoad",sourceLoadProgress:"createDataset/sourceLoadProgress",setPagination:"createDataset/setPagination",setOverlay:"settings/setOverlay"})),{},{onNext:function(){1===this.value&&this.onDownload(),this.value<this.list.length&&(this.value=this.value+1)},onPrev:function(){this.value>1&&(this.value=this.value-1)},onProgress:function(){var t=this;return Object(E["a"])(regeneratorRuntime.mark((function e(){var n,i;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.sourceLoadProgress();case 2:i=e.sent,null!==i&&void 0!==i&&null!==(n=i.data)&&void 0!==n&&n.finished?t.setOverlay(!1):t.debounce(!0);case 4:case"end":return e.stop()}}),e)})))()},onDownload:function(){var t=this;return Object(E["a"])(regeneratorRuntime.mark((function e(){var n,i,s,a;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return n=t.select,i=n.mode,s=n.value,e.next=3,t.setSourceLoad({mode:i,value:s});case 3:a=e.sent,a&&(t.setOverlay(!0),t.debounce(!0));case 5:case"end":return e.stop()}}),e)})))()}}),created:function(){var t=this;this.debounce=Object($["a"])((function(e){e&&t.onProgress()}),1e3)},beforeDestroy:function(){this.debounce(!1)}},Qt=qt,Zt=(n("0052"),Object(h["a"])(Qt,w,D,!1,null,null,null)),te=Zt.exports,ee={name:"Datasets",components:{Toolbar:v,Blocks:O,Params:te},data:function(){return{state:1}},methods:{onToolbar:function(t){console.log(t)}}},ne=ee,ie=(n("506e"),Object(h["a"])(ne,i,s,!1,null,null,null));e["default"]=ie.exports},"1ddc":function(t,e,n){"use strict";n("5630")},2855:function(t,e,n){"use strict";n("fab5")},"29b3":function(t,e,n){},"305f":function(t,e,n){},"37d0":function(t,e,n){},3916:function(t,e,n){"use strict";n("773a")},"3e0e":function(t,e,n){},4142:function(t,e,n){},4328:function(t,e,n){"use strict";var i=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"t-table"},[n("div",{staticClass:"t-table__header"}),n("div",{staticClass:"t-table__data"},[n("div",{staticClass:"t-table__col",style:{padding:"1px 0"}},t._l(6,(function(e,i){return n("div",{key:"idx_r_"+e,staticClass:"t-table__row"},[t._v(t._s(i||""))])})),0),n("div",{staticClass:"t-table__border"},t._l(t.origTable,(function(e,i){return n("div",{key:"row_"+i,class:["t-table__col",{"t-table__col--active":e.active}],style:t.getColor,on:{click:function(n){return t.select(e,n)}}},[t._l(e,(function(e,i){return[i<=5?n("div",{key:"item_"+i,staticClass:"t-table__row"},[t._v(" "+t._s(e)+" ")]):t._e()]})),n("div",{staticClass:"t-table__select"},t._l(t.all(e),(function(t,e){return n("div",{key:"all"+e,staticClass:"t-table__circle",style:t})})),0)],2)})),0)]),n("div",{directives:[{name:"click-outside",rawName:"v-click-outside",value:t.outside,expression:"outside"}],staticClass:"t-table__footer"},[n("span",[t._v(t._s(t.label))]),n("div",{staticClass:"t-table__footer--btn",on:{click:function(e){t.show=!0}}},[n("i",{staticClass:"t-icon icon-file-dot"})]),n("div",{directives:[{name:"show",rawName:"v-show",value:t.show,expression:"show"}],staticClass:"t-table__dropdown"},t._l(t.items,(function(e,i){var s=e.icon,a=e.event;return n("div",{key:"icon"+i,staticClass:"t-table__dropdown--item",on:{click:function(e){t.$emit("event",{label:t.label,event:a}),t.show=!1}}},[n("i",{class:["t-icon",s]})])})),0)])])},s=[],a=n("3835"),l=(n("a9e3"),n("7db0"),n("d81d"),n("caad"),n("2532"),n("4de4"),{name:"CardTable",props:{label:String,type:String,id:Number,cover:String,table:Array,value:String},data:function(){return{show:!1,items:[{icon:"icon-deploy-remove",event:"remove"}]}},computed:{getColor:function(){var t=this.handlers.find((function(t){return t.active}));return{borderColor:(null===t||void 0===t?void 0:t.color)||""}},origTable:function(){var t=this;return this.table.map((function(e){return e.active=t.selected.includes(e[0]),e}))},handlers:{set:function(t){this.$store.dispatch("tables/setHandlers",t)},get:function(){return this.$store.getters["tables/getHandlers"]}},selected:{set:function(t){var e=this;this.handlers=this.handlers.map((function(n){return n.active&&n.table[e.label]&&(n.table[e.label]=t),n}))},get:function(){var t=this.handlers.find((function(t){return t.active}));return t?t.table[this.label]:[]}}},methods:{outside:function(){this.show=!1},all:function(t){var e=this,n=Object(a["a"])(t,1),i=n[0];return this.handlers.filter((function(t){return t.table[e.label].includes(i)})).map((function(t,e){return{backgroundColor:t.color,top:-3*e+"px"}}))},select:function(t){var e=Object(a["a"])(t,1),n=e[0];this.selected.find((function(t){return t===n}))?this.selected=this.selected.filter((function(t){return t!==n})):this.selected.push(n)}}}),r=l,o=(n("ad58"),n("2877")),c=Object(o["a"])(r,i,s,!1,null,"137e7a5b",null);e["a"]=c.exports},"4a73":function(t,e,n){"use strict";n("29b3")},"4b70":function(t,e,n){"use strict";n("0213")},"4fce":function(t,e,n){},"506e":function(t,e,n){"use strict";n("be11")},5630:function(t,e,n){},6280:function(t,e,n){},"6d05":function(t,e,n){"use strict";n("4fce")},"730f":function(t,e,n){},"773a":function(t,e,n){},"7a68":function(t,e,n){"use strict";n("809b")},"809b":function(t,e,n){},8639:function(t,e,n){"use strict";n("730f")},a1a3:function(t,e,n){"use strict";n("4142")},ad58:function(t,e,n){"use strict";n("305f")},b0da:function(t,e,n){"use strict";n("f1cf")},b79b:function(t,e,n){"use strict";var i=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{directives:[{name:"click-outside",rawName:"v-click-outside",value:t.outside,expression:"outside"}],staticClass:"t-card-file",style:t.bc},[t.id?n("div",{staticClass:"t-card-file__header",style:t.bg},[t._v(t._s(t.title))]):t._e(),n("div",{class:["t-card-file__body","icon-file-"+t.type],style:t.img}),n("div",{staticClass:"t-card-file__footer"},[n("div",{staticClass:"t-card-file__footer--label"},[t._v(t._s(t.label))]),n("div",{staticClass:"t-card-file__footer--btn",on:{click:function(e){t.show=!0}}},[n("i",{staticClass:"t-icon icon-file-dot"})])]),n("div",{directives:[{name:"show",rawName:"v-show",value:t.show,expression:"show"}],staticClass:"t-card-file__dropdown"},t._l(t.items,(function(e,i){var s=e.icon,a=e.event;return n("div",{key:"icon"+i,staticClass:"t-card-file__dropdown--item",on:{click:function(e){t.$emit("event",{label:t.label,event:a}),t.show=!1}}},[n("i",{class:["t-icon",s]})])})),0)])},s=[],a=(n("a9e3"),{name:"t-card-file",props:{label:String,type:String,id:Number,cover:String},data:function(){return{show:!1,items:[{icon:"icon-deploy-remove",event:"remove"}]}},computed:{img:function(){return this.cover?{backgroundImage:"url('".concat(this.cover,"')"),backgroundPosition:"center",backgroundSize:"cover"}:{}},selectInputData:function(){return this.$store.getters["datasets/getInputDataByID"](this.id)||{}},title:function(){var t=this.selectInputData;return"input"===t.layer?"Входные данные "+t.id:"Выходные данные "+t.id},color:function(){return this.selectInputData.color||""},bg:function(){return{backgroundColor:this.id?this.color:""}},bc:function(){return{borderColor:this.id?this.color:""}}},methods:{outside:function(){this.show=!1}}}),l=a,r=(n("1ddc"),n("2877")),o=Object(r["a"])(l,i,s,!1,null,"8cd777c2",null);e["a"]=o.exports},be11:function(t,e,n){},f1cf:function(t,e,n){},f244:function(t,e,n){"use strict";n("3e0e")},f96a:function(t,e,n){"use strict";n("6280")},fab5:function(t,e,n){}}]);