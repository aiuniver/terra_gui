(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-a615f62a"],{"0052":function(t,e,n){"use strict";n("37d0")},"0213":function(t,e,n){},"0660":function(t,e,n){"use strict";n.r(e);var i=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"page-create"},[n("div",{staticClass:"page-create__toolbar"},[n("Toolbar",{on:{action:t.onToolbar}})],1),n("div",{staticClass:"page-create__main"},[n("Blocks")],1),n("div",{staticClass:"page-create__params"},[n("Params",{model:{value:t.state,callback:function(e){t.state=e},expression:"state"}})],1)])},s=[],a=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"d-create-toolbar"},[n("div",{staticClass:"d-create-toolbar__workspace mt-10"},[t._l(t.filter,(function(e){var i=e.color,s=e.type,a=e.typeBlock;return[n("div",{key:i,staticClass:"d-create-toolbar__item d-create-toolbar__item--no-hover",on:{click:function(e){return t.onAdd({type:s})}}},[n("d-icon-layer",t._b({},"d-icon-layer",{color:i,type:s,typeBlock:a},!1))],1)]}))],2)])},r=[],l=n("5530"),o=(n("caad"),n("7db0"),n("4de4"),n("2532"),n("2f62")),c=n("9e1e"),u={name:"d-toolbar",data:function(){return{types:c["e"],toolbar:[{id:3,filter:["data","handler","input"]},{id:4,filter:["data","handler","output"]}]}},computed:Object(l["a"])(Object(l["a"])({},Object(o["c"])({getPagination:"createDataset/getPagination"})),{},{isActive:function(){return Boolean([3,4].includes(this.getPagination))},filter:function(){var t,e=this,n=(null===(t=this.toolbar.find((function(t){return t.id===e.getPagination})))||void 0===t?void 0:t.filter)||[];return this.types.filter((function(t){return n.includes(t.type)}))}}),methods:Object(l["a"])(Object(l["a"])({},Object(o["b"])({add:"create/add"})),{},{onAdd:function(t){this.isActive&&this.add(t)}})},d=u,h=(n("6d05"),n("2877")),f=Object(h["a"])(d,a,r,!1,null,"0ade1807",null),v=f.exports,p=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{key:t.key,staticClass:"blocks",on:{contextmenu:t.contextmenu}},[t.isActive?t._e():n("div",{staticClass:"blocks__overlay"}),n("net",{staticClass:"blocks__center",attrs:{x:t.centerX,y:t.centerY,scale:t.scale}}),n("Link",{staticClass:"blocks__lines",attrs:{lines:t.lines}}),t._l(t.blocks,(function(e){return n("Block",t._b({key:e.id,ref:"block_"+e.id,refInFor:!0,attrs:{options:t.optionsForChild,linkingCheck:t.tempLink},on:{linkingStart:function(n){return t.linkingStart(e,n)},linkingStop:function(n){return t.linkingStop(e,n)},linkingBreak:function(n){return t.linkingBreak(e,n)},select:function(n){return t.blockSelect(e)},position:function(e){return t.position(e)}}},"Block",e,!1))}))],2)},g=[],m=n("3835"),b=n("2909"),_=n("b85c"),k=(n("159b"),n("99af"),n("d81d"),n("c8fb")),y=n("00b9"),j=n("2523"),C={name:"Blocks",components:{Block:y["default"],Link:j["default"]},props:{},data:function(){return{key:1,save:[],menu:{},dragging:!1,centerX:0,centerY:0,scale:1,tempLink:null,hasDragged:!1,mouseX:0,mouseY:0,lastMouseX:0,lastMouseY:0,minScale:.2,maxScale:5,linking:!1,linkStart:null,inputSlotClassName:"inputSlot"}},computed:Object(l["a"])(Object(l["a"])({},Object(o["c"])({getKeyEvent:"create/getKeyEvent",blocks:"create/getBlocks",links:"create/getLinks",getSelected:"create/getSelected",pagination:"createDataset/getPagination"})),{},{isActive:function(){return Boolean([3,4].includes(this.pagination))},keyEvent:{set:function(t){this.setKeyEvent(t)},get:function(){return this.getKeyEvent}},optionsForChild:function(){return{scale:this.scale,x:this.centerX,y:this.centerY}},container:function(){return{centerX:this.centerX,centerY:this.centerY,scale:this.scale}},lines:function(){var t,e=this,n=[],i=Object(_["a"])(this.links);try{var s=function(){var i=t.value,s=e.blocks.find((function(t){var e=t.id;return e===i.originID})),a=e.blocks.find((function(t){var e=t.id;return e===i.targetID}));if(!s||!a||s.id===a.id)return console.warn("Remove invalid link",i),e.removeLink(i.id),"continue";var r=e.getConnectionPos(s,i.originSlot,!1),l=e.getConnectionPos(a,i.targetSlot,!0);if(!r||!l)return console.log("Remove invalid link (slot not exist)",i),e.removeLink(i.id),"continue";var o=r.x,c=r.y,u=l.x,d=l.y;n.push({x1:o,y1:c,x2:u,y2:d,slot:i.originSlot,scale:e.scale,style:{stroke:"rgb(101, 124, 244)",strokeWidth:2*e.scale,fill:"none",zIndex:999},outlineStyle:{stroke:"#666",strokeWidth:2*e.scale,strokeOpacity:.6,fill:"none",zIndex:999}})};for(i.s();!(t=i.n()).done;)s()}catch(a){i.e(a)}finally{i.f()}return this.tempLink&&n.push(this.tempLink),n}}),methods:Object(l["a"])(Object(l["a"])({},Object(o["b"])("create",["add","cloneAll","align","distance","remove","clone","select","setKeyEvent","deselect","position","update","addLink","updateLink","removeLink"])),{},{event:function(t){this.menu={},console.log(t),"add"===t&&this.add({}),"delete"===t&&this.remove(),"clone"===t&&this.cloneAll(),"left"===t&&this.align("ArrowLeft"),"right"===t&&this.align("ArrowRight"),"up"===t&&this.align("ArrowUp"),"down"===t&&this.align("ArrowDown"),"center"===t&&this.align("center"),"vertical"===t&&this.distance("vertical"),"horizon"===t&&this.distance("horizon"),"select"===t&&this.deselect(!0),this.key+=1},contextmenu:function(t){console.log(t.clientX,t.clientY)},handleMauseOver:function(t){this.mouseIsOver="mouseenter"===t.type},keyup:function(t){var e=this;this.keyEvent=t;var n=t.code,i=t.ctrlKey,s=t.shiftKey,a=this.mouseIsOver;"keyup"===t.type&&(a&&"Delete"===n&&this.remove(),a&&"KeyA"===n&&i&&this.deselect(!0),a&&["ArrowLeft","ArrowRight","ArrowDown","ArrowUp"].includes(n)&&i&&!s&&(this.align(n),this.key+=1),a&&["ArrowUp"].includes(n)&&i&&s&&(this.distance("vertical"),this.key+=1),a&&"KeyC"===n&&i&&(this.save=this.blocks.filter((function(t){return t.selected}))),a&&"KeyX"===n&&i&&(this.save=JSON.parse(JSON.stringify(this.blocks.filter((function(t){return t.selected})))),console.log(this.save),this.remove()),a&&"KeyV"===n&&i&&(this.deselect(),this.save.forEach((function(t){e.clone(t)}))))},handleMove:function(t){var e=Object(k["b"])(this.$el,t);if(this.mouseX=e.x,this.mouseY=e.y,this.dragging){console.log("handleMove");var n=this.mouseX-this.lastMouseX,i=this.mouseY-this.lastMouseY;this.lastMouseX=this.mouseX,this.lastMouseY=this.mouseY,this.centerX+=n,this.centerY+=i,this.hasDragged=!0}if(this.linking&&this.linkStart){var s=this.getConnectionPos(this.linkStart.block,this.linkStart.slot,!1);this.tempLink={x1:s.x,y1:s.y,x2:this.mouseX,y2:this.mouseY,slot:this.linkStart.slot,style:{stroke:"#8f8f8f",strokeWidth:2*this.scale,fill:"none"}}}},handleDown:function(t){console.log("handleDown"),console.log(t);var e=t.target||t.srcElement;if((e===this.$el||e.matches("svg, svg *"))&&1===t.which){var n;this.dragging=!0;var i=Object(k["b"])(this.$el,t);this.mouseX=i.x,this.mouseY=i.y,this.lastMouseX=this.mouseX,this.lastMouseY=this.mouseY,null!==(n=this.keyEvent)&&void 0!==n&&n.ctrlKey||this.deselect(),t.preventDefault&&t.preventDefault()}},handleUp:function(t){console.log("handleUp");var e=t.target||t.srcElement;this.dragging&&(this.dragging=!1,this.hasDragged&&(this.hasDragged=!1)),!this.$el.contains(e)||"string"===typeof e.className&&e.className.includes(this.inputSlotClassName)||(this.linking=!1,this.tempLink=null,this.linkStart=null)},handleWheel:function(t){var e=t.target||t.srcElement;if(this.$el.contains(e)){var n=Math.pow(1.1,-.01*t.deltaY);if(this.scale*=n,this.scale<this.minScale)return void(this.scale=this.minScale);if(this.scale>this.maxScale)return void(this.scale=this.maxScale);var i=(this.mouseX-this.centerX)*(n-1),s=(this.mouseY-this.centerY)*(n-1);this.centerX-=i,this.centerY-=s}},getConnectionPos:function(t,e,n){var i,s,a;if(t&&-1!==e){var r=0,l=0;r+=t.position[0],l+=t.position[1];var o=(null===(i=this.$refs)||void 0===i||null===(s=i["block_"+t.id])||void 0===s||null===(a=s[0])||void 0===a?void 0:a.getH())||{},c=o.width,u=void 0===c?150:c,d=o.heigth,h=void 0===d?60:d;return n?(r+=u/2,l+=-3):(r+=u/2,l+=h),r*=this.scale,l*=this.scale,r+=this.centerX,l+=this.centerY,{x:r,y:l}}},linkingStart:function(t,e){console.log("linkingStart"),this.linkStart={block:t,slot:e};var n=this.getConnectionPos(t,e,!1);this.tempLink={x1:n.x,y1:n.y,x2:this.mouseX,y2:this.mouseY,style:{stroke:"#8f8f8f",strokeWidth:2*this.scale,fill:"none"}},this.linking=!0},linkingStop:function(t,e){if(console.log("linkingStop"),this.linkStart&&t&&e>-1){var n=this.linkStart,i=n.slot,s=n.block.id,a=t.id,r=e,l=this.links.filter((function(t){return!(t.targetID===a&&t.targetSlot===r&&t.originID===s&&t.originSlot===i)&&!(t.targetID===s&&t.originID===a||t.originID===s&&t.targetID===a)}));this.updateLink(l);var o=Math.max.apply(Math,[0].concat(Object(b["a"])(this.links.map((function(t){return t.id})))));if(this.linkStart.block.id!==t.id){var c=this.linkStart.block.id,u=this.linkStart.slot,d=t.id,h=e;this.addLink({id:o+1,originID:c,originSlot:u,targetID:d,targetSlot:h}),this.updateModel()}}this.linking=!1,this.tempLink=null,this.linkStart=null},linkingBreak:function(t,e){if(console.log("linkingBreak"),t&&e>-1){var n=this.links.find((function(n){var i=n.targetID,s=n.targetSlot;return i===t.id&&s===e}));if(n){var i=this.blocks.find((function(t){var e=t.id;return e===n.originID})),s=this.links.filter((function(n){var i=n.targetID,s=n.targetSlot;return!(i===t.id&&s===e)}));this.updateLink(s),this.linkingStart(i,n.originSlot),this.updateModel()}}},position:function(t){var e=t.left,n=t.top;if(!this.keyEvent.ctrlKey){var i=this.blocks.map((function(t){var i=Object(m["a"])(t.position,2),s=i[0],a=i[1],r=t.selected?[s+e,a+n]:[s,a];return Object(l["a"])(Object(l["a"])({},t),{},{position:r})}));this.update(i)}},blockSelect:function(t){var e=t.id,n=t.selected;console.log(e,n),n&&!this.keyEvent.ctrlKey||this.select({id:e})},updateModel:function(){}}),mounted:function(){var t=document.documentElement;this.$el.addEventListener("mouseenter",this.handleMauseOver),this.$el.addEventListener("mouseleave",this.handleMauseOver),t.addEventListener("keydown",this.keyup),t.addEventListener("keyup",this.keyup),t.addEventListener("mousemove",this.handleMove,!0),t.addEventListener("mousedown",this.handleDown,!0),t.addEventListener("mouseup",this.handleUp,!0),t.addEventListener("wheel",this.handleWheel,!0),this.centerX=this.$el.clientWidth/2,this.centerY=this.$el.clientHeight/2},beforeDestroy:function(){var t=document.documentElement;this.$el.removeEventListener("mouseenter",this.handleMauseOver),this.$el.removeEventListener("mouseleave",this.handleMauseOver),t.removeEventListener("keydown",this.keyup),t.removeEventListener("keyup",this.keyup),t.removeEventListener("mousemove",this.handleMove,!0),t.removeEventListener("mousedown",this.handleDown,!0),t.removeEventListener("mouseup",this.handleUp,!0),t.removeEventListener("wheel",this.handleWheel,!0)}},S=C,O=(n("8639"),Object(h["a"])(S,p,g,!1,null,null,null)),x=O.exports,D=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"params"},[n("div",{staticClass:"params__body"},[n("div",{staticClass:"params__header mb-2"},[t._v(t._s(t.getComp.title))]),n("scrollbar",[n("div",{staticClass:"params__inner"},[n(t.getComp.component,{tag:"component",attrs:{state:t.value}})],1)])],1),n("div",{staticClass:"params__footer"},[n("Pagination",{attrs:{value:t.value,list:t.list},on:{next:t.onNext,prev:t.onPrev,create:t.onCreate}})],1)])},w=[],E=n("1da1"),F=(n("96cf"),n("eb4c")),P=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"state-two"},[n("div",{staticClass:"state-two__cards"},[n("Cards",[t._l(t.getFiles,(function(e,i){return["folder"===e.type?n("CardFile",t._b({key:"files_"+i},"CardFile",e,!1)):t._e(),"table"===e.type?n("CardTable",t._b({key:"files_"+i},"CardTable",e,!1)):t._e()]}))],2)],1),n("div",{staticClass:"state-two__title mb-2"},[t._v("Файлы")]),n("div",{staticClass:"state-two__files"},[n("files-menu",{model:{value:t.filesSource,callback:function(e){t.filesSource=e},expression:"filesSource"}})],1)])},$=[],L=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"t-cards",style:t.style,on:{wheel:function(e){return e.preventDefault(),t.wheel.apply(null,arguments)}}},[n("scrollbar",{ref:"scrollCards",attrs:{ops:t.ops}},[n("div",{staticClass:"t-cards__items"},[n("div",{staticClass:"t-cards__items--item"},[t._t("default")],2)])])],1)},B=[],A={data:function(){return{wight:0,ops:{scrollPanel:{scrollingX:!0,scrollingY:!1}}}},computed:{style:function(){return{wight:this.wight+"px"}}},mounted:function(){this.wight=this.$el.clientWidth,console.log(this.$el.clientWidth)},methods:{wheel:function(t){t.stopPropagation(),this.$refs.scrollCards.scrollBy({dx:t.wheelDelta},200)}}},M=A,X=(n("60d2"),Object(h["a"])(M,L,B,!1,null,"4c2ae08b",null)),I=X.exports,T=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{directives:[{name:"click-outside",rawName:"v-click-outside",value:t.outside,expression:"outside"}],staticClass:"t-card-file",style:t.bc},[t.id?n("div",{staticClass:"t-card-file__header",style:t.bg},[t._v(t._s(t.title))]):t._e(),n("div",{class:["t-card-file__body","icon-file-"+t.type],style:t.img}),n("div",{staticClass:"t-card-file__footer"},[n("div",{staticClass:"t-card-file__footer--label"},[t._v(t._s(t.label))])])])},Y=[],V=(n("a9e3"),{name:"t-card-file",props:{label:String,type:String,id:Number,cover:String},data:function(){return{show:!1,items:[{icon:"icon-deploy-remove",event:"remove"}]}},computed:{img:function(){return this.cover?{backgroundImage:"url('".concat(this.cover,"')"),backgroundPosition:"center",backgroundSize:"cover"}:{}},selectInputData:function(){return this.$store.getters["datasets/getInputDataByID"](this.id)||{}},title:function(){var t=this.selectInputData;return"input"===t.layer?"Входные данные "+t.id:"Выходные данные "+t.id},color:function(){return this.selectInputData.color||""},bg:function(){return{backgroundColor:this.id?this.color:""}},bc:function(){return{borderColor:this.id?this.color:""}}},methods:{outside:function(){this.show=!1}}}),N=V,K=(n("f02a"),Object(h["a"])(N,T,Y,!1,null,"029a1084",null)),R=K.exports,U=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"t-table"},[n("div",{staticClass:"t-table__header"}),n("div",{staticClass:"t-table__data"},[n("div",{staticClass:"t-table__col",style:{padding:"1px 0"}},t._l(6,(function(e,i){return n("div",{key:"idx_r_"+e,staticClass:"t-table__row"},[t._v(t._s(i||""))])})),0),n("div",{staticClass:"t-table__border"},t._l(t.origTable,(function(e,i){return n("div",{key:"row_"+i,class:["t-table__col",{"t-table__col--active":e.active}],style:t.getColor,on:{click:function(n){return t.select(e,n)}}},[t._l(e,(function(e,i){return[i<=5?n("div",{key:"item_"+i,staticClass:"t-table__row"},[t._v(" "+t._s(e)+" ")]):t._e()]})),n("div",{staticClass:"t-table__select"},t._l(t.all(e),(function(t,e){return n("div",{key:"all"+e,staticClass:"t-table__circle",style:t})})),0)],2)})),0)])])},W=[],H={name:"CardTable",props:{label:String,type:String,id:Number,cover:String,table:Array,value:String},data:function(){return{show:!1,items:[{icon:"icon-deploy-remove",event:"remove"}]}},computed:{getColor:function(){var t=this.handlers.find((function(t){return t.active}));return{borderColor:(null===t||void 0===t?void 0:t.color)||""}},origTable:function(){var t=this;return this.table.map((function(e){return e.active=t.selected.includes(e[0]),e}))},handlers:{set:function(t){this.$store.dispatch("tables/setHandlers",t)},get:function(){return this.$store.getters["tables/getHandlers"]}},selected:{set:function(t){var e=this;this.handlers=this.handlers.map((function(n){return n.active&&n.table[e.label]&&(n.table[e.label]=t),n}))},get:function(){var t=this.handlers.find((function(t){return t.active}));return t?t.table[this.label]:[]}}},methods:{outside:function(){this.show=!1},all:function(t){var e=this,n=Object(m["a"])(t,1),i=n[0];return this.handlers.filter((function(t){return t.table[e.label].includes(i)})).map((function(t,e){return{backgroundColor:t.color,top:-3*e+"px"}}))},select:function(t){var e=Object(m["a"])(t,1),n=e[0];this.selected.find((function(t){return t===n}))?this.selected=this.selected.filter((function(t){return t!==n})):this.selected.push(n)}}},z=H,G=(n("1ec8"),Object(h["a"])(z,U,W,!1,null,"764fb40b",null)),J=G.exports,q={components:{CardFile:R,CardTable:J,Cards:I},props:{list:{type:Array,default:function(){return[]}}},data:function(){return{ops:{scrollPanel:{scrollingX:!0,scrollingY:!1}}}},computed:Object(l["a"])(Object(l["a"])({},Object(o["c"])({getFileManager:"createDataset/getFileManager",getFiles:"createDataset/getFiles"})),{},{filesSource:{set:function(t){this.$store.dispatch("datasets/setFilesSource",t)},get:function(){return this.getFileManager}}}),methods:{}},Q=q,Z=(n("8c41"),Object(h["a"])(Q,P,$,!1,null,"0392d2e5",null)),tt=Z.exports,et=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"panel-settings"},["data"===t.type?n("SettingData",{attrs:{selected:t.selected}}):t._e(),"handler"===t.type?n("SettingHandler",{attrs:{selected:t.selected,forms:t.forms}}):t._e(),["output","input"].includes(t.type)?n("SettingOutput",{attrs:{selected:t.selected}}):t._e(),t.type?t._e():n("SettingEmpty")],1)},nt=[],it=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"panel-settings-empty"},[n("div",{staticClass:"panel-settings-empty__title mb-4"},[t._v("Для создания датасета необходимо:")]),n("div",{staticClass:"panel-settings-empty__info pl-4"},[n("ul",[n("li",[n("span",[t._v("1. Выбрать входные чистые данные")]),n("d-svg",{attrs:{name:"sloy-start-add"}})],1),n("li",[n("span",[t._v("2. Задать обработчик")]),n("d-svg",{attrs:{name:"sloy-middle-add"}})],1),n("li",[n("span",[t._v("3. Соединить все в инпут вход один")]),n("d-svg",{attrs:{name:"sloy-handler-add"}})],1)])])])},st=[],at={},rt=at,lt=(n("f244"),Object(h["a"])(rt,it,st,!1,null,null,null)),ot=lt.exports,ct=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"panel-input"},[n("div",{staticClass:"panel-input__view"},[n("Cards",[t._l(t.getFile,(function(e,i){return["folder"===e.type?n("CardFile",t._b({key:"files_"+i},"CardFile",e,!1)):t._e(),"table"===e.type?n("CardTable",t._b({key:"files_"+i},"CardTable",e,!1)):t._e()]}))],2)],1),n("div",{staticClass:"panel-input__forms"},[n("t-field",{attrs:{label:"Входные данные"}},[n("d-multi-select",{attrs:{value:t.getValueData,placeholder:"Данные",list:t.listFiles},on:{change:t.onFile}})],1),t._v(" "+t._s(t.getParametrs)+" ")],1)])},ut=[],dt={components:{CardFile:R,CardTable:J,Cards:I},props:{selected:{type:Object,default:function(){return{}}}},data:function(){return{sel:{}}},computed:Object(l["a"])(Object(l["a"])({},Object(o["c"])({getFiles:"createDataset/getFiles",getFileManager:"createDataset/getFileManager",editBlock:"create/editBlock"})),{},{listFiles:function(){return this.getFileManager.map((function(t){return{label:t.title,value:t.value}}))},getFile:function(){var t=this;return this.getFiles.filter((function(e){return t.items.includes(e.label)}))},getValueData:function(){var t,e;return(null===(t=this.selected)||void 0===t||null===(e=t.parameters)||void 0===e?void 0:e.items)||[]},getParametrs:function(){var t;return(null===this||void 0===this||null===(t=this.selected)||void 0===t?void 0:t.parameters)||{}},id:function(){return this.selected.id},items:function(){var t,e;return(null===(t=this.selected)||void 0===t||null===(e=t.parameters)||void 0===e?void 0:e.items)||[]}}),methods:Object(l["a"])(Object(l["a"])({},Object(o["b"])({setParameters:"create/setParameters"})),{},{onFile:function(t){var e=this.items;e.includes(t.label)?e=e.filter((function(e){return e!==t.label})):e.push(t.label);var n=Object(l["a"])(Object(l["a"])({},this.parameters),{},{items:e});this.setParameters({id:this.id,parameters:n})}})},ht=dt,ft=(n("ad4b"),Object(h["a"])(ht,ct,ut,!1,null,null,null)),vt=ft.exports,pt=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",[t._l(t.forms,(function(e,i){return[n("t-auto-field-handler",t._b({key:t.id+i,attrs:{parameters:t.parameters,idKey:"key_"+i,root:""},on:{change:t.onChange}},"t-auto-field-handler",e,!1))]})),t._v(" "+t._s(t.selected)+" ")],2)},gt=[],mt=n("ade3"),bt=(n("b0c0"),{components:{},props:{selected:{type:Object,default:function(){return{}}},forms:{type:Array,default:function(){return[]}}},data:function(){return{debounce:null}},computed:{parameters:function(){var t;return(null===this||void 0===this||null===(t=this.selected)||void 0===t?void 0:t.parameters)||{}},id:function(){return this.selected.id}},methods:Object(l["a"])(Object(l["a"])({},Object(o["b"])({setParameters:"create/setParameters"})),{},{onChange:function(t){console.log(t);var e=Object(l["a"])(Object(l["a"])({},this.parameters),{},Object(mt["a"])({},t.name,t.value));this.setParameters({id:this.id,parameters:e})}}),created:function(){console.log("created"),this.debounce=F["a"]}}),_t=bt,kt=Object(h["a"])(_t,pt,gt,!1,null,null,null),yt=kt.exports,jt=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",[n("t-field",[n("d-input-text",{attrs:{value:t.value},on:{change:t.onChange}})],1)],1)},Ct=[],St={components:{},props:{selected:{type:Object,default:function(){return{}}}},computed:Object(l["a"])(Object(l["a"])({},Object(o["c"])({getFileManager:"createDataset/getFileManager",getDefault:"create/getDefault"})),{},{parameters:function(){var t;return(null===this||void 0===this||null===(t=this.selected)||void 0===t?void 0:t.parameters)||{}},value:function(){var t;return(null===(t=this.selected)||void 0===t?void 0:t.name)||""}}),methods:Object(l["a"])(Object(l["a"])({},Object(o["b"])({setParameters:"create/setParameters",editBlock:"create/editBlock"})),{},{onChange:function(t){var e=t.value,n=Object(l["a"])({},this.selected);n.name=e,this.editBlock(n)}})},Ot=St,xt=Object(h["a"])(Ot,jt,Ct,!1,null,null,null),Dt=xt.exports,wt={name:"DatasetSettings",components:{SettingEmpty:ot,SettingData:vt,SettingHandler:yt,SettingOutput:Dt},props:{state:{type:Number,default:1}},data:function(){return{show:!0,ops:{scrollPanel:{scrollingX:!0,scrollingY:!1}}}},computed:Object(l["a"])(Object(l["a"])({},Object(o["c"])({getSelected:"create/getSelected",getHandler:"createDataset/getHandler"})),{},{forms:function(){var t,e=3===this.state?"inputs":"outputs";return(null===(t=this.getHandler)||void 0===t?void 0:t[e])||[]},selected:function(){var t=this.getSelected.filter((function(t){return t.selected})).length;return 1===t?this.getSelected.find((function(t){return t.selected})):{}},type:function(){var t;return(null===this||void 0===this||null===(t=this.selected)||void 0===t?void 0:t.type)||""}}),methods:{}},Et=wt,Ft=(n("4b70"),Object(h["a"])(Et,et,nt,!1,null,null,null)),Pt=Ft.exports,$t=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"state-one"},[n("div",{staticClass:"state-one-list flex align-center"},t._l(t.items,(function(e){var i=e.text,s=e.mode;return n("div",{key:"tab_"+s,class:["state-one-list__item",{"state-one-list__item--active":t.isActive(s)}],on:{click:function(e){return t.onTabs(s)}}},[t._v(" "+t._s(i)+" ")])})),0),n("div",{staticClass:"state-one-content mt-10"},["GoogleDrive"===t.project.source.mode?n("t-field",{attrs:{icon:"google",label:"Выберите файл на Google диске"}},[n("d-auto-complete",{key:t.getValueSource,attrs:{value:t.getValueSource,placeholder:"Введите имя файла",list:t.getFilesSource},on:{click:t.getDatasetSources,change:function(e){return t.onSelect({mode:"GoogleDrive",value:e.value})}}})],1):n("t-field",{attrs:{icon:"link",label:"Загрузите по ссылке"}},[n("d-input-text",{attrs:{placeholder:"URL"},on:{blur:function(e){return t.onSelect({mode:"URL",value:e.target.value})}},model:{value:t.project.source.value,callback:function(e){t.$set(t.project.source,"value",e)},expression:"project.source.value"}})],1)],1),n("div",[n("t-field",{attrs:{label:"Название датасета"}},[n("d-input-text",{model:{value:t.project.name,callback:function(e){t.$set(t.project,"name",e)},expression:"project.name"}})],1),n("t-field",{attrs:{label:"Тип архитектуры"}},[n("d-auto-complete",{attrs:{value:t.getValueArchitectures,placeholder:"Архитектуры",list:t.getArchitectures},on:{change:t.onArchitectures}})],1),n("div",{staticClass:"mb-2"},[n("DTags",{model:{value:t.project.tags,callback:function(e){t.$set(t.project,"tags",e)},expression:"project.tags"}})],1)],1)])},Lt=[],Bt=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{class:["t-field",{"t-inline":t.inline}]},[n("label",{staticClass:"t-field__label",attrs:{for:t.parse}},[t._v(t._s(t.label))]),n("div",{staticClass:"d-tags"},[n("button",{staticClass:"d-tags__add mr-2",attrs:{type:"button"}},[n("i",{staticClass:"d-tags__add--icon t-icon icon-tag-plus",on:{click:t.create}}),n("input",{staticClass:"d-tags__add--input",attrs:{type:"text",disabled:t.tags.length>=3,placeholder:"Добавить"},on:{keypress:function(e){return!e.type.indexOf("key")&&t._k(e.keyCode,"enter",13,e.key,"Enter")?null:(e.preventDefault(),t.create.apply(null,arguments))}}})]),t._l(t.tags,(function(e,i){return[n("div",{key:"tag_"+i,staticClass:"d-tags__item mr-2"},[n("input",{staticClass:"d-tags__input",style:{width:12*(e.length+1)<=90?12*(e.length+1)+"px":"90px"},attrs:{"data-index":i,type:"text",autocomplete:"off"},domProps:{value:e},on:{input:t.change,blur:t.blur}}),n("i",{staticClass:"d-tags__remove--icon t-icon icon-tag-plus",on:{click:function(e){return t.removeTag(i)}}})])]}))],2)])},At=[],Mt={name:"t-input",props:{label:{type:String,default:"Теги"},type:{type:String,default:"text"},value:{type:Array},parse:String,name:String,inline:Boolean,disabled:Boolean},data:function(){return{}},computed:{tags:{set:function(t){this.$emit("input",t)},get:function(){return this.value}}},methods:{removeTag:function(t){this.tags=this.tags.filter((function(e,n){return n!==+t}))},create:function(){var t,e=null===(t=this.$el.getElementsByClassName("d-tags__add--input"))||void 0===t?void 0:t[0];console.log(e.value),e.value&&e.value.length>=3&&this.tags.length<=3&&(this.tags.push(e.value),this.tags=Object(b["a"])(this.tags),e.value="")},change:function(t){var e=t.target.dataset.index;console.log(e),t.target.value.length>=3&&(this.tags[+e].name=t.target.value)},blur:function(t){var e=t.target.dataset.index;t.target.value.length<=2&&(this.tags=this.tags.filter((function(t,n){return n!==+e})))}}},Xt=Mt,It=(n("2318"),Object(h["a"])(Xt,Bt,At,!1,null,"c79df15a",null)),Tt=It.exports,Yt={name:"DatasetDownloadTabs",components:{DTags:Tt},data:function(){return{items:[{text:"Google диск",mode:"GoogleDrive"},{text:"URL",mode:"URL"}]}},computed:Object(l["a"])(Object(l["a"])({},Object(o["c"])("createDataset",["getFilesSource","getProject","getArchitectures"])),{},{project:{set:function(t){this.setProject(t)},get:function(){return this.getProject}},getValueSource:function(){var t,e=this.project.source.value;return(null===(t=this.getFilesSource.find((function(t){return t.value===e})))||void 0===t?void 0:t.label)||""},getValueArchitectures:function(){var t,e=this.project.architecture;return(null===(t=this.getArchitectures.find((function(t){return t.value===e})))||void 0===t?void 0:t.label)||""}}),methods:Object(l["a"])(Object(l["a"])({},Object(o["b"])("createDataset",["getDatasetSources"])),{},{isActive:function(t){return this.project.source.mode===t},onSelect:function(t){var e=t.value,n=t.mode;this.project.source.mode=n,this.project.source.value=e},onTabs:function(t){this.project.source.mode=t,this.project.source.value=""},onArchitectures:function(t){console.log(t)}}),created:function(){this.getDatasetSources()}},Vt=Yt,Nt=(n("59d6"),Object(h["a"])(Vt,$t,Lt,!1,null,null,null)),Kt=Nt.exports,Rt=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"state-four"},[n("div",[n("t-field",{attrs:{label:"Название версии"}},[n("d-input-text",{model:{value:t.project.verName,callback:function(e){t.$set(t.project,"verName",e)},expression:"project.verName"}})],1),n("div",[n("DSlider",{model:{value:t.project.train,callback:function(e){t.$set(t.project,"train",e)},expression:"project.train"}})],1),n("t-field",{attrs:{label:"Перемешать"}},[n("d-checkbox",{model:{value:t.project.shuffle,callback:function(e){t.$set(t.project,"shuffle",e)},expression:"project.shuffle"}})],1)],1)])},Ut=[],Wt=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"t-field"},[n("div",{staticClass:"t-field__label"},[t._v("Train / Val")]),n("div",{ref:"slider",class:["d-slider",{"d-slider--disable":t.disable}],on:{mouseleave:t.stopDrag,mouseup:t.stopDrag}},[n("div",{staticClass:"d-slider__inputs"},[n("input",{attrs:{name:"[info][part][train]",type:"number","data-degree":t.degree},domProps:{value:t.btnFirstVal}}),n("input",{attrs:{name:"[info][part][validation]",type:"number","data-degree":t.degree},domProps:{value:100-t.btnFirstVal}})]),n("div",{staticClass:"d-slider__scales"},[n("div",{staticClass:"scales__first",style:t.firstScale},[n("input",{directives:[{name:"autowidth",rawName:"v-autowidth"}],key:t.key1,ref:"key1",attrs:{type:"number",autocomplete:"off"},domProps:{value:t.btnFirstVal},on:{keypress:function(e){return!e.type.indexOf("key")&&t._k(e.keyCode,"enter",13,e.key,"Enter")?null:t.inter(1,e)},blur:function(e){return t.clickInput(1,e)},focus:t.focus}})]),n("div",{staticClass:"scales__second",style:t.secondScale},[n("input",{directives:[{name:"autowidth",rawName:"v-autowidth"}],key:t.key2,ref:"key2",attrs:{type:"number",autocomplete:"off"},domProps:{value:100-t.btnFirstVal},on:{keypress:function(e){return!e.type.indexOf("key")&&t._k(e.keyCode,"enter",13,e.key,"Enter")?null:t.inter(2,e)},blur:function(e){return t.clickInput(2,e)},focus:t.focus}})])]),n("div",{ref:"between",staticClass:"d-slider__between"},[n("button",{staticClass:"d-slider__btn-1",style:t.sliderFirstStyle,on:{mousedown:t.startDragFirst,mouseup:t.stopDragFirst}})])])])},Ht=[],zt={name:"d-slider",props:{degree:Number,disable:Boolean,value:{type:Number,default:70}},data:function(){return{input:0,select:0,firstBtnDrag:!1,key1:1,key2:1}},methods:{focus:function(t){var e=t.target;e.select()},inter:function(t,e){var n=e.target;n.blur();var i=this.$refs["key".concat(t+1)];i&&(i.focus(),this.$nextTick((function(){i.select()})))},clickInput:function(t,e){var n=e.target,i=+n.value;1===t&&i>=0&&i<=90&&(this.btnFirstVal=i>10?i:10),2===t&&i>=0&&i<=90&&(this.btnFirstVal=i>10?100-i:10),this["key".concat(t)]+=1},stopDrag:function(){this.$refs.slider.removeEventListener("mousemove",this.firstBtn)},startDragFirst:function(){this.firstBtnDrag=!0,this.$refs.slider.addEventListener("mousemove",this.firstBtn)},stopDragFirst:function(){this.$refs.slider.removeEventListener("mousemove",this.firstBtn),this.firstBtnDrag=!1},firstBtn:function(t){if(this.firstBtnDrag){var e=document.querySelector(".d-slider__btn-1"),n=t.pageX-e.parentNode.getBoundingClientRect().x,i=this.$refs.slider.clientWidth;this.btnFirstVal=Math.round(n/i*100),this.btnFirstVal<10&&(this.btnFirstVal=10),this.btnFirstVal>90&&(this.btnFirstVal=90)}},diff:function(t){var e=arguments.length>1&&void 0!==arguments[1]?arguments[1]:90,n=arguments.length>2&&void 0!==arguments[2]?arguments[2]:10;return t<n&&(t=n),t>e&&(t=e),t}},computed:{sliderFirstStyle:function(){return{left:this.diff(this.btnFirstVal,90)+"%"}},sliderSecondStyle:function(){return{left:this.diff(this.btnFirstVal,90)+"%"}},firstScale:function(){return{width:this.diff(this.btnFirstVal,90)+"%"}},secondScale:function(){return{width:this.diff(100-this.btnFirstVal,90)+"%"}},btnFirstVal:{set:function(t){this.$emit("input",t/100)},get:function(){return Math.round(100*this.value)}}},watch:{disable:function(t){this.btnFirstVal=t?0:70}}},Gt=zt,Jt=(n("bb4b"),Object(h["a"])(Gt,Wt,Ht,!1,null,"05540f80",null)),qt=Jt.exports,Qt={name:"state-four",components:{DSlider:qt},data:function(){return{}},computed:Object(l["a"])(Object(l["a"])({},Object(o["c"])("createDataset",["getFilesSource","getProject"])),{},{project:{set:function(t){this.setProject(t)},get:function(){return this.getProject}}}),methods:Object(l["a"])({},Object(o["b"])("createDataset",["getDatasetSources","setSelectSource"]))},Zt=Qt,te=(n("4f57"),Object(h["a"])(Zt,Rt,Ut,!1,null,"7efce44a",null)),ee=te.exports,ne=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"d-pagination"},[n("button",{staticClass:"d-pagination__btn",attrs:{disabled:t.isDisabled},on:{click:function(e){return t.$emit("prev",e)}}},[n("d-svg",{attrs:{name:"arrow-carret-left-longer-big"}})],1),n("div",{staticClass:"d-pagination__inner"},[n("div",{staticClass:"d-pagination__list"},t._l(t.list.length,(function(e){return n("div",{key:e,class:["d-pagination__item",{"d-pagination__item--active":t.isActive(e)}]})})),0),n("div",{staticClass:"d-pagination__title"},[n("span",[t._v(t._s(t.getTitle))])])]),n("d-button",{staticStyle:{width:"40%"},attrs:{color:"secondary",direction:"left",text:t.getTextBtn,disabled:t.isStatus},on:{click:t.onNext}})],1)},ie=[],se={name:"DPagination",props:{list:{type:Array,default:function(){return[]}},value:{type:Number,default:0}},data:function(){return{}},computed:Object(l["a"])(Object(l["a"])({},Object(o["c"])({project:"createDataset/getProject"})),{},{isDisabled:function(){return 1===this.value},isStatus:function(){var t,e,n;return!(1!==this.value||null!==this&&void 0!==this&&null!==(t=this.project)&&void 0!==t&&t.source.value&&null!==this&&void 0!==this&&null!==(e=this.project)&&void 0!==e&&e.name&&null!==this&&void 0!==this&&null!==(n=this.project)&&void 0!==n&&n.architecture)},getTitle:function(){var t=this;return this.list.find((function(e){return e.id===t.value})).title},getTextBtn:function(){return this.value===this.list.length?"Создать":"Далее"}}),methods:{isActive:function(t){return this.value===t},onNext:function(t){this.$emit("next",t),this.value===this.list.length&&this.$emit("create",t)}}},ae=se,re=(n("f96a"),Object(h["a"])(ae,ne,ie,!1,null,null,null)),le=re.exports,oe={components:{StateOne:Kt,StateTwo:tt,StateThree:Pt,Pagination:le,StateFour:ee},data:function(){return{debounce:null,list:[{id:1,title:"Данные",component:"state-one"},{id:2,title:"Предпросмотр",component:"state-two"},{id:3,title:"Input",component:"state-three"},{id:4,title:"Output",component:"state-three"},{id:5,title:"Завершение",component:"state-four"}]}},computed:Object(l["a"])(Object(l["a"])({},Object(o["c"])({getPagination:"createDataset/getPagination"})),{},{getComp:function(){var t=this;return this.list.find((function(e){return e.id===t.value}))},value:{set:function(t){this.setPagination(t)},get:function(){return this.getPagination}}}),methods:Object(l["a"])(Object(l["a"])({},Object(o["b"])({setSourceLoad:"createDataset/setSourceLoad",sourceLoadProgress:"createDataset/sourceLoadProgress",setPagination:"createDataset/setPagination",create:"createDataset/create",setOverlay:"settings/setOverlay",blockSelect:"create/main"})),{},{onNext:function(){1===this.value&&this.onDownload(),this.value<this.list.length&&(this.value=this.value+1)},onPrev:function(){this.value>1&&(this.value=this.value-1)},onCreate:function(){console.log("create"),this.create()},onProgress:function(){var t=this;return Object(E["a"])(regeneratorRuntime.mark((function e(){var n,i;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.sourceLoadProgress();case 2:i=e.sent,null!==i&&void 0!==i&&null!==(n=i.data)&&void 0!==n&&n.finished?t.setOverlay(!1):t.debounce(!0);case 4:case"end":return e.stop()}}),e)})))()},onDownload:function(){var t=this;return Object(E["a"])(regeneratorRuntime.mark((function e(){var n;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.setSourceLoad();case 2:n=e.sent,n&&(t.setOverlay(!0),t.debounce(!0));case 4:case"end":return e.stop()}}),e)})))()}}),created:function(){var t=this;this.debounce=Object(F["a"])((function(e){e&&t.onProgress()}),1e3)},beforeDestroy:function(){this.debounce(!1)},watch:{value:function(t,e){console.log(t,e),this.blockSelect({value:t,old:e})}}},ce=oe,ue=(n("0052"),Object(h["a"])(ce,D,w,!1,null,null,null)),de=ue.exports,he={name:"Datasets",components:{Toolbar:v,Blocks:x,Params:de},data:function(){return{state:1}},methods:{onToolbar:function(t){console.log(t)}}},fe=he,ve=(n("506e"),Object(h["a"])(fe,i,s,!1,null,null,null));e["default"]=ve.exports},"1ec8":function(t,e,n){"use strict";n("39b9")},2318:function(t,e,n){"use strict";n("e27a")},3367:function(t,e,n){},"37d0":function(t,e,n){},"39b9":function(t,e,n){},"3e0e":function(t,e,n){},"4b70":function(t,e,n){"use strict";n("0213")},"4f57":function(t,e,n){"use strict";n("3367")},"4fce":function(t,e,n){},"506e":function(t,e,n){"use strict";n("be11")},"59d6":function(t,e,n){"use strict";n("b3ce")},"60d2":function(t,e,n){"use strict";n("c67c")},6280:function(t,e,n){},"62c6":function(t,e,n){},"6d05":function(t,e,n){"use strict";n("4fce")},"730f":function(t,e,n){},8639:function(t,e,n){"use strict";n("730f")},"8c41":function(t,e,n){"use strict";n("9d93")},"9d93":function(t,e,n){},aaf3:function(t,e,n){},ad4b:function(t,e,n){"use strict";n("aaf3")},b3ce:function(t,e,n){},bb4b:function(t,e,n){"use strict";n("fb9b")},be11:function(t,e,n){},c67c:function(t,e,n){},e27a:function(t,e,n){},f02a:function(t,e,n){"use strict";n("62c6")},f244:function(t,e,n){"use strict";n("3e0e")},f96a:function(t,e,n){"use strict";n("6280")},fb9b:function(t,e,n){}}]);