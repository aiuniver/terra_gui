(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-aa81cc34"],{"025d":function(t,e,n){"use strict";n("d07e")},1205:function(t,e,n){"use strict";n("39c4")},2394:function(t,e,n){},"2e96":function(t,e,n){},"39c4":function(t,e,n){},"43b4":function(t,e,n){"use strict";n("9e3e")},"488b":function(t,e,n){},"55c1":function(t,e,n){},"88de":function(t,e,n){"use strict";n("488b")},"895e":function(t,e,n){},"9e3e":function(t,e,n){},b077:function(t,e,n){"use strict";n("895e")},c934:function(t,e,n){"use strict";n("55c1")},d07e:function(t,e,n){},d88e:function(t,e,n){"use strict";n("2e96")},db96:function(t,e,n){"use strict";n("2394")},ebfc:function(t,e,n){"use strict";n.r(e);var i=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("main",{staticClass:"page-modeling"},[n("div",{staticClass:"cont"},[n("LoadModel",{model:{value:t.dialogLoadModel,callback:function(e){t.dialogLoadModel=e},expression:"dialogLoadModel"}}),n("SaveModel",{attrs:{image:t.imageModel},model:{value:t.dialogSaveModel,callback:function(e){t.dialogSaveModel=e},expression:"dialogSaveModel"}}),n("Toolbar",{on:{actions:t.actions}}),n("Blocks",{ref:"container",on:{blockSelect:function(e){t.selectBlock=e},blockDeselect:function(e){t.selectBlock=null},save:t.saveLayers}}),n("Params",{ref:"params",attrs:{selectBlock:t.selectBlock}})],1)])},s=[],o=n("1da1"),a=(n("caad"),n("2532"),n("96cf"),function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"toolbar"},[n("ul",{staticClass:"toolbar__menu"},[n("li",{staticClass:"toolbar__menu--item",attrs:{title:"Загрузить модель"},on:{click:function(e){return e.preventDefault(),t.click("load")}}},[n("i",{staticClass:"t-icon icon-model-load"})]),n("li",{staticClass:"toolbar__menu--item",attrs:{title:"Сохранить модель"},on:{click:function(e){return e.preventDefault(),t.click("save")}}},[n("i",{staticClass:"t-icon icon-model-save"})]),n("li",{staticClass:"toolbar__menu--item",attrs:{title:"Валидация"},on:{click:function(e){return e.preventDefault(),t.click("validation")}}},[n("i",{staticClass:"t-icon icon-model-validation"})]),n("hr"),n("li",{staticClass:"toolbar__menu--item",attrs:{title:"Входящий слой"},on:{click:function(e){return e.preventDefault(),t.click("input")}}},[n("i",{staticClass:"t-icon icon-layer-input-casc"})]),n("li",{staticClass:"toolbar__menu--item",attrs:{title:"Model"},on:{click:function(e){return e.preventDefault(),t.click("model")}}},[n("i",{staticClass:"t-icon icon-layer-model"})]),n("li",{staticClass:"toolbar__menu--item",attrs:{title:"Function"},on:{click:function(e){return e.preventDefault(),t.click("function")}}},[n("i",{staticClass:"t-icon icon-layer-function"})]),n("li",{staticClass:"toolbar__menu--item",attrs:{title:"Custom"},on:{click:function(e){return e.preventDefault(),t.click("custom")}}},[n("i",{staticClass:"t-icon icon-layer-custom"})]),n("li",{staticClass:"toolbar__menu--item",attrs:{title:"Исходящий слой"},on:{click:function(e){return e.preventDefault(),t.click("output")}}},[n("i",{staticClass:"t-icon icon-layer-output"})]),n("hr"),n("li",{staticClass:"toolbar__menu--item",attrs:{title:"Очистить"},on:{click:function(e){return e.preventDefault(),t.click("clear")}}},[n("i",{staticClass:"t-icon icon-clear-model"})])])])}),l=[],c=n("5530"),r=n("2f62"),u={name:"Toolbar",data:function(){return{}},computed:Object(c["a"])({},Object(r["b"])({blocks:"cascades/getBlocks",project:"projects/getProject"})),methods:{click:function(t,e){e||this.$emit("actions",t)}}},d=u,h=(n("1205"),n("2877")),m=Object(h["a"])(d,a,l,!1,null,"2b182ae8",null),v=m.exports,p=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"board"},[n("VueLink",{attrs:{lines:t.lines}}),t._l(t.blocks,(function(e){return n("VueBlock",t._b({key:e.id,attrs:{options:t.optionsForChild},on:{linkingStart:function(n){return t.linkingStart(e,n)},linkingStop:function(n){return t.linkingStop(e,n)},linkingBreak:function(n){return t.linkingBreak(e,n)},select:function(n){return t.blockSelect(e)},delete:function(n){return t.blockDelete(e)},position:function(n){return t.position(e,n)},moveBlock:t.moveBlock}},"VueBlock",e,!1))})),n("div",{staticClass:"btn-zoom"},[n("div",{staticClass:"btn-zoom__item"},[n("i",{staticClass:"t-icon icon-zoom-inc",on:{click:function(e){return t.zoom(1)}}})]),n("hr"),n("div",{staticClass:"btn-zoom__item"},[n("i",{staticClass:"t-icon icon-zoom-reset",on:{click:function(e){return t.zoom(0)}}})]),n("hr"),n("div",{staticClass:"btn-zoom__item"},[n("i",{staticClass:"t-icon icon-zoom-dec",on:{click:function(e){return t.zoom(-1)}}})])])],2)},f=[],g=n("2909"),k=n("b85c"),b=(n("7db0"),n("c740"),n("4de4"),n("99af"),n("d81d"),n("159b"),n("c7a7")),y=n.n(b),_=n("f69e"),x=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"vue-block",style:t.style,on:{mouseover:function(e){t.hover=!0},mouseleave:function(e){t.hover=!1}}},[n("div",{class:["header",t.group,{selected:t.selected}]},[n("div",{staticClass:"title",attrs:{title:t.name}},[t._v(t._s(t.name)+": "+t._s(t.type))]),n("div",{staticClass:"parametr",attrs:{title:t.parameters}},[t._v("[]")])]),t.group.includes("model")?n("div",{directives:[{name:"show",rawName:"v-show",value:t.hover||t.selected,expression:"hover || selected"}],staticClass:"hover-sloy"},[n("i",{staticClass:"t-icon icon-modeling-link-remove"}),n("i",{staticClass:"t-icon icon-modeling-remove",on:{click:t.deleteBlock}})]):n("div",{directives:[{name:"show",rawName:"v-show",value:t.hover||t.selected,expression:"hover || selected"}],staticClass:"hover-over"},[n("i",{staticClass:"t-icon icon-modeling-link-remove"})]),n("div",{staticClass:"inputs"},t._l(t.inputs,(function(e,i){return n("div",{key:"input"+i,staticClass:"input inputSlot",class:{active:e.active},on:{mouseup:function(e){return t.slotMouseUp(e,i)},mousedown:function(e){return t.slotBreak(e,i)}}})})),0),n("div",{staticClass:"outputs"},t._l(t.outputs,(function(e,i){return n("div",{key:"output"+i,staticClass:"output",class:[{active:e.active},t.typeLink[i]],on:{mousedown:function(e){return t.slotMouseDown(e,i)}}})})),0)])},C=[],w=(n("a9e3"),{name:"VueBlock",props:{id:{type:Number},name:{type:String},group:{type:String},position:{type:Array,default:function(){return[]},validator:function(t){return"number"===typeof t[0]&&"number"===typeof t[1]}},selected:Boolean,type:String,title:{type:String,default:"Title"},inputs:Array,outputs:Array,parameters:{type:Object,default:function(){}},options:{type:Object}},data:function(){return{hover:!1,hasDragged:!1,typeLink:["bottom","right","left"]}},created:function(){this.mouseX=0,this.mouseY=0,this.lastMouseX=0,this.lastMouseY=0,this.linking=!1,this.dragging=!1},mounted:function(){document.documentElement.addEventListener("mousemove",this.handleMove,!0),document.documentElement.addEventListener("mousedown",this.handleDown,!0),document.documentElement.addEventListener("mouseup",this.handleUp,!0)},beforeDestroy:function(){document.documentElement.removeEventListener("mousemove",this.handleMove,!0),document.documentElement.removeEventListener("mousedown",this.handleDown,!0),document.documentElement.removeEventListener("mouseup",this.handleUp,!0)},methods:{handleMove:function(t){if(this.mouseX=t.pageX||t.clientX+document.documentElement.scrollLeft,this.mouseY=t.pageY||t.clientY+document.documentElement.scrollTop,this.dragging&&!this.linking){var e=this.mouseX-this.lastMouseX,n=this.mouseY-this.lastMouseY;this.lastMouseX=this.mouseX,this.lastMouseY=this.mouseY,this.moveWithDiff(e,n),this.hasDragged=!0}},handleDown:function(t){this.mouseX=t.pageX||t.clientX+document.documentElement.scrollLeft,this.mouseY=t.pageY||t.clientY+document.documentElement.scrollTop,this.lastMouseX=this.mouseX,this.lastMouseY=this.mouseY;var e=t.target||t.srcElement;this.$el.contains(e)&&1===t.which&&(this.dragging=!0,this.$emit("select"),t.preventDefault&&t.preventDefault())},handleUp:function(){this.dragging&&(this.dragging=!1,this.hasDragged&&(this.$emit("moveBlock"),this.save(),this.hasDragged=!1)),this.linking&&(this.linking=!1)},slotMouseDown:function(t,e){this.linking=!0,this.$emit("linkingStart",e),t.preventDefault&&t.preventDefault()},slotMouseUp:function(t,e){this.$emit("linkingStop",e),t.preventDefault&&t.preventDefault()},slotBreak:function(t,e){this.linking=!0,this.$emit("linkingBreak",e),t.preventDefault&&t.preventDefault()},save:function(){this.$emit("update")},deleteBlock:function(){this.$emit("delete")},moveWithDiff:function(t,e){var n=this.position[0]+t/this.options.scale,i=this.position[1]+e/this.options.scale;this.$emit("position",[n,i])}},computed:{style:function(){return{left:this.options.center.x+this.position[0]*this.options.scale+"px",top:this.options.center.y+this.position[1]*this.options.scale+"px",width:this.options.width+"px",transform:"scale("+this.options.scale+")",transformOrigin:"top left"}}}}),D=w,S=(n("025d"),Object(h["a"])(D,x,C,!1,null,"552df3ca",null)),M=S.exports,L=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("svg",{attrs:{width:"100%",height:"100%"}},[t._l(t.renderedPathes,(function(e,i){return n("g",{key:"k"+i},[t.outline?n("path",{style:e.outlineStyle,attrs:{d:e.data}}):t._e(),n("path",{style:e.style,attrs:{d:e.data}})])})),n("g",t._l(t.renderedArrows,(function(t,e){return n("path",{key:"a"+e,style:t.style,attrs:{d:"M -1 -1 L 0 1 L 1 -1 z",transform:t.transform}})})),0)],2)},$=[],E={props:{lines:{type:Array,default:function(){return[]}},outline:{type:Boolean,default:!1}},methods:{distance:function(t,e,n,i){return Math.sqrt((n-t)*(n-t)+(i-e)*(i-e))},computeConnectionPoint:function(t,e,n,i,s){var o=this.distance(t,e,n,i),a={x:t,y:e},l={x:t+.25*o,y:e},c={x:n-.25*o,y:i},r={x:n,y:i},u=(1-s)*(1-s)*(1-s),d=(1-s)*(1-s)*3*s,h=3*(1-s)*(s*s),m=s*s*s,v=u*a.x+d*l.x+h*c.x+m*r.x,p=u*a.y+d*l.y+h*c.y+m*r.y;return{x:v,y:p}}},computed:{renderedPathes:function(){var t=this;if(!this.lines)return[];var e=[];return this.lines.forEach((function(n){var i=.2*t.distance(n.x1,n.y1,n.x2,n.y2),s=["M ".concat(n.x1,", ").concat(n.y1," C ").concat(n.x1,", ").concat(n.y1+50,", ").concat(n.x2,", ").concat(n.y2-50,", ").concat(n.x2,", ").concat(n.y2),"M ".concat(n.x1,", ").concat(n.y1," C ").concat(n.x1+i,", ").concat(n.y1,", ").concat(n.x2-i,", ").concat(n.y2,", ").concat(n.x2,", ").concat(n.y2),"M ".concat(n.x1,", ").concat(n.y1," C ").concat(n.x1-i,", ").concat(n.y1,", ").concat(n.x2+i,", ").concat(n.y2,", ").concat(n.x2,", ").concat(n.y2)];e.push({data:s[n.slot]||s[0],style:n.style,outlineStyle:n.outlineStyle})})),e},renderedArrows:function(){var t=this;if(!this.lines)return[];var e=[];return this.lines.forEach((function(n){var i=t.computeConnectionPoint(n.x1,n.y1,n.x2,n.y2,.5),s=t.computeConnectionPoint(n.x1,n.y1,n.x2,n.y2,.501),o=-Math.atan2(s.x-i.x,s.y-i.y),a=180*(o>=0?o:2*Math.PI+o)/Math.PI;e.push({transform:"translate(".concat(i.x,", ").concat(i.y,") rotate(").concat(a-15,")"),style:{stroke:"rgb(80, 125, 150)",strokeWidth:2*n.style.strokeWidth,fill:n.style.stroke}})})),e}}},B=E,O=Object(h["a"])(B,L,$,!1,null,"686bafd8",null),j=O.exports,X={name:"VueBlockContainer",components:{VueBlock:M,VueLink:j},props:{blocksContent:{type:Array,default:function(){return[]}},options:{type:Object}},data:function(){return{dragging:!1,centerX:0,centerY:0,scale:1,tempLink:null,selectedBlock:null,hasDragged:!1,mouseX:0,mouseY:0,lastMouseX:0,lastMouseY:0,minScale:.2,maxScale:5,linking:!1,linkStartData:null,inputSlotClassName:"inputSlot",defaultScene:{blocks:[],links:[],container:{}}}},computed:{blocks:{set:function(t){this.$store.dispatch("cascades/setBlocks",t)},get:function(){return this.$store.getters["cascades/getBlocks"]}},links:{set:function(t){console.log(t),this.$store.dispatch("cascades/setLinks",t)},get:function(){return this.$store.getters["cascades/getLinks"]}},optionsForChild:function(){return console.log(this.centerX,this.centerY),{width:200,titleHeight:48,scale:this.scale,inputSlotClassName:this.inputSlotClassName,center:{x:this.centerX,y:this.centerY}}},container:function(){return{centerX:this.centerX,centerY:this.centerY,scale:this.scale}},lines:function(){var t,e=this,n=[],i=Object(k["a"])(this.links);try{var s=function(){var i=t.value,s=e.blocks.find((function(t){return t.id===i.originID})),o=e.blocks.find((function(t){return t.id===i.targetID}));if(!s||!o)return console.log("Remove invalid link",i),e.removeLink(i.id),"continue";if(s.id===o.id)return console.log("Loop detected, remove link",i),e.removeLink(i.id),"continue";var a=e.getConnectionPos(s,i.originSlot,!1),l=e.getConnectionPos(o,i.targetSlot,!0);if(!a||!l)return console.log("Remove invalid link (slot not exist)",i),e.removeLink(i.id),"continue";var c=a.x,r=a.y,u=l.x,d=l.y;n.push({x1:c,y1:r,x2:u,y2:d,slot:i.originSlot,style:{stroke:"rgb(101, 185, 244)",strokeWidth:3*e.scale,fill:"none"},outlineStyle:{stroke:"#666",strokeWidth:6*e.scale,strokeOpacity:.6,fill:"none"}})};for(i.s();!(t=i.n()).done;)s()}catch(o){i.e(o)}finally{i.f()}return this.tempLink&&(this.tempLink.style={stroke:"#8f8f8f",strokeWidth:3*this.scale,fill:"none"},n.push(this.tempLink)),n}},methods:{handleMauseOver:function(t){this.mouseIsOver="mouseenter"===t.type},keyup:function(t){var e=t.code,n=t.ctrlKey,i=this.mouseIsOver;console.log(i,e),i&&"Delete"===e&&this.selectedBlock&&this.blockDelete(this.selectedBlock),i&&"KeyC"===e&&n&&this.selectedBlock&&this.blockDelete(this.selectedBlock)},zoom:function(t){if(0!==t){var e=1===t?1.1:.9090909090909091;if(this.scale*=e,this.scale<this.minScale)this.scale=this.minScale;else if(this.scale>this.maxScale)this.scale=this.maxScale;else{var n={x:this.mouseX,y:this.mouseY},i=(n.x-this.centerX)*(e-1),s=(n.y-this.centerY)*(e-1);this.centerX-=i,this.centerY-=s}}else this.scale=1},handleMove:function(t){var e=Object(_["c"])(this.$el,t);if(this.mouseX=e.x,this.mouseY=e.y,this.dragging){var n=this.mouseX-this.lastMouseX,i=this.mouseY-this.lastMouseY;this.lastMouseX=this.mouseX,this.lastMouseY=this.mouseY,this.centerX+=n,this.centerY+=i,this.hasDragged=!0}if(this.linking&&this.linkStartData){var s=this.getConnectionPos(this.linkStartData.block,this.linkStartData.slotNumber,!1);this.tempLink={x1:s.x,y1:s.y,x2:this.mouseX,y2:this.mouseY,slot:this.linkStartData.slotNumber}}},handleDown:function(t){console.log("handleDown");var e=t.target||t.srcElement;if((e===this.$el||e.matches("svg, svg *"))&&1===t.which){this.dragging=!0;var n=Object(_["c"])(this.$el,t);this.mouseX=n.x,this.mouseY=n.y,this.lastMouseX=this.mouseX,this.lastMouseY=this.mouseY,this.deselectAll(),t.preventDefault&&t.preventDefault()}},handleUp:function(t){console.log("handleUp");var e=t.target||t.srcElement;this.dragging&&(this.dragging=!1,this.hasDragged&&(this.hasDragged=!1)),!this.$el.contains(e)||"string"===typeof e.className&&-1!==e.className.indexOf(this.inputSlotClassName)||(this.linking=!1,this.tempLink=null,this.linkStartData=null)},handleWheel:function(t){var e=t.target||t.srcElement;if(this.$el.contains(e)){var n=Math.pow(1.1,-.01*t.deltaY);if(this.scale*=n,this.scale<this.minScale)return void(this.scale=this.minScale);if(this.scale>this.maxScale)return void(this.scale=this.maxScale);var i={x:this.mouseX,y:this.mouseY},s=(i.x-this.centerX)*(n-1),o=(i.y-this.centerY)*(n-1);this.centerX-=s,this.centerY-=o}},getConnectionPos:function(t,e,n){if(t&&-1!==e){var i=0,s=0;if(i+=t.position[0],s+=t.position[1],n&&t.inputs.length>e)1===t.inputs.length?i+=this.optionsForChild.width/2:(i+=this.optionsForChild.width/2-10*t.inputs.length/2,i+=20*e);else{if(n||!(t.outputs.length>e))return void console.error("slot "+e+" not found, is input: "+n,t);0===e&&(i+=this.optionsForChild.width/2,s+=50),1===e&&(i+=this.optionsForChild.width,s+=25),2===e&&(s+=25)}return i*=this.scale,s*=this.scale,i+=this.centerX,s+=this.centerY,{x:i,y:s}}},findindexBlock:function(t){return this.blocks.findIndex((function(e){return e.id===t}))},linkingStart:function(t,e){console.log("linkingStart"),this.linkStartData={block:t,slotNumber:e};var n=this.getConnectionPos(this.linkStartData.block,this.linkStartData.slotNumber,!1);this.tempLink={x1:n.x,y1:n.y,x2:this.mouseX,y2:this.mouseY},this.linking=!0},linkingStop:function(t,e){if(console.log("linkingStop"),this.linkStartData&&t&&e>-1){var n=this.linkStartData,i=n.slotNumber,s=n.block.id,o=t.id,a=e;this.links=this.links.filter((function(t){return!(t.targetID===o&&t.targetSlot===a&&t.originID===s&&t.originSlot===i)&&!(t.originID===s&&t.targetID===o)}));var l=Math.max.apply(Math,[0].concat(Object(g["a"])(this.links.map((function(t){return t.id})))));if(this.linkStartData.block.id!==t.id){var c=this.linkStartData.block.id,r=this.linkStartData.slotNumber,u=t.id,d=e;this.links.push({id:l+1,originID:c,originSlot:r,targetID:u,targetSlot:d}),this.$emit("save",!0)}}this.linking=!1,this.tempLink=null,this.linkStartData=null},linkingBreak:function(t,e){if(console.log("linkingBreak"),t&&e>-1){var n=this.links.find((function(n){return n.targetID===t.id&&n.targetSlot===e}));if(n){var i=this.blocks.find((function(t){return t.id===n.originID}));this.links=this.links.filter((function(n){return!(n.targetID===t.id&&n.targetSlot===e)})),this.$emit("save",!0),t.inputs[n.targetSlot].active=!1,i.outputs[n.originSlot].active=!1,this.linkingStart(i,n.originSlot)}}},removeLink:function(t){console.log("removeLink"),this.links=this.links.filter((function(e){return!(e.id===t)}))},getImages:function(){var t=this;return Object(o["a"])(regeneratorRuntime.mark((function e(){var n;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.prev=0,e.next=3,y.a.toPng(t.$el,{filter:function(t){return"btn-zoom"!==t.className}});case 3:return n=e.sent,e.abrupt("return",n);case 7:return e.prev=7,e.t0=e["catch"](0),console.log(e.t0),e.abrupt("return",null);case 11:case"end":return e.stop()}}),e,null,[[0,7]])})))()},addNewBlock:function(t,e,n){var i=Math.max.apply(Math,[0].concat(Object(g["a"])(this.blocks.map((function(t){return t.id}))))),s=Object(_["a"])(t,i+1);s?(void 0===e||void 0===n?(e=(this.$el.clientWidth/2-this.centerX)/this.scale,n=(this.$el.clientHeight/2-this.centerY)/this.scale):(e=(e-this.centerX)/this.scale,n=(n-this.centerY)/this.scale),s.position=[e,n],this.blocks.push(s),this.blocks=this.blocks):console.warn("block not create: "+s)},position:function(t,e){t.position=e},deselectAll:function(){var t=this,e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:null;this.blocks.forEach((function(n){n.id!==e&&n.selected&&t.blockDeselect(n)}))},blockSelect:function(t){t.selected=!0,this.selectedBlock=t,this.deselectAll(t.id),this.$emit("blockSelect",t)},blockDeselect:function(t){t.selected=!1,t&&this.selectedBlock&&this.selectedBlock.id===t.id&&(this.selectedBlock=null),this.$emit("blockDeselect",t)},blockDelete:function(t){var e=this;t.selected&&this.blockDeselect(t),this.links.forEach((function(n){n.originID!==t.id&&n.targetID!==t.id||e.removeLink(n.id)})),this.blocks=this.blocks.filter((function(e){return e.id!==t.id}))},moveBlock:function(){this.$store.dispatch("cascades/setButtons",{save:!0})},updateScene:function(){}},mounted:function(){this.$el.addEventListener("mouseenter",this.handleMauseOver),this.$el.addEventListener("mouseleave",this.handleMauseOver),document.documentElement.addEventListener("keyup",this.keyup),document.documentElement.addEventListener("mousemove",this.handleMove,!0),document.documentElement.addEventListener("mousedown",this.handleDown,!0),document.documentElement.addEventListener("mouseup",this.handleUp,!0),document.documentElement.addEventListener("wheel",this.handleWheel,!0),this.centerX=this.$el.clientWidth/2},beforeDestroy:function(){document.documentElement.removeEventListener("keyup",this.keyup),this.$el.removeEventListener("mouseenter",this.handleMauseOver),this.$el.removeEventListener("mouseleave",this.handleMauseOver),document.documentElement.removeEventListener("mousemove",this.handleMove,!0),document.documentElement.removeEventListener("mousedown",this.handleDown,!0),document.documentElement.removeEventListener("mouseup",this.handleUp,!0),document.documentElement.removeEventListener("wheel",this.handleWheel,!0)}},Y=X,I=(n("b077"),Object(h["a"])(Y,p,f,!1,null,"3967a2fe",null)),N=I.exports,R=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"params"},[n("Navbar"),n("scrollbar",[n("div",{staticClass:"params__items"},[n("div",{staticClass:"params__items--item"},[n("t-input",{attrs:{label:"Название слоя",type:"text",parse:"name",name:"name",disabled:!t.selectBlock},on:{change:t.saveModel},model:{value:t.block.name,callback:function(e){t.$set(t.block,"name",e)},expression:"block.name"}}),n("Autocomplete2",{attrs:{list:t.list,label:"Тип слоя",name:"type",disabled:!t.selectBlock},on:{change:t.saveModel},model:{value:t.block.type,callback:function(e){t.$set(t.block,"type",e)},expression:"block.type"}})],1),n("at-collapse",{attrs:{value:t.collapse}},[n("at-collapse-item",{directives:[{name:"show",rawName:"v-show",value:t.main.items.length,expression:"main.items.length"}],staticClass:"mb-3",attrs:{title:"Параметры слоя"}},[n("Forms",{attrs:{data:t.main},on:{change:t.change}})],1),n("at-collapse-item",{directives:[{name:"show",rawName:"v-show",value:t.extra.items.length,expression:"extra.items.length"}],staticClass:"mb-3",attrs:{title:"Дополнительные параметры"}},[n("Forms",{attrs:{data:t.extra},on:{change:t.change}})],1)],1),n("div",{staticClass:"params__items--item"},[n("button",{staticClass:"mb-1",attrs:{disabled:!t.buttonSave},on:{click:t.saveModel}},[t._v("Сохранить")]),n("button",{attrs:{disabled:"disabled"}},[t._v("Клонировать")])])],1)])],1)},P=[],T=(n("b64b"),n("b0c0"),function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"params__navbar"},[n("ul",{staticClass:"params__navbar--items"},[t._l(t.items,(function(e,i){var s=e.title,o=e.active;return[n("li",{key:"item_"+i,class:["params__navbar--item",{active:o}],attrs:{title:s},on:{click:function(e){return t.click(i)}}},[t._v(" "+t._s(s)+" ")])]}))],2)])}),z=[],A={name:"PNavbar",props:{list:{type:Array,default:function(){return[{title:"Слой",active:!0}]}}},data:function(){return{items:[]}},methods:{click:function(t){this.items=this.items.map((function(e,n){return e.active=t===n,e}))}},created:function(){this.items=this.list}},V=A,W=(n("88de"),Object(h["a"])(V,T,z,!1,null,"cb9c1202",null)),F=W.exports,U=n("6522"),H=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"form-inline-label"},[t._l(t.items,(function(e,i){var s=e.type,o=e.value,a=e.list,l=e.event,c=e.label,r=e.parse,u=e.name;return["tuple"===s?n("Input",{key:t.blockType+i,attrs:{value:t.getValue(t.valueDef[u],o),label:c,type:"text",parse:r,name:u,inline:""},on:{change:t.change}}):t._e(),"number"===s||"text"===s?n("Input",{key:t.blockType+i,attrs:{value:t.getValue(t.valueDef[u],o),label:c,type:s,parse:r,name:u,inline:""},on:{change:t.change}}):t._e(),"checkbox"===s?n("t-checkbox",{key:t.blockType+i,attrs:{inline:"",value:t.getValue(t.valueDef[u],o),label:c,type:"checkbox",parse:r,name:u,event:l},on:{change:t.change}}):t._e(),"select"===s?n("Select",{key:t.blockType+i,attrs:{value:t.getValue(t.valueDef[u],o),label:c,lists:a,parse:r,name:u},on:{change:t.change}}):t._e()]}))],2)},J=[],K=n("53ca"),q=(n("a15b"),n("7d6e")),G=n("7b8d"),Q={name:"Forms",components:{Input:q["a"],Select:G["a"]},props:{data:{type:Object,default:function(){return{type:"main",items:[],value:{}}}}},computed:{items:function(){var t;return(null===(t=this.data)||void 0===t?void 0:t.items)||[]},valueDef:function(){var t;return(null===(t=this.data)||void 0===t?void 0:t.value)||{}},type:function(){var t;return(null===(t=this.data)||void 0===t?void 0:t.type)||""},blockType:function(){var t;return(null===(t=this.data)||void 0===t?void 0:t.blockType)||""}},methods:{change:function(t){this.$emit("change",Object(c["a"])({type:this.type},t))},getValue:function(t,e){var n=null!==t&&void 0!==t?t:e;return"object"===Object(K["a"])(n)?n.join():n}},filters:{toString:function(t){return"object"===Object(K["a"])(t)?t.join():t},isCheck:function(t){return!!t}}},Z=Q,tt=Object(h["a"])(Z,H,J,!1,null,null,null),et=tt.exports,nt={name:"Params",props:{selectBlock:Object},components:{Autocomplete2:U["a"],Forms:et,Navbar:F},data:function(){return{collapse:[0,1],oldBlock:null}},computed:Object(c["a"])(Object(c["a"])({},Object(r["b"])({list:"cascades/getList",layers:"cascades/getLayersType",buttons:"cascades/getButtons"})),{},{block:{set:function(t){this.$store.dispatch("cascades/setBlock",t)},get:function(){return this.$store.getters["cascades/getBlock"]||{}}},buttonSave:function(){var t;return(null===(t=this.buttons)||void 0===t?void 0:t.save)||!1},main:function(){var t,e=null===(t=this.block)||void 0===t?void 0:t.type;if(Object.keys(this.layers).length&&e){var n,i,s,o=(null===(n=this.layers["Layer".concat(e,"Data")])||void 0===n?void 0:n.main)||[],a=(null===(i=this.block)||void 0===i||null===(s=i.parameters)||void 0===s?void 0:s.main)||{};return{type:"main",items:o,value:a,blockType:e}}return{type:"main",items:[],value:{}}},extra:function(){var t,e=null===(t=this.block)||void 0===t?void 0:t.type;if(Object.keys(this.layers).length&&e){var n,i,s,o=(null===(n=this.layers["Layer".concat(e,"Data")])||void 0===n?void 0:n.extra)||[],a=(null===(i=this.block)||void 0===i||null===(s=i.parameters)||void 0===s?void 0:s.extra)||{};return{type:"extra",items:o,value:a,blockType:e}}return{type:"extra",items:[],value:{}}}}),methods:{saveModel:function(){var t=this;return Object(o["a"])(regeneratorRuntime.mark((function e(){return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.$store.dispatch("cascades/saveModel",{});case 2:case"end":return e.stop()}}),e)})))()},change:function(t){var e=this;return Object(o["a"])(regeneratorRuntime.mark((function n(){var i,s,o;return regeneratorRuntime.wrap((function(n){while(1)switch(n.prev=n.next){case 0:i=t.type,s=t.name,o=t.value,console.log({type:i,name:s,value:o}),e.block.parameters?e.block.parameters[i][s]=o:e.oldBlock.parameters[i][s]=o,e.$emit("change"),e.saveModel();case 5:case"end":return n.stop()}}),n)})))()}},watch:{selectBlock:{handler:function(t,e){this.oldBlock=e,this.$store.dispatch("cascades/setSelect",null===t||void 0===t?void 0:t.id),console.log(t,e)}}}},it=nt,st=(n("d88e"),Object(h["a"])(it,R,P,!1,null,"79a02390",null)),ot=st.exports,at=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("at-modal",{attrs:{width:"680",showClose:""},model:{value:t.dialog,callback:function(e){t.dialog=e},expression:"dialog"}},[n("div",{staticStyle:{"text-align":"center"},attrs:{slot:"header"},slot:"header"},[n("span",[t._v("Загрузка модели")])]),n("div",{staticClass:"row at-row"},[n("div",{staticClass:"col-16 models-list scroll-area"},[n("scrollbar",[n("ul",{staticClass:"loaded-list"},[t._l(t.preset,(function(e,i){return n("li",{key:"preset_"+i,staticClass:"loaded-list__item",on:{click:function(n){return t.getModel(e)}}},[n("i",{staticClass:"loaded-list__item--icon"}),n("span",{staticClass:"loaded-list__item--text"},[t._v(t._s(e.label))])])})),t._l(t.custom,(function(e,i){return n("li",{key:"custom_"+i,staticClass:"loaded-list__item",on:{click:function(n){return t.getModel(e)}}},[n("i",{staticClass:"loaded-list__item--icon"}),n("span",{staticClass:"loaded-list__item--text"},[t._v(t._s(e.label))]),n("div",{staticClass:"loaded-list__item--empty"}),n("div",{staticClass:"loaded-list__item--remove"},[n("i")])])}))],2)])],1),n("div",{staticClass:"col-8"},[t.info.name?n("div",{staticClass:"model-arch"},[n("div",{staticClass:"wrapper hidden"},[n("div",{staticClass:"modal-arch-info"},[n("div",{staticClass:"model-arch-info-param name"},[t._v(" Name: "),n("span",[t._v(t._s(t.info.alias||""))])]),n("div",{staticClass:"model-arch-info-param input_shape"},[t._v(" Input shape: "),n("span",[t._v(t._s(t.info.input_shape||""))])]),n("div",{staticClass:"model-arch-info-param datatype"},[t._v(" Datatype: "),n("span",[t._v(t._s(t.info.name))])])]),n("div",{staticClass:"model-arch-img my-5"},[n("img",{attrs:{alt:"",width:"100",height:"200",src:"data:image/png;base64,"+t.info.image||!1}})]),n("div",{staticClass:"model-save-arch-btn"},[n("button",{attrs:{disabled:!t.model},on:{click:t.download}},[t._v("Загрузить")])])])]):t._e()])]),n("div",{attrs:{slot:"footer"},slot:"footer"})])},lt=[],ct={name:"ModalLoadModel",props:{value:Boolean},data:function(){return{lists:[],info:{},model:null}},computed:Object(c["a"])(Object(c["a"])({},Object(r["b"])({})),{},{preset:function(){var t;return(null===(t=this.lists[0])||void 0===t?void 0:t.models)||[]},custom:function(){var t;return(null===(t=this.lists[1])||void 0===t?void 0:t.models)||[]},dialog:{set:function(t){this.$emit("input",t)},get:function(){return this.value}}}),methods:{load:function(){var t=this;return Object(o["a"])(regeneratorRuntime.mark((function e(){var n,i;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.$store.dispatch("cascades/info",{});case 2:n=e.sent,i=n.data,i&&(t.lists=i);case 5:case"end":return e.stop()}}),e)})))()},getModel:function(t){var e=this;return Object(o["a"])(regeneratorRuntime.mark((function n(){var i,s;return regeneratorRuntime.wrap((function(n){while(1)switch(n.prev=n.next){case 0:return n.next=2,e.$store.dispatch("cascades/getModel",t);case 2:i=n.sent,s=i.data,s&&(e.info=s,e.model=t);case 5:case"end":return n.stop()}}),n)})))()},download:function(){var t=this;return Object(o["a"])(regeneratorRuntime.mark((function e(){return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.$store.dispatch("cascades/load",t.model);case 2:t.$emit("input",!1);case 3:case"end":return e.stop()}}),e)})))()}},watch:{dialog:{handler:function(t){t&&this.load()}}}},rt=ct,ut=(n("db96"),Object(h["a"])(rt,at,lt,!1,null,"fced48a8",null)),dt=ut.exports,ht=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("at-modal",{attrs:{width:"500",showClose:""},model:{value:t.dialog,callback:function(e){t.dialog=e},expression:"dialog"}},[n("div",{staticStyle:{"text-align":"center"},attrs:{slot:"header"},slot:"header"},[n("span",[t._v("Сохранить модель")])]),n("div",{staticClass:"model"},[t.image?n("div",{staticClass:"model__image"},[n("img",{attrs:{alt:"",width:"auto",height:"400",src:t.image||""}})]):n("Loading")],1),n("template",{slot:"footer"},[n("button",[t._v("Отменить")]),n("button",[t._v("Сохранить")])])],2)},mt=[],vt=n("8f6b"),pt={name:"ModalSaveModel",components:{Loading:vt["a"]},props:{value:Boolean,image:String},data:function(){return{}},computed:{dialog:{set:function(t){this.$emit("input",t)},get:function(){return this.value}}},methods:{save:function(){var t=this;return Object(o["a"])(regeneratorRuntime.mark((function e(){return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.$store.dispatch("deploy/SendDeploy",t.model);case 2:t.$emit("input",!1);case 3:case"end":return e.stop()}}),e)})))()}},watch:{dialog:{handler:function(t){t&&console.log(this.refs)}}}},ft=pt,gt=(n("43b4"),Object(h["a"])(ft,ht,mt,!1,null,"00b0aeb0",null)),kt=gt.exports,bt={name:"Cascades",components:{Toolbar:v,Blocks:N,Params:ot,LoadModel:dt,SaveModel:kt},data:function(){return{dialogLoadModel:!1,dialogSaveModel:!1,selectBlock:null,imageModel:null}},methods:{addBlock:function(t){console.log(t),this.create=!1,this.selectBlockType="",this.$refs.container.addNewBlock(t)},saveModel:function(){var t=this;return Object(o["a"])(regeneratorRuntime.mark((function e(){return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return t.imageModel=null,t.dialogSaveModel=!0,e.next=4,t.$refs.container.getImages();case 4:t.imageModel=e.sent;case 5:case"end":return e.stop()}}),e)})))()},saveLayers:function(){var t=this;return Object(o["a"])(regeneratorRuntime.mark((function e(){return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.$store.dispatch("cascades/saveModel",{});case 2:case"end":return e.stop()}}),e)})))()},actions:function(t){"load"===t&&(this.dialogLoadModel=!0),_["b"].includes(t)&&this.addBlock(t),"save"===t&&this.saveModel(),"validation"===t&&(console.log("hjkhjh"),this.$refs.params.saveModel()),"clear"===t&&this.$Modal.confirm({title:"Внимание!",content:"Очистить модель?",width:300,callback:function(t){"confirm"==t&&console.log("DELETE MODEL")}}),console.log(t)}}},yt=bt,_t=(n("c934"),Object(h["a"])(yt,i,s,!1,null,"294593f1",null));e["default"]=_t.exports}}]);