(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-24f5e6a7"],{"155d":function(t,e,n){},1739:function(t,e,n){"use strict";n("3762")},"220c":function(t,e,n){"use strict";n("155d")},3762:function(t,e,n){},3835:function(t,e,n){"use strict";function i(t){if(Array.isArray(t))return t}n.d(e,"a",(function(){return a}));n("a4d3"),n("e01a"),n("d3b7"),n("d28b"),n("3ca3"),n("ddb0");function o(t,e){var n=null==t?null:"undefined"!==typeof Symbol&&t[Symbol.iterator]||t["@@iterator"];if(null!=n){var i,o,s=[],r=!0,a=!1;try{for(n=n.call(t);!(r=(i=n.next()).done);r=!0)if(s.push(i.value),e&&s.length===e)break}catch(c){a=!0,o=c}finally{try{r||null==n["return"]||n["return"]()}finally{if(a)throw o}}return s}}var s=n("06c5");function r(){throw new TypeError("Invalid attempt to destructure non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")}function a(t,e){return i(t)||o(t,e)||Object(s["a"])(t,e)||r()}},5899:function(t,e){t.exports="\t\n\v\f\r                　\u2028\u2029\ufeff"},"58a8":function(t,e,n){var i=n("1d80"),o=n("5899"),s="["+o+"]",r=RegExp("^"+s+s+"*"),a=RegExp(s+s+"*$"),c=function(t){return function(e){var n=String(i(e));return 1&t&&(n=n.replace(r,"")),2&t&&(n=n.replace(a,"")),n}};t.exports={start:c(1),end:c(2),trim:c(3)}},7156:function(t,e,n){var i=n("861d"),o=n("d2bb");t.exports=function(t,e,n){var s,r;return o&&"function"==typeof(s=e.constructor)&&s!==n&&i(r=s.prototype)&&r!==n.prototype&&o(t,r),t}},"7db0":function(t,e,n){"use strict";var i=n("23e7"),o=n("b727").find,s=n("44d2"),r="find",a=!0;r in[]&&Array(1)[r]((function(){a=!1})),i({target:"Array",proto:!0,forced:a},{find:function(t){return o(this,t,arguments.length>1?arguments[1]:void 0)}}),s(r)},"8fc4":function(t,e,n){"use strict";n("a714")},"99af":function(t,e,n){"use strict";var i=n("23e7"),o=n("d039"),s=n("e8b5"),r=n("861d"),a=n("7b0b"),c=n("50c4"),l=n("8418"),u=n("65f0"),d=n("1dde"),f=n("b622"),h=n("2d00"),p=f("isConcatSpreadable"),m=9007199254740991,v="Maximum allowed index exceeded",g=h>=51||!o((function(){var t=[];return t[p]=!1,t.concat()[0]!==t})),k=d("concat"),b=function(t){if(!r(t))return!1;var e=t[p];return void 0!==e?!!e:s(t)},y=!g||!k;i({target:"Array",proto:!0,forced:y},{concat:function(t){var e,n,i,o,s,r=a(this),d=u(r,0),f=0;for(e=-1,i=arguments.length;e<i;e++)if(s=-1===e?r:arguments[e],b(s)){if(o=c(s.length),f+o>m)throw TypeError(v);for(n=0;n<o;n++,f++)n in s&&l(d,f,s[n])}else{if(f>=m)throw TypeError(v);l(d,f++,s)}return d.length=f,d}})},a509:function(t,e,n){"use strict";n("f486")},a714:function(t,e,n){},a9e3:function(t,e,n){"use strict";var i=n("83ab"),o=n("da84"),s=n("94ca"),r=n("6eeb"),a=n("5135"),c=n("c6b6"),l=n("7156"),u=n("c04e"),d=n("d039"),f=n("7c73"),h=n("241c").f,p=n("06cf").f,m=n("9bf2").f,v=n("58a8").trim,g="Number",k=o[g],b=k.prototype,y=c(f(b))==g,w=function(t){var e,n,i,o,s,r,a,c,l=u(t,!1);if("string"==typeof l&&l.length>2)if(l=v(l),e=l.charCodeAt(0),43===e||45===e){if(n=l.charCodeAt(2),88===n||120===n)return NaN}else if(48===e){switch(l.charCodeAt(1)){case 66:case 98:i=2,o=49;break;case 79:case 111:i=8,o=55;break;default:return+l}for(s=l.slice(2),r=s.length,a=0;a<r;a++)if(c=s.charCodeAt(a),c<48||c>o)return NaN;return parseInt(s,i)}return+l};if(s(g,!k(" 0o1")||!k("0b1")||k("+0x1"))){for(var x,_=function(t){var e=arguments.length<1?0:t,n=this;return n instanceof _&&(y?d((function(){b.valueOf.call(n)})):c(n)!=g)?l(new k(w(e)),n,_):w(e)},O=i?h(k):"MAX_VALUE,MIN_VALUE,NaN,NEGATIVE_INFINITY,POSITIVE_INFINITY,EPSILON,isFinite,isInteger,isNaN,isSafeInteger,MAX_SAFE_INTEGER,MIN_SAFE_INTEGER,parseFloat,parseInt,isInteger,fromString,range".split(","),N=0;O.length>N;N++)a(k,x=O[N])&&!a(_,x)&&m(_,x,p(k,x));_.prototype=b,b.constructor=_,r(o,g,_)}},c740:function(t,e,n){"use strict";var i=n("23e7"),o=n("b727").findIndex,s=n("44d2"),r="findIndex",a=!0;r in[]&&Array(1)[r]((function(){a=!1})),i({target:"Array",proto:!0,forced:a},{findIndex:function(t){return o(this,t,arguments.length>1?arguments[1]:void 0)}}),s(r)},e4f6:function(t,e,n){"use strict";n.r(e);var i=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("v-row",[n("NavDrawer"),n("v-dialog",{attrs:{persistent:"","max-width":"300px"},model:{value:t.dialog,callback:function(e){t.dialog=e},expression:"dialog"}},[n("v-card",[n("v-card-title",[n("span",{staticClass:"text-h5"},[t._v(t._s("Add "+t.nodeCategory[t.nodeType]+" layer"))])]),n("v-card-text",[n("v-container",[n("v-form",{ref:"form"},[n("v-row",[n("v-col",{attrs:{cols:"12"}},[n("v-text-field",{attrs:{label:"Name","prepend-icon":t.nodeIcons[t.nodeType],rules:[t.rules.length(3)]},model:{value:t.nodeLabel,callback:function(e){t.nodeLabel=e},expression:"nodeLabel"}})],1)],1)],1)],1)],1),n("v-card-actions",[n("v-spacer"),n("v-btn",{attrs:{color:"grey darken-1",text:""},on:{click:t.cancel}},[t._v(" Cancel ")]),n("v-btn",{attrs:{color:"blue darken-1",text:""},on:{click:t.add}},[t._v(" add ")])],1)],1)],1),n("v-col",{staticClass:"pa-0 accent",attrs:{cols:"12"}},[n("div",{staticClass:"d-flex flex-column float-left pt-5"},[n("v-btn",{attrs:{dark:"",plain:"",small:"",color:"text"},on:{click:function(e){t.dialog=!0}}},[n("v-icon",[t._v("mdi-plus")])],1),n("v-btn",{attrs:{small:"",plain:"",text:"",color:t.isColor(0)},on:{click:function(e){t.nodeType=0}}},[n("v-icon",[t._v("mdi-format-horizontal-align-left")])],1),n("v-btn",{attrs:{small:"",plain:"",text:"",color:t.isColor(1)},on:{click:function(e){t.nodeType=1}}},[n("v-icon",[t._v("mdi-format-horizontal-align-center")])],1),n("v-btn",{attrs:{small:"",plain:"",text:"",color:t.isColor(2)},on:{click:function(e){t.nodeType=2}}},[n("v-icon",[t._v("mdi-format-horizontal-align-right")])],1),n("v-btn",{attrs:{dark:"",plain:"",small:"",disabled:"",color:"text"},on:{click:t.save}},[n("v-icon",[t._v("mdi-cloud-download-outline")])],1),n("v-btn",{attrs:{dark:"",plain:"",small:"",color:"text"},on:{click:t.save}},[n("v-icon",[t._v("mdi-cloud-upload-outline")])],1)],1),n("div",[n("simple-flowchart",{staticClass:"accent",attrs:{scene:t.scene,height:800},on:{"update:scene":function(e){t.scene=e},nodeClick:t.nodeClick,nodeDelete:t.nodeDelete,linkBreak:t.linkBreak,linkAdded:t.linkAdded,canvasClick:t.canvasClick}})],1)])],1)},o=[],s=n("2909"),r=n("5530"),a=(n("99af"),n("d81d"),function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"flowchart-container",on:{mousemove:t.handleMove,mouseup:t.handleUp,mousedown:t.handleDown}},[n("svg",{attrs:{width:"100%",height:t.height+"px"}},t._l(t.lines,(function(e,i){return n("flowchart-link",t._b({key:"link"+i,on:{deleteLink:function(n){return t.linkDelete(e.id)}}},"flowchart-link",e,!1,!0))})),1),t._l(t.scene.nodes,(function(e,i){return n("flowchart-node",t._b({key:"node"+i,attrs:{options:t.nodeOptions},on:{linkingStart:function(n){return t.linkingStart(e.id)},linkingStop:function(n){return t.linkingStop(e.id)},nodeSelected:function(n){return t.nodeSelected(e.id,n)}}},"flowchart-node",e,!1,!0))}))],2)}),c=[],l=n("3835"),u=(n("a9e3"),n("7db0"),n("4de4"),n("c740"),function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("g",{on:{mouseover:t.handleMouseOver,mouseleave:t.handleMouseLeave}},[n("path",{style:t.pathStyle,attrs:{d:t.dAttr}}),t.show.delete?n("a",{on:{click:t.deleteLink}},[n("text",{attrs:{"text-anchor":"middle",transform:t.arrowTransform,"font-size":"22"}},[t._v("×")])]):n("path",{style:t.arrowStyle,attrs:{d:"M -1 -1 L 0 1 L 1 -1 z",transform:t.arrowTransform}})])}),d=[],f={name:"FlowchartLink",props:{start:{type:Array,default:function(){return[0,0]}},end:{type:Array,default:function(){return[0,0]}},id:Number},data:function(){return{show:{delete:!1}}},methods:{handleMouseOver:function(){this.id&&(this.show.delete=!0)},handleMouseLeave:function(){this.show.delete=!1},caculateCenterPoint:function(){var t=(this.end[0]-this.start[0])/2,e=(this.end[1]-this.start[1])/2;return[this.start[0]+t,this.start[1]+e]},caculateRotation:function(){var t=-Math.atan2(this.end[0]-this.start[0],this.end[1]-this.start[1]),e=180*t/Math.PI;return e<0?e+360:e},deleteLink:function(){this.$emit("deleteLink")}},computed:{pathStyle:function(){return{stroke:"rgb(101, 185, 244)",strokeWidth:2.73205,fill:"none"}},arrowStyle:function(){return{stroke:"rgb(101, 150, 250)",strokeWidth:6,fill:"none"}},arrowTransform:function(){var t=this.caculateCenterPoint(),e=Object(l["a"])(t,2),n=e[0],i=e[1],o=this.caculateRotation();return"translate(".concat(n,", ").concat(i,") rotate(").concat(o,")")},dAttr:function(){var t=this.start[0],e=this.start[1],n=this.end[0],i=this.end[1],o=t,s=e+50,r=n,a=i-50;return"M ".concat(t,", ").concat(e," C ").concat(o,", ").concat(s,", ").concat(r,", ").concat(a,", ").concat(n,", ").concat(i)}}},h=f,p=(n("1739"),n("2877")),m=Object(p["a"])(h,u,d,!1,null,"3852d28d",null),v=m.exports,g=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"flowchart-node",class:{selected:t.options.selected===t.id},style:t.nodeStyle,on:{mousedown:t.handleMousedown,mouseover:t.handleMouseOver,mouseleave:t.handleMouseLeave}},["input"!==t.type?n("div",{staticClass:"node-port node-input",on:{mousedown:t.inputMouseDown,mouseup:t.inputMouseUp}}):t._e(),n("div",{class:"node-main "+t.type},[n("div",{domProps:{textContent:t._s(t.type)}}),n("div",{domProps:{textContent:t._s(t.label)}})]),"output"!==t.type?n("div",{staticClass:"node-port node-output",on:{mousedown:t.outputMouseDown}}):t._e(),n("div",{directives:[{name:"show",rawName:"v-show",value:t.show.delete,expression:"show.delete"}],staticClass:"node-delete"},[t._v("×")])])},k=[],b={name:"FlowchartNode",props:{id:{type:Number,default:1e3,validator:function(t){return"number"===typeof t}},x:{type:Number,default:0,validator:function(t){return"number"===typeof t}},y:{type:Number,default:0,validator:function(t){return"number"===typeof t}},type:{type:String,default:"Default"},label:{type:String,default:"input name"},options:{type:Object,default:function(){return{centerX:1024,scale:1,centerY:140}}}},data:function(){return{show:{delete:!1}}},mounted:function(){},computed:{nodeStyle:function(){return{top:this.options.centerY+this.y*this.options.scale+"px",left:this.options.centerX+this.x*this.options.scale+"px",transform:"scale(".concat(this.options.scale,")")}}},methods:{handleMousedown:function(t){var e=t.target||t.srcElement;e.className.indexOf("node-input")<0&&e.className.indexOf("node-output")<0&&this.$emit("nodeSelected",t),t.preventDefault()},handleMouseOver:function(){this.show.delete=!0},handleMouseLeave:function(){this.show.delete=!1},outputMouseDown:function(t){this.$emit("linkingStart"),t.preventDefault()},inputMouseDown:function(t){t.preventDefault()},inputMouseUp:function(t){this.$emit("linkingStop"),t.preventDefault()}}},y=b,w=(n("220c"),Object(p["a"])(y,g,k,!1,null,"22858b8e",null)),x=w.exports;function _(t){var e=t.getBoundingClientRect(),n=window.pageYOffset,i=window.pageXOffset,o=e.top+n,s=e.left+i;return{top:Math.round(o),left:Math.round(s)}}function O(t,e){var n=e.pageX||e.clientX+document.documentElement.scrollLeft,i=e.pageY||e.clientY+document.documentElement.scrollTop,o=_(t),s=n-o.left,r=i-o.top;return[s,r]}var N={name:"VueFlowchart",props:{scene:{type:Object,default:function(){return{centerX:1024,scale:1,centerY:140,nodes:[],links:[]}}},height:{type:Number,default:400}},data:function(){return{action:{linking:!1,dragging:!1,scrolling:!1,selected:0},mouse:{x:0,y:0,lastX:0,lastY:0},draggingLink:null,rootDivOffset:{top:0,left:0}}},components:{FlowchartLink:v,FlowchartNode:x},computed:{nodeOptions:function(){return{centerY:this.scene.centerY,centerX:this.scene.centerX,scale:this.scene.scale,offsetTop:this.rootDivOffset.top,offsetLeft:this.rootDivOffset.left,selected:this.action.selected}},lines:function(){var t=this,e=this.scene.links.map((function(e){var n,i,o,s,r,a,c=t.findNodeWithID(e.from),u=t.findNodeWithID(e.to);n=t.scene.centerX+c.x,i=t.scene.centerY+c.y;var d=t.getPortPosition("bottom",n,i),f=Object(l["a"])(d,2);s=f[0],o=f[1],n=t.scene.centerX+u.x,i=t.scene.centerY+u.y;var h=t.getPortPosition("top",n,i),p=Object(l["a"])(h,2);return r=p[0],a=p[1],{start:[s,o],end:[r,a],id:e.id}}));if(this.draggingLink){var n,i,o,s,r=this.findNodeWithID(this.draggingLink.from);n=this.scene.centerX+r.x,i=this.scene.centerY+r.y;var a=this.getPortPosition("bottom",n,i),c=Object(l["a"])(a,2);s=c[0],o=c[1],e.push({start:[s,o],end:[this.draggingLink.mx,this.draggingLink.my]})}return e}},mounted:function(){this.rootDivOffset.top=this.$el?this.$el.offsetTop:0,this.rootDivOffset.left=this.$el?this.$el.offsetLeft:0},methods:{findNodeWithID:function(t){return this.scene.nodes.find((function(e){return t===e.id}))},getPortPosition:function(t,e,n){return"top"===t?[e+80,n]:"bottom"===t?[e+80,n+40]:void 0},linkingStart:function(t){this.action.linking=!0,this.draggingLink={from:t,mx:0,my:0}},linkingStop:function(t){var e=this;if(this.draggingLink&&this.draggingLink.from!==t){var n=this.scene.links.find((function(n){return n.from===e.draggingLink.from&&n.to===t}));if(!n){var i=Math.max.apply(Math,[0].concat(Object(s["a"])(this.scene.links.map((function(t){return t.id}))))),o={id:i+1,from:this.draggingLink.from,to:t};this.scene.links.push(o),this.$emit("linkAdded",o)}}this.draggingLink=null},linkDelete:function(t){var e=this.scene.links.find((function(e){return e.id===t}));e&&(this.scene.links=this.scene.links.filter((function(e){return e.id!==t})),this.$emit("linkBreak",e))},nodeSelected:function(t,e){this.action.dragging=t,this.action.selected=t,this.$emit("nodeClick",t),this.mouse.lastX=e.pageX||e.clientX+document.documentElement.scrollLeft,this.mouse.lastY=e.pageY||e.clientY+document.documentElement.scrollTop},handleMove:function(t){if(this.action.linking){var e=O(this.$el,t),n=Object(l["a"])(e,2);this.mouse.x=n[0],this.mouse.y=n[1];var i=[this.mouse.x,this.mouse.y];this.draggingLink.mx=i[0],this.draggingLink.my=i[1]}if(this.action.dragging){this.mouse.x=t.pageX||t.clientX+document.documentElement.scrollLeft,this.mouse.y=t.pageY||t.clientY+document.documentElement.scrollTop;var o=this.mouse.x-this.mouse.lastX,s=this.mouse.y-this.mouse.lastY;this.mouse.lastX=this.mouse.x,this.mouse.lastY=this.mouse.y,this.moveSelectedNode(o,s)}if(this.action.scrolling){var r=O(this.$el,t),a=Object(l["a"])(r,2);this.mouse.x=a[0],this.mouse.y=a[1];var c=this.mouse.x-this.mouse.lastX,u=this.mouse.y-this.mouse.lastY;this.mouse.lastX=this.mouse.x,this.mouse.lastY=this.mouse.y,this.scene.centerX+=c,this.scene.centerY+=u}},handleUp:function(t){var e=t.target||t.srcElement;this.$el.contains(e)&&(("string"!==typeof e.className||e.className.indexOf("node-input")<0)&&(this.draggingLink=null),"string"===typeof e.className&&e.className.indexOf("node-delete")>-1&&this.nodeDelete(this.action.dragging)),this.action.linking=!1,this.action.dragging=null,this.action.scrolling=!1},handleDown:function(t){var e=t.target||t.srcElement;if((e===this.$el||e.matches("svg, svg *"))&&1===t.which){this.action.scrolling=!0;var n=O(this.$el,t),i=Object(l["a"])(n,2);this.mouse.lastX=i[0],this.mouse.lastY=i[1],this.action.selected=null}this.$emit("canvasClick",t)},moveSelectedNode:function(t,e){var n=this,i=this.scene.nodes.findIndex((function(t){return t.id===n.action.dragging})),o=this.scene.nodes[i].x+t/this.scene.scale,s=this.scene.nodes[i].y+e/this.scene.scale;this.$set(this.scene.nodes,i,Object.assign(this.scene.nodes[i],{x:o,y:s}))},nodeDelete:function(t){this.scene.nodes=this.scene.nodes.filter((function(e){return e.id!==t})),this.scene.links=this.scene.links.filter((function(e){return e.from!==t&&e.to!==t})),this.$emit("nodeDelete",t)}}},L=N,C=(n("a509"),Object(p["a"])(L,a,c,!1,null,"f7d5f83e",null)),D=C.exports,M=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("v-navigation-drawer",{attrs:{width:"500",right:"",fixed:"",color:"accent",dark:""},model:{value:t.drawer,callback:function(e){t.drawer=e},expression:"drawer"}},[n("v-list-item",[n("v-list-item-content",[n("v-list-item-title",{staticClass:"text-h6"},[t._v(t._s(t.app.name))]),n("v-list-item-subtitle",[t._v(t._s("ver. "+t.app.version))])],1)],1),n("v-divider"),n("v-row",{staticClass:"pa-3"},[n("v-form",{ref:"form"},[n("v-col",{attrs:{cols:"12"}},[t._l(t.settings,(function(e,i){return["input"===e.type?n("v-text-field",{key:"input"+i,attrs:{label:e.label,placeholder:"Placeholder",outlined:"",dense:""},model:{value:e.value,callback:function(n){t.$set(e,"value",n)},expression:"setting.value"}}):t._e(),"select"===e.type?n("v-select",{key:"select"+i,attrs:{label:e.label,items:e.items,outlined:"",dense:""},model:{value:e.value,callback:function(n){t.$set(e,"value",n)},expression:"setting.value"}}):t._e()]})),n("v-btn",{attrs:{elevation:"2"},on:{click:t.click}},[t._v("Click")])],2)],1)],1)],1)},S=[],$=n("2f62"),I={data:function(){return{right:null,settings:[{value:"Test1",type:"input",label:"Name"},{value:"Test2",type:"input",label:"Title"},{value:2,type:"select",label:"Title",items:[1,2,3,4]}]}},computed:Object(r["a"])(Object(r["a"])({},Object($["b"])({app:"settings/getApp",menus:"settings/getMenus"})),{},{drawer:{set:function(t){this.$store.dispatch("settings/setDrawer",t)},get:function(){return this.$store.getters["settings/getDrawer"]}}}),methods:{click:function(){console.log(this.settings)}}},E=I,T=Object(p["a"])(E,M,S,!1,null,null,null),A=T.exports,j={name:"Modeling",components:{SimpleFlowchart:D,NavDrawer:A},data:function(){return{dialog:!1,nodeType:1,nodeLabel:"",nodeCategory:["input","action","output"],nodeIcons:["mdi-format-horizontal-align-left","mdi-format-horizontal-align-center","mdi-format-horizontal-align-right"],rules:{length:function(t){return function(e){return(e||"").length>=t||"Length < ".concat(t)}}}}},computed:Object(r["a"])(Object(r["a"])({},Object($["b"])({scene:"data/getData"})),{},{drawer:{set:function(t){this.$store.dispatch("settings/setDrawer",t)},get:function(){return this.$store.getters["settings/getDrawer"]}}}),methods:{canvasClick:function(t){console.log("canvas Click, event:",t),console.log(t.type)},add:function(){if(this.$refs.form.validate()){var t=Math.max.apply(Math,[0].concat(Object(s["a"])(this.scene.nodes.map((function(t){return t.id})))));this.scene.nodes.push({id:t+1,x:-400,y:-100,type:this.nodeCategory[this.nodeType],label:this.nodeLabel?this.nodeLabel:"test".concat(t+1)}),this.nodeLabel="",this.dialog=!1}},cancel:function(){console.log(this.$refs.form.reset()),this.nodeLabel="",this.dialog=!1},save:function(){var t=this.scene,e=t.nodes,n=t.links;console.log({nodes:e,links:n}),alert(JSON.stringify({nodes:e,links:n}))},nodeClick:function(t){console.log("node click",t)},nodeDelete:function(t){console.log("node delete",t)},linkBreak:function(t){console.log("link break",t)},linkAdded:function(t){console.log("new link added:",t)},isColor:function(t){return this.nodeType!==t?"text":"white"}}},X=j,Y=(n("8fc4"),Object(p["a"])(X,i,o,!1,null,"3064c422",null));e["default"]=Y.exports},f486:function(t,e,n){}}]);
//# sourceMappingURL=chunk-24f5e6a7.efe8475e.js.map