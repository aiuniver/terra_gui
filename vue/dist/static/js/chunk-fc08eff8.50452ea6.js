(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-fc08eff8"],{"002d":function(e,t,r){"use strict";var s=function(){var e=this,t=e.$createElement,r=e._self._c||t;return r("div",{class:["t-input"]},["checkbox"===(e.type||"text")?r("input",e._b({directives:[{name:"model",rawName:"v-model",value:e.input,expression:"input"}],class:["t-input__input",{"t-input__input--error":e.error},{"t-input__input--small":e.small}],attrs:{name:e.name||e.parse,"data-degree":e.degree,autocomplete:"off",disabled:e.isDisabled,type:"checkbox"},domProps:{checked:Array.isArray(e.input)?e._i(e.input,null)>-1:e.input},on:{blur:e.change,focus:e.focus,mouseover:function(t){e.hover=!0},mouseleave:function(t){e.hover=!1},change:function(t){var r=e.input,s=t.target,n=!!s.checked;if(Array.isArray(r)){var i=null,a=e._i(r,i);s.checked?a<0&&(e.input=r.concat([i])):a>-1&&(e.input=r.slice(0,a).concat(r.slice(a+1)))}else e.input=n}}},"input",e.$attrs,!1)):"radio"===(e.type||"text")?r("input",e._b({directives:[{name:"model",rawName:"v-model",value:e.input,expression:"input"}],class:["t-input__input",{"t-input__input--error":e.error},{"t-input__input--small":e.small}],attrs:{name:e.name||e.parse,"data-degree":e.degree,autocomplete:"off",disabled:e.isDisabled,type:"radio"},domProps:{checked:e._q(e.input,null)},on:{blur:e.change,focus:e.focus,mouseover:function(t){e.hover=!0},mouseleave:function(t){e.hover=!1},change:function(t){e.input=null}}},"input",e.$attrs,!1)):r("input",e._b({directives:[{name:"model",rawName:"v-model",value:e.input,expression:"input"}],class:["t-input__input",{"t-input__input--error":e.error},{"t-input__input--small":e.small}],attrs:{name:e.name||e.parse,"data-degree":e.degree,autocomplete:"off",disabled:e.isDisabled,type:e.type||"text"},domProps:{value:e.input},on:{blur:e.change,focus:e.focus,mouseover:function(t){e.hover=!0},mouseleave:function(t){e.hover=!1},input:function(t){t.target.composing||(e.input=t.target.value)}}},"input",e.$attrs,!1)),e.error&&e.hover?r("div",{class:["t-field__hint",{"t-field__hint--big":!e.small}]},[r("span",[e._v(e._s(e.error))])]):e._e()])},n=[],i=(r("a9e3"),r("caad"),r("2532"),r("b0c0"),{name:"t-input-new",props:{type:String,value:[String,Number],parse:String,name:String,small:Boolean,error:String,degree:Number,disabled:[Boolean,Array]},data:function(){return{isChange:!1,hover:!1}},computed:{isDisabled:function(){return Array.isArray(this.disabled)?!!this.disabled.includes(this.name):this.disabled},input:{set:function(e){this.$emit("input",e),this.isChange=!0},get:function(){return this.value}}},methods:{label:function(){this.$el.children[0].focus()},focus:function(e){this.$emit("focus",e),this.error&&this.$emit("clean",!0)},change:function(e){var t=e.target;if(this.isChange){var r="number"===this.type?+t.value:t.value;this.$emit("change",{name:this.name,value:r}),this.$emit("parse",{name:this.name,parse:this.parse,value:r}),this.isChange=!1}}},created:function(){this.input=this.value}}),a=i,o=(r("0cf2"),r("2877")),c=Object(o["a"])(a,s,n,!1,null,"c540ef36",null);t["a"]=c.exports},"0cf2":function(e,t,r){"use strict";r("a5a4")},"214c":function(e,t,r){"use strict";r("c9b3")},"39b9a":function(e,t,r){"use strict";r.r(t);var s=function(){var e=this,t=e.$createElement,r=e._self._c||t;return r("div",{staticClass:"page-servers"},[r("scrollbar",{staticClass:"page-servers__scroll",on:{"handle-scroll":e.handleScroll}},[r("span",{staticClass:"page-servers__btn",style:{left:e.scrollLeft+"px",top:e.scrollTop+"px"},on:{click:function(t){e.addNew=!0}}},[r("i",{staticClass:"ci-icon ci-plus_circle"}),r("span",[e._v("Добавить сервер")])]),r("div",{staticClass:"page-servers__list"},[r("LoadSpiner",{directives:[{name:"show",rawName:"v-show",value:e.fetchingServers,expression:"fetchingServers"}],attrs:{text:"Получение списка серверов"}}),r("ServerTable",{directives:[{name:"show",rawName:"v-show",value:e.showTable&&!e.fetchingServers,expression:"showTable && !fetchingServers"}],attrs:{servers:e.servers},on:{instruction:e.openInstruction}}),r("p",{directives:[{name:"show",rawName:"v-show",value:!e.showTable&&!e.fetchingServers,expression:"!showTable && !fetchingServers"}],staticClass:"page-servers__noserver"},[e._v(" Нет добавленных серверов демо-панелий ")])],1)]),r("div",{staticClass:"page-servers__new"},[e.addNew?r("NewServer",{on:{addServer:e.newServer}}):e._e()],1),r("at-modal",{staticClass:"modal",attrs:{okText:"Читать инструкцию"},on:{"on-confirm":function(t){return e.openInstruction(e.serverID)}},scopedSlots:e._u([{key:"header",fn:function(){return[r("span",{staticClass:"modal-title"},[e._v("Сервер демо-панели добавлен")])]},proxy:!0}]),model:{value:e.serverModal,callback:function(t){e.serverModal=t},expression:"serverModal"}},[r("p",[e._v(" Ознакомьтесь с дальнейшими действиями в "),r("span",{staticClass:"clickable",on:{click:function(t){return e.openInstruction(e.serverID)}}},[e._v("Инструкции")])]),r("p",[e._v("Вы также сможете найти ее в таблице серверов на владке Серверы демо-панелей в вашем Профиле")])]),r("at-modal",{staticClass:"modal",attrs:{showConfirmButton:!1,showCancelButton:!1,width:600},on:{"on-cancel":function(t){e.buffer=""}},scopedSlots:e._u([{key:"header",fn:function(){return[r("span",{staticClass:"modal-title"},[e._v("Инструкция по настройке сервера демо-панели")])]},proxy:!0}]),model:{value:e.InstructionModal,callback:function(t){e.InstructionModal=t},expression:"InstructionModal"}},[r("div",{class:["server-state",""+e.selectedServer.state.name]},[e._v(e._s(e.selectedServer.state.value))]),r("div",{staticClass:"server-info"},e._l(e.selectedServer.info,(function(t,s){return r("div",{key:s,staticClass:"server-info__item"},[r("p",{staticClass:"label"},[e._v(e._s(s))]),r("p",{staticClass:"value"},[e._v(e._s(t))])])})),0),r("div",{staticClass:"ssh-wrapper"},[r("div",{staticClass:"ssh-wrapper__item"},[r("span",{staticClass:"ssh"},[e._v("Приватный SSH-ключ")]),r("i",{staticClass:"btn-copy",attrs:{title:"Скопировать"},on:{click:function(t){return e.copy("private")}}}),r("a",{staticClass:"clickable",attrs:{href:e.privateURI,download:"id_rsa"}},[e._v("Скачать")]),r("span",{directives:[{name:"show",rawName:"v-show",value:"private"===e.buffer,expression:"buffer === 'private'"}],staticClass:"buffer"},[e._v("Ключ скопирован в буффер обмена")])]),r("div",{staticClass:"ssh-wrapper__item"},[r("span",{staticClass:"ssh"},[e._v("Публичный SSH-ключ")]),r("i",{staticClass:"btn-copy",attrs:{title:"Скопировать"},on:{click:function(t){return e.copy("public")}}}),r("a",{staticClass:"clickable",attrs:{href:e.publicURI,download:"id_rsa.pub"}},[e._v("Скачать")]),r("span",{directives:[{name:"show",rawName:"v-show",value:"public"===e.buffer,expression:"buffer === 'public'"}],staticClass:"buffer"},[e._v("Ключ скопирован в буффер обмена")])])]),r("hr"),r("div",{staticClass:"instruction",domProps:{innerHTML:e._s(e.instruction)}})])],1)},n=[],i=r("1da1"),a=r("5530"),o=(r("96cf"),r("b64b"),r("7db0"),function(){var e=this,t=e.$createElement,r=e._self._c||t;return r("table",{staticClass:"server-table"},[e._m(0),r("tbody",e._l(e.servers,(function(t){return r("tr",{key:t.id},[r("td",[e._v(e._s(t.domain_name))]),r("td",[e._v(e._s(t.ip_address))]),r("td",[e._v(e._s(t.user))]),r("td",[e._v(e._s(t.port_ssh))]),r("td",[e._v(e._s(t.port_http))]),r("td",[e._v(e._s(t.port_https))]),r("td",{staticClass:"clickable"},[r("span",{on:{click:function(r){return e.instruction(t.id)}}},[e._v("Открыть")])]),r("td",{class:[""+t.state.name]},[e._v(e._s(t.state.value))]),r("td",{staticClass:"clickable",on:{click:function(r){return e.setup(t.id)}}},[r("i",{class:["ci-icon",e.getIcon(t.state.name)]}),r("span",[e._v(e._s(e.getAction(t.state.name)))])])])})),0)])}),c=[function(){var e=this,t=e.$createElement,r=e._self._c||t;return r("thead",[r("tr",[r("th",[e._v("Доменное имя")]),r("th",[e._v("IP адрес")]),r("th",[e._v("Имя пользователя")]),r("th",[e._v("SSH порт")]),r("th",[e._v("HTTP порт")]),r("th",[e._v("HTTPS порт")]),r("th",[e._v("Инструкция")]),r("th",[e._v("Состояние")]),r("th",{staticStyle:{"min-width":"144px"}})])])}],l={name:"ServerTable",props:{servers:[Array,Object]},methods:{instruction:function(e){this.$emit("instruction",e)},setup:function(e){this.$store.dispatch("servers/setup",{id:e})},getIcon:function(e){return"ready"===e?"ci-refresh":"waiting"===e?"":"ci-play_arrow"},getAction:function(e){return"ready"===e?"Обновить":"waiting"===e?"":"Установить"}}},u=l,p=(r("d5f2"),r("2877")),d=Object(p["a"])(u,o,c,!1,null,"66848bf4",null),h=d.exports,v=function(){var e=this,t=e.$createElement,r=e._self._c||t;return r("div",{staticClass:"new-server"},[r("p",{staticClass:"new-server__header"},[e._v("Добавление сервера демо-панели")]),r("form",{staticClass:"new-server__form",on:{submit:function(t){return t.preventDefault(),e.addServer.apply(null,arguments)}}},[r("div",{directives:[{name:"show",rawName:"v-show",value:e.loading,expression:"loading"}],staticClass:"new-server__form--overlay"}),r("t-field",{attrs:{label:"Доменное имя"}},[r("TInputNew",{attrs:{placeholder:""},model:{value:e.domain_name,callback:function(t){e.domain_name=t},expression:"domain_name"}})],1),r("t-field",{attrs:{label:"IP адрес"}},[r("VueIP",{attrs:{ip:e.ip_address,onChange:e.change}})],1),r("t-field",{attrs:{label:"Имя пользователя"}},[r("TInputNew",{attrs:{placeholder:""},model:{value:e.user,callback:function(t){e.user=t},expression:"user"}})],1),r("div",{staticClass:"new-server__ports"},[r("t-field",{attrs:{label:"SSH порт"}},[r("d-input-number",{attrs:{placeholder:""},model:{value:e.port_ssh,callback:function(t){e.port_ssh=t},expression:"port_ssh"}})],1),r("t-field",{attrs:{label:"HTTP порт"}},[r("d-input-number",{attrs:{placeholder:""},model:{value:e.port_http,callback:function(t){e.port_http=t},expression:"port_http"}})],1),r("t-field",{attrs:{label:"HTTPS порт"}},[r("d-input-number",{attrs:{placeholder:""},model:{value:e.port_https,callback:function(t){e.port_https=t},expression:"port_https"}})],1)],1),r("d-button",{staticClass:"new-server__btn",attrs:{disabled:!e.validForm||e.loading}},[e._v("Добавить")])],1),r("LoadSpiner",{directives:[{name:"show",rawName:"v-show",value:e.loading,expression:"loading"}],attrs:{text:""}})],1)},f=[],m=function(){var e=this,t=e.$createElement,r=e._self._c||t;return r("span",{staticClass:"vue-ip",class:{"show-port":!1!==e.portCopy,"material-theme":"material"===e.theme,active:e.active,valid:e.valid}},e._l(e.ipCopy,(function(t,s){return r("div",{key:s,staticClass:"segment"},[r("input",{directives:[{name:"model",rawName:"v-model",value:e.ipCopy[s],expression:"ipCopy[index]"}],ref:"ipSegment",refInFor:!0,attrs:{type:"number",placeholder:"___",maxlength:"3"},domProps:{value:e.ipCopy[s]},on:{paste:function(t){return e.paste(t)},keydown:function(t){return e.ipKeydown(t,s)},focus:function(t){return e.ipFocus(s)},blur:e.blur,input:[function(t){t.target.composing||e.$set(e.ipCopy,s,t.target.value)},e.input]}})])})),0)},_=[],b=(r("a9e3"),r("ac1f"),r("1276"),r("d81d"),r("a15b"),{props:{onChange:Function,ip:{required:!0,type:String},port:{type:[String,Number,Boolean],default:!1},placeholder:{type:[Boolean],default:!1},theme:{type:[String,Boolean],default:!1}},data:function(){return{ipCopy:["","","",""],portCopy:null,valid:!1,active:!1}},beforeMount:function(){this.copyValue(this.ip,this.port)},watch:{ip:function(e){this.copyValue(e,this.port)},port:function(e){this.copyValue(this.ip,e)}},methods:{input:function(e){e.target.value||(e.target.value="")},placeholderPos:function(e){if(!this.placeholder)return"";switch(e){case 0:return"192";case 1:return"168";case 2:return"0";case 3:return"1"}},ipFocus:function(e){this.active=!0,this.ipCopy[e]="",this.changed()},clearAll:function(){this.ipCopy=["","","",""],this.portCopy=null,this.valid=!1},blur:function(){this.active=!1},portFocus:function(){this.active=!0,this.portCopy=null,this.changed()},paste:function(e){this.$refs.ipSegment[0].focus();var t=e.clipboardData.getData("text/plain"),r=t.indexOf(":");if(!1===this.port){console.warn("A IP address with a port has been entered but this module has no port attribute. Please enable it update the port."),this.clearAll();var s=t.split(":");return this.copyValue(s[0],!1),void this.$refs.ipSegment[0].blur()}switch(r){case-1:this.copyValue(t,null),this.changed(),this.$refs.ipSegment[0].blur();break;default:var n=t.split(":");this.copyValue(n[0],n[1]),this.changed(),this.$refs.ipSegment[0].blur();break}},ipKeydown:function(e,t){var r=this,s=e.keyCode||e.which;8!==s&&37!==s||0===this.ipCopy[t].length&&void 0!==this.ipCopy[t-1]&&this.$refs.ipSegment[t-1].focus(),setTimeout((function(){"0"===r.ipCopy[t]?r.moveToNextIpSegment(t,!1):r.moveToNextIpSegment(t),r.changed()}))},moveToNextIpSegment:function(e){var t=!(arguments.length>1&&void 0!==arguments[1])||arguments[1];t?this.ipCopy[e].length>=3&&void 0!==this.ipCopy[e+1]&&this.$refs.ipSegment[e+1].focus():t||void 0!==this.ipCopy[e+1]&&this.$refs.ipSegment[e+1].focus()},changed:function(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:this.ipCopy,t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:this.portCopy,r=this.arrayToIp(e);this.onChange(r,t,this.validateIP(e))},copyValue:function(e,t){e&&this.ipToArray(e),this.portCopy=t,this.valid=this.validateIP(this.ipCopy),this.changed()},ipToArray:function(e){var t=[];e.split(".").map((function(e){(isNaN(e)||e<0||e>255)&&(e=255),t.push(e)})),4!==t.length?(console.error("Not valid, so clearing ip",t),this.clearAll()):this.ipCopy=t},arrayToIp:function(e){return e.join(".")},validateIP:function(e){var t=this.arrayToIp(e);return/^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/.test(t)}}}),g=b,y=(r("bd97"),Object(p["a"])(g,m,_,!1,null,"74740840",null)),w=y.exports,C=r("1636"),S=r("002d"),k={name:"NewServer",components:{VueIP:w,LoadSpiner:C["default"],TInputNew:S["a"]},data:function(){return{ip_address:"",ipValid:null,domain_name:"",user:"",port_ssh:22,port_http:80,port_https:443,loading:!1}},computed:{validForm:function(){return!!(this.ipValid&&this.domain_name&&this.user&&this.port_ssh&&this.port_http&&this.port_https)}},methods:{change:function(e,t,r){this.ip_address=e,this.ipValid=r},addServer:function(){var e=this;return Object(i["a"])(regeneratorRuntime.mark((function t(){var r,s;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return e.loading=!0,t.next=3,e.$store.dispatch("servers/addServer",{domain_name:e.domain_name,ip_address:e.ip_address,user:e.user,port_ssh:e.port_ssh,port_http:e.port_http,port_https:e.port_https});case 3:r=t.sent,s=r.id,e.$emit("addServer",s),e.loading=!1;case 7:case"end":return t.stop()}}),t)})))()}}},x=k,I=(r("214c"),Object(p["a"])(x,v,f,!1,null,"fd66c1aa",null)),T=I.exports,$=r("2f62"),N={name:"servers",components:{ServerTable:h,NewServer:T,LoadSpiner:C["default"]},data:function(){return{addNew:!1,serverModal:!1,InstructionModal:!1,serverID:null,private_key:null,public_key:null,instruction:null,fetchingServers:!1,buffer:"",scrollLeft:0,scrollTop:0,intervalID:null}},computed:Object(a["a"])(Object(a["a"])({},Object($["c"])({servers:"servers/getServers"})),{},{showTable:function(){return!!Object.keys(this.$store.getters["servers/getServers"]).length},getServer:function(){var e=this;return this.servers.find((function(t){return t.id===e.serverID}))},privateURI:function(){return"data:application/octet-stream;charset=utf-8,"+encodeURIComponent(this.private_key)},publicURI:function(){return"data:application/octet-stream;charset=utf-8,"+encodeURIComponent(this.public_key)},selectedServer:function(){var e=this,t=this.$store.getters["servers/getServers"].find((function(t){return t.id===e.serverID}));return t?{info:{"Доменное имя":t.domain_name,"Имя пользователя":t.user,"HTTP порт":t.port_http,"IP адерс":t.ip_address,"SSH порт":t.port_ssh,"HTTPS порт":t.port_https},state:t.state}:{state:""}}}),methods:{copy:function(e){var t="private"===e?this.private_key:this.public_key,r=document.createElement("input");document.body.appendChild(r),r.value=t,r.select(),document.execCommand("copy"),r.remove(),this.buffer=e},newServer:function(e){this.serverID=e,this.addNew=!1,this.serverModal=!0},openInstruction:function(e){var t=this;return Object(i["a"])(regeneratorRuntime.mark((function r(){var s,n;return regeneratorRuntime.wrap((function(r){while(1)switch(r.prev=r.next){case 0:return t.serverID=e,t.serverModal=!1,t.InstructionModal=!0,r.next=5,t.$store.dispatch("servers/getInstruction",{id:e});case 5:s=r.sent,n=s.data,t.private_key=n.private_ssh_key,t.public_key=n.public_ssh_key,t.instruction=n.instruction;case 10:case"end":return r.stop()}}),r)})))()},handleScroll:function(e,t){this.scrollLeft=t.scrollLeft,this.scrollTop=e.scrollTop}},created:function(){var e=this;return Object(i["a"])(regeneratorRuntime.mark((function t(){return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return e.fetchingServers=!0,t.next=3,e.$store.dispatch("servers/getServers");case 3:e.intervalID=setInterval(Object(i["a"])(regeneratorRuntime.mark((function t(){return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("servers/getServers");case 2:case"end":return t.stop()}}),t)}))),6e4),e.fetchingServers=!1,e.$router.afterEach((function(){clearInterval(e.intervalID)}));case 6:case"end":return t.stop()}}),t)})))()}},P=N,D=(r("bae3"),Object(p["a"])(P,s,n,!1,null,"27dd62e0",null));t["default"]=D.exports},"9cab":function(e,t,r){},a5a4:function(e,t,r){},bae3:function(e,t,r){"use strict";r("d099")},bd97:function(e,t,r){"use strict";r("9cab")},c9b3:function(e,t,r){},d099:function(e,t,r){},d5f2:function(e,t,r){"use strict";r("efd4")},efd4:function(e,t,r){}}]);