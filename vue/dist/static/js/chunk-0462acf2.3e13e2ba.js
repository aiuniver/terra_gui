(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-0462acf2"],{"323f":function(e,t,s){},"51ba":function(e,t,s){"use strict";var r=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("table",{staticClass:"server-table"},[e._m(0),s("tbody",e._l(e.servers,(function(t){return s("tr",{key:t.id},[s("td",[e._v(e._s(t.domain_name))]),s("td",[e._v(e._s(t.ip_address))]),s("td",[e._v(e._s(t.user))]),s("td",[e._v(e._s(t.port_ssh))]),s("td",[e._v(e._s(t.port_http))]),s("td",[e._v(e._s(t.port_https))]),s("td",{staticClass:"clickable"},[s("span",{on:{click:function(s){return e.instruction(t.id)}}},[e._v("Открыть")])]),s("td",{class:[""+t.state.name]},[e._v(e._s(t.state.value))]),s("td",{staticClass:"clickable",on:{click:function(s){return e.setup(t.id)}}},[s("i",{class:["ci-icon",e.getIcon(t.state.name)]}),s("span",[e._v(e._s(e.getAction(t.state.name)))])])])})),0)])},n=[function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("thead",[s("tr",[s("th",[e._v("Доменное имя")]),s("th",[e._v("IP адрес")]),s("th",[e._v("Имя пользователя")]),s("th",[e._v("SSH порт")]),s("th",[e._v("HTTP порт")]),s("th",[e._v("HTTPS порт")]),s("th",[e._v("Инструкция")]),s("th",[e._v("Состояние")]),s("th",{staticStyle:{"min-width":"144px"}})])])}],a={name:"ServerTable",props:{servers:[Array,Object]},methods:{instruction:function(e){this.$emit("instruction",e)},setup:function(e){this.$store.dispatch("servers/setup",{id:e})},getIcon:function(e){return"ready"===e?"ci-refresh":"waiting"===e?"":"ci-play_arrow"},getAction:function(e){return"ready"===e?"Обновить":"waiting"===e?"":"Установить"}}},i=a,o=(s("d5f2"),s("2877")),c=Object(o["a"])(i,r,n,!1,null,"66848bf4",null);t["a"]=c.exports},"5bc3":function(e,t,s){"use strict";s.r(t);var r=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"page-servers"},[s("scrollbar",{staticClass:"page-servers__scroll",on:{"handle-scroll":e.handleScroll}},[s("span",{staticClass:"page-servers__btn",style:{left:e.scrollLeft+"px",top:e.scrollTop+"px"},on:{click:function(t){e.addNew=!0}}},[s("i",{staticClass:"ci-icon ci-plus_circle"}),s("span",[e._v("Добавить сервер")])]),s("div",{staticClass:"page-servers__list"},[s("LoadSpiner",{directives:[{name:"show",rawName:"v-show",value:e.fetchingServers,expression:"fetchingServers"}],attrs:{text:"Получение списка серверов"}}),s("ServerTable",{directives:[{name:"show",rawName:"v-show",value:e.showTable&&!e.fetchingServers,expression:"showTable && !fetchingServers"}],attrs:{servers:e.servers},on:{instruction:e.openInstruction}}),s("p",{directives:[{name:"show",rawName:"v-show",value:!e.showTable&&!e.fetchingServers,expression:"!showTable && !fetchingServers"}],staticClass:"page-servers__noserver"},[e._v(" Нет добавленных серверов демо-панелий ")])],1)]),s("div",{staticClass:"page-servers__new"},[e.addNew?s("NewServer",{on:{addServer:e.newServer}}):e._e()],1),s("at-modal",{staticClass:"modal",attrs:{okText:"Читать инструкцию"},on:{"on-confirm":function(t){return e.openInstruction(e.serverID)}},scopedSlots:e._u([{key:"header",fn:function(){return[s("span",{staticClass:"modal-title"},[e._v("Сервер демо-панели добавлен")])]},proxy:!0}]),model:{value:e.serverModal,callback:function(t){e.serverModal=t},expression:"serverModal"}},[s("p",[e._v(" Ознакомьтесь с дальнейшими действиями в "),s("span",{staticClass:"clickable",on:{click:function(t){return e.openInstruction(e.serverID)}}},[e._v("Инструкции")])]),s("p",[e._v("Вы также сможете найти ее в таблице серверов на владке Серверы демо-панелей в вашем Профиле")])]),s("at-modal",{staticClass:"modal",attrs:{showConfirmButton:!1,showCancelButton:!1,width:600},on:{"on-cancel":function(t){e.buffer=""}},scopedSlots:e._u([{key:"header",fn:function(){return[s("span",{staticClass:"modal-title"},[e._v("Инструкция по настройке сервера демо-панели")])]},proxy:!0}]),model:{value:e.InstructionModal,callback:function(t){e.InstructionModal=t},expression:"InstructionModal"}},[s("div",{class:["server-state",""+e.selectedServer.state.name]},[e._v(e._s(e.selectedServer.state.value))]),s("div",{staticClass:"server-info"},e._l(e.selectedServer.info,(function(t,r){return s("div",{key:r,staticClass:"server-info__item"},[s("p",{staticClass:"label"},[e._v(e._s(r))]),s("p",{staticClass:"value"},[e._v(e._s(t))])])})),0),s("div",{staticClass:"ssh-wrapper"},[s("div",{staticClass:"ssh-wrapper__item"},[s("span",{staticClass:"ssh"},[e._v("Приватный SSH-ключ")]),s("i",{staticClass:"btn-copy",attrs:{title:"Скопировать"},on:{click:function(t){return e.copy("private")}}}),s("a",{staticClass:"clickable",attrs:{href:e.privateURI,download:"id_rsa"}},[e._v("Скачать")]),s("span",{directives:[{name:"show",rawName:"v-show",value:"private"===e.buffer,expression:"buffer === 'private'"}],staticClass:"buffer"},[e._v("Ключ скопирован в буффер обмена")])]),s("div",{staticClass:"ssh-wrapper__item"},[s("span",{staticClass:"ssh"},[e._v("Публичный SSH-ключ")]),s("i",{staticClass:"btn-copy",attrs:{title:"Скопировать"},on:{click:function(t){return e.copy("public")}}}),s("a",{staticClass:"clickable",attrs:{href:e.publicURI,download:"id_rsa.pub"}},[e._v("Скачать")]),s("span",{directives:[{name:"show",rawName:"v-show",value:"public"===e.buffer,expression:"buffer === 'public'"}],staticClass:"buffer"},[e._v("Ключ скопирован в буффер обмена")])])]),s("hr"),s("div",{staticClass:"instruction",domProps:{innerHTML:e._s(e.instruction)}})])],1)},n=[],a=s("1da1"),i=s("5530"),o=(s("96cf"),s("b64b"),s("7db0"),s("d3b7"),s("51ba")),c=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"new-server"},[s("p",{staticClass:"new-server__header"},[e._v("Добавление сервера демо-панели")]),s("form",{staticClass:"new-server__form",on:{submit:function(t){return t.preventDefault(),e.addServer.apply(null,arguments)}}},[s("div",{directives:[{name:"show",rawName:"v-show",value:e.loading,expression:"loading"}],staticClass:"new-server__form--overlay"}),s("t-field",{attrs:{label:"Доменное имя"}},[s("t-input-new",{attrs:{placeholder:""},model:{value:e.domain_name,callback:function(t){e.domain_name=t},expression:"domain_name"}})],1),s("t-field",{attrs:{label:"IP адрес"}},[s("VueIP",{attrs:{ip:e.ip_address,onChange:e.change}})],1),s("t-field",{attrs:{label:"Имя пользователя"}},[s("t-input-new",{attrs:{placeholder:""},model:{value:e.user,callback:function(t){e.user=t},expression:"user"}})],1),s("div",{staticClass:"new-server__ports"},[s("t-field",{attrs:{label:"SSH порт"}},[s("d-input-number",{attrs:{placeholder:""},model:{value:e.port_ssh,callback:function(t){e.port_ssh=t},expression:"port_ssh"}})],1),s("t-field",{attrs:{label:"HTTP порт"}},[s("d-input-number",{attrs:{placeholder:""},model:{value:e.port_http,callback:function(t){e.port_http=t},expression:"port_http"}})],1),s("t-field",{attrs:{label:"HTTPS порт"}},[s("d-input-number",{attrs:{placeholder:""},model:{value:e.port_https,callback:function(t){e.port_https=t},expression:"port_https"}})],1)],1),s("t-button",{staticClass:"new-server__btn",attrs:{disabled:!e.validForm||e.loading}},[e._v("Добавить")])],1),s("LoadSpiner",{directives:[{name:"show",rawName:"v-show",value:e.loading,expression:"loading"}],attrs:{text:""}})],1)},l=[],p=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("span",{staticClass:"vue-ip",class:{"show-port":!1!==e.portCopy,"material-theme":"material"===e.theme,active:e.active,valid:e.valid}},e._l(e.ipCopy,(function(t,r){return s("div",{key:r,staticClass:"segment"},[s("input",{directives:[{name:"model",rawName:"v-model",value:e.ipCopy[r],expression:"ipCopy[index]"}],ref:"ipSegment",refInFor:!0,attrs:{type:"number",placeholder:"___",maxlength:"3"},domProps:{value:e.ipCopy[r]},on:{paste:function(t){return e.paste(t)},keydown:function(t){return e.ipKeydown(t,r)},focus:function(t){return e.ipFocus(r)},blur:e.blur,input:[function(t){t.target.composing||e.$set(e.ipCopy,r,t.target.value)},e.input]}})])})),0)},u=[],d=(s("a9e3"),s("ac1f"),s("1276"),s("d81d"),s("a15b"),s("00b4"),{props:{onChange:Function,ip:{required:!0,type:String},port:{type:[String,Number,Boolean],default:!1},placeholder:{type:[Boolean],default:!1},theme:{type:[String,Boolean],default:!1}},data:function(){return{ipCopy:["","","",""],portCopy:null,valid:!1,active:!1}},beforeMount:function(){this.copyValue(this.ip,this.port)},watch:{ip:function(e){this.copyValue(e,this.port)},port:function(e){this.copyValue(this.ip,e)}},methods:{input:function(e){e.target.value||(e.target.value="")},placeholderPos:function(e){if(!this.placeholder)return"";switch(e){case 0:return"192";case 1:return"168";case 2:return"0";case 3:return"1"}},ipFocus:function(e){this.active=!0,this.ipCopy[e]="",this.changed()},clearAll:function(){this.ipCopy=["","","",""],this.portCopy=null,this.valid=!1},blur:function(){this.active=!1},portFocus:function(){this.active=!0,this.portCopy=null,this.changed()},paste:function(e){this.$refs.ipSegment[0].focus();var t=e.clipboardData.getData("text/plain"),s=t.indexOf(":");if(!1===this.port){console.warn("A IP address with a port has been entered but this module has no port attribute. Please enable it update the port."),this.clearAll();var r=t.split(":");return this.copyValue(r[0],!1),void this.$refs.ipSegment[0].blur()}switch(s){case-1:this.copyValue(t,null),this.changed(),this.$refs.ipSegment[0].blur();break;default:var n=t.split(":");this.copyValue(n[0],n[1]),this.changed(),this.$refs.ipSegment[0].blur();break}},ipKeydown:function(e,t){var s=this,r=e.keyCode||e.which;8!==r&&37!==r||0===this.ipCopy[t].length&&void 0!==this.ipCopy[t-1]&&this.$refs.ipSegment[t-1].focus(),setTimeout((function(){"0"===s.ipCopy[t]?s.moveToNextIpSegment(t,!1):s.moveToNextIpSegment(t),s.changed()}))},moveToNextIpSegment:function(e){var t=!(arguments.length>1&&void 0!==arguments[1])||arguments[1];t?this.ipCopy[e].length>=3&&void 0!==this.ipCopy[e+1]&&this.$refs.ipSegment[e+1].focus():t||void 0!==this.ipCopy[e+1]&&this.$refs.ipSegment[e+1].focus()},changed:function(){var e=arguments.length>0&&void 0!==arguments[0]?arguments[0]:this.ipCopy,t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:this.portCopy,s=this.arrayToIp(e);this.onChange(s,t,this.validateIP(e))},copyValue:function(e,t){e&&this.ipToArray(e),this.portCopy=t,this.valid=this.validateIP(this.ipCopy),this.changed()},ipToArray:function(e){var t=[];e.split(".").map((function(e){(isNaN(e)||e<0||e>255)&&(e=255),t.push(e)})),4!==t.length?(console.error("Not valid, so clearing ip",t),this.clearAll()):this.ipCopy=t},arrayToIp:function(e){return e.join(".")},validateIP:function(e){var t=this.arrayToIp(e);return/^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/.test(t)}}}),v=d,h=(s("a34e"),s("2877")),f=Object(h["a"])(v,p,u,!1,null,"64c5dbba",null),_=f.exports,m=s("1636"),b={name:"NewServer",components:{VueIP:_,LoadSpiner:m["default"]},data:function(){return{ip_address:"",ipValid:null,domain_name:"",user:"",port_ssh:22,port_http:80,port_https:443,loading:!1}},computed:{validForm:function(){return!!(this.ipValid&&this.domain_name&&this.user&&this.port_ssh&&this.port_http&&this.port_https)}},methods:{change:function(e,t,s){this.ip_address=e,this.ipValid=s},addServer:function(){var e=this;return Object(a["a"])(regeneratorRuntime.mark((function t(){var s,r;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return e.loading=!0,t.next=3,e.$store.dispatch("servers/addServer",{domain_name:e.domain_name,ip_address:e.ip_address,user:e.user,port_ssh:e.port_ssh,port_http:e.port_http,port_https:e.port_https});case 3:s=t.sent,r=s.id,e.$emit("addServer",r),e.loading=!1;case 7:case"end":return t.stop()}}),t)})))()}}},g=b,w=(s("867f"),Object(h["a"])(g,c,l,!1,null,"128ed206",null)),y=w.exports,C=s("2f62"),S={name:"servers",components:{ServerTable:o["a"],NewServer:y,LoadSpiner:m["default"]},data:function(){return{addNew:!1,serverModal:!1,InstructionModal:!1,serverID:null,private_key:null,public_key:null,instruction:null,fetchingServers:!1,buffer:"",scrollLeft:0,scrollTop:0,intervalID:null}},computed:Object(i["a"])(Object(i["a"])({},Object(C["c"])({servers:"servers/getServers"})),{},{showTable:function(){return!!Object.keys(this.$store.getters["servers/getServers"]).length},getServer:function(){var e=this;return this.servers.find((function(t){return t.id===e.serverID}))},privateURI:function(){return"data:application/octet-stream;charset=utf-8,"+encodeURIComponent(this.private_key)},publicURI:function(){return"data:application/octet-stream;charset=utf-8,"+encodeURIComponent(this.public_key)},selectedServer:function(){var e=this,t=this.$store.getters["servers/getServers"].find((function(t){return t.id===e.serverID}));return t?{info:{"Доменное имя":t.domain_name,"Имя пользователя":t.user,"HTTP порт":t.port_http,"IP адерс":t.ip_address,"SSH порт":t.port_ssh,"HTTPS порт":t.port_https},state:t.state}:{state:""}}}),methods:{copy:function(e){var t="private"===e?this.private_key:this.public_key,s=document.createElement("input");document.body.appendChild(s),s.value=t,s.select(),document.execCommand("copy"),s.remove(),this.buffer=e},newServer:function(e){this.serverID=e,this.addNew=!1,this.serverModal=!0},openInstruction:function(e){var t=this;return Object(a["a"])(regeneratorRuntime.mark((function s(){var r,n;return regeneratorRuntime.wrap((function(s){while(1)switch(s.prev=s.next){case 0:return t.serverID=e,t.serverModal=!1,t.InstructionModal=!0,s.next=5,t.$store.dispatch("servers/getInstruction",{id:e});case 5:r=s.sent,n=r.data,t.private_key=n.private_ssh_key,t.public_key=n.public_ssh_key,t.instruction=n.instruction;case 10:case"end":return s.stop()}}),s)})))()},handleScroll:function(e,t){this.scrollLeft=t.scrollLeft,this.scrollTop=e.scrollTop}},created:function(){var e=this;return Object(a["a"])(regeneratorRuntime.mark((function t(){return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return e.fetchingServers=!0,t.next=3,e.$store.dispatch("servers/getServers");case 3:e.intervalID=setInterval(Object(a["a"])(regeneratorRuntime.mark((function t(){return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("servers/getServers");case 2:case"end":return t.stop()}}),t)}))),6e4),e.fetchingServers=!1,e.$router.afterEach((function(){clearInterval(e.intervalID)}));case 6:case"end":return t.stop()}}),t)})))()}},k=S,I=(s("8a1c"),Object(h["a"])(k,r,n,!1,null,"65777922",null));t["default"]=I.exports},"867f":function(e,t,s){"use strict";s("323f")},"8a1c":function(e,t,s){"use strict";s("ec3e")},a34e:function(e,t,s){"use strict";s("a5df")},a5df:function(e,t,s){},d5f2:function(e,t,s){"use strict";s("efd4")},ec3e:function(e,t,s){},efd4:function(e,t,s){}}]);