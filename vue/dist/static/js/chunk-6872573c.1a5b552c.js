(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-6872573c"],{"0427":function(e,t,a){},3403:function(e,t,a){"use strict";a("3ad9")},"3ad9":function(e,t,a){},4973:function(e,t,a){"use strict";a.r(t);var s=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{key:"key_update-"+e.updateKey,staticClass:"params deploy"},[a("scrollbar",[a("div",{staticClass:"params__body"},[a("div",{staticClass:"params__items"},[a("at-collapse",{attrs:{value:e.collapse}},e._l(e.params,(function(t,s){var c=t.visible,r=t.name,n=t.fields;return a("at-collapse-item",{directives:[{name:"show",rawName:"v-show",value:c&&"server"!==s,expression:"visible && key !== 'server'"}],key:s,staticClass:"mt-3",attrs:{name:s,title:r||""}},[a("div",{staticClass:"params__fields"},[e._l(n,(function(t,c){return[a("t-auto-field-cascade",e._b({key:s+c,attrs:{big:"type"===s,parameters:e.parameters,inline:!1},on:{change:e.parse}},"t-auto-field-cascade",t,!1)),a("t-button",{key:"key"+c,attrs:{disabled:e.overlayStatus},on:{click:e.onStart}},[e._v("Подготовить")])]}))],2)])})),1)],1),e.paramsDownloaded.isParamsSettingsLoad?a("div",{staticClass:"params__items"},[a("div",{staticClass:"params-container pa-5"},[a("div",{staticClass:"t-input"},[a("label",{staticClass:"label",attrs:{for:"deploy[deploy]"}},[e._v("Название папки")]),a("div",{staticClass:"t-input__label"},[e._v(" "+e._s("https://srv1.demo.neural-university.ru/"+e.userData.login+"/"+e.projectData.name_alias+"/"+e.deploy)+" ")]),a("input",{directives:[{name:"model",rawName:"v-model",value:e.deploy,expression:"deploy"}],staticClass:"t-input__input",attrs:{type:"text",id:"deploy[deploy]",name:"deploy[deploy]",autocomplete:"off"},domProps:{value:e.deploy},on:{input:function(t){t.target.composing||(e.deploy=t.target.value)}}})]),a("Autocomplete2",{attrs:{autocomplete:"off",value:e.serverLabel,list:e.list,name:"deploy[server]",label:"Сервер"},on:{focus:e.focus,change:e.selected}}),a("Checkbox",{staticClass:"pd__top",attrs:{label:"Перезаписать с таким же названием папки",type:"checkbox",parse:"replace",name:"replace"},on:{change:e.onChange}}),a("Checkbox",{attrs:{label:"Использовать пароль для просмотра страницы",parse:"replace",name:"use_sec",type:"checkbox"},on:{change:e.onChange}}),e.use_sec?a("div",{staticClass:"password"},[a("div",{staticClass:"t-input"},["checkbox"===(e.passwordShow?"text":"password")?a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec,expression:"sec"}],attrs:{placeholder:"Введите пароль",type:"checkbox"},domProps:{checked:Array.isArray(e.sec)?e._i(e.sec,null)>-1:e.sec},on:{change:function(t){var a=e.sec,s=t.target,c=!!s.checked;if(Array.isArray(a)){var r=null,n=e._i(a,r);s.checked?n<0&&(e.sec=a.concat([r])):n>-1&&(e.sec=a.slice(0,n).concat(a.slice(n+1)))}else e.sec=c}}}):"radio"===(e.passwordShow?"text":"password")?a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec,expression:"sec"}],attrs:{placeholder:"Введите пароль",type:"radio"},domProps:{checked:e._q(e.sec,null)},on:{change:function(t){e.sec=null}}}):a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec,expression:"sec"}],attrs:{placeholder:"Введите пароль",type:e.passwordShow?"text":"password"},domProps:{value:e.sec},on:{input:function(t){t.target.composing||(e.sec=t.target.value)}}}),a("div",{staticClass:"password__icon"},[a("i",{class:["t-icon",e.passwordShow?"icon-deploy-password-open":"icon-deploy-password-close"],attrs:{title:"show password"},on:{click:function(t){e.passwordShow=!e.passwordShow}}})])]),a("div",{staticClass:"t-input"},["checkbox"===(e.passwordShow?"text":"password")?a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec_accept,expression:"sec_accept"}],attrs:{placeholder:"Подтверждение пароля",type:"checkbox"},domProps:{checked:Array.isArray(e.sec_accept)?e._i(e.sec_accept,null)>-1:e.sec_accept},on:{change:function(t){var a=e.sec_accept,s=t.target,c=!!s.checked;if(Array.isArray(a)){var r=null,n=e._i(a,r);s.checked?n<0&&(e.sec_accept=a.concat([r])):n>-1&&(e.sec_accept=a.slice(0,n).concat(a.slice(n+1)))}else e.sec_accept=c}}}):"radio"===(e.passwordShow?"text":"password")?a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec_accept,expression:"sec_accept"}],attrs:{placeholder:"Подтверждение пароля",type:"radio"},domProps:{checked:e._q(e.sec_accept,null)},on:{change:function(t){e.sec_accept=null}}}):a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec_accept,expression:"sec_accept"}],attrs:{placeholder:"Подтверждение пароля",type:e.passwordShow?"text":"password"},domProps:{value:e.sec_accept},on:{input:function(t){t.target.composing||(e.sec_accept=t.target.value)}}}),a("div",{staticClass:"password__icon"},[a("i",{class:["t-icon",e.checkCorrect],attrs:{title:"is correct"}})])]),a("div",{staticClass:"password__rule"},[a("p",[e._v("Пароль должен содержать не менее 6 символов")])])]):e._e(),e.paramsDownloaded.isSendParamsDeploy?e._e():a("t-button",{attrs:{disabled:e.send_disabled},on:{click:e.sendDeployData}},[e._v(" Загрузить ")]),e.paramsDownloaded.isSendParamsDeploy?a("div",{staticClass:"req-ans"},[a("div",{staticClass:"answer__success"},[e._v("Загрузка завершена!")]),a("div",{staticClass:"answer__label"},[e._v("Ссылка на сформированную загрузку")]),a("div",{staticClass:"answer__url"},[a("i",{class:["t-icon","icon-deploy-copy"],attrs:{title:"copy"},on:{click:function(t){return e.copy(e.moduleList.url)}}}),a("a",{attrs:{href:e.moduleList.url,target:"_blank"}},[e._v(" "+e._s(e.moduleList.url)+" ")])])]):e._e(),e.paramsDownloaded.isSendParamsDeploy?a("ModuleList",{attrs:{moduleList:e.moduleList.api_text}}):e._e()],1)]):e._e()])])],1)},c=[],r=a("1da1"),n=a("5530"),o=(a("96cf"),a("b0c0"),a("ac1f"),a("5319"),function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"t-field t-inline"},[a("label",{staticClass:"t-field__label",attrs:{for:e.parse}},[e._v(e._s(e.label))]),a("div",{staticClass:"t-field__switch"},[a("input",{staticClass:"t-field__input",attrs:{id:e.parse,type:e.type,name:e.parse},domProps:{checked:e.checked?"checked":"",value:e.checked},on:{change:e.change}}),a("span")])])}),i=[],l=(a("caad"),a("2532"),a("56d7")),d={props:{label:{type:String,default:"Label"},type:{type:String,default:"text"},value:{type:[Boolean]},name:{type:String},parse:{type:String},event:{type:Array,default:function(){return[]}}},data:function(){return{checked:null}},methods:{change:function(e){var t=e.target.checked;this.$emit("change",{name:this.name,value:t}),l["bus"].$emit("change",{event:this.name,value:t})}},created:function(){var e=this;this.checked=this.value,this.event.length&&l["bus"].$on("change",(function(t){var a=t.event;e.event.includes(a)&&(e.checked=!1)}))},destroyed:function(){this.event.length&&(l["bus"].$off(),console.log("destroyed",this.name))}},u=d,p=(a("a868"),a("2877")),h=Object(p["a"])(u,o,i,!1,null,"96e0dd2e",null),m=h.exports,v=a("6522"),f=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"load__answer"},[a("scrollbar",{attrs:{ops:e.ops}},[a("div",{staticClass:"answer__text"},[e._v(" "+e._s(e.moduleList)+" ")])])],1)},_=[],y={name:"ModuleList",props:{moduleList:{type:String,default:function(){return""}}},data:function(){return{ops:{scrollPanel:{scrollingX:!1,scrollingY:!0}}}}},w=y,b=(a("7435"),Object(p["a"])(w,f,_,!1,null,"93845714",null)),g=b.exports,k=["icon-deploy-password-correct","icon-deploy-password-incorrect"],C=["type","server"],x={name:"Settings",components:{Checkbox:m,ModuleList:g,Autocomplete2:v["a"]},props:{params:{type:[Object,Array],default:function(){return{}}},moduleList:{type:Object,default:function(){return{}}},projectData:{type:[Array,Object],default:function(){return[]}},userData:{type:[Object,Array],default:function(){}},paramsDownloaded:{type:Object,default:function(){return{}}},overlayStatus:{type:Boolean,default:!1}},data:function(){return{updateKey:0,collapse:C,deploy:"",server:"",serverLabel:"",replace:!1,use_sec:!1,sec:"",sec_accept:"",passwordShow:!1,parameters:{},list:[]}},computed:{checkCorrect:function(){return this.sec==this.sec_accept?k[0]:k[1]},send_disabled:function(){return!(this.use_sec&&this.sec==this.sec_accept&&this.sec.length>5&&0!=this.deploy.length)&&0==this.deploy.length},isLoad:function(){return!(this.parameters.type&&this.parameters.name)}},methods:{parse:function(e){var t=e.value,a=e.name;"type"===a&&(this.parameters["name"]=null),this.parameters[a]=t,this.parameters=Object(n["a"])({},this.parameters)},onChange:function(e){var t=e.name,a=e.value;this[t]=a},copy:function(e){var t=document.createElement("textarea");t.value=e,t.style.top="0",t.style.left="0",t.style.position="fixed",document.body.appendChild(t),t.focus(),t.select();try{document.execCommand("copy")}catch(a){console.error("Fallback: Oops, unable to copy",a)}document.body.removeChild(t)},onStart:function(){var e=this;return Object(r["a"])(regeneratorRuntime.mark((function t(){return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return e.$emit("downloadSettings",e.parameters),t.next=3,e.focus();case 3:case"end":return t.stop()}}),t)})))()},focus:function(){var e=this;return Object(r["a"])(regeneratorRuntime.mark((function t(){var a,s,c,r,n;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("servers/ready");case 2:s=t.sent,s.data&&(e.list=(null===s||void 0===s?void 0:s.data)||[]),c=null===(a=e.list)||void 0===a?void 0:a[0],r=c.value,n=c.label,e.serverLabel||(e.serverLabel=n,e.server=r);case 6:case"end":return t.stop()}}),t)})))()},selected:function(e){var t=e.value;this.server=t},sendDeployData:function(){var e=this;return Object(r["a"])(regeneratorRuntime.mark((function t(){var a;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:a={deploy:e.deploy,server:e.server,replace:e.replace,use_sec:e.use_sec},e.use_sec&&(a["sec"]=e.sec),e.$emit("sendParamsDeploy",a);case 3:case"end":return t.stop()}}),t)})))()}},beforeDestroy:function(){this.$emit("clear")},watch:{params:function(){this.updateKey++}}},S=x,L=(a("3403"),Object(p["a"])(S,s,c,!1,null,"35b4f30c",null));t["default"]=L.exports},6522:function(e,t,a){"use strict";var s=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{class:["dropdown",{"dropdown--active":e.show}]},[a("label",{attrs:{for:e.name}},[e._v(e._s(e.label))]),a("input",{directives:[{name:"model",rawName:"v-model",value:e.search,expression:"search"}],staticClass:"dropdown__input",attrs:{autocomplete:"off",id:e.name,name:e.name,disabled:e.disabled,placeholder:e.placeholder},domProps:{value:e.search},on:{focus:e.focus,blur:function(t){return e.select(!1)},input:[function(t){t.target.composing||(e.search=t.target.value)},function(t){e.changed=!0}]}}),a("div",{directives:[{name:"show",rawName:"v-show",value:e.show,expression:"show"}],staticClass:"dropdown__content"},[e._l(e.filterList,(function(t,s){return a("div",{key:s,on:{mousedown:function(a){return e.select(t)}}},[e._v(" "+e._s(t.label)+" ")])})),e.filterList.length?e._e():a("div",[e._v("Нет данных")])],2)])},c=[],r=(a("ac1f"),a("841c"),a("4de4"),a("caad"),a("2532"),{name:"Autocomplete",props:{name:String,list:{type:Array,required:!0,default:function(){return[]}},placeholder:String,disabled:Boolean,label:String,value:String},data:function(){return{selected:{},show:!1,search:"",changed:null}},created:function(){this.search=this.value,this.changed=!1},computed:{filterList:function(){var e=this;return this.list?this.list.filter((function(t){var a=e.search;return!a||!e.changed||t.label.toLowerCase().includes(a.toLowerCase())})):[]}},methods:{select:function(e){e?(this.selected=e,this.show=!1,this.search=e.label,this.$emit("input",this.selected.value),this.$emit("change",e)):(this.search=this.selected.label||this.value,this.show=!1,this.changed=!1)},focus:function(e){var t=e.target;t.select(),this.show=!0,this.$emit("focus",!0)}},watch:{value:{handler:function(e){this.show=!1,this.search=e}}}}),n=r,o=(a("9b90"),a("2877")),i=Object(o["a"])(n,s,c,!1,null,"93faebd0",null);t["a"]=i.exports},7435:function(e,t,a){"use strict";a("7442")},7442:function(e,t,a){},"9b90":function(e,t,a){"use strict";a("cfe0")},a868:function(e,t,a){"use strict";a("0427")},cfe0:function(e,t,a){}}]);