(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-4826e7d3"],{"0427":function(e,t,s){},1622:function(e,t,s){"use strict";s("e2cf")},"33ad":function(e,t,s){},4973:function(e,t,s){"use strict";s.r(t);var a=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{key:"key_update-"+e.updateKey,staticClass:"params"},[s("scrollbar",[s("div",{staticClass:"params__body"},[s("div",{staticClass:"params__items"},[s("at-collapse",{attrs:{value:e.collapse}},e._l(e.params,(function(t,a){var c=t.visible,r=t.name,n=t.fields;return s("at-collapse-item",{directives:[{name:"show",rawName:"v-show",value:c&&"server"!==a,expression:"visible && key !== 'server'"}],key:a,staticClass:"mt-3",attrs:{name:a,title:r||""}},[s("div",{staticClass:"params__fields"},[e._l(n,(function(t,c){return[s("t-auto-field-cascade",e._b({key:a+c,attrs:{big:"type"===a,parameters:e.parameters,inline:!1},on:{change:e.parse}},"t-auto-field-cascade",t,!1)),s("t-button",{key:"key"+c,attrs:{disabled:e.overlayStatus},on:{click:function(t){return e.$emit("downloadSettings",e.parameters)}}},[e._v(" Подготовить ")])]}))],2)])})),1)],1),e.paramsDownloaded.isParamsSettingsLoad?s("div",{staticClass:"params__items"},[s("div",{staticClass:"params-container pa-5"},[s("div",{staticClass:"t-input"},[s("label",{staticClass:"label",attrs:{for:"deploy[deploy]"}},[e._v("Название папки")]),s("div",{staticClass:"t-input__label"},[e._v(" https://srv1.demo.neural-university.ru/"+e._s(e.userData.login)+"/"+e._s(e.projectData.name_alias)+"/"+e._s(e.deploy)+" ")]),s("input",{directives:[{name:"model",rawName:"v-model",value:e.deploy,expression:"deploy"}],staticClass:"t-input__input",attrs:{type:"text",id:"deploy[deploy]",name:"deploy[deploy]"},domProps:{value:e.deploy},on:{input:function(t){t.target.composing||(e.deploy=t.target.value)}}})]),s("Autocomplete2",{attrs:{list:e.list,name:"deploy[server]",label:"Сервер"},on:{focus:e.focus,change:e.selected}}),s("Checkbox",{staticClass:"pd__top",attrs:{label:"Перезаписать с таким же названием папки",type:"checkbox",parse:"replace",name:"replace"},on:{change:e.onChange}}),s("Checkbox",{attrs:{label:"Использовать пароль для просмотра страницы",parse:"replace",name:"use_sec",type:"checkbox"},on:{change:e.onChange}}),e.use_sec?s("div",{staticClass:"password"},[s("div",{staticClass:"t-input"},["checkbox"===(e.passwordShow?"text":"password")?s("input",{directives:[{name:"model",rawName:"v-model",value:e.sec,expression:"sec"}],attrs:{placeholder:"Введите пароль",type:"checkbox"},domProps:{checked:Array.isArray(e.sec)?e._i(e.sec,null)>-1:e.sec},on:{change:function(t){var s=e.sec,a=t.target,c=!!a.checked;if(Array.isArray(s)){var r=null,n=e._i(s,r);a.checked?n<0&&(e.sec=s.concat([r])):n>-1&&(e.sec=s.slice(0,n).concat(s.slice(n+1)))}else e.sec=c}}}):"radio"===(e.passwordShow?"text":"password")?s("input",{directives:[{name:"model",rawName:"v-model",value:e.sec,expression:"sec"}],attrs:{placeholder:"Введите пароль",type:"radio"},domProps:{checked:e._q(e.sec,null)},on:{change:function(t){e.sec=null}}}):s("input",{directives:[{name:"model",rawName:"v-model",value:e.sec,expression:"sec"}],attrs:{placeholder:"Введите пароль",type:e.passwordShow?"text":"password"},domProps:{value:e.sec},on:{input:function(t){t.target.composing||(e.sec=t.target.value)}}}),s("div",{staticClass:"password__icon"},[s("i",{class:["t-icon",e.passwordShow?"icon-deploy-password-open":"icon-deploy-password-close"],attrs:{title:"show password"},on:{click:function(t){e.passwordShow=!e.passwordShow}}})])]),s("div",{staticClass:"t-input"},["checkbox"===(e.passwordShow?"text":"password")?s("input",{directives:[{name:"model",rawName:"v-model",value:e.sec_accept,expression:"sec_accept"}],attrs:{placeholder:"Подтверждение пароля",type:"checkbox"},domProps:{checked:Array.isArray(e.sec_accept)?e._i(e.sec_accept,null)>-1:e.sec_accept},on:{change:function(t){var s=e.sec_accept,a=t.target,c=!!a.checked;if(Array.isArray(s)){var r=null,n=e._i(s,r);a.checked?n<0&&(e.sec_accept=s.concat([r])):n>-1&&(e.sec_accept=s.slice(0,n).concat(s.slice(n+1)))}else e.sec_accept=c}}}):"radio"===(e.passwordShow?"text":"password")?s("input",{directives:[{name:"model",rawName:"v-model",value:e.sec_accept,expression:"sec_accept"}],attrs:{placeholder:"Подтверждение пароля",type:"radio"},domProps:{checked:e._q(e.sec_accept,null)},on:{change:function(t){e.sec_accept=null}}}):s("input",{directives:[{name:"model",rawName:"v-model",value:e.sec_accept,expression:"sec_accept"}],attrs:{placeholder:"Подтверждение пароля",type:e.passwordShow?"text":"password"},domProps:{value:e.sec_accept},on:{input:function(t){t.target.composing||(e.sec_accept=t.target.value)}}}),s("div",{staticClass:"password__icon"},[s("i",{class:["t-icon",e.checkCorrect],attrs:{title:"is correct"}})])]),s("div",{staticClass:"password__rule"},[s("p",[e._v("Пароль должен содержать не менее 6 символов")])])]):e._e(),e.paramsDownloaded.isSendParamsDeploy?e._e():s("t-button",{attrs:{disabled:e.send_disabled},on:{click:e.sendDeployData}},[e._v(" Загрузить ")]),e.paramsDownloaded.isSendParamsDeploy?s("div",{staticClass:"req-ans"},[s("div",{staticClass:"answer__success"},[e._v("Загрузка завершена!")]),s("div",{staticClass:"answer__label"},[e._v("Ссылка на сформированную загрузку")]),s("div",{staticClass:"answer__url"},[s("i",{class:["t-icon","icon-deploy-copy"],attrs:{title:"copy"},on:{click:function(t){return e.copy(e.moduleList.url)}}}),s("a",{attrs:{href:e.moduleList.url,target:"_blank"}},[e._v(" "+e._s(e.moduleList.url)+" ")])])]):e._e(),e.paramsDownloaded.isSendParamsDeploy?s("ModuleList",{attrs:{moduleList:e.moduleList.api_text}}):e._e()],1)]):e._e()])])],1)},c=[],r=s("1da1"),n=s("5530"),o=(s("96cf"),s("b0c0"),s("ac1f"),s("5319"),function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"t-field t-inline"},[s("label",{staticClass:"t-field__label",attrs:{for:e.parse}},[e._v(e._s(e.label))]),s("div",{staticClass:"t-field__switch"},[s("input",{staticClass:"t-field__input",attrs:{id:e.parse,type:e.type,name:e.parse},domProps:{checked:e.checked?"checked":"",value:e.checked},on:{change:e.change}}),s("span")])])}),i=[],l=(s("caad"),s("2532"),s("56d7")),d={props:{label:{type:String,default:"Label"},type:{type:String,default:"text"},value:{type:[Boolean]},name:{type:String},parse:{type:String},event:{type:Array,default:function(){return[]}}},data:function(){return{checked:null}},methods:{change:function(e){var t=e.target.checked;this.$emit("change",{name:this.name,value:t}),l["bus"].$emit("change",{event:this.name,value:t})}},created:function(){var e=this;this.checked=this.value,this.event.length&&l["bus"].$on("change",(function(t){var s=t.event;e.event.includes(s)&&(e.checked=!1)}))},destroyed:function(){this.event.length&&(l["bus"].$off(),console.log("destroyed",this.name))}},u=d,p=(s("a868"),s("2877")),h=Object(p["a"])(u,o,i,!1,null,"96e0dd2e",null),m=h.exports,v=s("6522"),f=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"load__answer"},[s("scrollbar",{attrs:{ops:e.ops}},[s("div",{staticClass:"answer__text"},[e._v(" "+e._s(e.moduleList)+" ")])])],1)},_=[],y={name:"ModuleList",props:{moduleList:{type:String,default:function(){return""}}},data:function(){return{ops:{scrollPanel:{scrollingX:!1,scrollingY:!0}}}}},w=y,b=(s("7435"),Object(p["a"])(w,f,_,!1,null,"93845714",null)),g=b.exports,k=["icon-deploy-password-correct","icon-deploy-password-incorrect"],C=["type","server"],x={name:"Settings",components:{Checkbox:m,ModuleList:g,Autocomplete2:v["a"]},props:{params:{type:[Object,Array],default:function(){return{}}},moduleList:{type:Object,default:function(){return{}}},projectData:{type:[Array,Object],default:function(){return[]}},userData:{type:[Object,Array],default:function(){}},paramsDownloaded:{type:Object,default:function(){return{}}},overlayStatus:{type:Boolean,default:!1}},data:function(){return{updateKey:0,collapse:C,deploy:"",server:"",replace:!1,use_sec:!1,sec:"",sec_accept:"",passwordShow:!1,parameters:{},list:[]}},computed:{checkCorrect:function(){return this.sec==this.sec_accept?k[0]:k[1]},send_disabled:function(){return!(this.use_sec&&this.sec==this.sec_accept&&this.sec.length>5&&0!=this.deploy.length)&&0==this.deploy.length},isLoad:function(){return!(this.parameters.type&&this.parameters.name)}},methods:{parse:function(e){var t=e.value,s=e.name;this.parameters[s]=t,this.parameters=Object(n["a"])({},this.parameters)},onChange:function(e){var t=e.name,s=e.value;this[t]=s},copy:function(e){var t=document.createElement("textarea");t.value=e,t.style.top="0",t.style.left="0",t.style.position="fixed",document.body.appendChild(t),t.focus(),t.select();try{document.execCommand("copy")}catch(s){console.error("Fallback: Oops, unable to copy",s)}document.body.removeChild(t)},focus:function(){var e=this;return Object(r["a"])(regeneratorRuntime.mark((function t(){var s;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("servers/ready");case 2:s=t.sent,s.data&&(e.list=(null===s||void 0===s?void 0:s.data)||[]),console.log(s);case 5:case"end":return t.stop()}}),t)})))()},selected:function(e){var t=e.value;this.server=t},sendDeployData:function(){var e=this;return Object(r["a"])(regeneratorRuntime.mark((function t(){var s;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:s={deploy:e.deploy,server:e.server,replace:e.replace,use_sec:e.use_sec},e.use_sec&&(s["sec"]=e.sec),e.$emit("sendParamsDeploy",s);case 3:case"end":return t.stop()}}),t)})))()}},beforeDestroy:function(){this.$emit("clear")},watch:{params:function(){this.updateKey++}}},S=x,L=(s("1622"),Object(p["a"])(S,a,c,!1,null,"66a210d0",null));t["default"]=L.exports},5997:function(e,t,s){"use strict";s("33ad")},6522:function(e,t,s){"use strict";var a=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{class:["dropdown",{"dropdown--active":e.show}]},[s("label",{attrs:{for:e.name}},[e._v(e._s(e.label))]),s("input",{directives:[{name:"model",rawName:"v-model",value:e.search,expression:"search"}],staticClass:"dropdown__input",attrs:{id:e.name,name:e.name,disabled:e.disabled,placeholder:e.placeholder,autocomplete:"off"},domProps:{value:e.search},on:{focus:e.focus,blur:function(t){return e.select(!1)},input:function(t){t.target.composing||(e.search=t.target.value)}}}),s("div",{directives:[{name:"show",rawName:"v-show",value:e.show,expression:"show"}],staticClass:"dropdown__content"},[e._l(e.filterList,(function(t,a){return s("div",{key:a,on:{mousedown:function(s){return e.select(t)}}},[e._v(" "+e._s(t.label)+" ")])})),e.filterList.length?e._e():s("div",[e._v("Нет данных")])],2)])},c=[],r=(s("ac1f"),s("841c"),s("4de4"),s("caad"),s("2532"),{name:"Autocomplete",props:{name:String,list:{type:Array,required:!0,default:function(){return[]}},placeholder:String,disabled:Boolean,label:String,value:String},data:function(){return{selected:{},show:!1,search:""}},created:function(){this.search=this.value},computed:{filterList:function(){var e=this;return this.list?this.list.filter((function(t){var s=e.search;return!s||t.label.toLowerCase().includes(s.toLowerCase())})):[]}},methods:{select:function(e){e?(this.selected=e,this.show=!1,this.search=e.label,this.$emit("input",this.selected.value),this.$emit("change",e)):(this.search=this.selected.label||this.value,this.show=!1)},focus:function(e){var t=e.target;t.select(),this.show=!0,this.$emit("focus",!0)}},watch:{value:{handler:function(e){this.show=!1,this.search=e}}}}),n=r,o=(s("5997"),s("2877")),i=Object(o["a"])(n,a,c,!1,null,"3c6819c5",null);t["a"]=i.exports},7435:function(e,t,s){"use strict";s("7442")},7442:function(e,t,s){},a868:function(e,t,s){"use strict";s("0427")},e2cf:function(e,t,s){}}]);