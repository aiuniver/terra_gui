(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-1d23fc83"],{"040a":function(e,t,a){"use strict";a("32b7")},"0427":function(e,t,a){},"27db":function(e,t,a){"use strict";a.r(t);var s=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("main",{staticClass:"page-deploy"},[a("div",{staticClass:"cont"},[a("Deploy"),a("Params",{on:{overlay:e.setOverlay}})],1),e.overlay?a("div",{staticClass:"overlay"}):e._e()])},r=[],n=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"params"},[e._e(),a("scrollbar",[a("div",{staticClass:"params__body"},[a("div",{staticClass:"params__items"},[a("at-collapse",{key:e.key,attrs:{value:e.collapse},on:{"on-change":e.onchange}},e._l(e.params,(function(t,s){var r=t.visible,n=t.name,o=t.fields;return a("at-collapse-item",{directives:[{name:"show",rawName:"v-show",value:r&&"server"!==s,expression:"visible && key !== 'server'"}],key:s,staticClass:"mt-3",attrs:{name:s,title:n||""}},[a("div",{staticClass:"params__fields"},[e._l(o,(function(t,r){return[a("t-auto-field-cascade",e._b({key:s+r,attrs:{big:"type"===s,parameters:e.parameters,inline:!1},on:{change:e.parse}},"t-auto-field-cascade",t,!1)),a("t-button",{key:"key"+r,attrs:{disabled:e.isLoad},on:{click:e.handleDownload}},[e._v("Загрузить")])]}))],2)])})),1)],1),e.load?a("div",{staticClass:"params__items"},[a("div",{staticClass:"params-container pa-5"},[a("div",{staticClass:"t-input"},[a("label",{staticClass:"label",attrs:{for:"deploy[deploy]"}},[e._v("Название папки")]),a("div",{staticClass:"t-input__label"},[e._v(" https://srv1.demo.neural-university.ru/"+e._s(e.userData.login)+"/"+e._s(e.projectData.name_alias)+"/"+e._s(e.deploy)+" ")]),a("input",{directives:[{name:"model",rawName:"v-model",value:e.deploy,expression:"deploy"}],staticClass:"t-input__input",attrs:{type:"text",id:"deploy[deploy]",name:"deploy[deploy]"},domProps:{value:e.deploy},on:{blur:function(t){return e.$emit("blur",t.target.value)},input:function(t){t.target.composing||(e.deploy=t.target.value)}}})]),a("Checkbox",{staticClass:"pd__top",attrs:{label:"Перезаписать с таким же названием папки",type:"checkbox",parse:"deploy[overwrite]",name:"deploy[overwrite]"},on:{change:e.UseReplace}}),a("Checkbox",{attrs:{label:"Использовать пароль для просмотра страницы",parse:"deploy[use_password]",name:"deploy[use_password]",type:"checkbox"},on:{change:e.UseSec}}),e.use_sec?a("div",{staticClass:"password"},[a("div",{staticClass:"t-input"},["checkbox"===(e.passwordShow?"text":"password")?a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec,expression:"sec"}],attrs:{placeholder:"Введите пароль",type:"checkbox"},domProps:{checked:Array.isArray(e.sec)?e._i(e.sec,null)>-1:e.sec},on:{change:function(t){var a=e.sec,s=t.target,r=!!s.checked;if(Array.isArray(a)){var n=null,o=e._i(a,n);s.checked?o<0&&(e.sec=a.concat([n])):o>-1&&(e.sec=a.slice(0,o).concat(a.slice(o+1)))}else e.sec=r}}}):"radio"===(e.passwordShow?"text":"password")?a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec,expression:"sec"}],attrs:{placeholder:"Введите пароль",type:"radio"},domProps:{checked:e._q(e.sec,null)},on:{change:function(t){e.sec=null}}}):a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec,expression:"sec"}],attrs:{placeholder:"Введите пароль",type:e.passwordShow?"text":"password"},domProps:{value:e.sec},on:{input:function(t){t.target.composing||(e.sec=t.target.value)}}}),a("div",{staticClass:"password__icon"},[a("i",{class:["t-icon",e.passwordShow?"icon-deploy-password-open":"icon-deploy-password-close"],attrs:{title:"show password"},on:{click:function(t){e.passwordShow=!e.passwordShow}}})])]),a("div",{staticClass:"t-input"},["checkbox"===(e.passwordShow?"text":"password")?a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec_accept,expression:"sec_accept"}],attrs:{placeholder:"Подтверждение пароля",type:"checkbox"},domProps:{checked:Array.isArray(e.sec_accept)?e._i(e.sec_accept,null)>-1:e.sec_accept},on:{change:function(t){var a=e.sec_accept,s=t.target,r=!!s.checked;if(Array.isArray(a)){var n=null,o=e._i(a,n);s.checked?o<0&&(e.sec_accept=a.concat([n])):o>-1&&(e.sec_accept=a.slice(0,o).concat(a.slice(o+1)))}else e.sec_accept=r}}}):"radio"===(e.passwordShow?"text":"password")?a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec_accept,expression:"sec_accept"}],attrs:{placeholder:"Подтверждение пароля",type:"radio"},domProps:{checked:e._q(e.sec_accept,null)},on:{change:function(t){e.sec_accept=null}}}):a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec_accept,expression:"sec_accept"}],attrs:{placeholder:"Подтверждение пароля",type:e.passwordShow?"text":"password"},domProps:{value:e.sec_accept},on:{input:function(t){t.target.composing||(e.sec_accept=t.target.value)}}}),a("div",{staticClass:"password__icon"},[a("i",{class:["t-icon",e.checkCorrect],attrs:{title:"is correct"}})])]),a("div",{staticClass:"password__rule"},[a("p",[e._v("Пароль должен содержать не менее 6 символов")])])]):e._e(),e.DataSent?e._e():a("t-button",{attrs:{disabled:e.send_disabled},on:{click:e.SendData}},[e._v("Загрузить")]),e.DataLoading?a("div",{staticClass:"loader"},[a("div",{staticClass:"loader__title"},[e._v("Дождитесь окончания загрузки")]),a("div",{staticClass:"loader__progress"},[a("load-spiner")],1)]):e._e(),e.DataSent?a("div",{staticClass:"req-ans"},[a("div",{staticClass:"answer__success"},[e._v("Загрузка завершена!")]),a("div",{staticClass:"answer__label"},[e._v("Ссылка на сформированную загрузку")]),a("div",{staticClass:"answer__url"},[a("i",{class:["t-icon","icon-deploy-copy"],attrs:{title:"copy"},on:{click:function(t){return e.Copy(e.moduleList.url)}}}),a("a",{attrs:{href:e.moduleList.url,target:"_blank"}},[e._v(" "+e._s(e.moduleList.url)+"sdfasadfasdfasgdfhasiofhusduifhasiodcfuisfhoadsifisdhfiosdup ")])])]):e._e(),e.DataSent?a("ModuleList",{attrs:{moduleList:e.moduleList.api_text}}):e._e()],1)]):e._e()])])],1)},o=[],c=a("1da1"),i=a("5530"),l=(a("96cf"),a("b0c0"),a("7db0"),a("ac1f"),a("5319"),a("2f62")),d=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"t-field t-inline"},[a("label",{staticClass:"t-field__label",attrs:{for:e.parse}},[e._v(e._s(e.label))]),a("div",{staticClass:"t-field__switch"},[a("input",{staticClass:"t-field__input",attrs:{id:e.parse,type:e.type,name:e.parse},domProps:{checked:e.checked?"checked":"",value:e.checked},on:{change:e.change}}),a("span")])])},u=[],p=(a("caad"),a("2532"),a("56d7")),h={props:{label:{type:String,default:"Label"},type:{type:String,default:"text"},value:{type:[Boolean]},name:{type:String},parse:{type:String},event:{type:Array,default:function(){return[]}}},data:function(){return{checked:null}},methods:{change:function(e){var t=e.target.checked;this.$emit("change",{name:this.name,value:t}),p["bus"].$emit("change",{event:this.name,value:t})}},created:function(){var e=this;this.checked=this.value,this.event.length&&p["bus"].$on("change",(function(t){var a=t.event;e.event.includes(a)&&(e.checked=!1)}))},destroyed:function(){this.event.length&&(p["bus"].$off(),console.log("destroyed",this.name))}},v=h,m=(a("a868"),a("2877")),f=Object(m["a"])(v,d,u,!1,null,"96e0dd2e",null),y=f.exports,_=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"load__answer"},[a("scrollbar",{attrs:{ops:e.ops}},[a("div",{staticClass:"answer__text"},[e._v(" "+e._s(e.moduleList)+" ")])])],1)},g=[],b={name:"ModuleList",props:{moduleList:{type:String,default:function(){return""}}},data:function(){return{ops:{scrollPanel:{scrollingX:!1,scrollingY:!0}}}}},w=b,C=(a("7435"),Object(m["a"])(w,_,g,!1,null,"93845714",null)),k=C.exports,x=a("1636"),S=a("eb4c"),D={name:"Settings",components:{Checkbox:y,ModuleList:k,LoadSpiner:x["default"]},data:function(){return{debounce:null,collapse:["type","server"],load:!1,key:"1212",downloadSettings:{},trainSettings:{},deploy:"",replace:!1,use_sec:!1,sec:"",sec_accept:"",DataSent:!1,DataLoading:!1,passwordShow:!1,parameters:{},ops:{scrollPanel:{scrollingX:!1,scrollingY:!0}}}},computed:Object(i["a"])(Object(i["a"])({},Object(l["b"])({params:"deploy/getParams",height:"settings/height",moduleList:"deploy/getModuleList",projectData:"projects/getProject",userData:"projects/getUser"})),{},{state:{set:function(e){this.$store.dispatch("deploy/setStateParams",e)},get:function(){return this.$store.getters["deploy/getStateParams"]}},checkCorrect:function(){return this.sec==this.sec_accept?"icon-deploy-password-correct":"icon-deploy-password-incorrect"},send_disabled:function(){if(this.DataLoading)return!0;if(this.use_sec){if(this.sec==this.sec_accept&&this.sec.length>5&&0!=this.deploy.length)return!1}else if(0!=this.deploy.length)return!1;return!0},isLoad:function(){var e,t,a=(null===(e=this.parameters)||void 0===e?void 0:e.type)||"",s=(null===(t=this.parameters)||void 0===t?void 0:t.name)||"";return!(s&&a)}}),methods:{handleDownload:function(){var e=this;return Object(c["a"])(regeneratorRuntime.mark((function t(){var a,s,r,n,o;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:if(r=(null===(a=e.parameters)||void 0===a?void 0:a.type)||"",n=(null===(s=e.parameters)||void 0===s?void 0:s.name)||"",!r||!n){t.next=7;break}return t.next=5,e.$store.dispatch("deploy/DownloadSettings",{type:r,name:n});case 5:o=t.sent,null!==o&&void 0!==o&&o.success&&(e.$store.dispatch("settings/setOverlay",!0),e.debounce(!0));case 7:case"end":return t.stop()}}),t)})))()},progressGet:function(){var e=this;return Object(c["a"])(regeneratorRuntime.mark((function t(){var a,s,r,n,o;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("deploy/progress",{});case 2:a=t.sent,console.log(a),a&&null!==a&&void 0!==a&&a.data&&(s=a.data,r=s.finished,n=s.message,o=s.percent,e.$store.dispatch("messages/setProgressMessage",n),e.$store.dispatch("messages/setProgress",o),r?(e.$store.dispatch("settings/setOverlay",!1),e.$store.dispatch("projects/get"),e.load=!0):e.debounce(!0)),null!==a&&void 0!==a&&a.error&&e.$store.dispatch("settings/setOverlay",!1);case 6:case"end":return t.stop()}}),t)})))()},parse:function(e){var t=e.id,a=e.value,s=e.name,r=e.root;console.log(t,a,s,r),this.parameters[s]=a,this.parameters=Object(i["a"])({},this.parameters)},onchange:function(e){console.log(e)},click:function(){console.log()},Percents:function(e){var t=document.querySelector(".progress-bar > .loading");t.style.width=e+"%",t.find("span").value=e},Copy:function(e){var t=document.createElement("textarea");t.value=e,t.style.top="0",t.style.left="0",t.style.position="fixed",document.body.appendChild(t),t.focus(),t.select();try{document.execCommand("copy")}catch(a){console.error("Fallback: Oops, unable to copy",a)}document.body.removeChild(t)},UseSec:function(e){this.use_sec=e.value},UseReplace:function(e){this.replace=e.value},progress:function(){var e=this;return Object(c["a"])(regeneratorRuntime.mark((function t(){var a;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("deploy/CheckProgress");case 2:a=t.sent,console.log(a),a?(e.DataLoading=!1,e.DataSent=!0,e.$emit("overlay",e.DataLoading)):e.getProgress();case 5:case"end":return t.stop()}}),t)})))()},getProgress:function(){setTimeout(this.progress,2e3)},SendData:function(){var e=this;return Object(c["a"])(regeneratorRuntime.mark((function t(){var a,s,r,n;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return a={deploy:e.deploy,replace:e.replace,use_sec:e.use_sec},e.use_sec&&(a["sec"]=e.sec),t.next=4,e.$store.dispatch("deploy/SendDeploy",a);case 4:s=t.sent,console.log(s),s&&(r=s.error,n=s.success,console.log(r,n),!r&&n&&(e.DataLoading=!0,e.$emit("overlay",e.DataLoading),e.getProgress()));case 7:case"end":return t.stop()}}),t)})))()}},created:function(){var e=this;this.debounce=Object(S["a"])((function(t){t&&e.progressGet()}),1e3)},beforeDestroy:function(){this.debounce(!1),this.$store.dispatch("deploy/clear")},watch:{params:function(){this.key="dsdsdsd"}}},$=D,j=(a("040a"),Object(m["a"])($,n,o,!1,null,"286c12e8",null)),L=j.exports,O=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"board"},[a("scrollbar",[a("div",{staticClass:"wrapper"},[a("div",{staticClass:"content"},[a("div",{staticClass:"board__data-field"},[a("div",[a("div",{staticClass:"board__title"},[e._v("Исходные данные / Предсказанные данные")]),e.isTable?a("div",{staticClass:"board__data"},["DataframeRegression"===e.type?a("Table",e._b({key:e.random,on:{reload:e.ReloadCard,reloadAll:e.ReloadAll}},"Table",e.deploy,!1)):e._e(),"DataframeClassification"===e.type?a("TableClass",e._b({key:e.random,on:{reload:e.ReloadCard,reloadAll:e.ReloadAll}},"TableClass",e.deploy,!1)):e._e()],1):a("div",{staticClass:"board__data"},e._l(e.Cards,(function(t,s){return a("IndexCard",e._b({key:"card-"+s,attrs:{card:t,color_map:e.deploy.color_map,index:s},on:{reload:e.ReloadCard}},"IndexCard",t,!1))})),1)])])])])])],1)},R=[],P=(a("d3b7"),a("3ca3"),a("ddb0"),a("25f0"),{components:{IndexCard:function(){return Promise.all([a.e("chunk-743e06ca"),a.e("chunk-2d0b290a"),a.e("chunk-3711c166")]).then(a.bind(null,"3af9"))},Table:function(){return a.e("chunk-3c37d762").then(a.bind(null,"5448"))},TableClass:function(){return a.e("chunk-dfd64006").then(a.bind(null,"6eb8"))}},data:function(){return{random:"sd32efl"}},computed:Object(i["a"])(Object(i["a"])({},Object(l["b"])({dataLoaded:"deploy/getDataLoaded",Cards:"deploy/getCards",height:"settings/autoHeight",type:"deploy/getDeployType",deploy:"deploy/getDeploy"})),{},{isTable:function(){return["DataframeClassification","DataframeRegression"].includes(this.type)}}),methods:{ReloadCard:function(e){var t=this;return Object(c["a"])(regeneratorRuntime.mark((function a(){var s;return regeneratorRuntime.wrap((function(a){while(1)switch(a.prev=a.next){case 0:return a.next=2,t.$store.dispatch("deploy/ReloadCard",e);case 2:return s=a.sent,a.next=5,t.$store.dispatch("deploy/random");case 5:return t.random=a.sent,a.abrupt("return",s);case 7:case"end":return a.stop()}}),a)})))()},ReloadAll:function(){var e=this;return Object(c["a"])(regeneratorRuntime.mark((function t(){var a,s;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:for(a=[],s=0;s<e.Cards.length;s++)a.push(s.toString());return t.next=4,e.$store.dispatch("deploy/ReloadCard",a);case 4:case"end":return t.stop()}}),t)})))()}},mounted:function(){console.log(this.deploy)}}),A=P,T=(a("f09e"),Object(m["a"])(A,O,R,!1,null,"80eec96a",null)),N=T.exports,E={name:"Datasets",components:{Params:L,Deploy:N},data:function(){return{overlay:!1}},methods:{setOverlay:function(e){this.overlay=e}}},M=E,U=(a("c60e"),Object(m["a"])(M,s,r,!1,null,"3f9d1b9e",null));t["default"]=U.exports},2913:function(e,t,a){},"32b7":function(e,t,a){},"3fcb":function(e,t,a){},7435:function(e,t,a){"use strict";a("7442")},7442:function(e,t,a){},a868:function(e,t,a){"use strict";a("0427")},c60e:function(e,t,a){"use strict";a("3fcb")},eb4c:function(e,t,a){"use strict";a.d(t,"a",(function(){return s}));var s=function(e,t,a){var s;return function(){var r=this,n=arguments,o=function(){s=null,a||e.apply(r,n)},c=a&&!s;clearTimeout(s),s=setTimeout(o,t),c&&e.apply(r,n)}}},f09e:function(e,t,a){"use strict";a("2913")}}]);