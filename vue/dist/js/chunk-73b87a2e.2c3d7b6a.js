(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-73b87a2e"],{"0427":function(e,t,a){},"1dfe":function(e,t,a){},"27db":function(e,t,a){"use strict";a.r(t);var s=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("main",{staticClass:"page-deploy"},[a("div",{staticClass:"cont"},[a("Deploy"),a("Params")],1)])},c=[],n=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"params"},[a("scrollbar",[a("div",{staticClass:"params-container__name"},[e._v("Загрузка в демо-панель")]),a("div",{staticClass:"params-container pa-5"},[a("div",{staticClass:"label"},[e._v("Название папки")]),a("div",{staticClass:"t-input"},[a("label",{staticClass:"t-input__label"},[e._v(" https://demo.neural-university.ru/"+e._s(e.userData.login)+"/"+e._s(e.projectData.name_alias)+"/"+e._s(e.deploy)+" ")]),a("input",{directives:[{name:"model",rawName:"v-model",value:e.deploy,expression:"deploy"}],staticClass:"t-input__input",attrs:{type:"text",name:"deploy[deploy]"},domProps:{value:e.deploy},on:{blur:function(t){return e.$emit("blur",t.target.value)},input:function(t){t.target.composing||(e.deploy=t.target.value)}}})]),a("Checkbox",{staticClass:"pd__top",attrs:{label:"Перезаписать с таким же названием папки",type:"checkbox"},on:{change:e.UseReplace}}),a("Checkbox",{attrs:{label:"Использовать пароль для просмотра страницы",type:"checkbox"},on:{change:e.UseSec}}),e.use_sec?a("div",{staticClass:"password"},[a("div",{staticClass:"t-input"},["checkbox"===(e.passwordShow?"text":"password")?a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec,expression:"sec"}],attrs:{placeholder:"Введите пароль",type:"checkbox"},domProps:{checked:Array.isArray(e.sec)?e._i(e.sec,null)>-1:e.sec},on:{change:function(t){var a=e.sec,s=t.target,c=!!s.checked;if(Array.isArray(a)){var n=null,r=e._i(a,n);s.checked?r<0&&(e.sec=a.concat([n])):r>-1&&(e.sec=a.slice(0,r).concat(a.slice(r+1)))}else e.sec=c}}}):"radio"===(e.passwordShow?"text":"password")?a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec,expression:"sec"}],attrs:{placeholder:"Введите пароль",type:"radio"},domProps:{checked:e._q(e.sec,null)},on:{change:function(t){e.sec=null}}}):a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec,expression:"sec"}],attrs:{placeholder:"Введите пароль",type:e.passwordShow?"text":"password"},domProps:{value:e.sec},on:{input:function(t){t.target.composing||(e.sec=t.target.value)}}}),a("div",{staticClass:"password__icon"},[a("i",{class:["t-icon",e.passwordShow?"icon-deploy-password-open":"icon-deploy-password-close"],attrs:{title:"show password"},on:{click:function(t){e.passwordShow=!e.passwordShow}}})])]),a("div",{staticClass:"t-input"},["checkbox"===(e.passwordShow?"text":"password")?a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec_accept,expression:"sec_accept"}],attrs:{placeholder:"Подтверждение пароля",type:"checkbox"},domProps:{checked:Array.isArray(e.sec_accept)?e._i(e.sec_accept,null)>-1:e.sec_accept},on:{change:function(t){var a=e.sec_accept,s=t.target,c=!!s.checked;if(Array.isArray(a)){var n=null,r=e._i(a,n);s.checked?r<0&&(e.sec_accept=a.concat([n])):r>-1&&(e.sec_accept=a.slice(0,r).concat(a.slice(r+1)))}else e.sec_accept=c}}}):"radio"===(e.passwordShow?"text":"password")?a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec_accept,expression:"sec_accept"}],attrs:{placeholder:"Подтверждение пароля",type:"radio"},domProps:{checked:e._q(e.sec_accept,null)},on:{change:function(t){e.sec_accept=null}}}):a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec_accept,expression:"sec_accept"}],attrs:{placeholder:"Подтверждение пароля",type:e.passwordShow?"text":"password"},domProps:{value:e.sec_accept},on:{input:function(t){t.target.composing||(e.sec_accept=t.target.value)}}}),a("div",{staticClass:"password__icon"},[a("i",{class:["t-icon",e.checkCorrect],attrs:{title:"is correct"}})])])]):e._e(),e.DataSent?e._e():a("button",{attrs:{disabled:e.send_disabled},on:{click:e.SendData}},[e._v("Загрузить")]),e.DataLoading?a("div",{staticClass:"loader"},[a("div",{staticClass:"loader__title"},[e._v("Дождитесь окончания загрузки")]),a("div",{staticClass:"loader__progress"},[a("load-spiner")],1)]):e._e(),e.DataSent?a("div",{staticClass:"req-ans"},[a("div",{staticClass:"answer__success"},[e._v("Загрузка завершена!")]),a("div",{staticClass:"answer__label"},[e._v("Ссылка на сформированную загрузку")]),a("div",{staticClass:"answer__url"},[a("i",{class:["t-icon","icon-deploy-copy"],attrs:{title:"copy"},on:{click:function(t){return e.Copy(e.moduleList.url)}}}),a("a",{attrs:{href:e.moduleList.url,target:"_blank"}},[e._v(e._s(e.moduleList.url))])])]):e._e(),e.DataSent?a("ModuleList",{attrs:{moduleList:e.moduleList.api_text}}):e._e()],1)])],1)},r=[],i=a("1da1"),o=a("5530"),l=(a("96cf"),a("7db0"),a("ac1f"),a("5319"),a("2f62")),d=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"t-field t-inline"},[a("label",{staticClass:"t-field__label",attrs:{for:e.parse}},[e._v(e._s(e.label))]),a("div",{staticClass:"t-field__switch"},[a("input",{staticClass:"t-field__input",attrs:{id:e.parse,type:e.type,name:e.parse},domProps:{checked:e.checked?"checked":"",value:e.checked},on:{change:e.change}}),a("span")])])},u=[],p=(a("b0c0"),a("caad"),a("2532"),a("56d7")),_={props:{label:{type:String,default:"Label"},type:{type:String,default:"text"},value:{type:[Boolean]},name:{type:String},parse:{type:String},event:{type:Array,default:function(){return[]}}},data:function(){return{checked:null}},methods:{change:function(e){var t=e.target.checked;this.$emit("change",{name:this.name,value:t}),p["bus"].$emit("change",{event:this.name,value:t})}},created:function(){var e=this;this.checked=this.value,this.event.length&&p["bus"].$on("change",(function(t){var a=t.event;e.event.includes(a)&&(e.checked=!1)}))},destroyed:function(){this.event.length&&(p["bus"].$off(),console.log("destroyed",this.name))}},h=_,m=(a("a868"),a("2877")),f=Object(m["a"])(h,d,u,!1,null,"96e0dd2e",null),v=f.exports,g=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"load__answer"},[a("scrollbar",[a("div",{staticClass:"apiBlock"},[e._v(" "+e._s(e.moduleList)+" ")])])],1)},y=[],b={name:"ModuleList",props:{moduleList:{type:String,default:function(){return""}}}},C=b,x=(a("46b3"),Object(m["a"])(C,g,y,!1,null,"ef881ec4",null)),w=x.exports,k=a("1636"),S={name:"Settings",components:{Checkbox:v,ModuleList:w,LoadSpiner:k["a"]},data:function(){return{deploy:"",replace:!1,use_sec:!1,sec:"",sec_accept:"",send_disabled:!0,DataSent:!1,DataLoading:!1,passwordShow:!1,pattern:/^(?=[a-zA-Z])[A-Z_a-z0-9]+$/}},computed:Object(o["a"])(Object(o["a"])({},Object(l["b"])({height:"settings/height",moduleList:"deploy/getModuleList",projectData:"projects/getProject",userData:"projects/getUser"})),{},{checkCorrect:function(){return this.sec==this.sec_accept&&this.sec.length>5?"icon-deploy-password-correct":"icon-deploy-password-incorrect"}}),mounted:function(){},watch:{deploy:function(e){""!==e&&this.pattern.test(this.deploy)?this.send_disabled=!1:this.send_disabled=!0},sec_accept:function(e){this.use_sec&&(e==this.sec&&this.pattern.test(this.deploy)?this.send_disabled=!1:this.send_disabled=!0)}},methods:{click:function(){console.log()},Percents:function(e){var t=document.querySelector(".progress-bar > .loading");t.style.width=e+"%",t.find("span").value=e},Copy:function(e){var t=document.createElement("textarea");t.value=e,t.style.top="0",t.style.left="0",t.style.position="fixed",document.body.appendChild(t),t.focus(),t.select();try{document.execCommand("copy")}catch(a){console.error("Fallback: Oops, unable to copy",a)}document.body.removeChild(t)},UseSec:function(e){this.use_sec=e.value},UseReplace:function(e){this.replace=e.value},progress:function(){var e=this;return Object(i["a"])(regeneratorRuntime.mark((function t(){var a;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("deploy/CheckProgress");case 2:a=t.sent,console.log(a),a?(e.DataLoading=!1,e.DataSent=!0):e.getProgress();case 5:case"end":return t.stop()}}),t)})))()},getProgress:function(){setTimeout(this.progress,2e3)},SendData:function(){var e=this;return Object(i["a"])(regeneratorRuntime.mark((function t(){var a,s,c,n;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return a={deploy:e.deploy,replace:e.replace,use_sec:e.use_sec},e.use_sec&&(a["sec"]=e.sec),t.next=4,e.$store.dispatch("deploy/SendDeploy",a);case 4:s=t.sent,console.log(s),s&&(c=s.error,n=s.success,console.log(c,n),!c&&n&&(e.DataLoading=!0,e.getProgress()));case 7:case"end":return t.stop()}}),t)})))()}}},D=S,j=(a("d6c5"),Object(m["a"])(D,n,r,!1,null,"74af094d",null)),L=j.exports,O=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"board"},[a("scrollbar",[a("div",{staticClass:"wrapper"},[a("div",{staticClass:"content"},["table"!=e.Cards[0].type?a("button",{staticClass:"board__reload-all",on:{click:e.ReloadAll}},[a("i",{class:["t-icon","icon-deploy-reload"],attrs:{title:"reload"}}),a("span",[e._v("Перезагрузить все")])]):e._e(),"table"!=e.Cards[0].type?a("div",{staticClass:"board__data-field"},[a("div",{staticClass:"board__title"},[e._v("Исходные данные / Предсказанные данные")]),a("div",{staticClass:"board__data"},e._l(e.Cards,(function(t,s){return a("IndexCard",e._b({key:"card-"+s},"IndexCard",t,!1))})),1)]):e._e()])])])],1)},A=[],P=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"card"},[a("div",{staticClass:"card__content"},["card"==e.type?a("div",[a("div",{staticClass:"card__original"},["image"==e.original.type?a("ImgCard",{attrs:{imgUrl:e.original.imgUrl}}):e._e(),"text"==e.original.type?a("TextCard",{style:e.origTextStyle},[e._v(e._s(e.original.data))]):e._e()],1),a("div",{staticClass:"card__result"},["image"==e.result.type?a("ImgCard",{attrs:{imgUrl:e.result.imgUrl}}):e._e(),"text"==e.result.type?a("TextCard",{style:"image"==e.original.type?{width:"224px"}:{}},[e._v(e._s(e.result.data))]):e._e()],1)]):e._e(),"graphic"==e.type?a("div",{staticClass:"card__graphic"},[a("Plotly",{attrs:{data:e.data,layout:e.layout,"display-mode-bar":!1}})],1):e._e()]),"table"!=e.type?a("div",{staticClass:"card__reload"},[a("button",{staticClass:"btn-reload",on:{click:e.ReloadCard}},[a("i",{class:["t-icon","icon-deploy-reload"],attrs:{title:"reload"}})])]):e._e()])},$=[],R=function(){var e=this,t=e.$createElement,s=e._self._c||t;return s("div",{staticClass:"img-card"},[s("img",{staticClass:"img-card__image",attrs:{src:a("d12e")("./"+e.imgUrl),alt:e.ImgAlt}})])},E=[],T={name:"ImgCard",props:{imgUrl:{type:String,default:"img.png"},imgAlt:{type:String,default:"image"}}},U=T,I=(a("906e"),Object(m["a"])(U,R,E,!1,null,"ed392374",null)),N=I.exports,q=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"text-card"},[a("scrollbar",[e._t("default",(function(){return[e._v("TEXT")]}))],2)],1)},M=[],z={name:"TextCard",props:{text:{type:String,default:"text"}}},B=z,J=(a("65b8"),Object(m["a"])(B,q,M,!1,null,"33d62058",null)),Z=J.exports,F=a("04d11"),G={name:"IndexCard",components:{ImgCard:N,TextCard:Z,Plotly:F["Plotly"]},data:function(){return{}},props:{original:{type:Object,default:function(){return{}}},result:{type:Object,default:function(){return{}}},type:{type:String,default:""}},mounted:function(){console.log(this.graphicData)},methods:{ReloadCard:function(){console.log("RELOAD_CARD")}},computed:Object(o["a"])(Object(o["a"])({},Object(l["b"])({graphicData:"deploy/getGraphicData",defaultLayout:"deploy/getDefaultLayout",origTextStyle:"deploy/getOrigTextStyle"})),{},{layout:function(){var e=this.defaultLayout;return this.char&&(e.title.text=this.char.title||"",e.xaxis.title=this.char.xaxis.title||"",e.yaxis.title=this.char.yaxis.title||""),e},data:function(){var e=[this.graphicData]||!1;return e}})},H=G,X=(a("7318"),Object(m["a"])(H,P,$,!1,null,"16903c12",null)),K=X.exports,Q={components:{IndexCard:K},data:function(){return{}},computed:Object(o["a"])({},Object(l["b"])({dataLoaded:"deploy/getDataLoaded",Cards:"deploy/getCards",height:"settings/autoHeight"})),methods:{click:function(e){console.log(e)},ReloadAll:function(){console.log("RELOAD_DATA")}}},V=Q,W=(a("59b9"),Object(m["a"])(V,O,A,!1,null,"b81dbfc6",null)),Y=W.exports,ee={name:"Datasets",components:{Params:L,Deploy:Y}},te=ee,ae=(a("f2af"),Object(m["a"])(te,s,c,!1,null,"31143d8c",null));t["default"]=ae.exports},"36f6":function(e,t,a){},"46b3":function(e,t,a){"use strict";a("f099")},"59b9":function(e,t,a){"use strict";a("9410")},"65b8":function(e,t,a){"use strict";a("1dfe")},7318:function(e,t,a){"use strict";a("c8ce")},"906e":function(e,t,a){"use strict";a("36f6")},9410:function(e,t,a){},a868:function(e,t,a){"use strict";a("0427")},c8ce:function(e,t,a){},cd06:function(e,t,a){},d6c5:function(e,t,a){"use strict";a("faaf")},f099:function(e,t,a){},f2af:function(e,t,a){"use strict";a("cd06")},faaf:function(e,t,a){}}]);