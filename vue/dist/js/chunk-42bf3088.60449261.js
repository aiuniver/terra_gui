(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-42bf3088"],{"0427":function(e,t,a){},"07ce":function(e,t,a){},"0da8":function(e,t,a){},"27db":function(e,t,a){"use strict";a.r(t);var s=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("main",{staticClass:"page-deploy"},[a("div",{staticClass:"cont"},[a("Deploy"),a("Params",{on:{overlay:e.setOverlay}})],1),e.overlay?a("div",{staticClass:"overlay"}):e._e()])},c=[],r=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"params"},[a("scrollbar",{attrs:{ops:e.ops}},[a("div",{staticClass:"params-container__name"},[e._v("Загрузка в демо-панель")]),a("div",{staticClass:"params-container pa-5"},[a("div",{staticClass:"t-input"},[a("label",{staticClass:"label",attrs:{for:"deploy[deploy]"}},[e._v("Название папки")]),a("div",{staticClass:"t-input__label"},[e._v(" https://demo.neural-university.ru/"+e._s(e.userData.login)+"/"+e._s(e.projectData.name_alias)+"/"+e._s(e.deploy)+" ")]),a("input",{directives:[{name:"model",rawName:"v-model",value:e.deploy,expression:"deploy"}],staticClass:"t-input__input",attrs:{type:"text",id:"deploy[deploy]",name:"deploy[deploy]"},domProps:{value:e.deploy},on:{blur:function(t){return e.$emit("blur",t.target.value)},input:function(t){t.target.composing||(e.deploy=t.target.value)}}})]),a("Checkbox",{staticClass:"pd__top",attrs:{label:"Перезаписать с таким же названием папки",type:"checkbox",parse:"deploy[overwrite]",name:"deploy[overwrite]"},on:{change:e.UseReplace}}),a("Checkbox",{attrs:{label:"Использовать пароль для просмотра страницы",parse:"deploy[use_password]",name:"deploy[use_password]",type:"checkbox"},on:{change:e.UseSec}}),e.use_sec?a("div",{staticClass:"password"},[a("div",{staticClass:"t-input"},["checkbox"===(e.passwordShow?"text":"password")?a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec,expression:"sec"}],attrs:{placeholder:"Введите пароль",type:"checkbox"},domProps:{checked:Array.isArray(e.sec)?e._i(e.sec,null)>-1:e.sec},on:{change:function(t){var a=e.sec,s=t.target,c=!!s.checked;if(Array.isArray(a)){var r=null,n=e._i(a,r);s.checked?n<0&&(e.sec=a.concat([r])):n>-1&&(e.sec=a.slice(0,n).concat(a.slice(n+1)))}else e.sec=c}}}):"radio"===(e.passwordShow?"text":"password")?a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec,expression:"sec"}],attrs:{placeholder:"Введите пароль",type:"radio"},domProps:{checked:e._q(e.sec,null)},on:{change:function(t){e.sec=null}}}):a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec,expression:"sec"}],attrs:{placeholder:"Введите пароль",type:e.passwordShow?"text":"password"},domProps:{value:e.sec},on:{input:function(t){t.target.composing||(e.sec=t.target.value)}}}),a("div",{staticClass:"password__icon"},[a("i",{class:["t-icon",e.passwordShow?"icon-deploy-password-open":"icon-deploy-password-close"],attrs:{title:"show password"},on:{click:function(t){e.passwordShow=!e.passwordShow}}})])]),a("div",{staticClass:"t-input"},["checkbox"===(e.passwordShow?"text":"password")?a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec_accept,expression:"sec_accept"}],attrs:{placeholder:"Подтверждение пароля",type:"checkbox"},domProps:{checked:Array.isArray(e.sec_accept)?e._i(e.sec_accept,null)>-1:e.sec_accept},on:{change:function(t){var a=e.sec_accept,s=t.target,c=!!s.checked;if(Array.isArray(a)){var r=null,n=e._i(a,r);s.checked?n<0&&(e.sec_accept=a.concat([r])):n>-1&&(e.sec_accept=a.slice(0,n).concat(a.slice(n+1)))}else e.sec_accept=c}}}):"radio"===(e.passwordShow?"text":"password")?a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec_accept,expression:"sec_accept"}],attrs:{placeholder:"Подтверждение пароля",type:"radio"},domProps:{checked:e._q(e.sec_accept,null)},on:{change:function(t){e.sec_accept=null}}}):a("input",{directives:[{name:"model",rawName:"v-model",value:e.sec_accept,expression:"sec_accept"}],attrs:{placeholder:"Подтверждение пароля",type:e.passwordShow?"text":"password"},domProps:{value:e.sec_accept},on:{input:function(t){t.target.composing||(e.sec_accept=t.target.value)}}}),a("div",{staticClass:"password__icon"},[a("i",{class:["t-icon",e.checkCorrect],attrs:{title:"is correct"}})])])]):e._e(),e.DataSent?e._e():a("button",{attrs:{disabled:e.send_disabled},on:{click:e.SendData}},[e._v("Загрузить")]),e.DataLoading?a("div",{staticClass:"loader"},[a("div",{staticClass:"loader__title"},[e._v("Дождитесь окончания загрузки")]),a("div",{staticClass:"loader__progress"},[a("load-spiner")],1)]):e._e(),e.DataSent?a("div",{staticClass:"req-ans"},[a("div",{staticClass:"answer__success"},[e._v("Загрузка завершена!")]),a("div",{staticClass:"answer__label"},[e._v("Ссылка на сформированную загрузку")]),a("div",{staticClass:"answer__url"},[a("i",{class:["t-icon","icon-deploy-copy"],attrs:{title:"copy"},on:{click:function(t){return e.Copy(e.moduleList.url)}}}),a("a",{attrs:{href:e.moduleList.url,target:"_blank"}},[e._v(e._s(e.moduleList.url))])])]):e._e(),e.DataSent?a("ModuleList",{attrs:{moduleList:e.moduleList.api_text}}):e._e()],1)])],1)},n=[],o=a("1da1"),i=a("5530"),l=(a("96cf"),a("7db0"),a("ac1f"),a("5319"),a("2f62")),d=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"t-field t-inline"},[a("label",{staticClass:"t-field__label",attrs:{for:e.parse}},[e._v(e._s(e.label))]),a("div",{staticClass:"t-field__switch"},[a("input",{staticClass:"t-field__input",attrs:{id:e.parse,type:e.type,name:e.parse},domProps:{checked:e.checked?"checked":"",value:e.checked},on:{change:e.change}}),a("span")])])},u=[],p=(a("b0c0"),a("caad"),a("2532"),a("56d7")),h={props:{label:{type:String,default:"Label"},type:{type:String,default:"text"},value:{type:[Boolean]},name:{type:String},parse:{type:String},event:{type:Array,default:function(){return[]}}},data:function(){return{checked:null}},methods:{change:function(e){var t=e.target.checked;this.$emit("change",{name:this.name,value:t}),p["bus"].$emit("change",{event:this.name,value:t})}},created:function(){var e=this;this.checked=this.value,this.event.length&&p["bus"].$on("change",(function(t){var a=t.event;e.event.includes(a)&&(e.checked=!1)}))},destroyed:function(){this.event.length&&(p["bus"].$off(),console.log("destroyed",this.name))}},_=h,m=(a("a868"),a("2877")),f=Object(m["a"])(_,d,u,!1,null,"96e0dd2e",null),v=f.exports,g=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"load__answer"},[a("scrollbar",[a("div",{staticClass:"apiBlock"},[e._v(" "+e._s(e.moduleList)+" ")])])],1)},y=[],b={name:"ModuleList",props:{moduleList:{type:String,default:function(){return""}}}},C=b,w=(a("46b3"),Object(m["a"])(C,g,y,!1,null,"ef881ec4",null)),x=w.exports,k=a("1636"),S={name:"Settings",components:{Checkbox:v,ModuleList:x,LoadSpiner:k["a"]},data:function(){return{deploy:"",replace:!1,use_sec:!1,sec:"",sec_accept:"",DataSent:!1,DataLoading:!1,passwordShow:!1,ops:{scrollPanel:{scrollingX:!1,scrollingY:!0}}}},computed:Object(i["a"])(Object(i["a"])({},Object(l["b"])({height:"settings/height",moduleList:"deploy/getModuleList",projectData:"projects/getProject",userData:"projects/getUser"})),{},{checkCorrect:function(){return this.sec==this.sec_accept?"icon-deploy-password-correct":"icon-deploy-password-incorrect"},send_disabled:function(){if(this.DataLoading)return!0;if(this.use_sec){if(this.sec==this.sec_accept&&this.sec.length>5&&0!=this.deploy.length)return!1}else if(0!=this.deploy.length)return!1;return!0}}),methods:{click:function(){console.log()},Percents:function(e){var t=document.querySelector(".progress-bar > .loading");t.style.width=e+"%",t.find("span").value=e},Copy:function(e){var t=document.createElement("textarea");t.value=e,t.style.top="0",t.style.left="0",t.style.position="fixed",document.body.appendChild(t),t.focus(),t.select();try{document.execCommand("copy")}catch(a){console.error("Fallback: Oops, unable to copy",a)}document.body.removeChild(t)},UseSec:function(e){this.use_sec=e.value},UseReplace:function(e){this.replace=e.value},progress:function(){var e=this;return Object(o["a"])(regeneratorRuntime.mark((function t(){var a;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("deploy/CheckProgress");case 2:a=t.sent,console.log(a),a?(e.DataLoading=!1,e.DataSent=!0,e.$emit("overlay",e.DataLoading)):e.getProgress();case 5:case"end":return t.stop()}}),t)})))()},getProgress:function(){setTimeout(this.progress,2e3)},SendData:function(){var e=this;return Object(o["a"])(regeneratorRuntime.mark((function t(){var a,s,c,r;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return a={deploy:e.deploy,replace:e.replace,use_sec:e.use_sec},e.use_sec&&(a["sec"]=e.sec),t.next=4,e.$store.dispatch("deploy/SendDeploy",a);case 4:s=t.sent,console.log(s),s&&(c=s.error,r=s.success,console.log(c,r),!c&&r&&(e.DataLoading=!0,e.$emit("overlay",e.DataLoading),e.getProgress()));case 7:case"end":return t.stop()}}),t)})))()}}},j=S,O=(a("e9ed"),Object(m["a"])(j,r,n,!1,null,"415976da",null)),D=O.exports,L=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"board"},[a("scrollbar",[a("div",{staticClass:"wrapper"},[a("div",{staticClass:"content"},e._l(e.Cards,(function(t,s){return a("div",{key:"block-"+s,staticClass:"board__data-field"},[a("button",{staticClass:"board__reload-all",on:{click:function(t){return e.ReloadAll(s)}}},[a("i",{class:["t-icon","icon-deploy-reload"],attrs:{title:"reload"}}),a("span",[e._v("Перезагрузить все")])]),a("div",{staticClass:"board__title"},[e._v("Исходные данные / Предсказанные данные")]),a("div",{staticClass:"board__data"},e._l(t.data,(function(c,r){return a("IndexCard",e._b({key:"card-"+r,attrs:{type:t.type,block:s,index:r},on:{reload:e.ReloadCard}},"IndexCard",c,!1))})),1)])})),0)])])],1)},P=[],$=(a("d3b7"),a("25f0"),function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"card"},[a("div",{staticClass:"card__content"},["ImageClassification"==e.type?a("div",[a("div",{staticClass:"card__original"},[a("ImgCard",{attrs:{imgUrl:e.source}})],1),a("div",{staticClass:"card__result"},[a("TextCard",{style:{width:"224px"}},[e._v(e._s(e.imageClassificationText))])],1)]):e._e(),"ImageSegmentation"==e.type?a("div",[a("div",{staticClass:"card__original"},[a("ImgCard",{attrs:{imgUrl:e.source}})],1),a("div",{staticClass:"card__result"},[a("ImgCard",{attrs:{imgUrl:e.segment}})],1)]):e._e(),"graphic"==e.type?a("div",{staticClass:"card__graphic"},[a("Plotly",{attrs:{data:e.data,layout:e.layout,"display-mode-bar":!1}})],1):e._e()]),"table"!=e.type?a("div",{staticClass:"card__reload"},[a("button",{staticClass:"btn-reload",on:{click:e.ReloadCard}},[a("i",{class:["t-icon","icon-deploy-reload"],attrs:{title:"reload"}})])]):e._e()])}),R=[],A=(a("a9e3"),a("4e82"),a("99af"),function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"img-card"},[a("img",{staticClass:"img-card__image",attrs:{src:e.src}})])}),I=[],E={name:"ImgCard",props:{imgUrl:{type:String,default:"img.png"}},computed:Object(i["a"])(Object(i["a"])({},Object(l["b"])({id:"deploy/getRandId"})),{},{src:function(){return"/_media/blank/?path=".concat(this.imgUrl,"&r=").concat(this.id)}})},T=E,U=(a("7e8a"),Object(m["a"])(T,A,I,!1,null,"5ab7da04",null)),N=U.exports,q=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"text-card"},[a("scrollbar",{attrs:{ops:e.ops}},[a("pre",[e._t("default",(function(){return[e._v("TEXT")]}))],2)])],1)},M=[],X={name:"TextCard",props:{text:{type:String,default:"text"}},data:function(){return{ops:{scrollPanel:{scrollingX:!1,scrollingY:!0}}}}},B=X,J=(a("f4ce"),Object(m["a"])(B,q,M,!1,null,"6e13d9d6",null)),Y=J.exports,F=a("04d11"),G={name:"IndexCard",components:{ImgCard:N,TextCard:Y,Plotly:F["Plotly"]},data:function(){return{}},props:{source:{type:String,default:""},segment:{type:String,default:""},data:{type:[Array,Object,String],default:function(){return{}}},block:String,index:[String,Number],type:String},methods:{ReloadCard:function(){this.$emit("reload",{id:this.block,indexes:[this.index.toString()]})}},computed:Object(i["a"])(Object(i["a"])({},Object(l["b"])({graphicData:"deploy/getGraphicData",defaultLayout:"deploy/getDefaultLayout",origTextStyle:"deploy/getOrigTextStyle"})),{},{layout:function(){var e=this.defaultLayout;return this.char&&(e.title.text=this.char.title||"",e.xaxis.title=this.char.xaxis.title||"",e.yaxis.title=this.char.yaxis.title||""),e},imageClassificationText:function(){var e=this.data,t="";e.sort((function(e,t){return e[1]<t[1]?1:-1}));for(var a=0;a<e.length;a++){if(a>2)break;t+="".concat(e[a][0]," - ").concat(e[a][1],"% \n")}return t}})},H=G,z=(a("3655"),Object(m["a"])(H,$,R,!1,null,"5b028f26",null)),K=z.exports,Q={components:{IndexCard:K},data:function(){return{}},computed:Object(i["a"])({},Object(l["b"])({dataLoaded:"deploy/getDataLoaded",Cards:"deploy/getCards",height:"settings/autoHeight"})),created:function(){this.$store.dispatch("projects/get")},methods:{ReloadCard:function(e){var t=this;return Object(o["a"])(regeneratorRuntime.mark((function a(){return regeneratorRuntime.wrap((function(a){while(1)switch(a.prev=a.next){case 0:return a.next=2,t.$store.dispatch("deploy/ReloadCard",e);case 2:case"end":return a.stop()}}),a)})))()},ReloadAll:function(e){var t=this;return Object(o["a"])(regeneratorRuntime.mark((function a(){var s,c;return regeneratorRuntime.wrap((function(a){while(1)switch(a.prev=a.next){case 0:for(s=[],c=0;c<t.Cards[e].data.length;c++)s.push(c.toString());return a.next=4,t.$store.dispatch("deploy/ReloadCard",{id:e,indexes:s});case 4:case"end":return a.stop()}}),a)})))()}}},V=Q,W=(a("85d7"),Object(m["a"])(V,L,P,!1,null,"123d1a0c",null)),Z=W.exports,ee={name:"Datasets",components:{Params:D,Deploy:Z},data:function(){return{overlay:!1}},methods:{setOverlay:function(e){this.overlay=e}}},te=ee,ae=(a("fcfc"),Object(m["a"])(te,s,c,!1,null,"7f880b75",null));t["default"]=ae.exports},3014:function(e,t,a){},3180:function(e,t,a){},3655:function(e,t,a){"use strict";a("d951")},"46b3":function(e,t,a){"use strict";a("f099")},"686d":function(e,t,a){},"7e8a":function(e,t,a){"use strict";a("0da8")},"85d7":function(e,t,a){"use strict";a("3014")},a868:function(e,t,a){"use strict";a("0427")},d951:function(e,t,a){},e9ed:function(e,t,a){"use strict";a("686d")},f099:function(e,t,a){},f4ce:function(e,t,a){"use strict";a("07ce")},fcfc:function(e,t,a){"use strict";a("3180")}}]);