(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-cfe7a8f6"],{"0359":function(t,e,a){"use strict";a("f5cb")},"0a01":function(t,e,a){},"1a06":function(t,e,a){"use strict";a("9684")},"1dfe":function(t,e,a){},"24c7":function(t,e,a){},"27db":function(t,e,a){"use strict";a.r(e);var s=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("main",{staticClass:"page-deploy"},[a("div",{staticClass:"cont"},[a("Deploy"),a("Params")],1)])},c=[],r=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"params"},[a("div",{staticClass:"params-container__name"},[t._v("Загрузка в демо-панель")]),a("div",{staticClass:"params-container pa-5"},[a("div",{staticClass:"label"},[t._v("Название папки")]),a("div",{staticClass:"t-input"},[a("label",{staticClass:"t-input__label"},[t._v("https://demo.neural-university.ru/"+t._s(t.userData.login)+"/"+t._s(t.projectData.name_alias)+"/"+t._s(t.deploy))]),a("input",{directives:[{name:"model",rawName:"v-model",value:t.deploy,expression:"deploy"}],staticClass:"t-input__input",attrs:{type:"text",name:"deploy[deploy]"},domProps:{value:t.deploy},on:{blur:function(e){return t.$emit("blur",e.target.value)},input:function(e){e.target.composing||(t.deploy=e.target.value)}}})]),a("Checkbox",{staticClass:"pd__top",attrs:{label:"Перезаписать с таким же названием папки",type:"checkbox"},on:{change:t.UseReplace}}),a("Checkbox",{attrs:{label:"Использовать пароль для просмотра страницы",type:"checkbox"},on:{change:t.UseSec}}),t.use_sec?a("div",{staticClass:"password"},[a("div",{staticClass:"t-input"},["checkbox"===(t.passwordShow?"text":"password")?a("input",{directives:[{name:"model",rawName:"v-model",value:t.sec,expression:"sec"}],attrs:{placeholder:"Введите пароль",type:"checkbox"},domProps:{checked:Array.isArray(t.sec)?t._i(t.sec,null)>-1:t.sec},on:{change:function(e){var a=t.sec,s=e.target,c=!!s.checked;if(Array.isArray(a)){var r=null,i=t._i(a,r);s.checked?i<0&&(t.sec=a.concat([r])):i>-1&&(t.sec=a.slice(0,i).concat(a.slice(i+1)))}else t.sec=c}}}):"radio"===(t.passwordShow?"text":"password")?a("input",{directives:[{name:"model",rawName:"v-model",value:t.sec,expression:"sec"}],attrs:{placeholder:"Введите пароль",type:"radio"},domProps:{checked:t._q(t.sec,null)},on:{change:function(e){t.sec=null}}}):a("input",{directives:[{name:"model",rawName:"v-model",value:t.sec,expression:"sec"}],attrs:{placeholder:"Введите пароль",type:t.passwordShow?"text":"password"},domProps:{value:t.sec},on:{input:function(e){e.target.composing||(t.sec=e.target.value)}}}),a("div",{staticClass:"password__icon"},[a("i",{class:["t-icon",t.passwordShow?"icon-deploy-password-open":"icon-deploy-password-close"],attrs:{title:"show password"},on:{click:function(e){t.passwordShow=!t.passwordShow}}})])]),a("div",{staticClass:"t-input"},["checkbox"===(t.passwordShow?"text":"password")?a("input",{directives:[{name:"model",rawName:"v-model",value:t.sec_accept,expression:"sec_accept"}],attrs:{placeholder:"Подтверждение пароля",type:"checkbox"},domProps:{checked:Array.isArray(t.sec_accept)?t._i(t.sec_accept,null)>-1:t.sec_accept},on:{change:function(e){var a=t.sec_accept,s=e.target,c=!!s.checked;if(Array.isArray(a)){var r=null,i=t._i(a,r);s.checked?i<0&&(t.sec_accept=a.concat([r])):i>-1&&(t.sec_accept=a.slice(0,i).concat(a.slice(i+1)))}else t.sec_accept=c}}}):"radio"===(t.passwordShow?"text":"password")?a("input",{directives:[{name:"model",rawName:"v-model",value:t.sec_accept,expression:"sec_accept"}],attrs:{placeholder:"Подтверждение пароля",type:"radio"},domProps:{checked:t._q(t.sec_accept,null)},on:{change:function(e){t.sec_accept=null}}}):a("input",{directives:[{name:"model",rawName:"v-model",value:t.sec_accept,expression:"sec_accept"}],attrs:{placeholder:"Подтверждение пароля",type:t.passwordShow?"text":"password"},domProps:{value:t.sec_accept},on:{input:function(e){e.target.composing||(t.sec_accept=e.target.value)}}}),a("div",{staticClass:"password__icon"},[a("i",{class:["t-icon",t.checkCorrect],attrs:{title:"is correct"}})])])]):t._e(),t.DataSent?t._e():a("button",{attrs:{disabled:t.send_disabled},on:{click:t.SendData}},[t._v("Отправить")]),t.DataLoading?a("div",{staticClass:"loader"},[a("div",{staticClass:"loader__title"},[t._v("Дождитесь окончания загрузки")]),a("div",{staticClass:"loader__progress"},[a("load-spiner")],1)]):t._e(),t.DataSent?a("div",{staticClass:"req-ans"},[a("div",{staticClass:"answer__success"},[t._v("Загрузка завершена!")]),a("div",{staticClass:"answer__label"},[t._v("Ссылка на сформированную загрузку")]),a("div",{staticClass:"answer__url"},[a("i",{class:["t-icon","icon-deploy-copy"],attrs:{title:"copy"},on:{click:function(e){return t.Copy(t.moduleList.url)}}}),a("a",{attrs:{href:t.moduleList.url,target:"_blank"}},[t._v(t._s(t.moduleList.url))])])]):t._e(),t.DataSent?a("ModuleList",{attrs:{moduleList:t.moduleList.api_text}}):t._e()],1)])},i=[],o=a("1da1"),n=a("5530"),l=(a("96cf"),a("7db0"),a("ac1f"),a("5319"),a("2f62")),d=a("e902"),u=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"load__answer"},[a("scrollbar",[a("div",{staticClass:"apiBlock"},[t._v(" "+t._s(t.moduleList)+" ")])])],1)},p=[],_={name:"ModuleList",props:{moduleList:{type:String,default:function(){return""}}}},v=_,f=(a("46b3"),a("2877")),m=Object(f["a"])(v,u,p,!1,null,"ef881ec4",null),h=m.exports,g=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("svg",{staticClass:"spinner spinner--circle",style:t.styles,attrs:{viewBox:"0 0 66 66",xmlns:"http://www.w3.org/2000/svg"}},[a("circle",{staticClass:"path",attrs:{fill:"none","stroke-width":"6","stroke-linecap":"round",cx:"33",cy:"33",r:"30"}})])},b=[],y={name:"load-spiner",props:{size:{default:"40px"}},computed:{styles:function(){return{width:this.size,height:this.size}}}},C=y,w=(a("41f1"),Object(f["a"])(C,g,b,!1,null,"763f1670",null)),x=w.exports,k={name:"Settings",components:{Checkbox:d["a"],ModuleList:h,LoadSpiner:x},data:function(){return{deploy:"",replace:!1,use_sec:!1,sec:"",sec_accept:"",send_disabled:!0,DataSent:!1,DataLoading:!1,passwordShow:!1}},computed:Object(n["a"])(Object(n["a"])({},Object(l["b"])({height:"settings/height",moduleList:"deploy/getModuleList",projectData:"projects/getProject",userData:"projects/getUser"})),{},{checkCorrect:function(){return this.sec==this.sec_accept&&this.sec.length>5?"icon-deploy-password-correct":"icon-deploy-password-incorrect"}}),mounted:function(){},watch:{deploy:function(t){this.send_disabled=""===t},sec_accept:function(t){this.use_sec&&(t==this.sec?this.send_disabled=!1:this.send_disabled=!0)}},methods:{click:function(){console.log()},Percents:function(t){var e=document.querySelector(".progress-bar > .loading");e.style.width=t+"%",e.find("span").value=t},Copy:function(t){var e=document.createElement("textarea");e.value=t,e.style.top="0",e.style.left="0",e.style.position="fixed",document.body.appendChild(e),e.focus(),e.select();try{document.execCommand("copy")}catch(a){console.error("Fallback: Oops, unable to copy",a)}document.body.removeChild(e)},UseSec:function(t){this.use_sec=t.value},UseReplace:function(t){this.replace=t.value},progress:function(){var t=this;return Object(o["a"])(regeneratorRuntime.mark((function e(){var a;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.$store.dispatch("deploy/CheckProgress");case 2:a=e.sent,console.log(a),a?(t.DataLoading=!1,t.DataSent=!0):t.getProgress();case 5:case"end":return e.stop()}}),e)})))()},getProgress:function(){setTimeout(this.progress,2e3)},SendData:function(){var t=this;return Object(o["a"])(regeneratorRuntime.mark((function e(){var a;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return a={deploy:t.deploy,replace:t.replace,use_sec:t.use_sec},t.use_sec&&(a["sec"]=t.sec),t.DataLoading=!0,e.next=5,t.$store.dispatch("deploy/SendDeploy",a);case 5:t.getProgress();case 6:case"end":return e.stop()}}),e)})))()}}},S=k,j=(a("1a06"),Object(f["a"])(S,r,i,!1,null,"e52d2d26",null)),D=j.exports,O=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"board"},[a("div",{staticClass:"wrapper"},[a("div",{staticClass:"content"},[a("div",{staticClass:"board__data-field"},[a("div",{staticClass:"board__title"},[t._v("Исходные данные / Предсказанные данные")]),a("div",{staticClass:"board__data"},t._l(t.Cards,(function(e,s){return a("IndexCard",t._b({key:"card-"+s},"IndexCard",e,!1))})),1)])])])])},L=[],N=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"card"},[a("div",{staticClass:"card__content"},["card"==t.type?a("div",[a("div",{staticClass:"card__original"},["image"==t.original.type?a("ImgCard"):t._e(),"text"==t.original.type?a("TextCard",{style:t.origTextStyle},[t._v(t._s(t.original.data))]):t._e()],1),a("div",{staticClass:"card__result"},["image"==t.result.type?a("ImgCard"):t._e(),"text"==t.result.type?a("TextCard",[t._v(t._s(t.result.data))]):t._e()],1)]):t._e(),"graphic"==t.type?a("div",{staticClass:"card__graphic"},[a("Plotly",{attrs:{data:t.data,layout:t.layout,"display-mode-bar":!1}})],1):t._e(),"table"==t.type?a("div",{staticClass:"card__table"},[a("Table")],1):t._e()]),"table"!=t.type?a("div",{staticClass:"card__reload"},[a("button",{staticClass:"btn-reload"},[a("i",{class:["t-icon","icon-deploy-reload"],attrs:{title:"reload"}})])]):t._e()])},P=[],T=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"img-card"},[s("img",{staticClass:"img-card__image",attrs:{src:a("90d4"),alt:t.ImgAlt}})])},$=[],E={name:"ImgCard",props:{ImgUrl:{type:String,default:"@/../public/imgs/img.png"},ImgAlt:{type:String,default:"image"}}},I=E,A=(a("abf7"),Object(f["a"])(I,T,$,!1,null,"ec4a2fcc",null)),R=A.exports,U=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"text-card"},[a("scrollbar",[t._t("default",(function(){return[t._v("TEXT")]}))],2)],1)},q=[],M={name:"TextCard",props:{text:{type:String,default:"text"}}},z=M,B=(a("65b8"),Object(f["a"])(z,U,q,!1,null,"33d62058",null)),J=B.exports,F=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"content"},[a("table",{staticClass:"table"},[a("tr",{staticClass:"table__title-row"},[a("td",[a("button",{staticClass:"table__reload-all"},[a("i",{class:["t-icon","icon-deploy-reload"],attrs:{title:"reload"}}),a("span",[t._v("Перезагрузить все")])])]),a("td",[t._v("Предсказанные данные")]),a("td",[t._v("Комнат")]),a("td",[t._v("Метро / ЖД станции")]),a("td",[t._v("От станции")]),a("td",[t._v("Дом")]),a("td",[t._v("Балкон")]),a("td",[t._v("Санузел")]),a("td",[t._v("Площадь")]),a("td",[t._v("Цена, руб.")]),a("td",[t._v("ГРМ")]),a("td",[t._v("Бонус агенству")])]),a("tr",[a("td",{staticClass:"table__td-reload"},[a("button",{staticClass:"td-reload__btn-reload"},[a("i",{class:["t-icon","icon-deploy-reload"],attrs:{title:"reload"}})])]),a("td",{staticClass:"table__result-data"},[t._v("Мастер и Маргарита, Михаил Афанасьевич Булгаков ")]),a("td",[t._v("1")]),a("td",[t._v("Шелепиха м.")]),a("td",[t._v("12п")]),a("td",[t._v("35/37 М")]),a("td",[t._v("NaN")]),a("td",[t._v("2")]),a("td",[t._v("64.1/23/20")]),a("td",[t._v("19500000.0")]),a("td",[t._v("NaN")]),a("td",[t._v("NaN")])]),a("tr",[a("td",{staticClass:"table__td-reload"},[a("button",{staticClass:"td-reload__btn-reload"},[a("i",{class:["t-icon","icon-deploy-reload"],attrs:{title:"reload"}})])]),a("td",{staticClass:"table__result-data"},[t._v("Мастер и Маргарита, Михаил Афанасьевич Булгаков ")]),a("td",[t._v("1")]),a("td",[t._v("Шелепиха м.")]),a("td",[t._v("12п")]),a("td",[t._v("35/37 М")]),a("td",[t._v("NaN")]),a("td",[t._v("2")]),a("td",[t._v("64.1/23/20")]),a("td",[t._v("19500000.0")]),a("td",[t._v("NaN")]),a("td",[t._v("NaN")])])])])},G=[],H={name:"Table"},X=H,K=(a("bd46"),Object(f["a"])(X,F,G,!1,null,"30d56c34",null)),Q=K.exports,V=a("04d11"),W={name:"IndexCard",components:{ImgCard:R,TextCard:J,Table:Q,Plotly:V["Plotly"]},data:function(){return{}},props:{original:{type:Object,default:function(){return{}}},result:{type:Object,default:function(){return{}}},type:{type:String,default:""}},mounted:function(){console.log(this.graphicData)},computed:Object(n["a"])(Object(n["a"])({},Object(l["b"])({graphicData:"deploy/getGraphicData",defaultLayout:"deploy/getDefaultLayout",origTextStyle:"deploy/getOrigTextStyle"})),{},{layout:function(){var t=this.defaultLayout;return this.char&&(t.title.text=this.char.title||"",t.xaxis.title=this.char.xaxis.title||"",t.yaxis.title=this.char.yaxis.title||""),t},data:function(){var t=[this.graphicData]||!1;return t}})},Y=W,Z=(a("0359"),Object(f["a"])(Y,N,P,!1,null,"108aab53",null)),tt=Z.exports,et={components:{IndexCard:tt},data:function(){return{}},computed:Object(n["a"])({},Object(l["b"])({dataLoaded:"deploy/getDataLoaded",Cards:"deploy/getCards",height:"settings/autoHeight"})),methods:{click:function(t){console.log(t)}}},at=et,st=(a("4d69"),Object(f["a"])(at,O,L,!1,null,"bdb2a7e4",null)),ct=st.exports,rt={name:"Datasets",components:{Params:D,Deploy:ct}},it=rt,ot=(a("f2af"),Object(f["a"])(it,s,c,!1,null,"31143d8c",null));e["default"]=ot.exports},"41f1":function(t,e,a){"use strict";a("24c7")},"46b3":function(t,e,a){"use strict";a("f099")},"4d69":function(t,e,a){"use strict";a("93a6")},"65b8":function(t,e,a){"use strict";a("1dfe")},"90d4":function(t,e,a){t.exports=a.p+"img/img.83f4a1c9.png"},"93a6":function(t,e,a){},9684:function(t,e,a){},abf7:function(t,e,a){"use strict";a("0a01")},bd46:function(t,e,a){"use strict";a("ddfc")},cd06:function(t,e,a){},ddfc:function(t,e,a){},f099:function(t,e,a){},f2af:function(t,e,a){"use strict";a("cd06")},f5cb:function(t,e,a){}}]);