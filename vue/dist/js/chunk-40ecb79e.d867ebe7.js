(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-40ecb79e"],{"27db":function(e,t,a){"use strict";a.r(t);var s=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("main",{staticClass:"page-deploy"},[a("div",{staticClass:"cont"},[a("Deploy"),a("Params",{on:{overlay:e.setOverlay}})],1),e.overlay?a("div",{staticClass:"overlay"}):e._e()])},n=[],r=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"params"},[e._e(),a("scrollbar",[a("div",{staticClass:"params__body"},[a("div",{staticClass:"params__items"},[a("at-collapse",{key:e.key,attrs:{value:e.collapse},on:{"on-change":e.onchange}},e._l(e.params,(function(t,s){var n=t.visible,r=t.name,o=t.fields;return a("at-collapse-item",{directives:[{name:"show",rawName:"v-show",value:n,expression:"visible"}],key:s,staticClass:"mt-3",attrs:{name:s,title:r||""}},["outputs"!==s?a("div",{staticClass:"params__fields"},[e._l(o,(function(t,n){return[a("t-auto-field-trainings",e._b({key:s+n,class:"params__fields--"+s,attrs:{state:e.state,inline:!1},on:{parse:e.parse}},"t-auto-field-trainings",t,!1))]}))],2):a("div",{staticClass:"blocks-layers"},[e._l(o,(function(t,s){return[a("div",{key:"block_layers_"+s,staticClass:"block-layers"},[a("div",{staticClass:"block-layers__header"},[e._v(" "+e._s(t.name)+" ")]),a("div",{staticClass:"block-layers__body"},[e._l(t.fields,(function(t,s){return[a("t-auto-field-trainings",e._b({key:"checkpoint_"+s+t.parse,attrs:{state:e.state,inline:!0},on:{parse:e.parse}},"t-auto-field-trainings",t,!1))]}))],2)])]}))],2)])})),1)],1)])])],1)},o=[],c=a("1da1"),l=a("5530"),i=(a("96cf"),a("7db0"),a("ac1f"),a("5319"),a("2f62")),d={name:"Settings",components:{},data:function(){return{collapse:["main","fit","outputs","checkpoint","yolo"],key:"1212",deploy:"",replace:!1,use_sec:!1,sec:"",sec_accept:"",DataSent:!1,DataLoading:!1,passwordShow:!1,ops:{scrollPanel:{scrollingX:!1,scrollingY:!0}}}},computed:Object(l["a"])(Object(l["a"])({},Object(i["b"])({params:"deploy/getParams",height:"settings/height",moduleList:"deploy/getModuleList",projectData:"projects/getProject",userData:"projects/getUser"})),{},{checkCorrect:function(){return this.sec==this.sec_accept?"icon-deploy-password-correct":"icon-deploy-password-incorrect"},send_disabled:function(){if(this.DataLoading)return!0;if(this.use_sec){if(this.sec==this.sec_accept&&this.sec.length>5&&0!=this.deploy.length)return!1}else if(0!=this.deploy.length)return!1;return!0}}),methods:{onchange:function(e){console.log(e)},click:function(){console.log()},Percents:function(e){var t=document.querySelector(".progress-bar > .loading");t.style.width=e+"%",t.find("span").value=e},Copy:function(e){var t=document.createElement("textarea");t.value=e,t.style.top="0",t.style.left="0",t.style.position="fixed",document.body.appendChild(t),t.focus(),t.select();try{document.execCommand("copy")}catch(a){console.error("Fallback: Oops, unable to copy",a)}document.body.removeChild(t)},UseSec:function(e){this.use_sec=e.value},UseReplace:function(e){this.replace=e.value},progress:function(){var e=this;return Object(c["a"])(regeneratorRuntime.mark((function t(){var a;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("deploy/CheckProgress");case 2:a=t.sent,console.log(a),a?(e.DataLoading=!1,e.DataSent=!0,e.$emit("overlay",e.DataLoading)):e.getProgress();case 5:case"end":return t.stop()}}),t)})))()},getProgress:function(){setTimeout(this.progress,2e3)},SendData:function(){var e=this;return Object(c["a"])(regeneratorRuntime.mark((function t(){var a,s,n,r;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return a={deploy:e.deploy,replace:e.replace,use_sec:e.use_sec},e.use_sec&&(a["sec"]=e.sec),t.next=4,e.$store.dispatch("deploy/SendDeploy",a);case 4:s=t.sent,console.log(s),s&&(n=s.error,r=s.success,console.log(n,r),!n&&r&&(e.DataLoading=!0,e.$emit("overlay",e.DataLoading),e.getProgress()));case 7:case"end":return t.stop()}}),t)})))()}}},u=d,p=(a("52e8"),a("2877")),f=Object(p["a"])(u,r,o,!1,null,"76339ba5",null),h=f.exports,y=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"board"},[a("scrollbar",[a("div",{staticClass:"wrapper"},[a("div",{staticClass:"content"},[a("div",{staticClass:"board__data-field"},[a("div",[a("div",{staticClass:"board__title"},[e._v("Исходные данные / Предсказанные данные")]),e.isTable?a("div",{staticClass:"board__data"},["DataframeRegression"===e.type?a("Table",e._b({key:e.RandId,on:{reload:e.ReloadCard,reloadAll:e.ReloadAll}},"Table",e.deploy,!1)):e._e(),"DataframeClassification"===e.type?a("TableClass",e._b({key:e.RandId,on:{reload:e.ReloadCard,reloadAll:e.ReloadAll}},"TableClass",e.deploy,!1)):e._e()],1):a("div",{staticClass:"board__data"},e._l(e.Cards,(function(t,s){return a("IndexCard",e._b({key:"card-"+s,attrs:{card:t,color_map:e.deploy.color_map,index:s},on:{reload:e.ReloadCard}},"IndexCard",t,!1))})),1)])])])])])],1)},b=[],m=(a("d3b7"),a("3ca3"),a("ddb0"),a("caad"),a("25f0"),{components:{IndexCard:function(){return Promise.all([a.e("chunk-743e06ca"),a.e("chunk-24caae1c")]).then(a.bind(null,"3af9"))},Table:function(){return a.e("chunk-3c37d762").then(a.bind(null,"5448"))},TableClass:function(){return a.e("chunk-dfd64006").then(a.bind(null,"6eb8"))}},data:function(){return{}},computed:Object(l["a"])(Object(l["a"])({},Object(i["b"])({dataLoaded:"deploy/getDataLoaded",Cards:"deploy/getCards",height:"settings/autoHeight",type:"deploy/getDeployType",deploy:"deploy/getDeploy",RandId:"deploy/getRandId"})),{},{isTable:function(){return["DataframeClassification","DataframeRegression"].includes(this.type)}}),methods:{ReloadCard:function(e){var t=this;return Object(c["a"])(regeneratorRuntime.mark((function a(){return regeneratorRuntime.wrap((function(a){while(1)switch(a.prev=a.next){case 0:return a.next=2,t.$store.dispatch("deploy/ReloadCard",e);case 2:case"end":return a.stop()}}),a)})))()},ReloadAll:function(){var e=this;return Object(c["a"])(regeneratorRuntime.mark((function t(){var a,s;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:for(a=[],s=0;s<e.Cards.length;s++)a.push(s.toString());return t.next=4,e.$store.dispatch("deploy/ReloadCard",a);case 4:case"end":return t.stop()}}),t)})))()}},mounted:function(){console.log(this.deploy)}}),g=m,v=(a("8187"),Object(p["a"])(g,y,b,!1,null,"0e8a5027",null)),_=v.exports,C={name:"Datasets",components:{Params:h,Deploy:_},data:function(){return{overlay:!1}},methods:{setOverlay:function(e){this.overlay=e}}},k=C,w=(a("c60e"),Object(p["a"])(k,s,n,!1,null,"3f9d1b9e",null));t["default"]=w.exports},"3fcb":function(e,t,a){},"52e8":function(e,t,a){"use strict";a("ef1c")},8187:function(e,t,a){"use strict";a("bd29")},bd29:function(e,t,a){},c60e:function(e,t,a){"use strict";a("3fcb")},ef1c:function(e,t,a){}}]);