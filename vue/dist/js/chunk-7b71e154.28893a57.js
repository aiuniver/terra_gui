(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-7b71e154"],{"27db":function(e,t,a){"use strict";a.r(t);var r=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("main",{staticClass:"page-deploy"},[a("div",{staticClass:"cont"},[a("div",{staticClass:"board"},[a("scrollbar",[a("div",{staticClass:"wrapper"},[a("div",{staticClass:"content"},[a("div",{staticClass:"board__data-field"},[a("div",[a("div",{staticClass:"board__title"},[e._v("Исходные данные / Предсказанные данные")]),e.isTable?a("div",{staticClass:"board__data"},["DataframeRegression"===e.type?a("Table",e._b({key:"#board-"+e.updateKey,on:{reload:e.reload,reloadAll:e.reloadAll}},"Table",e.deploy,!1)):e._e(),"DataframeClassification"===e.type?a("TableClass",e._b({key:"#board-"+e.updateKey,on:{reload:e.reload,reloadAll:e.reloadAll}},"TableClass",e.deploy,!1)):e._e()],1):a("div",{staticClass:"board__data"},e._l(e.cards,(function(t,r){return a("IndexCard",{key:"#board-card-"+r,attrs:{card:t,"color-map":e.deploy.color_map,index:r,"default-layout":e.defaultLayout,type:e.type},on:{reload:e.reload}})})),1)])])])])])],1),a("Params",{attrs:{params:e.params,"module-list":e.moduleList,"project-data":e.projectData,"user-data":e.userData,"sent-deploy":e.paramsSettings.isSendParamsDeploy,"params-downloaded":e.paramsSettings,"overlay-status":e.overlay},on:{downloadSettings:e.downloadSettings,overlay:e.setOverlay,sendParamsDeploy:e.sendParamsDeploy,clear:e.clearParams}})],1),e.overlay?a("div",{staticClass:"overlay"}):e._e()])},n=[],s=a("1da1"),o=a("5530"),i=(a("96cf"),a("d3b7"),a("3ca3"),a("ddb0"),a("caad"),a("b0c0"),a("2f62")),c=a("eb4c"),d={name:"Datasets",components:{Params:function(){return a.e("chunk-5809d306").then(a.bind(null,"4973"))},IndexCard:function(){return a.e("chunk-4ea898e2").then(a.bind(null,"3af9"))},Table:function(){return a.e("chunk-1c0abd43").then(a.bind(null,"5448"))},TableClass:function(){return a.e("chunk-1f1083e0").then(a.bind(null,"6eb8"))}},computed:Object(o["a"])(Object(o["a"])({},Object(i["b"])({defaultLayout:"deploy/getDefaultLayout",dataLoaded:"deploy/getDataLoaded",cards:"deploy/getCards",autoHeight:"settings/autoHeight",type:"deploy/getDeployType",deploy:"deploy/getDeploy",params:"deploy/getParams",height:"settings/height",moduleList:"deploy/getModuleList",projectData:"projects/getProject",userData:"projects/getUser"})),{},{isTable:function(){return["DataframeClassification","DataframeRegression"].includes(this.type)}}),data:function(){return{overlay:!1,updateKey:0,debounce:null,idCheckProgressSendDeploy:null,paramsSettings:{isSendParamsDeploy:!1,isParamsSettingsLoad:!1}}},created:function(){var e=this;this.debounce=Object(c["a"])((function(t){t&&e.checkProgressDownload()}),1e3)},beforeDestroy:function(){this.debounce(!1)},methods:{clearParams:function(){this.$store.dispatch("deploy/clear")},setOverlay:function(e){this.overlay=e},checkProgressSendDeploy:function(){var e=this;return Object(s["a"])(regeneratorRuntime.mark((function t(){var a;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("deploy/checkProgress");case 2:a=t.sent,console.log(a),a&&(e.overlay=!0,e.paramsSettings.isSendParamsDeploy=!0,clearTimeout(e.idCheckProgressSendDeploy));case 5:case"end":return t.stop()}}),t)})))()},sendParamsDeploy:function(e){var t=this;return Object(s["a"])(regeneratorRuntime.mark((function a(){var r,n,s;return regeneratorRuntime.wrap((function(a){while(1)switch(a.prev=a.next){case 0:return a.next=2,t.$store.dispatch("deploy/sendDeploy",e);case 2:r=a.sent,r&&(n=r.error,s=r.success,!n&&s&&(t.overlay=!0,t.idCheckProgressSendDeploy=setTimeout(t.checkProgressSendDeploy,2e3)));case 4:case"end":return a.stop()}}),a)})))()},checkProgressDownload:function(){var e=this;return Object(s["a"])(regeneratorRuntime.mark((function t(){var a,r,n,s,o;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("deploy/progress",{});case 2:if(a=t.sent,!a||null===a||void 0===a||!a.data){t.next=15;break}if(r=a.data,n=r.finished,s=r.message,o=r.percent,e.$store.dispatch("messages/setProgressMessage",s),e.$store.dispatch("messages/setProgress",o),n){t.next=12;break}return t.next=10,e.debounce(!0);case 10:t.next=15;break;case 12:e.$store.dispatch("projects/get"),e.paramsSettings.isParamsSettingsLoad=!0,e.overlay=!1;case 15:null!==a&&void 0!==a&&a.error&&(e.overlay=!1);case 16:case"end":return t.stop()}}),t)})))()},downloadSettings:function(e){var t=this;return Object(s["a"])(regeneratorRuntime.mark((function a(){var r,n,s,o,i;return regeneratorRuntime.wrap((function(a){while(1)switch(a.prev=a.next){case 0:if(r=e.type,n=void 0===r?null:r,s=e.name,o=void 0===s?null:s,!n||!o){a.next=9;break}return t.overlay=!0,a.next=5,t.$store.dispatch("deploy/downloadSettings",{type:n,name:o});case 5:if(i=a.sent,null===i||void 0===i||!i.success){a.next=9;break}return a.next=9,t.debounce(!0);case 9:case"end":return a.stop()}}),a)})))()},reload:function(e){var t=this;return Object(s["a"])(regeneratorRuntime.mark((function a(){return regeneratorRuntime.wrap((function(a){while(1)switch(a.prev=a.next){case 0:return t.updateKey++,a.next=3,t.$store.dispatch("deploy/reloadCard",[String(e)]);case 3:case"end":return a.stop()}}),a)})))()},reloadAll:function(){var e=this;return Object(s["a"])(regeneratorRuntime.mark((function t(){var a,r;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:for(a=[],r=0;r<e.cards.length;r++)a.push(String(r));return t.next=4,e.$store.dispatch("deploy/reloadCard",a);case 4:case"end":return t.stop()}}),t)})))()}}},l=d,u=(a("a2e6"),a("2877")),p=Object(u["a"])(l,r,n,!1,null,"2f53276c",null);t["default"]=p.exports},"5dfb":function(e,t,a){},a2e6:function(e,t,a){"use strict";a("5dfb")},eb4c:function(e,t,a){"use strict";a.d(t,"a",(function(){return r}));var r=function(e,t,a){var r;return function(){var n=this,s=arguments,o=function(){r=null,a||e.apply(n,s)},i=a&&!r;clearTimeout(r),r=setTimeout(o,t),i&&e.apply(n,s)}}}}]);