(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-b25fb372"],{"528f":function(e,t,a){},"6f6c":function(e,t,a){"use strict";a.r(t);var r=function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("div",{staticClass:"page-deploy"},[a("div",{staticClass:"cont"},[a("div",{staticClass:"board"},[a("scrollbar",[a("div",{staticClass:"wrapper"},[a("div",{staticClass:"content"},[a("div",{staticClass:"board__data-field"},[a("div",[a("div",{staticClass:"board__title"},[e._v("Исходные данные / Предсказанные данные")]),e.isTable?a("div",{staticClass:"board__data"},["table_data_regression"===e.type?a("Table",e._b({key:"#board-"+e.updateKey,on:{reload:e.reload,reloadAll:e.reloadAll}},"Table",e.deploy,!1)):e._e(),"table_data_classification"===e.type?a("TableClass",e._b({key:"#board-"+e.updateKey,on:{reload:e.reload,reloadAll:e.reloadAll}},"TableClass",e.deploy,!1)):e._e()],1):a("div",{staticClass:"board__data"},e._l(e.cards,(function(t,r){return a("IndexCard",{key:"#board-card-"+r,attrs:{card:t,"color-map":e.deploy.color_map,index:r,"default-layout":e.defaultLayout,type:e.type},on:{reload:e.reload}})})),1)])])])])])],1),a("Params",{attrs:{params:e.params,"module-list":e.moduleList,"project-data":e.projectData,"user-data":e.userData,"sent-deploy":e.paramsSettings.isSendParamsDeploy,"params-downloaded":e.paramsSettings,"overlay-status":e.isOverlay},on:{downloadSettings:e.getData,overlay:e.setOverlay,sendParamsDeploy:e.uploadData,clear:e.clearParams}})],1)])},s=[],n=a("1da1"),o=a("5530"),i=(a("96cf"),a("d3b7"),a("3ca3"),a("ddb0"),a("caad"),a("b0c0"),a("2f62")),d=a("eb4c"),l={name:"Datasets",components:{Params:function(){return Promise.all([a.e("styles"),a.e("chunk-8762b9c6")]).then(a.bind(null,"ecf0"))},IndexCard:function(){return Promise.all([a.e("styles"),a.e("chunk-5cddc26d")]).then(a.bind(null,"3af9"))},Table:function(){return Promise.all([a.e("styles"),a.e("chunk-42f8169c")]).then(a.bind(null,"5448"))},TableClass:function(){return Promise.all([a.e("styles"),a.e("chunk-749d6fa2")]).then(a.bind(null,"6eb8"))}},computed:Object(o["a"])(Object(o["a"])({},Object(i["c"])({defaultLayout:"deploy/getDefaultLayout",dataLoaded:"deploy/getDataLoaded",cards:"deploy/getCards",autoHeight:"settings/autoHeight",type:"deploy/getDeployType",deploy:"deploy/getDeploy",params:"deploy/getParams",height:"settings/height",isOverlay:"settings/getOverlay",moduleList:"deploy/getModuleList",projectData:"projects/getProject",userData:"projects/getUser"})),{},{isTable:function(){return["table_data_classification","table_data_regression"].includes(this.type)}}),data:function(){return{updateKey:0,debounceProgressData:null,debounceProgressUpload:null,idCheckProgressSendDeploy:null,paramsSettings:{isSendParamsDeploy:!1,isParamsSettingsLoad:!1}}},created:function(){var e=this;this.debounceProgressData=Object(d["a"])((function(t){t&&e.progressData()}),1e3),this.debounceProgressUpload=Object(d["a"])((function(t){t&&e.progressUpload()}),1e3)},beforeDestroy:function(){this.debounceProgressUpload(!1),this.debounceProgressData(!1)},methods:{clearParams:function(){this.$store.dispatch("deploy/clear")},setOverlay:function(e){this.$store.dispatch("settings/setOverlay",e)},progressUpload:function(){var e=this;return Object(n["a"])(regeneratorRuntime.mark((function t(){var a,r,s,n,o;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("deploy/progressUpload");case 2:if(a=t.sent,console.log(a),null===a||void 0===a||!a.data){t.next=16;break}if(r=a.data,s=r.finished,n=r.message,o=r.percent,e.$store.dispatch("messages/setProgressMessage",n),e.$store.dispatch("messages/setProgress",o),s||a.error){t.next=13;break}return t.next=11,e.debounceProgressUpload(!0);case 11:t.next=16;break;case 13:e.$store.dispatch("projects/get"),e.paramsSettings.isSendParamsDeploy=!0,e.setOverlay(!1);case 16:case"end":return t.stop()}}),t)})))()},uploadData:function(e){var t=this;return Object(n["a"])(regeneratorRuntime.mark((function a(){var r,s,n;return regeneratorRuntime.wrap((function(a){while(1)switch(a.prev=a.next){case 0:return a.next=2,t.$store.dispatch("deploy/uploadData",e);case 2:if(r=a.sent,!r){a.next=9;break}if(s=r.error,n=r.success,s||!n){a.next=9;break}return t.setOverlay(!0),a.next=9,t.debounceProgressUpload(!0);case 9:case"end":return a.stop()}}),a)})))()},progressData:function(){var e=this;return Object(n["a"])(regeneratorRuntime.mark((function t(){var a,r,s,n,o;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:return t.next=2,e.$store.dispatch("deploy/progressData",{});case 2:if(a=t.sent,!a||null===a||void 0===a||!a.data){t.next=15;break}if(r=a.data,s=r.finished,n=r.message,o=r.percent,e.$store.dispatch("messages/setProgressMessage",n),e.$store.dispatch("messages/setProgress",o),s){t.next=12;break}return t.next=10,e.debounceProgressData(!0);case 10:t.next=15;break;case 12:e.$store.dispatch("projects/get"),e.paramsSettings.isParamsSettingsLoad=!0,e.setOverlay(!1);case 15:null!==a&&void 0!==a&&a.error&&e.setOverlay(!1);case 16:case"end":return t.stop()}}),t)})))()},getData:function(e){var t=this;return Object(n["a"])(regeneratorRuntime.mark((function a(){var r,s,n,o,i;return regeneratorRuntime.wrap((function(a){while(1)switch(a.prev=a.next){case 0:if(r=e.type,s=void 0===r?null:r,n=e.name,o=void 0===n?null:n,!s||!o){a.next=9;break}return t.setOverlay(!0),a.next=5,t.$store.dispatch("deploy/getData",{type:s,name:o});case 5:if(i=a.sent,null===i||void 0===i||!i.success){a.next=9;break}return a.next=9,t.debounceProgressData(!0);case 9:case"end":return a.stop()}}),a)})))()},reload:function(e){var t=this;return Object(n["a"])(regeneratorRuntime.mark((function a(){return regeneratorRuntime.wrap((function(a){while(1)switch(a.prev=a.next){case 0:return t.updateKey++,a.next=3,t.$store.dispatch("deploy/reloadCard",[String(e)]);case 3:case"end":return a.stop()}}),a)})))()},reloadAll:function(){var e=this;return Object(n["a"])(regeneratorRuntime.mark((function t(){var a,r;return regeneratorRuntime.wrap((function(t){while(1)switch(t.prev=t.next){case 0:for(a=[],r=0;r<e.cards.length;r++)a.push(String(r));return t.next=4,e.$store.dispatch("deploy/reloadCard",a);case 4:case"end":return t.stop()}}),t)})))()}}},c=l,u=(a("a4d0"),a("2877")),p=Object(u["a"])(c,r,s,!1,null,"1a1694e4",null);t["default"]=p.exports},a4d0:function(e,t,a){"use strict";a("528f")}}]);