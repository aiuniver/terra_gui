(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-22580e1a"],{"070b":function(t,e,s){},"0ae7":function(t,e,s){"use strict";s("eca8")},1255:function(t,e,s){"use strict";s("d015")},1640:function(t,e,s){},1845:function(t,e,s){},"1fcc":function(t,e,s){"use strict";s("8b03")},"2e98":function(t,e,s){"use strict";s("7538")},"337a":function(t,e,s){},"3b64":function(t,e,s){},"4a13":function(t,e,s){"use strict";s("ff49")},5633:function(t,e,s){"use strict";s("3b64")},6914:function(t,e,s){},"6c76":function(t,e,s){"use strict";s("070b")},"6ef8":function(t,e,s){},7538:function(t,e,s){},"7b8a":function(t,e,s){},"7d57":function(t,e,s){"use strict";s("e0a5")},"7e79":function(t,e,s){},"8b03":function(t,e,s){},9605:function(t,e,s){},9849:function(t,e,s){"use strict";s("337a")},b707:function(t,e,s){"use strict";s.r(e);var a=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("main",{staticClass:"page-training"},[s("div",{staticClass:"cont"},[s("Toolbar",{attrs:{collabse:t.collabse}}),s("Graphics",{on:{collabse:function(e){t.collabse=e}}}),s("Params")],1)])},i=[],n=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"toolbar"},[s("ul",{staticClass:"toolbar__menu"},t._l(t.items,(function(e,a){var i=e.title,n=e.disabled,l=e.icon;return s("li",{key:"items_"+a,class:["toolbar__menu--item"],attrs:{disabled:n}},[s("i",{class:["t-icon",l,{active:t.active(a)}],attrs:{title:i}})])})),0)])},l=[],r=s("5530"),c=(s("caad"),s("2532"),s("d3b7"),s("25f0"),s("2f62")),o={name:"Toolbar",props:{collabse:Array},data:function(){return{}},computed:Object(r["a"])({},Object(c["b"])({items:"trainings/getToolbar"})),methods:{active:function(t){return this.collabse.includes(t.toString())},click:function(){}}},_=o,u=(s("f7f8"),s("2877")),d=Object(u["a"])(_,n,l,!1,null,"559a36b5",null),v=d.exports,m=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"board"},[s("scrollbar",{attrs:{ops:{scrollPanel:{scrollingX:!1}}}},[s("div",{staticClass:"wrapper"},[s("at-collapse",{staticClass:"mt-3",on:{"on-change":t.change}},[s("at-collapse-item",{staticClass:"mt-3",attrs:{title:"Лоссы",center:""}},[s("LoadSpiner",{directives:[{name:"show",rawName:"v-show",value:t.loading,expression:"loading"}]}),t.show?s("LossGraphs",{on:{isLoad:function(e){t.loading=!1}}}):t._e()],1),s("at-collapse-item",{staticClass:"mt-3",attrs:{title:"Метрики",center:""}},[s("MetricGraphs",{on:{isLoad:function(e){t.loading=!1}}})],1),s("at-collapse-item",{staticClass:"mt-3",attrs:{title:"Промежуточные результаты",center:""}},[s("PrePesults"),s("Prediction")],1),s("at-collapse-item",{staticClass:"mt-3",attrs:{title:"Прогресс обучения",center:""}},[s("Progress")],1),s("at-collapse-item",{staticClass:"mt-3",attrs:{title:"Таблица прогресса обучения",center:""}},[s("Texts")],1),s("at-collapse-item",{staticClass:"mt-3",attrs:{title:"Статистические данные",center:""}},[s("Stats")],1),s("at-collapse-item",{staticClass:"mt-3",attrs:{title:"Баланс данных",center:""}},[s("Balance")],1)],1)],1)])],1)},p=[],f=(s("3ca3"),s("ddb0"),function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"texts"},[s("div",{staticClass:"texts__header"},[s("div",{staticClass:"texts__header__out"},[s("p",[t._v("output 1")]),s("t-checkbox",{attrs:{inline:!0,label:"loss"}}),s("t-checkbox",{attrs:{inline:!0,label:"данные метрики"}})],1),s("div",{staticClass:"texts__header__out"},[s("p",[t._v("output 2")]),s("t-checkbox",{attrs:{inline:!0,label:"loss"}}),s("t-checkbox",{attrs:{inline:!0,label:"данные метрики"}})],1),s("button",[t._v("Показать")])]),s("div",{staticClass:"texts__content"},[s("div",{staticClass:"inner"},[s("div",{staticClass:"epochs"},[s("Table",{attrs:{data:t.progressTable}})],1)])])])}),b=[],h=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("table",[s("thead",[s("tr",{staticClass:"outputs_heads"},[s("th",{attrs:{rowspan:"2"}},[t._v("Эпоха")]),t._m(0),t._l(t.outputs,(function(e,a){return s("th",{key:a,attrs:{colspan:"4"}},[t._v(t._s(a))])}))],2),s("tr",{staticClass:"callbacks_heads"},[t._l(t.output,(function(e,a){return t._l(e,(function(e,i){return s("th",{key:a+i},[t._v(t._s(i))])}))}))],2)]),s("tbody",t._l(t.data,(function(e,a,i){var n=e.time,l=e.data;return s("tr",{key:"epoch_"+i},[s("td",{staticClass:"epoch_num"},[t._v(t._s(a))]),s("td",[t._v(t._s(t._f("int")(n)))]),t._l(l,(function(e,a){return[t._l(e,(function(e,i){return t._l(e,(function(e,n){return s("td",{key:a+"t"+i+"r"+n,staticClass:"value"},[s("span",[t._v(t._s(t._f("int")(e)))]),s("i",[t._v(".")]),t._v(t._s(t._f("drob")(e)))])}))}))]}))],2)})),0),s("tfoot",[s("tr",[s("th",{attrs:{colspan:"6"}},[t._v(t._s("summary"))])])])])},g=[function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("th",{attrs:{rowspan:"2"}},[t._v("Время"),s("br"),t._v("(сек.)")])}],C=(s("07ac"),s("fb6a"),s("b680"),{name:"TTable",props:{data:{type:Object,default:function(){}}},computed:{outputs:function(){var t,e;return(null===(t=this.data)||void 0===t||null===(e=t[1])||void 0===e?void 0:e.data)||{}},output:function(){return Object.values(this.outputs)[0]}},filters:{int:function(t){return~~t},drob:function(t){return(t%1).toFixed(9).slice(2)}}}),y=C,x=(s("1fcc"),Object(u["a"])(y,h,g,!1,null,"1aaa0f69",null)),k=x.exports,w={name:"Texts",components:{Table:k},computed:{progressTable:function(){return this.$store.getters["trainings/getTrainData"]("progress_table")||[]}}},O=w,j=(s("1255"),Object(u["a"])(O,f,b,!1,null,"000e6f02",null)),T=j.exports,S=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"t-progress"},[s("div",{staticClass:"t-progress__item t-progress__item--timers"},[s("Timers",t._b({},"Timers",t.timings,!1))],1),s("div",{staticClass:"t-progress__item t-progress__item--info"},[s("Sysinfo",{attrs:{usage:t.usage}})],1)])},$=[],z=function(){var t=this,e=t.$createElement,s=t._self._c||e;return t.isEmptyAll()?s("div",{staticClass:"t-sysinfo"},[s("div",{staticClass:"t-sysinfo__label"},[t._v("Информация об устройстве")]),s("div",{staticClass:"t-sysinfo__grid"},[t.isEmpty(t.gpu)?s("div",[s("div",{staticClass:"t-sysinfo__grid--item"},[t._v("GPU")]),s("div",{class:["t-sysinfo__grid--item",{warning:t.isWarning(t.gpu.gpu_utilization)}]},[s("p",{staticClass:"t-sysinfo__gpu-name"},[t._v("NVIDIA GeForce GTX 1060 6 GB")]),s("p",[t._v(t._s((t.gpu.gpu_utilization||"0%")+" ("+(t.gpu.gpu_memory_used||"0")+" / "+(t.gpu.gpu_memory_total||"0")+")"))]),s("div",{staticClass:"t-sysinfo__progress-bar"},[s("div",{staticClass:"t-sysinfo__progress-bar--fill",style:{width:(t.gpu.gpu_utilization||0)+"%"}})])])]):t._e(),t.isEmpty(t.cpu)?s("div",[s("div",{staticClass:"t-sysinfo__grid--item"},[t._v("CPU")]),s("div",{class:["t-sysinfo__grid--item",{warning:t.isWarning(t.cpu.cpu_utilization)}]},[s("p",[t._v(t._s((t.cpu.cpu_utilization||"0%")+" ("+(t.cpu.cpu_memory_used||"0")+" / "+(t.cpu.cpu_memory_total||"0")+")"))]),s("div",{staticClass:"t-sysinfo__progress-bar"},[s("div",{staticClass:"t-sysinfo__progress-bar--fill",style:{width:(t.cpu.cpu_utilization||0)+"%"}})])])]):t._e(),t.isEmpty(t.ram)?s("div",[s("div",{staticClass:"t-sysinfo__grid--item"},[t._v("RAM")]),s("div",{class:["t-sysinfo__grid--item",{warning:t.isWarning(t.ram.ram_utilization)}]},[s("p",[t._v(" "+t._s((t.ram.ram_utilization+"% "||!1)+" ("+(t.ram.ram_memory_used||"0")+" / "+(t.ram.ram_memory_total||"0")+")")+" ")]),s("div",{staticClass:"t-sysinfo__progress-bar"},[s("div",{staticClass:"t-sysinfo__progress-bar--fill",style:{width:(t.ram.ram_utilization||0)+"%"}})])])]):t._e(),t.isEmpty(t.disk)?s("div",[s("div",{staticClass:"t-sysinfo__grid-item"},[t._v("Disk")]),s("div",{class:["t-sysinfo__grid-item",{warning:t.isWarning(t.disk.disk_utilization)}]},[s("p",[t._v(" "+t._s((t.disk.disk_utilization+"% "||!1)+" ("+(t.disk.disk_memory_used||"0")+" / "+(t.disk.disk_memory_total||"0")+")")+" ")]),s("div",{staticClass:"t-sysinfo__progress-bar"},[s("div",{staticClass:"t-sysinfo__progress-bar--fill",style:{width:(t.disk.disk_utilization||0)+"%"}})])])]):t._e()])]):t._e()},E=[],P=(s("b64b"),s("498a"),s("ac1f"),s("5319"),{name:"",props:{usage:Object},computed:{disk:function(){var t;return(null===(t=this.usage)||void 0===t?void 0:t.Disk)||{}},gpu:function(){var t;return(null===(t=this.usage)||void 0===t?void 0:t.GPU)||{}},cpu:function(){var t;return(null===(t=this.usage)||void 0===t?void 0:t.CPU)||{}},ram:function(){var t;return(null===(t=this.usage)||void 0===t?void 0:t.RAM)||{}}},methods:{isEmptyAll:function(){return this.isEmpty(this.disk)&&(this.isEmpty(this.gpu)||this.isEmpty(this.cpu))&&this.isEmpty(this.ram)},isEmpty:function(t){return 0!==Object.keys(t).length},isWarning:function(t){var e,s=null===t||void 0===t||null===(e=t.trim())||void 0===e?void 0:e.replace("%","");return console.log(),+s>50}}}),G=P,R=(s("5633"),Object(u["a"])(G,z,E,!1,null,"b47ad76e",null)),A=R.exports,D=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"t-time"},[s("div",{staticClass:"t-time__train"},[s("div",{staticClass:"t-time__timer-wrapper"},[s("div",{staticClass:"t-time__timer"},[s("span",[t._v("Расчетное время обучения")]),s("div",[t._v(t._s(t.formatTime(t.avg_epoch_time)))])]),s("div",{staticClass:"t-time__timer"},[s("span",[t._v("Прошло времени")]),s("div",[t._v(t._s(t.formatTime(t.avg_epoch_time)))])]),s("div",{staticClass:"t-time__timer"},[s("span",[t._v("Время до окончания обучения")]),s("div",[t._v(t._s(t.formatTime(t.avg_epoch_time)))])]),s("div",{staticClass:"t-time__timer"},[s("span",[t._v("Эпоха")]),s("div",[t._v(t._s(t.epoch.current||0))]),t._v(" / "),s("div",[t._v(t._s(t.epoch.total||0))])])]),t._m(0)]),s("div",{staticClass:"t-time__age"},[s("div",{staticClass:"t-time__timer-wrapper"},[s("div",{staticClass:"t-time__timer"},[s("span",[t._v("Среднее время эпохи")]),s("div",[t._v(t._s(t.formatTime(t.avg_epoch_time)))])]),s("div",{staticClass:"t-time__timer"},[s("span",[t._v("Прошло времени на эпоху")]),s("div",[t._v(t._s(t.formatTime(t.avg_epoch_time)))])]),s("div",{staticClass:"t-time__timer"},[s("span",[t._v("Время до окончания текущей эпохи")]),s("div",[t._v(t._s(t.formatTime(t.avg_epoch_time)))])]),s("div",{staticClass:"t-time__timer"},[s("span",[t._v("Батч")]),s("div",[t._v(t._s(t.batch.current||0))]),t._v(" / "),s("div",[t._v(t._s(t.batch.total||0))])])]),t._m(1)])])},M=[function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"t-time__progress-bar"},[s("div",{staticClass:"t-time__progress-bar--fill",staticStyle:{width:"553px"}})])},function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"t-time__progress-bar"},[s("div",{staticClass:"t-time__progress-bar--fill",staticStyle:{width:"743px"}})])}],N=(s("a9e3"),s("99af"),{name:"t-train-time",props:{avg_epoch_time:{type:Number,default:0},elapsed_epoch_time:{type:Number,default:0},elapsed_time:{type:Number,default:0},estimated_time:{type:Number,default:0},still_epoch_time:{type:Number,default:0},still_time:{type:Number,default:0},batch:{type:Object,default:function(){return{current:0,total:0}}},epoch:{type:Object,default:function(){return{current:0,total:0}}}},methods:{formatTime:function(t){return"".concat(Math.floor(t/60/60)," : ").concat(Math.floor(t/60)," : ").concat(t)}}}),V=N,F=(s("e106"),Object(u["a"])(V,D,M,!1,null,"6e5a11ea",null)),K=F.exports,U={name:"t-progress",components:{Sysinfo:A,Timers:K},computed:{lossGraphs:function(){return this.$store.getters["trainings/getTrainUsage"]||{}},usage:function(){var t;return console.log(this.lossGraphs),(null===(t=this.lossGraphs)||void 0===t?void 0:t.hard_usage)||{}},timings:function(){var t;return(null===(t=this.lossGraphs)||void 0===t?void 0:t.timings)||{}}},mounted:function(){console.log(this.data)}},L=U,I=(s("c73a"),Object(u["a"])(L,S,$,!1,null,"ee005578",null)),B=I.exports,W=s("1636"),H=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"t-scatters"},[s("div",{staticClass:"t-scatters__header"},[s("div",{staticClass:"checks"},[t._l(t.statisticData,(function(e,a,i){return[e.labels?s("t-checkbox",{key:"check_"+i,attrs:{inline:!0,label:"Выход "+a,value:!0,name:a},on:{change:function(e){return t.change(e,a)}}}):t._e()]})),s("t-checkbox",{attrs:{inline:!0,label:"Автоотбновление"}})],2),s("button",{on:{click:function(e){t.showContent=!t.showContent}}},[t._v("Показать")])]),t.showContent?s("div",{staticClass:"t-scatters__content"},[t._l(t.statisticData,(function(e,a,i){return[e.data_array&&t.isShowKeys.includes(a)?s("Matrix",t._b({key:i},"Matrix",e,!1)):t._e()]}))],2):t._e()])},J=[],X=s("2909"),q=(s("4de4"),function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"t-matrix"},[s("div",{staticClass:"t-matrix__gradient"},[s("div",{staticClass:"colors"}),s("div",{staticClass:"values"},t._l(t.stepValues,(function(e,a){return s("div",{key:a},[t._v(t._s(e))])})),0)]),s("div",{staticClass:"t-matrix__table"},[s("div",{staticClass:"t-matrix__label--top"},[t._v(t._s(t.graph_name))]),s("div",{staticClass:"t-matrix__grid--wrapper"},[s("div",{staticClass:"legend--left"},t._l(t.labels,(function(e){return s("div",{key:e},[t._v(t._s(e))])})),0),s("div",{staticClass:"t-matrix__grid",style:{gridTemplate:"repeat("+t.labels.length+", 40px) / repeat("+t.labels.length+", 40px)"}},t._l(t.values,(function(e,a){return s("div",{key:"col_"+a,staticClass:"t-matrix__grid--item",style:{background:t.getColor(t.percent[a])},attrs:{title:e+" / "+t.percent[a]+"%"}},[t._v(" "+t._s(e+", "+t.percent[a]+"%")+" ")])})),0),s("div",{staticClass:"legend--bottom"},t._l(t.labels,(function(e){return s("div",{key:e},[t._v(t._s(e))])})),0),s("div",{staticClass:"t-matrix__label--left"},[t._v(t._s(t.y_label))])]),s("div",{staticClass:"t-matrix__label--bottom"},[t._v(t._s(t.x_label))])])])}),Y=[],Q=(s("d81d"),{name:"t-matrix",props:{id:Number,task_type:String,graph_name:String,x_label:String,y_label:String,labels:Array,data_array:Array,data_percent_array:Array},data:function(){return{}},computed:{values:function(){var t;return(t=[]).concat.apply(t,Object(X["a"])(this.data_array))},percent:function(){var t;return(t=[]).concat.apply(t,Object(X["a"])(this.data_percent_array))},stepValues:function(){var t=this;return[4,3,2,1,0].map((function(e){return t.max/4*e})).reverse()},max:function(){return 100*Math.round(this.maxValue/100)},maxValue:function(){return Math.max.apply(Math,Object(X["a"])(this.values))}},methods:{getColor:function(t){return"rgb(".concat(0+t,", ").concat(50+t,", ").concat(150+t,")")}}}),Z=Q,tt=(s("6c76"),Object(u["a"])(Z,q,Y,!1,null,"e73210fe",null)),et=tt.exports,st={name:"t-scatters",components:{Matrix:et},computed:{statisticData:function(){return this.$store.getters["trainings/getTrainData"]("statistic_data")||[]}},data:function(){return{showContent:!0,isShowKeys:[]}},methods:{change:function(t,e){console.log(t),this.isShowKeys=this.isShowKeys.includes(e)?this.isShowKeys.filter((function(t){return t!==e})):[].concat(Object(X["a"])(this.isShowKeys),[e])}},created:function(){this.isShowKeys=Object.keys(this.statisticData)}},at=st,it=(s("c038"),Object(u["a"])(at,H,J,!1,null,"b61e2386",null)),nt=it.exports,lt=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"t-balance"},[s("div",{staticClass:"t-balance__header"},[s("p",[t._v("Параметры")]),s("div",{staticClass:"t-balance__wrapper"},[s("div",{staticClass:"t-balance__checks"},[s("t-checkbox",{attrs:{inline:!0,label:"Показать тренировочную выборку"}}),s("t-checkbox",{attrs:{inline:!0,label:"Показать проверочную выборку"}})],1),s("Select",{attrs:{small:!0,inline:!0,label:"Сортировать",lists:[],width:"180px"}}),s("button",[t._v("Показать")])],1)]),s("div",{staticClass:"t-balance__graphs"},[t._l(t.dataDalance,(function(e,a){return[t._l(e,(function(e,i){return[s("Graph",t._b({key:"graph_"+a+"/"+i},"Graph",e,!1))]}))]}))],2)])},rt=[],ct=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{class:["t-field",""+t.selectPos]},[s("label",{staticClass:"t-field__label"},[t._v(t._s(t.label))]),s("input",{staticStyle:{display:"none"},attrs:{name:t.parse},domProps:{value:t.select}}),s("at-select",{class:["t-field__select",{"t-field__error":t.error}],attrs:{clearable:"",size:"small",disabled:t.disabled,width:t.width},on:{"on-change":t.change,click:t.cleanError},model:{value:t.select,callback:function(e){t.select=e},expression:"select"}},t._l(t.items,(function(e,a){var i=e.label,n=e.value;return s("at-option",{key:"item_"+a,attrs:{value:n,title:i}},[t._v(" "+t._s(i)+" ")])})),1)],1)},ot=[],_t=(s("b0c0"),{name:"t-select",props:{label:{type:String,default:"Label"},type:{type:String,default:""},value:{type:[String,Number]},name:{type:String},parse:{type:String},lists:{type:[Array,Object]},disabled:Boolean,error:String,width:String,selectPos:{type:String,default:"left",required:!1}},data:function(){return{select:""}},computed:{items:function(){return Array.isArray(this.lists)?this.lists.map((function(t){return t||""})):Object.keys(this.lists)}},methods:{cleanError:function(){this.error&&this.$emit("cleanError")},change:function(t){this.$emit("input",t),this.$emit("change",{name:this.name,value:t})}},created:function(){this.select=this.value},destroyed:function(){}}),ut=_t,dt=(s("7d57"),Object(u["a"])(ut,ct,ot,!1,null,"40e1241c",null)),vt=dt.exports,mt=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("scrollbar",{attrs:{ops:t.ops}},[s("div",{staticClass:"t-graph"},[s("p",[t._v(t._s(t.graph_name||""))]),s("div",{staticClass:"t-graph__wrapper"},[s("div",{staticClass:"t-graph__x-label"},[t._v(t._s(t.x_label))]),s("div",{staticClass:"t-graph__y-label"},[t._v(t._s(t.y_label))]),s("div",{staticClass:"t-graph__values"},t._l(t.stepValues,(function(e,a){return s("div",{key:a},[t._v(t._s(e))])})),0),s("div",{staticClass:"t-graph__diagram"},t._l(t.values,(function(e,a){return s("div",{key:a,staticClass:"t-graph__diagram-item"},[s("span",[t._v(t._s(e))]),s("div",{staticClass:"t-graph__diagram-fill",style:{height:(e/t.max*100).toFixed()+"%"}}),s("div",{staticClass:"t-graph__diagram-label",attrs:{title:t.labels[a]}},[t._v(t._s(t.labels[a]))])])})),0)])])])},pt=[],ft={name:"t-graph",props:{id:Number,graph_name:String,x_label:String,y_label:String,plot_data:Array},data:function(){return{ops:{scrollPanel:{scrollingX:!0,scrollingY:!1}}}},computed:{values:function(){var t,e;return console.log(this.plot_data[0]),(null===(t=this.plot_data)||void 0===t||null===(e=t[0])||void 0===e?void 0:e.values)||[]},labels:function(){var t,e;return(null===(t=this.plot_data)||void 0===t||null===(e=t[0])||void 0===e?void 0:e.labels)||[]},stepValues:function(){var t=this;return[4,3,2,1,0].map((function(e){return t.max/4*e}))},max:function(){return 100*Math.round(this.maxValue/100)},maxValue:function(){return Math.max.apply(Math,Object(X["a"])(this.values))}}},bt=ft,ht=(s("d8af"),Object(u["a"])(bt,mt,pt,!1,null,"39bf538a",null)),gt=ht.exports,Ct={name:"t-balance",components:{Select:vt,Graph:gt},computed:{dataDalance:function(){return this.$store.getters["trainings/getTrainData"]("data_balance")||[]}}},yt=Ct,xt=(s("cf10"),Object(u["a"])(yt,lt,rt,!1,null,"717b6050",null)),kt=xt.exports,wt=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"predictions"},[s("h3",[t._v("Параметры")]),s("div",{staticClass:"predictions__params"},[s("div",{staticClass:"predictions__param"},[s("t-field",{attrs:{inline:"",label:"Показать тренировочную выборку"}},[s("t-checkbox-new",{attrs:{value:!0,small:""}})],1)],1),s("div",{staticClass:"predictions__param"},[s("t-field",{attrs:{inline:"",label:"Данные для расчета"}},[s("t-select-new",{staticStyle:{width:"180px"},attrs:{small:""}})],1),s("t-field",{attrs:{inline:"",label:"Тип выбора данных"}},[s("t-select-new",{staticStyle:{width:"180px"},attrs:{small:""}})],1)],1),s("div",{staticClass:"predictions__param"},[s("t-field",{attrs:{inline:"",label:"Показать примеров"}},[s("t-input-new",{staticStyle:{width:"32px","text-align":"center"},attrs:{value:10,type:"number",small:""}})],1),s("t-field",{attrs:{inline:"",label:"Показать статистику"}},[s("t-checkbox-new",{attrs:{value:!0,small:""}})],1)],1),s("div",{staticClass:"predictions__param"},[s("t-field",{attrs:{inline:"",label:"Автообновление"}},[s("t-checkbox-new",{attrs:{small:""}})],1)],1),s("div",{staticClass:"predictions__param"},[s("t-button",{staticStyle:{width:"150px",height:"40px"},nativeOn:{click:function(e){t.showTextTable=!t.showTextTable}}},[t._v("Показать")])],1)]),s("TextTable",{attrs:{show:t.showTextTable,predict:t.predictData}})],1)},Ot=[],jt=function(){var t=this,e=t.$createElement,s=t._self._c||e;return t.show?s("div",{staticClass:"table"},[t._m(0)]):t._e()},Tt=[function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"table__columns"},[s("div",{staticClass:"table__column"},[s("div",{staticClass:"table__item title size-1"},[t._v("Слой")]),s("div",{staticClass:"table__item"},[t._v("1")]),s("div",{staticClass:"table__item"},[t._v("2")]),s("div",{staticClass:"table__item"},[t._v("3")]),s("div",{staticClass:"table__item"},[t._v("4")])]),s("div",{staticClass:"table__column"},[s("div",{staticClass:"table__item"},[s("div",{staticClass:"table__column"},[s("div",{staticClass:"table__item title size-2"},[t._v(" Исходные данные ")]),s("div",{staticClass:"table__row"},[s("div",{staticClass:"table__column"},[s("div",{staticClass:"table__item title size-2"},[t._v("Вход 1")]),s("div",{staticClass:"table__item"},[t._v("FROG")]),s("div",{staticClass:"table__item"},[t._v("TRUCK")]),s("div",{staticClass:"table__item"},[t._v("SHIP")]),s("div",{staticClass:"table__item"},[t._v("FROG")])])])])])]),s("div",{staticClass:"table__column"},[s("div",{staticClass:"table__item"},[s("div",{staticClass:"table__column"},[s("div",{staticClass:"table__item title size-2"},[t._v(" Истинное значение ")]),s("div",{staticClass:"table__row"},[s("div",{staticClass:"table__column"},[s("div",{staticClass:"table__item title size-2"},[t._v("Выход 2")]),s("div",{staticClass:"table__item"},[t._v("FROG")]),s("div",{staticClass:"table__item"},[t._v("TRUCK")]),s("div",{staticClass:"table__item"},[t._v("SHIP")]),s("div",{staticClass:"table__item"},[t._v("FROG")])])])])])]),s("div",{staticClass:"table__column"},[s("div",{staticClass:"table__item"},[s("div",{staticClass:"table__column"},[s("div",{staticClass:"table__column title size-2"},[t._v(" Предсказание ")]),s("div",{staticClass:"table__row"},[s("div",{staticClass:"table__column"},[s("div",{staticClass:"table__item title size-2"},[t._v("Выход 3")]),s("div",{staticClass:"table__item"},[t._v("FROG")]),s("div",{staticClass:"table__item"},[t._v("TRUCK")]),s("div",{staticClass:"table__item"},[t._v("SHIP")]),s("div",{staticClass:"table__item"},[t._v("FROG")])])])])])]),s("div",{staticClass:"table__column"},[s("div",{staticClass:"table__item"},[s("div",{staticClass:"table__column"},[s("div",{staticClass:"table__item title size-2"},[t._v(" Статистика примеров ")]),s("div",{staticClass:"table__row"},[s("div",{staticClass:"table__column"},[s("div",{staticClass:"table__item title size-2"},[t._v("airplane")]),s("div",{staticClass:"table__item"},[t._v("0.9%")]),s("div",{staticClass:"table__item"},[t._v("0.9%")]),s("div",{staticClass:"table__item"},[t._v("0.9%")]),s("div",{staticClass:"table__item"},[t._v("0.9%")])]),s("div",{staticClass:"table__column"},[s("div",{staticClass:"table__item title size-2"},[t._v("frog")]),s("div",{staticClass:"table__item"},[t._v("0.9%")]),s("div",{staticClass:"table__item"},[t._v("0.9%")]),s("div",{staticClass:"table__item"},[t._v("0.9%")]),s("div",{staticClass:"table__item"},[t._v("0.9%")])]),s("div",{staticClass:"table__column"},[s("div",{staticClass:"table__item title size-2"},[t._v("truck")]),s("div",{staticClass:"table__item"},[t._v("0.9%")]),s("div",{staticClass:"table__item"},[t._v("0.9%")]),s("div",{staticClass:"table__item"},[t._v("0.9%")]),s("div",{staticClass:"table__item"},[t._v("0.9%")])]),s("div",{staticClass:"table__column"},[s("div",{staticClass:"table__item title size-2"},[t._v("ship")]),s("div",{staticClass:"table__item"},[t._v("0.9%")]),s("div",{staticClass:"table__item"},[t._v("0.9%")]),s("div",{staticClass:"table__item"},[t._v("0.9%")]),s("div",{staticClass:"table__item"},[t._v("0.9%")])]),s("div",{staticClass:"table__column"},[s("div",{staticClass:"table__item title size-2"},[t._v("horse")]),s("div",{staticClass:"table__item"},[t._v("0.9%")]),s("div",{staticClass:"table__item"},[t._v("0.9%")]),s("div",{staticClass:"table__item"},[t._v("0.9%")]),s("div",{staticClass:"table__item"},[t._v("0.9%")])])])])])])])}],St={name:"TextTableTest",props:{show:Boolean,predict:{type:Object,default:function(){return{}}}},data:function(){return{}}},$t=St,zt=(s("9849"),Object(u["a"])($t,jt,Tt,!1,null,"2ea71951",null)),Et=zt.exports,Pt={name:"Predictions",components:{TextTable:Et},data:function(){return{showTextTable:!1}},computed:Object(r["a"])({},Object(c["b"])({predictData:"trainings/getPredict"}))},Gt=Pt,Rt=(s("2e98"),Object(u["a"])(Gt,wt,Ot,!1,null,"d401d5b8",null)),At=Rt.exports,Dt={name:"Graphics",components:{Prediction:At,Texts:T,Progress:B,LoadSpiner:W["a"],Stats:nt,Balance:kt,LossGraphs:function(){return Promise.all([s.e("chunk-743e06ca"),s.e("chunk-b4690670")]).then(s.bind(null,"e04c"))},MetricGraphs:function(){return Promise.all([s.e("chunk-743e06ca"),s.e("chunk-ce8631dc")]).then(s.bind(null,"f9f5"))}},data:function(){return{collabse:[],loading:!0}},computed:Object(r["a"])(Object(r["a"])({},Object(c["b"])({})),{},{show:function(){return this.collabse.includes("0")}}),methods:{change:function(t){this.$emit("collabse",this.collabse),this.collabse=t,console.log(t)}}},Mt=Dt,Nt=(s("e88c"),Object(u["a"])(Mt,m,p,!1,null,"b615cb2e",null)),Vt=Nt.exports,Ft=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"params"},[s("div",{staticClass:"params__body"},[s("scrollbar",[s("div",{staticClass:"params__items"},[s("at-collapse",{attrs:{value:t.collapse}},[s("at-collapse-item",{staticClass:"mt-3",attrs:{title:""}},[t._l(t.main.fields,(function(e,a){return[s("t-auto-field-trainings",t._b({key:"main_"+a,attrs:{state:t.state,inline:!1},on:{parse:t.parse}},"t-auto-field-trainings",e,!1))]}))],2),s("at-collapse-item",{staticClass:"mt-3",attrs:{title:""}},[s("div",{staticClass:"fit"},[t._l(t.fit.fields,(function(e,a){return[s("t-auto-field-trainings",t._b({key:"fit_"+a,staticClass:"fit__item",attrs:{state:t.state,inline:!0},on:{parse:t.parse}},"t-auto-field-trainings",e,!1))]}))],2)]),s("at-collapse-item",{staticClass:"mt-3",attrs:{title:t.optimizer.name}},[s("div",{staticClass:"optimizer"},[t._l(t.optimizerFields,(function(e,a){return[s("t-auto-field-trainings",t._b({key:"optimizer_"+a,staticClass:"optimizer__item",attrs:{state:t.state,inline:""},on:{parse:t.parse}},"t-auto-field-trainings",e,!1))]}))],2)]),s("at-collapse-item",{staticClass:"mt-3",attrs:{title:t.outputs.name}},[s("div",{staticClass:"blocks-layers"},[t._l(t.outputs.fields,(function(e,a){return[s("div",{key:"block_layers_"+a,staticClass:"block-layers"},[s("div",{staticClass:"block-layers__header"},[t._v(" "+t._s(e.name)+" ")]),s("div",{staticClass:"block-layers__body"},[t._l(e.fields,(function(e,a){return[s("t-auto-field-trainings",t._b({key:"checkpoint_"+a,attrs:{state:t.state,inline:!0},on:{parse:t.parse}},"t-auto-field-trainings",e,!1))]}))],2)])]}))],2)]),s("at-collapse-item",{staticClass:"mt-3",attrs:{title:t.checkpoint.name}},[s("div",{staticClass:"checkpoint"},[s("t-field",{staticClass:"checkpoint__item",attrs:{inline:"",label:"Функция"}},[s("t-select-new",{attrs:{list:t.func,small:"",update:"",name:"metric_name",parse:"architecture[parameters][checkpoint][metric_name]",value:t.getValue},on:{parse:t.parse}})],1),t._l(t.checkpoint.fields,(function(e,a){return[s("t-auto-field-trainings",t._b({key:"outputs_"+a,staticClass:"checkpoint__item",attrs:{state:t.state,inline:!0},on:{parse:t.parse}},"t-auto-field-trainings",e,!1))]}))],2)])],1)],1)])],1),s("div",{staticClass:"params__footer"},[s("div",[s("t-button",{on:{click:t.start}},[t._v("Обучить")]),s("t-button",{on:{click:t.stop}},[t._v("Остановить")])],1),s("div",[s("t-button",{on:{click:t.save}},[t._v("Сохранить")]),s("t-button",{on:{click:t.clear}},[t._v("Сбросить")])],1)])])},Kt=[],Ut=s("ade3"),Lt=s("1da1"),It=(s("96cf"),s("4d63"),s("466d"),/(\[[^\[\]]*\])/g);function Bt(t){var e=[],s=/^([^\[\]]*)/,a=new RegExp(It),i=s.exec(t);i[1]&&e.push(i[1]);while(null!==(i=a.exec(t)))e.push(i[1]);return e}function Wt(t,e,s){if(0===e.length)return t=s,t;var a=e.shift(),i=a.match(/^\[(.+?)\]$/);if("[]"===a)return t=t||[],Array.isArray(t)?t.push(Wt(null,e,s)):(t._values=t._values||[],t._values.push(Wt(null,e,s))),t;if(i){var n=i[1],l=+n;isNaN(l)?(t=t||{},t[n]=Wt(t[n],e,s)):(t=t||[],t[l]=Wt(t[l],e,s))}else t[a]=Wt(t[a],e,s);return t}function Ht(t,e,s){var a=e.match(It);if(a){var i=Bt(e);Wt(t,i,s)}else{var n=t[e];n?(Array.isArray(n)||(t[e]=[n]),t[e].push(s)):t[e]=s}return t}var Jt=Ht,Xt={name:"params-traning",components:{},data:function(){return{obj:{},collapse:[0,1,2,3,4],optimizerValue:"",metricData:""}},computed:Object(r["a"])(Object(r["a"])({},Object(c["b"])({params:"trainings/getParams"})),{},{getValue:function(){var t,e;return null!==(t=null===(e=this.state)||void 0===e?void 0:e["architecture[parameters][checkpoint][metric_name]"])&&void 0!==t?t:"Accuracy"},state:{set:function(t){this.$store.dispatch("trainings/setStateParams",t)},get:function(){return this.$store.getters["trainings/getStateParams"]}},main:function(){var t;return(null===(t=this.params)||void 0===t?void 0:t.main)||{}},fit:function(){var t;return(null===(t=this.params)||void 0===t?void 0:t.fit)||{}},outputs:function(){var t,e;return console.log((null===(t=this.params)||void 0===t?void 0:t.outputs)||{}),(null===(e=this.params)||void 0===e?void 0:e.outputs)||{}},optimizerFields:function(){var t,e,s;return(null===(t=this.params)||void 0===t||null===(e=t.optimizer)||void 0===e||null===(s=e.fields)||void 0===s?void 0:s[this.optimizerValue])||[]},optimizer:function(){var t;return(null===(t=this.params)||void 0===t?void 0:t.optimizer)||{}},checkpoint:function(){var t;return(null===(t=this.params)||void 0===t?void 0:t.checkpoint)||{}},func:function(){var t,e,s,a,i,n=(null===(t=this.obj)||void 0===t||null===(e=t.architecture)||void 0===e||null===(s=e.parameters)||void 0===s?void 0:s.outputs)||[];return n=(null===(a=n)||void 0===a||null===(i=a[this.metricData])||void 0===i?void 0:i.metrics)||[],n=n.map((function(t){return{label:t,value:t}})),n}}),methods:{click:function(t){console.log(t)},start:function(){var t=this;return Object(Lt["a"])(regeneratorRuntime.mark((function e(){var s,a;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return console.log(JSON.stringify(t.obj,null,2)),e.next=3,t.$store.dispatch("trainings/start",t.obj);case 3:s=e.sent,s&&(a=s.data,a.status&&t.progress()),console.log(s);case 6:case"end":return e.stop()}}),e)})))()},stop:function(){var t=this;return Object(Lt["a"])(regeneratorRuntime.mark((function e(){var s;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.$store.dispatch("trainings/stop",{});case 2:s=e.sent,console.log(s);case 4:case"end":return e.stop()}}),e)})))()},clear:function(){var t=this;return Object(Lt["a"])(regeneratorRuntime.mark((function e(){var s;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.$store.dispatch("trainings/clear",{});case 2:s=e.sent,console.log(s);case 4:case"end":return e.stop()}}),e)})))()},save:function(){var t=this;return Object(Lt["a"])(regeneratorRuntime.mark((function e(){var s;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.$store.dispatch("trainings/save",{});case 2:s=e.sent,console.log(s);case 4:case"end":return e.stop()}}),e)})))()},progress:function(){var t=this;return Object(Lt["a"])(regeneratorRuntime.mark((function e(){return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:setTimeout(Object(Lt["a"])(regeneratorRuntime.mark((function e(){var s,a,i,n,l,r,c,o,_,u;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return e.next=2,t.$store.dispatch("trainings/progress",{});case 2:s=e.sent,console.log(s),s?(a=s.data,i=a.finished,n=a.message,l=a.percent,r=a.data,console.log(l),t.$store.dispatch("messages/setProgressMessage",n),t.$store.dispatch("messages/setProgress",l),r&&(c=r.info,o=r.states,_=r.train_data,u=r.train_usage,t.$store.dispatch("trainings/setInfo",c),t.$store.dispatch("trainings/setStates",o),t.$store.dispatch("trainings/setTrainData",_),t.$store.dispatch("trainings/setTrainUsage",u)),i?console.log(s):t.progress()):console.log(s);case 5:case"end":return e.stop()}}),e)}))),1e3);case 1:case"end":return e.stop()}}),e)})))()},parse:function(t){var e=t.parse,s=t.value,a=t.name;this.state=Object(Ut["a"])({},"".concat(e),s),Jt(this.obj,e,s),this.obj=Object(r["a"])({},this.obj),"architecture_parameters_checkpoint_layer"===a&&(this.metricData=s),"optimizer"===a&&(this.optimizerValue=s)}},created:function(){}},qt=Xt,Yt=(s("4a13"),Object(u["a"])(qt,Ft,Kt,!1,null,"1d7b3e0d",null)),Qt=Yt.exports,Zt={name:"Training",components:{Toolbar:v,Graphics:Vt,Params:Qt},data:function(){return{collabse:[]}},methods:{},created:function(){}},te=Zt,ee=(s("0ae7"),Object(u["a"])(te,a,i,!1,null,"dcc032b4",null));e["default"]=ee.exports},c038:function(t,e,s){"use strict";s("7b8a")},c73a:function(t,e,s){"use strict";s("1845")},cf10:function(t,e,s){"use strict";s("7e79")},d015:function(t,e,s){},d8af:function(t,e,s){"use strict";s("1640")},e0a5:function(t,e,s){},e106:function(t,e,s){"use strict";s("6914")},e88c:function(t,e,s){"use strict";s("9605")},eca8:function(t,e,s){},f7f8:function(t,e,s){"use strict";s("6ef8")},ff49:function(t,e,s){}}]);