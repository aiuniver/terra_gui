(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-3c37d762"],{"510c":function(t,l,a){"use strict";a("c691")},5448:function(t,l,a){"use strict";a.r(l);var e=function(){var t=this,l=t.$createElement,a=t._self._c||l;return a("div",{staticClass:"content"},[a("table",{staticClass:"table"},[a("tr",{staticClass:"table__title-row"},[a("td",[a("button",{staticClass:"table__reload-all",on:{click:t.ReloadAll}},[a("i",{class:["t-icon","icon-deploy-reload"],attrs:{title:"reload"}}),a("span",[t._v("Перезагрузить все")])])]),a("td",[t._v("Предсказанные данные")]),t._l(t.columns,(function(l,e){return a("td",{key:"col_"+e},[t._v(t._s(l))])}))],2),t._l(t.data,(function(l,e){var n=l.preset,o=l.label;return a("tr",{key:"row_"+e},[a("td",{staticClass:"table__td-reload"},[a("button",{staticClass:"td-reload__btn-reload",on:{click:function(l){return t.ReloadRow(e)}}},[a("i",{class:["t-icon","icon-deploy-reload"],attrs:{title:"reload"}})])]),a("td",{staticClass:"table__result-data"},[t._v(t._s(o))]),t._l(n,(function(l,n){return a("td",{key:"data_"+e+n},[a("TableText",t._b({style:{width:"100%"}},"TableText",{value:l},!1))],1)})),a("td",{staticClass:"table__result-data"},[t._v(t._s(o))])],2)}))],2)])},n=[],o=(a("d3b7"),a("3ca3"),a("ddb0"),a("25f0"),{name:"Table",components:{TableText:function(){return a.e("chunk-0ff35222").then(a.bind(null,"f9c5"))}},props:{data:Array,columns:Array},methods:{ReloadRow:function(t){this.$emit("reload",[t.toString()])},ReloadAll:function(){this.$emit("reloadAll")}},mounted:function(){console.log(12312312),console.log(this.data),console.log(this.columns)}}),s=o,c=(a("510c"),a("2877")),i=Object(c["a"])(s,e,n,!1,null,"7a55e615",null);l["default"]=i.exports},c691:function(t,l,a){}}]);