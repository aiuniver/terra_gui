(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-cff85eac"],{"6c7e":function(t,e,n){"use strict";n("85e4")},"85e4":function(t,e,n){},f9c5:function(t,e,n){"use strict";n.r(e);var i=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"t-predict-text"},[n("p",{staticClass:"t-predict-text__text"},[t._v(t._s(t.text))]),t.length?n("span",{staticClass:"t-predict-text__more",on:{click:function(e){return t.show(t.idx)}}},[t._v(t._s(t.textBtn[Number(t.isExpanded)]))]):t._e()])},s=[],a=(n("caad"),n("2532"),{name:"table-text",props:{value:{type:String,default:""},color_mark:{type:Array,default:function(){return[]}},idx:{default:null}},data:function(){return{isShow:!1,textBtn:["Показать больше","Скрыть"]}},mounted:function(){},computed:{length:function(){return this.value.length>=50},text:function(){return this.isExpanded?this.value:this.value.substring(0,49)+"..."},isExpanded:function(){return this.$store.getters["trainings/getExpandedIdx"].includes(this.idx)}},methods:{show:function(t){this.isShow=!this.isShow,this.$store.dispatch("trainings/setExpandedIdx",{value:this.isShow,idx:t})}},watch:{isExpanded:function(t){this.isShow=t}}}),u=a,c=(n("6c7e"),n("2877")),r=Object(c["a"])(u,i,s,!1,null,"46761dbd",null);e["default"]=r.exports}}]);