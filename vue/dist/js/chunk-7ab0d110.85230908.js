(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-7ab0d110"],{"8e7c":function(t,e,n){"use strict";n("b6e6")},b6e6:function(t,e,n){},f9c5:function(t,e,n){"use strict";n.r(e);var i=function(){var t=this,e=t.$createElement,n=t._self._c||e;return n("div",{staticClass:"t-predict-text"},[n("p",{staticClass:"t-predict-text__text"},[t._v(t._s(t.text))]),t.length?n("button",{nativeOn:{click:function(e){return t.show.apply(null,arguments)}}},[t._v("Показать больше")]):t._e()])},s=[],u={name:"table-text",props:{value:{type:String,default:""},color_mark:{type:Array,default:function(){return[]}}},data:function(){return{text:"",isShow:!1}},mounted:function(){this.text=this.length?this.value.substring(0,99)+"...":this.value},computed:{length:function(){return this.value.length>=100}},methods:{show:function(){this.length&&this.isShow?this.text=this.value:this.text=this.value.substring(0,99)+"...",this.isShow=!this.isShow}}},c=u,a=(n("8e7c"),n("2877")),h=Object(a["a"])(c,i,s,!1,null,"3bbfc49e",null);e["default"]=h.exports}}]);