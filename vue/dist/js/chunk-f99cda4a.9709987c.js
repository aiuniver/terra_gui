(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-f99cda4a"],{"0502":function(t,e,s){"use strict";s("3089")},3089:function(t,e,s){},f9c5:function(t,e,s){"use strict";s.r(e);var n=function(){var t=this,e=t.$createElement,s=t._self._c||e;return s("div",{staticClass:"t-predict-text"},[s("p",{staticClass:"t-predict-text__text"},[t._v(t._s(t.text))]),t.length?s("span",{staticClass:"t-predict-text__more",on:{click:t.show}},[t._v(t._s(t.textBtn[Number(t.isShow)]))]):t._e()])},i=[],u={name:"table-text",props:{value:{type:String,default:""},color_mark:{type:Array,default:function(){return[]}}},data:function(){return{text:"",isShow:!1,textBtn:["Показать больше","Скрыть"]}},mounted:function(){this.text=this.length?this.value.substring(0,99)+"...":this.value},computed:{length:function(){return this.value.length>=100}},methods:{show:function(){this.isShow=!this.isShow,this.text=this.isShow?this.value:this.value.substring(0,99)+"..."}}},a=u,c=(s("0502"),s("2877")),o=Object(c["a"])(a,n,i,!1,null,"31d883b0",null);e["default"]=o.exports}}]);