(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-5fa381c2"],{2606:function(t,e,i){"use strict";i.r(e);var s=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{directives:[{name:"click-outside",rawName:"v-click-outside",value:t.outside,expression:"outside"}],class:["t-predict-image",{"t-predict-image--large":t.isLarge}]},[i("img",{key:t.src,attrs:{width:"auto",height:t.isLarge?300:120,src:t.src,alt:"value"},on:{click:function(e){return t.click(!0)}}})])},c=[],a=(i("99af"),{name:"t-table-image",props:{value:{type:String,default:""},update:String,size:String},data:function(){return{show:!1}},computed:{src:function(){return"/_media/blank/?path=".concat(this.value,"&r=").concat(this.update)},isLarge:function(){return"large"===this.size}},methods:{click:function(t){this.isLarge&&(this.show=t),t&&this.$store.dispatch("trainings/setLargeImg",this.src)},outside:function(){this.show=!1}}}),n=a,r=(i("73b3"),i("2877")),u=Object(r["a"])(n,s,c,!1,null,"5f8ccdae",null);e["default"]=u.exports},"3c53":function(t,e,i){},"73b3":function(t,e,i){"use strict";i("3c53")}}]);