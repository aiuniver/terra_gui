(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-268a064e"],{2563:function(t,e,i){"use strict";i("423a")},"423a":function(t,e,i){},c5f0:function(t,e,i){"use strict";i.r(e);var s=function(){var t=this,e=t.$createElement,i=t._self._c||e;return i("div",{directives:[{name:"click-outside",rawName:"v-click-outside",value:t.outside,expression:"outside"}],class:["t-predict-image",{"t-predict-image--large":t.isLarge}]},[i("img",{key:t.src,attrs:{width:"auto",height:t.isLarge?300:120,src:t.src,alt:"value"},on:{click:function(e){return t.click(!0)}}})])},a=[],n=(i("99af"),{name:"t-table-image",props:{value:{type:String,default:""},update:String,size:String},data:function(){return{show:!1}},computed:{src:function(){return"/_media/blank/?path=".concat(this.value,"&r=").concat(this.update)},isLarge:function(){return"large"===this.size}},methods:{click:function(t){this.isLarge&&(this.show=t),t&&this.$store.dispatch("trainings/setLargeImg",this.src)},outside:function(){this.show=!1}}}),c=n,r=(i("2563"),i("2877")),u=Object(r["a"])(c,s,a,!1,null,"eb218698",null);e["default"]=u.exports}}]);