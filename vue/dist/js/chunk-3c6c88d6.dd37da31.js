(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-3c6c88d6"],{"4ade":function(t,i,e){"use strict";e("add8")},add8:function(t,i,e){},c5f0:function(t,i,e){"use strict";e.r(i);var c=function(){var t=this,i=t.$createElement,e=t._self._c||i;return e("div",{directives:[{name:"click-outside",rawName:"v-click-outside",value:t.outside,expression:"outside"}],class:["t-predict-image",{"t-predict-image--large":t.isLarge}]},[t.isLarge&&t.show?e("div",{staticClass:"t-predict-image__mask",on:{click:function(i){return t.click(!1)}}}):t._e(),t.isLarge&&t.show?e("div",{staticClass:"t-predict-image__fixed"},[e("i",{staticClass:"ci-icon ci-close_big",on:{click:function(i){return t.click(!1)}}}),e("img",{key:t.src,attrs:{width:"auto",height:600,src:t.src,alt:"value"}})]):t._e(),e("img",{key:t.src,attrs:{width:"auto",height:t.isLarge?300:120,src:t.src,alt:"value"},on:{click:function(i){return t.click(!0)}}})])},s=[],a=(e("99af"),{name:"t-table-image",props:{value:{type:String,default:""},update:String,size:String},data:function(){return{show:!1}},computed:{src:function(){return"/_media/blank/?path=".concat(this.value,"&r=").concat(this.update)},isLarge:function(){return"large"===this.size}},methods:{click:function(t){this.isLarge&&(this.show=t),t&&this.$store.dispatch("trainings/setLargeImg",this.src)},outside:function(){this.show=!1}}}),n=a,r=(e("4ade"),e("2877")),u=Object(r["a"])(n,c,s,!1,null,"7d6968a9",null);i["default"]=u.exports}}]);