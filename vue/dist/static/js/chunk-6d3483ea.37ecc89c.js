(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-6d3483ea"],{"10a9":function(t,a,e){},3731:function(t,a,e){"use strict";e("10a9")},f2ca:function(t,a,e){"use strict";e.r(a);var s=function(){var t=this,a=t.$createElement,e=t._self._c||a;return e("div",{staticClass:"t-heatmap",class:["t-heatmap",{"t-heatmap--small":t.isSmallMap}]},[e("div",{ref:"scale",staticClass:"t-heatmap__scale"},[e("div",{staticClass:"t-heatmap__scale--gradient"}),e("div",{staticClass:"t-heatmap__scale--values"},t._l(t.stepValues,(function(a,s){return e("span",{key:s,staticClass:"value"},[t._v(t._s(a.toFixed()))])})),0),e("div",{staticClass:"t-heatmap__y-label"},[t._v(t._s(t.y_label))])]),e("div",{ref:"label",staticClass:"t-heatmap__grid--y-labels"},t._l(t.labels,(function(a,s){return e("span",{key:s},[t._v(t._s(a))])})),0),e("div",{staticClass:"t-heatmap__body",style:{maxWidth:t.bodyWidth}},[e("p",{staticClass:"t-heatmap__title"},[t._v(t._s(t.graph_name))]),e("div",{staticClass:"t-heatmap__x-label"},[t._v(t._s(t.x_label))]),e("scrollbar",{attrs:{ops:t.ops}},[e("div",{staticClass:"t-heatmap__wrapper"},[e("div",{staticClass:"t-heatmap__grid",style:t.gridTemplate},[e("div",{staticClass:"t-heatmap__grid--x-labels"},t._l(t.labels,(function(a,s){return e("span",{key:s,attrs:{title:a}},[t._v(t._s(a))])})),0),t._l(t.values,(function(a,s){return e("div",{key:"col_"+s,staticClass:"t-heatmap__grid--item",style:{background:t.getColor(t.percent[s])},attrs:{title:a+" / "+t.percent[s]+"%"}},[t._v(" "+t._s(""+a)+" "),e("br"),t._v(" "+t._s(t.percent[s]+"%")+" ")])}))],2)])])],1)])},r=[],l=e("2909"),n=(e("a9e3"),e("99af"),e("d3b7"),e("ddb0"),e("d81d"),{name:"t-heatmap",props:{id:Number,task_type:String,graph_name:String,x_label:String,y_label:String,labels:Array,data_array:Array,data_percent_array:Array},data:function(){return{ops:{scrollPanel:{scrollingX:!0,scrollingY:!1}},width:null}},computed:{values:function(){var t;return(t=[]).concat.apply(t,Object(l["a"])(this.data_array))},percent:function(){var t;return(t=[]).concat.apply(t,Object(l["a"])(this.data_percent_array))},averageVal:function(){return this.values.reduce((function(t,a){return t+a}))/this.values.length},stepValues:function(){var t=this;return[4,3,2,1,0].map((function(a){return t.max/4*a}))},max:function(){return 10*Math.ceil(this.maxValue/10)},maxValue:function(){return Math.max.apply(Math,Object(l["a"])(this.values))},bodyWidth:function(){return"calc(100% - ".concat(this.width-10,"px)")},isSmallMap:function(){return this.data_array.length<5},gridTemplate:function(){var t=this.isSmallMap?"80px":"40px";return{gridTemplate:"repeat(".concat(this.data_array.length,", ").concat(t,") / repeat(").concat(this.data_array.length,", ").concat(t,")")}}},methods:{getColor:function(t){var a=66-t/100*41;return"hsl(212, 100%, ".concat(a,"%)")}},mounted:function(){this.width=this.$refs.label.offsetWidth+this.$refs.scale.offsetWidth}}),i=n,c=(e("3731"),e("2877")),u=Object(c["a"])(i,s,r,!1,null,"325e5f38",null);a["default"]=u.exports}}]);