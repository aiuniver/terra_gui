(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-4ea898e2"],{"3af9":function(t,e,a){"use strict";a.r(e);var i=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"card"},[a("div",{staticClass:"card__content"},["ImageClassification"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("ImgCard",{attrs:{imgUrl:t.card.source}})],1),a("div",{staticClass:"card__result"},[a("TextCard",{style:{width:"224px",height:"80px"}},[t._v(t._s(t.classificationResult))])],1)]):t._e(),"TextClassification"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("TextCard",{style:{width:"600px",color:"#A7BED3",height:"324px"}},[t._v(t._s(t.card.source))])],1),a("div",{staticClass:"card__result"},[a("TextCard",{style:{width:"600px",height:"80px"}},[t._v(t._s(t.classificationResult))])],1)]):t._e(),"TextSegmentation"==t.type?a("div",[a("div",{staticClass:"card__original segmentation__original",style:{height:"324px"}},[a("scrollbar",{attrs:{ops:t.ops}},[a("TableTextSegmented",t._b({},"TableTextSegmented",{value:t.card.format,tags_color:{segmentationLayer:t.segmentationLayer},layer:"segmentationLayer",block_width:"598px"},!1))],1)],1),a("div",{staticClass:"card__result"},[a("SegmentationTags",{style:{width:"600px",height:"50px"},attrs:{tags:t.segmentationLayer}})],1)]):t._e(),"AudioClassification"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("AudioCard",{attrs:{value:t.card.source}})],1),a("div",{staticClass:"card__result"},[a("TextCard",{style:{width:"600px",height:"80px"}},[t._v(t._s(t.classificationResult))])],1)]):t._e(),"ImageSegmentation"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("ImgCard",{attrs:{imgUrl:t.card.source}})],1),a("div",{staticClass:"card__result"},[a("ImgCard",{attrs:{imgUrl:t.card.segment}})],1)]):t._e(),"VideoObjectDetection"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("TableVideo",{attrs:{value:t.card.source}})],1),a("div",{staticClass:"card__result"},[a("TableVideo",{attrs:{value:t.card.predict}})],1)]):t._e(),"YoloV3"==t.type||"YoloV4"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("TableImage",{attrs:{size:"large",value:t.card.source}})],1),a("div",{staticClass:"card__result"},[a("TableImage",{attrs:{size:"large",value:t.card.predict}})],1)]):t._e(),"Timeseries"==t.type?a("div",{staticClass:"card__graphic"},[a("GraphicCard",t._b({key:"graphic_"+t.index},"GraphicCard",t.card,!1))],1):t._e(),"TimeseriesTrend"==t.type?a("div",{staticClass:"card__graphic"},[a("div",{staticClass:"card__original"},[a("GraphicCardPredict",{attrs:{data:t.card.predict}})],1),a("div",{staticClass:"card__result"},[a("GraphicCardSource",{attrs:{data:t.card.source}})],1)]):t._e()]),a("div",{staticClass:"card__reload"},[a("button",{staticClass:"btn-reload",on:{click:t.reload}},[a("i",{class:["t-icon","icon-deploy-reload"],attrs:{title:"reload"}})])])])},r=[],n=a("1da1"),c=(a("96cf"),a("d3b7"),a("3ca3"),a("ddb0"),a("a9e3"),a("caad"),a("2532"),a("fb6a"),a("4e82"),a("99af"),{name:"IndexCard",components:{ImgCard:function(){return a.e("chunk-12dc8d07").then(a.bind(null,"07be"))},TableVideo:function(){return a.e("chunk-2d0e4ff4").then(a.bind(null,"9337"))},TextCard:function(){return a.e("chunk-6773b306").then(a.bind(null,"67d1"))},GraphicCard:function(){return Promise.all([a.e("chunk-743e06ca"),a.e("chunk-a7162ab8")]).then(a.bind(null,"1f33"))},GraphicCardSource:function(){return Promise.all([a.e("chunk-743e06ca"),a.e("chunk-1d4234b4")]).then(a.bind(null,"5366"))},GraphicCardPredict:function(){return Promise.all([a.e("chunk-743e06ca"),a.e("chunk-9280b2c8")]).then(a.bind(null,"1e1b"))},AudioCard:function(){return a.e("chunk-125a555c").then(a.bind(null,"bf2e"))},TableTextSegmented:function(){return a.e("chunk-4346e76e").then(a.bind(null,"1039"))},SegmentationTags:function(){return a.e("chunk-b4ad1b6a").then(a.bind(null,"c29f"))},TableImage:function(){return a.e("chunk-5f596762").then(a.bind(null,"c5f0"))}},data:function(){return{ops:{scrollPanel:{scrollingX:!1,scrollingY:!0}}}},props:{card:{type:Object,default:function(){return{}}},colorMap:{type:Array,default:function(){return[]}},index:{type:[String,Number],required:!0},defaultLayout:{type:Object,default:function(){return{}}},type:{type:String,default:""}},methods:{reload:function(){var t=this;return Object(n["a"])(regeneratorRuntime.mark((function e(){return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:t.$emit("reload",String(t.index));case 1:case"end":return e.stop()}}),e)})))()}},computed:{layout:function(){var t=this.defaultLayout;return this.char&&(t.title.text=this.char.title||"",t.xaxis.title=this.char.xaxis.title||"",t.yaxis.title=this.char.yaxis.title||""),t},segmentationLayer:function(){var t={};for(var e in this.colorMap)if(!this.colorMap[e][0].includes("p")){var a=this.colorMap[e][0].slice(1,this.colorMap[e][0].length-1);t[a]=this.colorMap[e][2]}return t},classificationResult:function(){var t=this.card.data,e="";t.sort((function(t,e){return t[1]<e[1]?1:-1}));for(var a=0;a<t.length;a++)e+="".concat(t[a][0]," - ").concat(t[a][1],"% \n");return e}}}),s=c,l=(a("70f6"),a("2877")),d=Object(l["a"])(s,i,r,!1,null,"1f091c1b",null);e["default"]=d.exports},"65ae":function(t,e,a){},"70f6":function(t,e,a){"use strict";a("65ae")}}]);