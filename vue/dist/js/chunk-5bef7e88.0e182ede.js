(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-5bef7e88"],{"25bc":function(t,a,e){"use strict";e("a900")},"3af9":function(t,a,e){"use strict";e.r(a);var i=function(){var t=this,a=t.$createElement,e=t._self._c||a;return e("div",{staticClass:"card"},[e("div",{staticClass:"card__content"},["image_classification"==t.type?e("div",[e("div",{staticClass:"card__original"},[e("ImgCard",{attrs:{imgUrl:t.card.source}})],1),e("div",{staticClass:"card__result"},[e("TextCard",{style:{width:"224px",height:"80px"}},[t._v(t._s(t.classificationResult))])],1)]):t._e(),"video_classification"==t.type?e("div",[e("div",{staticClass:"card__original"},[e("TableVideo",{attrs:{value:t.card.source}})],1),e("div",{staticClass:"card__result"},[e("TextCard",{style:{width:"300px",height:"80px"}},[e("div",{staticClass:"video_classification"},[t._l(t.getData,(function(a){var i=a.name,r=a.value;return[e("div",{key:i,staticClass:"video_classification__item"},[t._v(t._s(i+": "+r+"%"))])]}))],2)])],1)]):t._e(),"text_classification"==t.type?e("div",[e("div",{staticClass:"card__original"},[e("TextCard",{style:{width:"600px",color:"#A7BED3",height:"324px"}},[t._v(t._s(t.card.source))])],1),e("div",{staticClass:"card__result"},[e("TextCard",{style:{width:"600px",height:"80px"}},[t._v(t._s(t.classificationResult))])],1)]):t._e(),"text_segmentation"==t.type?e("div",[e("div",{staticClass:"card__original segmentation__original",style:{height:"324px"}},[e("scrollbar",{attrs:{ops:t.ops}},[e("TableTextSegmented",t._b({},"TableTextSegmented",{value:t.card.format,tags_color:{segmentationLayer:t.segmentationLayer},layer:"segmentationLayer",block_width:"598px"},!1))],1)],1),e("div",{staticClass:"card__result"},[e("SegmentationTags",{style:{width:"600px",height:"50px"},attrs:{tags:t.segmentationLayer}})],1)]):t._e(),"audio_classification"==t.type?e("div",[e("div",{staticClass:"card__original"},[e("AudioCard",{attrs:{value:t.card.source}})],1),e("div",{staticClass:"card__result"},[e("TextCard",{style:{width:"600px",height:"80px"}},[t._v(t._s(t.classificationResult))])],1)]):t._e(),"text_to_audio"==t.type?e("div",[e("div",{staticClass:"card__result"},[e("TextCard",{style:{width:"600px"}},[t._v(t._s(t.card.source))])],1),e("div",{staticClass:"card__original"},[e("AudioCard",{key:t.card.source,attrs:{value:t.card.predict}})],1)]):t._e(),"audio_to_text"==t.type?e("div",[e("div",{staticClass:"card__original"},[e("AudioCard",{key:t.card.predict,attrs:{value:t.card.source}})],1),e("div",{staticClass:"card__result"},[e("TextCard",{style:{width:"600px"}},[t._v(t._s(t.card.predict))])],1)]):t._e(),"image_segmentation"==t.type?e("div",[e("div",{staticClass:"card__original"},[e("ImgCard",{attrs:{imgUrl:t.card.source}})],1),e("div",{staticClass:"card__result"},[e("ImgCard",{attrs:{imgUrl:t.card.segment}})],1)]):t._e(),"video_object_detection"==t.type?e("div",[e("div",{staticClass:"card__original"},[e("TableVideo",{attrs:{value:t.card.source}})],1),e("div",{staticClass:"card__result"},[e("TableVideo",{attrs:{value:t.card.predict}})],1)]):t._e(),"object_detection"==t.type?e("div",[e("div",{staticClass:"card__original"},[e("TableImage",{attrs:{size:"large",value:t.card.source}})],1),e("div",{staticClass:"card__result"},[e("TableImage",{attrs:{size:"large",value:t.card.predict}})],1)]):t._e(),"time_series"==t.type?e("div",{staticClass:"card__graphic"},[e("GraphicCard",t._b({key:"graphic_"+t.index},"GraphicCard",t.card,!1))],1):t._e(),"time_series_trend"==t.type?e("div",{staticClass:"card__graphic"},[e("div",{staticClass:"card__original"},[e("GraphicCardPredict",{attrs:{data:t.card.predict}})],1),e("div",{staticClass:"card__result"},[e("GraphicCardSource",{attrs:{data:t.card.source}})],1)]):t._e()]),e("div",{staticClass:"card__reload"},[e("button",{staticClass:"btn-reload",on:{click:t.reload}},[e("i",{class:["t-icon","icon-deploy-reload"],attrs:{title:"reload"}})])])])},r=[],s=e("1da1"),c=(e("96cf"),e("d3b7"),e("3ca3"),e("ddb0"),e("a9e3"),e("caad"),e("2532"),e("fb6a"),e("4e82"),e("99af"),e("d81d"),{name:"IndexCard",components:{ImgCard:function(){return e.e("chunk-12dc8d07").then(e.bind(null,"07be"))},TableVideo:function(){return e.e("chunk-2d0e4ff4").then(e.bind(null,"9337"))},TextCard:function(){return e.e("chunk-5b768e32").then(e.bind(null,"67d1"))},GraphicCard:function(){return Promise.all([e.e("chunk-743e06ca"),e.e("chunk-a7162ab8")]).then(e.bind(null,"1f33"))},GraphicCardSource:function(){return Promise.all([e.e("chunk-743e06ca"),e.e("chunk-1d4234b4")]).then(e.bind(null,"5366"))},GraphicCardPredict:function(){return Promise.all([e.e("chunk-743e06ca"),e.e("chunk-9280b2c8")]).then(e.bind(null,"1e1b"))},AudioCard:function(){return e.e("chunk-125a555c").then(e.bind(null,"bf2e"))},TableTextSegmented:function(){return e.e("chunk-4346e76e").then(e.bind(null,"1039"))},SegmentationTags:function(){return e.e("chunk-b4ad1b6a").then(e.bind(null,"c29f"))},TableImage:function(){return e.e("chunk-5f596762").then(e.bind(null,"c5f0"))}},data:function(){return{ops:{scrollPanel:{scrollingX:!1,scrollingY:!0}}}},props:{card:{type:Object,default:function(){return{}}},colorMap:{type:Array,default:function(){return[]}},index:{type:[String,Number],required:!0},defaultLayout:{type:Object,default:function(){return{}}},type:{type:String,default:""}},methods:{reload:function(){var t=this;return Object(s["a"])(regeneratorRuntime.mark((function a(){return regeneratorRuntime.wrap((function(a){while(1)switch(a.prev=a.next){case 0:t.$emit("reload",String(t.index));case 1:case"end":return a.stop()}}),a)})))()}},computed:{layout:function(){var t=this.defaultLayout;return this.char&&(t.title.text=this.char.title||"",t.xaxis.title=this.char.xaxis.title||"",t.yaxis.title=this.char.yaxis.title||""),t},segmentationLayer:function(){var t={};for(var a in this.colorMap)if(!this.colorMap[a][0].includes("p")){var e=this.colorMap[a][0].slice(1,this.colorMap[a][0].length-1);t[e]=this.colorMap[a][2]}return t},classificationResult:function(){var t=this.card.data,a="";t.sort((function(t,a){return t[1]<a[1]?1:-1}));for(var e=0;e<t.length;e++)a+="".concat(t[e][0]," - ").concat(t[e][1],"%");return a},getData:function(){var t,a=(null===(t=this.card)||void 0===t?void 0:t.data)||[],e=a.map((function(t){return{name:t[0],value:t[1]}}));return e}}}),n=c,d=(e("25bc"),e("2877")),l=Object(d["a"])(n,i,r,!1,null,"6a64b1aa",null);a["default"]=l.exports},a900:function(t,a,e){}}]);