(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-e40a7756"],{3617:function(t,e,a){"use strict";a("689b")},"3af9":function(t,e,a){"use strict";a.r(e);var i=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"card"},[a("div",{staticClass:"card__content"},[["image_gan","image_cgan"].includes(t.type)?a("div",[a("div",{staticClass:"card__original"},[a("ImgCard",{attrs:{imgUrl:t.card.source}})],1),a("div",[a("TextCard",{style:{width:"224px",height:"20px"}},[t._v(" "+t._s(t.card.actual)+" ")])],1)]):t._e(),"image_classification"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("ImgCard",{attrs:{imgUrl:t.card.source}})],1),a("div",{staticClass:"card__result"},[a("TextCard",{style:{width:"224px",height:"80px"}},[t._l(t.getData,(function(e){var i=e.name,r=e.value;return[a("div",{key:i,staticClass:"video_classification__item"},[t._v(t._s(i+": "+r+"%"))])]}))],2)],1)]):t._e(),"video_classification"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("TableVideo",{attrs:{value:t.card.source}})],1),a("div",{staticClass:"card__result"},[a("TextCard",{style:{width:"300px",height:"80px"}},[a("div",{staticClass:"video_classification"},[t._l(t.getData,(function(e){var i=e.name,r=e.value;return[a("div",{key:i,staticClass:"video_classification__item"},[t._v(t._s(i+": "+r+"%"))])]}))],2)])],1)]):t._e(),"text_classification"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("TextCard",{style:{width:"600px",color:"#A7BED3",height:"324px"}},[t._v(t._s(t.card.source))])],1),a("div",{staticClass:"card__result"},[a("TextCard",{style:{width:"600px",height:"80px"}},t._l(t.getData,(function(e){var i=e.name,r=e.value;return a("div",{key:i},[t._v(t._s(i+": "+r+"%"))])})),0)],1)]):t._e(),"text_segmentation"==t.type?a("div",[a("div",{staticClass:"card__original segmentation__original",style:{height:"324px"}},[a("scrollbar",{attrs:{ops:t.ops}},[a("TableTextSegmented",t._b({},"TableTextSegmented",{value:t.card.format,tags_color:{segmentationLayer:t.segmentationLayer},layer:"segmentationLayer",block_width:"598px"},!1))],1)],1),a("div",{staticClass:"card__result"},[a("SegmentationTags",{style:{width:"600px",height:"50px"},attrs:{tags:t.segmentationLayer}})],1)]):t._e(),"audio_classification"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("AudioCard",{key:t.card.source,attrs:{value:t.card.source}})],1),a("div",{staticClass:"card__result"},[a("TextCard",{style:{width:"600px",height:"80px"}},t._l(t.getData,(function(e){var i=e.name,r=e.value;return a("div",{key:i},[t._v(t._s(i+": "+r+"%"))])})),0)],1)]):t._e(),"text_to_audio"==t.type?a("div",[a("div",{staticClass:"card__result"},[a("TextCard",{style:{width:"600px",height:"200px"}},[t._v(" "+t._s(t.card.source)+" ")])],1),a("div",{staticClass:"card__original"},[a("AudioCard",{key:t.card.source,attrs:{value:t.card.predict}})],1)]):t._e(),"audio_to_text"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("AudioCard",{key:t.card.predict,attrs:{value:t.card.source}})],1),a("div",{staticClass:"card__result"},[a("TextCard",{style:{width:"600px",height:"200px"}},[t._v(" "+t._s(t.card.predict)+" ")])],1)]):t._e(),"image_segmentation"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("ImgCard",{attrs:{imgUrl:t.card.source}})],1),a("div",{staticClass:"card__result"},[a("ImgCard",{attrs:{imgUrl:t.card.segment}})],1)]):t._e(),"video_object_detection"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("TableVideo",{attrs:{value:t.card.source}})],1),a("div",{staticClass:"card__result"},[a("TableVideo",{attrs:{value:t.card.predict}})],1)]):t._e(),"object_detection"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("TableImage",{attrs:{size:"large",value:t.card.source}})],1),a("div",{staticClass:"card__result"},[a("TableImage",{attrs:{size:"large",value:t.card.predict}})],1)]):t._e(),"time_series"==t.type?a("div",{staticClass:"card__graphic"},[a("GraphicCard",t._b({key:"graphic_"+t.index},"GraphicCard",t.card,!1))],1):t._e(),"time_series_trend"==t.type?a("div",{staticClass:"card__graphic"},[a("div",{staticClass:"card__original"},[a("GraphicCardPredict",{attrs:{data:t.card.predict}})],1),a("div",{staticClass:"card__result"},[a("GraphicCardSource",{attrs:{data:t.card.source}})],1)]):t._e()]),a("div",{staticClass:"card__reload"},[a("button",{staticClass:"btn-reload",on:{click:t.reload}},[a("i",{class:["t-icon","icon-deploy-reload"],attrs:{title:"reload"}})])])])},r=[],s=a("1da1"),c=(a("96cf"),a("d3b7"),a("3ca3"),a("ddb0"),a("a9e3"),a("caad"),a("2532"),a("fb6a"),a("d81d"),{name:"IndexCard",components:{ImgCard:function(){return Promise.all([a.e("styles"),a.e("chunk-82e98350")]).then(a.bind(null,"07be"))},TableVideo:function(){return a.e("chunk-2d0e4ff4").then(a.bind(null,"9337"))},TextCard:function(){return Promise.all([a.e("styles"),a.e("chunk-dd62d286")]).then(a.bind(null,"67d1"))},GraphicCard:function(){return Promise.all([a.e("styles"),a.e("chunk-64a5a4dc")]).then(a.bind(null,"1f33"))},GraphicCardSource:function(){return Promise.all([a.e("styles"),a.e("chunk-6437b668")]).then(a.bind(null,"5366"))},GraphicCardPredict:function(){return Promise.all([a.e("styles"),a.e("chunk-0becb838")]).then(a.bind(null,"1e1b"))},AudioCard:function(){return Promise.all([a.e("styles"),a.e("chunk-2c05d2d0")]).then(a.bind(null,"bf2e"))},TableTextSegmented:function(){return Promise.all([a.e("styles"),a.e("chunk-e9656254")]).then(a.bind(null,"1039"))},SegmentationTags:function(){return Promise.all([a.e("styles"),a.e("chunk-4faaa094")]).then(a.bind(null,"c29f"))},TableImage:function(){return Promise.all([a.e("styles"),a.e("chunk-d9c913e0")]).then(a.bind(null,"c5f0"))}},data:function(){return{ops:{scrollPanel:{scrollingX:!1,scrollingY:!0}}}},props:{card:{type:Object,default:function(){return{}}},colorMap:{type:Array,default:function(){return[]}},index:{type:[String,Number],required:!0},defaultLayout:{type:Object,default:function(){return{}}},type:{type:String,default:""}},methods:{reload:function(){var t=this;return Object(s["a"])(regeneratorRuntime.mark((function e(){return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:t.$emit("reload",String(t.index));case 1:case"end":return e.stop()}}),e)})))()}},computed:{layout:function(){var t=this.defaultLayout;return this.char&&(t.title.text=this.char.title||"",t.xaxis.title=this.char.xaxis.title||"",t.yaxis.title=this.char.yaxis.title||""),t},segmentationLayer:function(){var t={};for(var e in this.colorMap)if(!this.colorMap[e][0].includes("p")){var a=this.colorMap[e][0].slice(1,this.colorMap[e][0].length-1);t[a]=this.colorMap[e][2]}return t},getData:function(){var t,e=(null===(t=this.card)||void 0===t?void 0:t.data)||[],a=e.map((function(t){return{name:t[0],value:t[1]}}));return a}}}),d=c,n=(a("3617"),a("2877")),l=Object(n["a"])(d,i,r,!1,null,"78b513c7",null);e["default"]=l.exports},"689b":function(t,e,a){}}]);