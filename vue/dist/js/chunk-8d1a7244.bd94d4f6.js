(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-8d1a7244","chunk-58f9b478"],{"07ce":function(t,e,a){},"0da8":function(t,e,a){},1039:function(t,e,a){"use strict";a.r(e);var r=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"t-text-segmented",style:{width:t.block_width}},[a("div",{staticClass:"t-text-segmented__content"},t._l(t.arrText,(function(e,r){var s=e.tags,i=e.word;return a("div",{key:"word_"+r,staticClass:"t-text-segmented__word"},[s.includes("p1")?a("div",{staticClass:"t-text-segmented__text"},[t._v(t._s(i))]):a("at-tooltip",[a("div",{staticClass:"t-text-segmented__text"},[t._v(t._s(i))]),a("template",{slot:"content"},t._l(s,(function(e){return a("div",{key:"colors_"+e,staticClass:"t-text-segmented__colors",style:"background-color: rgb("+t.rgb(e)+");"},[t._v(t._s(e))])})),0)],2),t._l(s,(function(e,s){return a("div",{key:"tags_"+r+"_"+s,staticClass:"t-text-segmented__line",style:"background-color: rgb("+t.rgb(e)+");"})}))],2)})),0)])},s=[],i=(a("ac1f"),a("466d"),a("d81d"),a("a15b"),a("5319"),a("1276"),{name:"TableTextSegmented",props:{value:{type:String,default:""},color_mark:{type:Array,default:function(){return[]}},tags_color:{type:Object,default:function(){}},layer:{type:String,default:""},block_width:{type:String,default:"400px"}},computed:{tags:function(){var t;return(null===(t=this.tags_color)||void 0===t?void 0:t[this.layer])||{}},arrText:function(){var t=this,e=this.value.replaceAll(" ",""),a=e.match(/(<[sp][0-9]>)+([^<\/>]+)(<\/[sp][0-9]>)+/g);return a.map((function(e){return t.convert(e)}))}},methods:{rgb:function(t){var e=this.tags[t]||[];return e.join(" ")},convert:function(t){t=t.replace(/(<\/[^>]+>)+/g,"");var e=t.replace(/(<[^>]+>)+/g,"");t=t.replace(/></g,",");var a=t.match(/<(.+)>/)[1].split(",");return{tags:a,word:e}}}}),n=i,c=(a("ca51"),a("2877")),l=Object(c["a"])(n,r,s,!1,null,"be918f14",null);e["default"]=l.exports},"3af9":function(t,e,a){"use strict";a.r(e);var r=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"card"},[a("div",{staticClass:"card__content"},["ImageClassification"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("ImgCard",{attrs:{imgUrl:t.card.source}})],1),a("div",{staticClass:"card__result"},[a("TextCard",{style:{width:"224px",height:"80px"}},[t._v(t._s(t.ClassificationResult))])],1)]):t._e(),"TextClassification"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("TextCard",{style:{width:"600px",color:"#A7BED3",height:"324px"}},[t._v(t._s(t.card.source))])],1),a("div",{staticClass:"card__result"},[a("TextCard",{style:{width:"600px",height:"80px"}},[t._v(t._s(t.ClassificationResult))])],1)]):t._e(),"TextSegmentation"==t.type?a("div",[a("div",{staticClass:"card__original segmentation__original",style:{height:"324px"}},[a("scrollbar",{attrs:{ops:t.ops}},[a("TableTextSegmented",t._b({key:t.RandId},"TableTextSegmented",{value:t.card.format,tags_color:{segmentationLayer:t.segmentationLayer},layer:"segmentationLayer",block_width:"598px"},!1))],1)],1),a("div",{staticClass:"card__result"},[a("SegmentationTags",{style:{width:"600px",height:"80px"},attrs:{tags:t.segmentationLayer}})],1)]):t._e(),"AudioClassification"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("AudioCard",{attrs:{value:t.card.source,update:t.RandId}})],1),a("div",{staticClass:"card__result"},[a("TextCard",{style:{width:"600px",height:"80px"}},[t._v(t._s(t.ClassificationResult))])],1)]):t._e(),"ImageSegmentation"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("ImgCard",{attrs:{imgUrl:t.card.source}})],1),a("div",{staticClass:"card__result"},[a("ImgCard",{attrs:{imgUrl:t.card.segment}})],1)]):t._e(),"graphic"==t.type?a("div",{staticClass:"card__graphic"},[a("Plotly",{attrs:{data:t.card.data,layout:t.layout,"display-mode-bar":!1}})],1):t._e()]),a("div",{staticClass:"card__reload"},[a("button",{staticClass:"btn-reload",on:{click:t.ReloadCard}},[a("i",{class:["t-icon","icon-deploy-reload"],attrs:{title:"reload"}})])])])},s=[],i=a("5530"),n=(a("a9e3"),a("d3b7"),a("25f0"),a("caad"),a("2532"),a("fb6a"),a("4e82"),a("99af"),function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"img-card"},[a("img",{staticClass:"img-card__image",attrs:{src:t.src}})])}),c=[],l=a("2f62"),o={name:"ImgCard",props:{imgUrl:{type:String,default:"img.png"}},computed:Object(i["a"])(Object(i["a"])({},Object(l["b"])({id:"deploy/getRandId"})),{},{src:function(){return"/_media/blank/?path=".concat(this.imgUrl,"&r=").concat(this.id)}})},d=o,u=(a("7e8a"),a("2877")),g=Object(u["a"])(d,n,c,!1,null,"5ab7da04",null),p=g.exports,_=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"text-card"},[a("scrollbar",{attrs:{ops:t.ops}},[a("pre",[t._t("default",(function(){return[t._v("TEXT")]}))],2)])],1)},f=[],v={name:"TextCard",props:{text:{type:String,default:"text"}},data:function(){return{ops:{scrollPanel:{scrollingX:!1,scrollingY:!0}}}}},h=v,m=(a("f4ce"),Object(u["a"])(h,_,f,!1,null,"6e13d9d6",null)),y=m.exports,x=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"item"},[a("div",{staticClass:"audio"},[a("div",{staticClass:"audio__card"},[a("av-waveform",{attrs:{"canv-width":500,"canv-height":23,"played-line-color":"#65B9F4","noplayed-line-color":"#2B5278","played-line-width":0,playtime:!1,"canv-top":!0,"canv-class":"custom-player","canv-fill-color":"#2B5278","audio-src":t.src}})],1)])])},b=[],C={name:"AudioCard",props:{value:{type:String,default:""},update:{type:String,default:""}},computed:{src:function(){return"/_media/blank/?path=".concat(this.value,"&r=").concat(this.update)}},mounted:function(){console.log(this.update),this.$el.querySelector("audio").setAttribute("controlsList","nodownload")}},T=C,w=(a("7bf5"),Object(u["a"])(T,x,b,!1,null,null,null)),S=w.exports,k=a("1039"),j=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"card"},[a("scrollbar",{attrs:{ops:t.ops}},[a("div",{staticClass:"card__text"},t._l(t.tags,(function(e,r,s){return a("div",{key:"tag-line_"+s,staticClass:"card__text--line"},[a("p",{style:{"background-color":t.rgb(e)}},[t._v(t._s(r))]),t._v(" - Название "+t._s(s)+" ")])})),0)])],1)},O=[],I={name:"SegmentationTags",props:{tags:{type:Object,default:function(){return{}}}},methods:{rgb:function(t){var e="#";for(var a in t){var r=t[a].toString(16);e+=1==r.length?"0"+r:r}return e}},data:function(){return{ops:{scrollPanel:{scrollingX:!1,scrollingY:!0}}}}},R=I,A=(a("cd1e"),Object(u["a"])(R,j,O,!1,null,"37c346ca",null)),L=A.exports,E=a("04d11"),$={name:"IndexCard",components:{ImgCard:p,TextCard:y,Plotly:E["Plotly"],AudioCard:S,TableTextSegmented:k["default"],SegmentationTags:L},data:function(){return{ops:{scrollPanel:{scrollingX:!1,scrollingY:!0}}}},props:{card:{type:Object,default:function(){return{}}},index:[String,Number],extra:{type:Array,default:function(){return[]}}},methods:{ReloadCard:function(){this.$emit("reload",[this.index.toString()])}},computed:Object(i["a"])(Object(i["a"])({},Object(l["b"])({graphicData:"deploy/getGraphicData",defaultLayout:"deploy/getDefaultLayout",origTextStyle:"deploy/getOrigTextStyle",type:"deploy/getDeployType",RandId:"deploy/getRandId"})),{},{layout:function(){var t=this.defaultLayout;return this.char&&(t.title.text=this.char.title||"",t.xaxis.title=this.char.xaxis.title||"",t.yaxis.title=this.char.yaxis.title||""),t},segmentationLayer:function(){var t={};for(var e in this.extra)if(!this.extra[e][0].includes("p")){var a=this.extra[e][0].slice(1,this.extra[e][0].length-1);t[a]=this.extra[e][2]}return t},ClassificationResult:function(){var t=this.card.data,e="";t.sort((function(t,e){return t[1]<e[1]?1:-1}));for(var a=0;a<t.length;a++)e+="".concat(t[a][0]," - ").concat(t[a][1],"% \n");return e}})},P=$,D=(a("ed70"),Object(u["a"])(P,r,s,!1,null,"9e2bda20",null));e["default"]=D.exports},"7bf5":function(t,e,a){"use strict";a("bf44")},"7e8a":function(t,e,a){"use strict";a("0da8")},"9a1e":function(t,e,a){},b022:function(t,e,a){},bf44:function(t,e,a){},ca51:function(t,e,a){"use strict";a("b022")},cd1e:function(t,e,a){"use strict";a("de5c")},de5c:function(t,e,a){},ed70:function(t,e,a){"use strict";a("9a1e")},f4ce:function(t,e,a){"use strict";a("07ce")}}]);