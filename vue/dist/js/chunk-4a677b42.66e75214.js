(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-4a677b42","chunk-39c73cc3"],{"07ce":function(t,e,a){},"0da8":function(t,e,a){},1039:function(t,e,a){"use strict";a.r(e);var r=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"t-text-segmented",style:{width:t.block_width}},[a("div",{staticClass:"t-text-segmented__content"},t._l(t.arrText,(function(e,r){var i=e.tags,s=e.word;return a("div",{key:"word_"+r,staticClass:"t-text-segmented__word"},[i.includes("p1")?a("div",{staticClass:"t-text-segmented__text"},[t._v(t._s(s))]):a("at-tooltip",[a("div",{staticClass:"t-text-segmented__text"},[t._v(t._s(s))]),a("template",{slot:"content"},t._l(i,(function(e){return a("div",{key:"colors_"+e,staticClass:"t-text-segmented__colors",style:"background-color: rgb("+t.rgb(e)+");"},[t._v(t._s(e))])})),0)],2),t._l(i,(function(e,i){return a("div",{key:"tags_"+r+"_"+i,staticClass:"t-text-segmented__line",style:"background-color: rgb("+t.rgb(e)+");"})}))],2)})),0)])},i=[],s=(a("5b81"),a("ac1f"),a("466d"),a("d81d"),a("a15b"),a("5319"),a("1276"),{name:"TableTextSegmented",props:{value:{type:String,default:""},color_mark:{type:Array,default:function(){return[]}},tags_color:{type:Object,default:function(){}},layer:{type:String,default:""},block_width:{type:String,default:"400px"}},computed:{tags:function(){var t;return(null===(t=this.tags_color)||void 0===t?void 0:t[this.layer])||{}},arrText:function(){var t=this,e=this.value.replaceAll(" ",""),a=e.match(/(<[sp][0-9]>)+([^<\/>]+)(<\/[sp][0-9]>)+/g);return a.map((function(e){return t.convert(e)}))}},methods:{rgb:function(t){var e=this.tags[t]||[];return e.join(" ")},convert:function(t){t=t.replace(/(<\/[^>]+>)+/g,"");var e=t.replace(/(<[^>]+>)+/g,"");t=t.replace(/></g,",");var a=t.match(/<(.+)>/)[1].split(",");return{tags:a,word:e}}}}),n=s,l=(a("ca51"),a("2877")),c=Object(l["a"])(n,r,i,!1,null,"be918f14",null);e["default"]=c.exports},"1e6f":function(t,e,a){"use strict";a("b55a")},"3af9":function(t,e,a){"use strict";a.r(e);var r=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"card"},[a("div",{staticClass:"card__content"},["ImageClassification"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("ImgCard",{attrs:{imgUrl:t.card.source}})],1),a("div",{staticClass:"card__result"},[a("TextCard",{style:{width:"224px",height:"80px"}},[t._v(t._s(t.ClassificationResult))])],1)]):t._e(),"TextClassification"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("TextCard",{style:{width:"600px",color:"#A7BED3",height:"324px"}},[t._v(t._s(t.card.source))])],1),a("div",{staticClass:"card__result"},[a("TextCard",{style:{width:"600px",height:"80px"}},[t._v(t._s(t.ClassificationResult))])],1)]):t._e(),"TextSegmentation"==t.type?a("div",[a("div",{staticClass:"card__original segmentation__original",style:{height:"324px"}},[a("scrollbar",{attrs:{ops:t.ops}},[a("TableTextSegmented",t._b({key:t.RandId},"TableTextSegmented",{value:t.card.format,tags_color:{segmentationLayer:t.segmentationLayer},layer:"segmentationLayer",block_width:"598px"},!1))],1)],1),a("div",{staticClass:"card__result"},[a("SegmentationTags",{style:{width:"600px",height:"80px"},attrs:{tags:t.segmentationLayer}})],1)]):t._e(),"AudioClassification"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("AudioCard",{attrs:{value:t.card.source,update:t.RandId}})],1),a("div",{staticClass:"card__result"},[a("TextCard",{style:{width:"600px",height:"80px"}},[t._v(t._s(t.ClassificationResult))])],1)]):t._e(),"ImageSegmentation"==t.type?a("div",[a("div",{staticClass:"card__original"},[a("ImgCard",{attrs:{imgUrl:t.card.source}})],1),a("div",{staticClass:"card__result"},[a("ImgCard",{attrs:{imgUrl:t.card.segment}})],1)]):t._e(),"graphic"==t.type?a("div",{staticClass:"card__graphic"},[a("Plotly",{attrs:{data:t.card.data,layout:t.layout,"display-mode-bar":!1}})],1):t._e()]),a("div",{staticClass:"card__reload"},[a("button",{staticClass:"btn-reload",on:{click:t.ReloadCard}},[a("i",{class:["t-icon","icon-deploy-reload"],attrs:{title:"reload"}})])])])},i=[],s=a("5530"),n=(a("a9e3"),a("d3b7"),a("25f0"),a("caad"),a("2532"),a("fb6a"),a("4e82"),a("99af"),function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"img-card"},[a("img",{staticClass:"img-card__image",attrs:{src:t.src}})])}),l=[],c=a("2f62"),o={name:"ImgCard",props:{imgUrl:{type:String,default:"img.png"}},computed:Object(s["a"])(Object(s["a"])({},Object(c["b"])({id:"deploy/getRandId"})),{},{src:function(){return"/_media/blank/?path=".concat(this.imgUrl,"&r=").concat(this.id)}})},d=o,u=(a("7e8a"),a("2877")),g=Object(u["a"])(d,n,l,!1,null,"5ab7da04",null),p=g.exports,f=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"text-card"},[a("scrollbar",{attrs:{ops:t.ops}},[a("pre",[t._t("default",(function(){return[t._v("TEXT")]}))],2)])],1)},_=[],v={name:"TextCard",props:{text:{type:String,default:"text"}},data:function(){return{ops:{scrollPanel:{scrollingX:!1,scrollingY:!0}}}}},m=v,h=(a("f4ce"),Object(u["a"])(m,f,_,!1,null,"6e13d9d6",null)),y=h.exports,b=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"item"},[a("div",{staticClass:"audio"},[a("div",{staticClass:"audio__card"},[a("av-waveform",{attrs:{"canv-width":500,"canv-height":23,"played-line-color":"#65B9F4","noplayed-line-color":"#2B5278","played-line-width":0,playtime:!1,"canv-top":!0,"canv-class":"custom-player","canv-fill-color":"#2B5278","audio-src":t.src}})],1)])])},x=[],C={name:"AudioCard",props:{value:{type:String,default:""},update:{type:String,default:""}},computed:{src:function(){return"/_media/blank/?path=".concat(this.value,"&r=").concat(this.update)}},mounted:function(){console.log(this.update),this.$el.querySelector("audio").setAttribute("controlsList","nodownload")}},T=C,w=(a("7bf5"),Object(u["a"])(T,b,x,!1,null,null,null)),S=w.exports,k=a("1039"),O=function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("div",{staticClass:"card"},[a("scrollbar",{attrs:{ops:t.ops}},[a("div",{staticClass:"card__text"},t._l(t.tags,(function(e,r,i){return a("div",{key:"tag-line_"+i,staticClass:"card__text--line"},[a("p",{style:{"background-color":t.rgb(e)}},[t._v(t._s(r))]),t._v(" - Название "+t._s(i)+" ")])})),0)])],1)},j=[],I={name:"SegmentationTags",props:{tags:{type:Object,default:function(){return{}}}},methods:{rgb:function(t){var e="#";for(var a in t){var r=t[a].toString(16);e+=1==r.length?"0"+r:r}return e}},data:function(){return{ops:{scrollPanel:{scrollingX:!1,scrollingY:!0}}}}},R=I,A=(a("cd1e"),Object(u["a"])(R,O,j,!1,null,"37c346ca",null)),E=A.exports,L=a("04d11"),$={name:"IndexCard",components:{ImgCard:p,TextCard:y,Plotly:L["Plotly"],AudioCard:S,TableTextSegmented:k["default"],SegmentationTags:E},data:function(){return{ops:{scrollPanel:{scrollingX:!1,scrollingY:!0}}}},props:{card:{type:Object,default:function(){return{}}},index:[String,Number],color_map:{type:Array,default:function(){return[]}}},methods:{ReloadCard:function(){this.$emit("reload",[this.index.toString()])}},computed:Object(s["a"])(Object(s["a"])({},Object(c["b"])({graphicData:"deploy/getGraphicData",defaultLayout:"deploy/getDefaultLayout",origTextStyle:"deploy/getOrigTextStyle",type:"deploy/getDeployType",RandId:"deploy/getRandId"})),{},{layout:function(){var t=this.defaultLayout;return this.char&&(t.title.text=this.char.title||"",t.xaxis.title=this.char.xaxis.title||"",t.yaxis.title=this.char.yaxis.title||""),t},segmentationLayer:function(){var t={};for(var e in this.color_map)if(!this.color_map[e][0].includes("p")){var a=this.color_map[e][0].slice(1,this.color_map[e][0].length-1);t[a]=this.color_map[e][2]}return t},ClassificationResult:function(){var t=this.card.data,e="";t.sort((function(t,e){return t[1]<e[1]?1:-1}));for(var a=0;a<t.length;a++)e+="".concat(t[a][0]," - ").concat(t[a][1],"% \n");return e}}),mounted:function(){console.log(this.color_map)}},P=$,D=(a("1e6f"),Object(u["a"])(P,r,i,!1,null,"e241e6e4",null));e["default"]=D.exports},"5b81":function(t,e,a){"use strict";var r=a("23e7"),i=a("1d80"),s=a("1626"),n=a("44e7"),l=a("577e"),c=a("dc4a"),o=a("ad6d"),d=a("0cb2"),u=a("b622"),g=a("c430"),p=u("replace"),f=RegExp.prototype,_=Math.max,v=function(t,e,a){return a>t.length?-1:""===e?a:t.indexOf(e,a)};r({target:"String",proto:!0},{replaceAll:function(t,e){var a,r,u,m,h,y,b,x,C,T=i(this),w=0,S=0,k="";if(null!=t){if(a=n(t),a&&(r=l(i("flags"in f?t.flags:o.call(t))),!~r.indexOf("g")))throw TypeError("`.replaceAll` does not allow non-global regexes");if(u=c(t,p),u)return u.call(t,T,e);if(g&&a)return l(T).replace(t,e)}m=l(T),h=l(t),y=s(e),y||(e=l(e)),b=h.length,x=_(1,b),w=v(m,h,0);while(-1!==w)C=y?l(e(h,w,m)):d(h,m,w,[],void 0,e),k+=m.slice(S,w)+C,S=w+b,w=v(m,h,w+x);return S<m.length&&(k+=m.slice(S)),k}})},"7bf5":function(t,e,a){"use strict";a("bf44")},"7e8a":function(t,e,a){"use strict";a("0da8")},b022:function(t,e,a){},b55a:function(t,e,a){},bf44:function(t,e,a){},ca51:function(t,e,a){"use strict";a("b022")},cd1e:function(t,e,a){"use strict";a("de5c")},de5c:function(t,e,a){},f4ce:function(t,e,a){"use strict";a("07ce")}}]);